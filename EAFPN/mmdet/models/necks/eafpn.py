import torch
from mmcv.cnn import kaiming_init, xavier_init
from mmcv.cnn import ConvModule
import torch.nn.functional as F
import torch.nn as nn
from utils.sffm import SFFM_Four_Level
from utils.pam import PAM
from utils.cfam import CFAM
from ..builder import NECKS
from mmcv.runner import auto_fp16



@NECKS.register_module()
class EA_FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 pool_ratios=[0.1, 0.2, 0.3],
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(EA_FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.adaptive_pool_output_ratio = pool_ratios

        self.SFFM_0 = SFFM_Four_Level(fea_channel_list=[256, 512, 1024, 2048], cur_level=0)
        self.SFFM_1 = SFFM_Four_Level(fea_channel_list=[256, 512, 1024, 2048], cur_level=1)
        self.SFFM_2 = SFFM_Four_Level(fea_channel_list=[256, 512, 1024, 2048], cur_level=2)
        self.SFFM_3 = SFFM_Four_Level(fea_channel_list=[256, 512, 1024, 2048], cur_level=3)

        self.path_aug = PAM(in_channels=256, out_channels=256)

        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        self.CFAM = CFAM(out_channels)

        # self.grads = {}
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.high_lateral_conv = nn.ModuleList()
        for i in range(len(self.adaptive_pool_output_ratio)):
            high_lateral_conv = ConvModule(
                in_channels=self.in_channels[-1],
                out_channels=self.out_channels,
                kernel_size=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False
            )
            self.high_lateral_conv.append(high_lateral_conv)

        self.high_lateral_conv_attention = nn.Sequential(
            nn.Conv2d(out_channels * (len(self.adaptive_pool_output_ratio)), out_channels, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, len(self.adaptive_pool_output_ratio), 3, padding=1)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function"""
        assert len(inputs) == len(self.in_channels)

        asff_out_0 = self.SFFM_0(inputs)
        asff_out_1 = self.SFFM_1(inputs)
        asff_out_2 = self.SFFM_2(inputs)
        asff_out_3 = self.SFFM_3(inputs)
        inputs = [asff_out_0, asff_out_1, asff_out_2, asff_out_3]

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        att_list = self.CFAM(laterals)

        # MFEM
        laterals = [(1 + att_list[i]) * laterals[i] for i in range(len(laterals))]

        # RFAM
        h, w = inputs[-1].size(2), inputs[-1].size(3)
        AdapPool_Features = []
        for i in range(len(self.adaptive_pool_output_ratio)):
            context = F.adaptive_avg_pool2d(
                input=inputs[-1],
                output_size=(
                    max(1, int(h * self.adaptive_pool_output_ratio[i])),
                    max(1, int(w * self.adaptive_pool_output_ratio[i]))
                )
            )
            context = self.high_lateral_conv[i](context)
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
            AdapPool_Features.append(context)

        Concat_AdapPool_Features = torch.cat(AdapPool_Features, dim=1)
        fusion_weights = self.high_lateral_conv_attention(Concat_AdapPool_Features)
        fusion_weights = torch.sigmoid(fusion_weights)
        adap_pool_fusion = 0
        for i in range(len(self.adaptive_pool_output_ratio)):
            adap_pool_fusion += torch.unsqueeze(fusion_weights[:, i, :, :], dim=1) * AdapPool_Features[i]

        laterals[-1] += adap_pool_fusion

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                # MFAM
                prev_shape = laterals[i - 1].shape[2:]
                att_2x = F.interpolate(att_list[i], size=prev_shape, **self.upsample_cfg)
                att_insec = att_list[i - 1] * att_2x
                select_gate = att_insec
                laterals[i - 1] = laterals[i - 1] + select_gate * F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # MFEM
        outs = [
            (1 + att_list[i]) * self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # PAM
        outs = self.path_aug(outs)

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)