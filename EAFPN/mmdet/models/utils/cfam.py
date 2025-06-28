import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.cnn import kaiming_init, xavier_init
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
            padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 2, 5, 1)):
        super().__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


class CFAM(nn.Module):
    def __init__(self, inplanes, reduction_ratio=1, fpn_lvl=4):
        super(CFAM, self).__init__()
        self.fpn_lvl = fpn_lvl
        self.dila_conv = nn.Sequential(nn.Conv2d(inplanes * fpn_lvl // reduction_ratio, inplanes // reduction_ratio,
                                                 kernel_size=3, stride=1, padding=1),
                                       ASPP(inplanes // reduction_ratio, inplanes // (4 * reduction_ratio)),
                                       CBAMLayer(channel=inplanes // reduction_ratio),
                                       nn.Conv2d(inplanes // reduction_ratio, inplanes // reduction_ratio,
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(inplanes // reduction_ratio),
                                       nn.ReLU(inplace=False)
                                       )
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        self.upsample_cfg = dict(mode='nearest')
        self.down_conv = nn.ModuleList()
        self.att_conv = nn.ModuleList()
        for i in range(self.fpn_lvl):
            self.att_conv.append(nn.Conv2d(inplanes // reduction_ratio,
                                           1,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1))
            if i == 0:
                down_stride = 1
            else:
                down_stride = 2
            self.down_conv.append(
                nn.Conv2d(inplanes // reduction_ratio, inplanes // reduction_ratio, kernel_size=3, stride=down_stride,
                          padding=1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    @auto_fp16()
    def forward(self, x):
        prev_shape = x[0].shape[2:]
        multi_feats = [x[0]]

        for i in range(1, len(x)):
            pyr_feats_2x = F.interpolate(x[i], size=prev_shape, **self.upsample_cfg)
            multi_feats.append(pyr_feats_2x)

        multi_feats = torch.cat(multi_feats, 1)
        lvl_fea = self.dila_conv(multi_feats)

        multi_atts = []

        for i in range(self.fpn_lvl):
            lvl_fea = self.down_conv[i](lvl_fea)
            lvl_att = self.att_conv[i](lvl_fea)
            multi_atts.append(self.sigmoid(lvl_att))

        return multi_atts
