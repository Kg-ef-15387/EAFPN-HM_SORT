import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        leaky=True
):
    stage = nn.Sequential()
    padding = (kernel_size - 1) // 2

    stage.add_module('conv',
                     nn.Conv2d(
                         in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=(kernel_size, kernel_size),
                         stride=(stride, stride),
                         padding=(padding, padding),
                         bias=False
                              )
                     )

    stage.add_module('bn',
                     nn.BatchNorm2d(out_channels)
                     )

    if leaky:
        stage.add_module('leaky',
                        nn.LeakyReLU(0.1)
                         )

    else:
        stage.add_module('relu6',
                         nn.ReLU6(inplace=True)

                         )

    return stage


class SFFM_Four_Level(nn.Module):
    def __init__(
            self,
            fea_channel_list=[256, 512, 1024, 2048],
            cur_level=0,
            rfb=False,
            vis=False
    ):
        super(SFFM_Four_Level, self).__init__()

        assert len(fea_channel_list) == 4, 'The number of output feature maps is not 4!'

        self.fea_channel_list = fea_channel_list
        self.cur_level = cur_level
        self.cur_channel = self.fea_channel_list[self.cur_level]
        self.vis = vis

        # level 0
        if self.cur_level == 0:

            self.channel_set_1 = conv_block(
                in_channels=self.fea_channel_list[1],
                out_channels=self.cur_channel,
                kernel_size=1,
                stride=1
            )

            self.channel_set_2 = conv_block(
                in_channels=self.fea_channel_list[2],
                out_channels=self.cur_channel,
                kernel_size=1,
                stride=1
            )

            self.channel_set_3 = conv_block(
                in_channels=self.fea_channel_list[3],
                out_channels=self.cur_channel,
                kernel_size=1,
                stride=1
            )

            self.expand = conv_block(
                in_channels=self.cur_channel,
                out_channels=self.cur_channel,
                kernel_size=3,
                stride=1
            )

        # level 1
        elif self.cur_level == 1:

            self.down_sample_0 = conv_block(
                in_channels=self.fea_channel_list[0],
                out_channels=self.cur_channel,
                kernel_size=3,
                stride=2
            )

            self.channel_set_2 = conv_block(
                in_channels=self.fea_channel_list[2],
                out_channels=self.cur_channel,
                kernel_size=1,
                stride=1
            )

            self.channel_set_3 = conv_block(
                in_channels=self.fea_channel_list[3],
                out_channels=self.cur_channel,
                kernel_size=1,
                stride=1
            )

            self.expand = conv_block(
                in_channels=self.cur_channel,
                out_channels=self.cur_channel,
                kernel_size=3,
                stride=1
            )

        # level 2
        elif self.cur_level == 2:

            self.down_sample_0 = conv_block(
                in_channels=self.fea_channel_list[0],
                out_channels=self.cur_channel,
                kernel_size=3,
                stride=2
            )

            self.down_sample_1 = conv_block(
                in_channels=self.fea_channel_list[1],
                out_channels=self.cur_channel,
                kernel_size=3,
                stride=2
            )

            self.channel_set_3 = conv_block(
                in_channels=self.fea_channel_list[3],
                out_channels=self.cur_channel,
                kernel_size=1,
                stride=1
            )

            self.expand = conv_block(
                in_channels=self.cur_channel,
                out_channels=self.cur_channel,
                kernel_size=3,
                stride=1
            )

        # level 3
        elif self.cur_level == 3:

            self.down_sample_0 = nn.Sequential(
                conv_block(
                    in_channels=self.fea_channel_list[0],
                    out_channels=self.cur_channel,
                    kernel_size=3,
                    stride=2
                ),
                conv_block(
                    in_channels=self.cur_channel,
                    out_channels=self.cur_channel,
                    kernel_size=3,
                    stride=2
                )
            )

            self.down_sample_1 = conv_block(
                in_channels=self.fea_channel_list[1],
                out_channels=self.cur_channel,
                kernel_size=3,
                stride=2
            )

            self.down_sample_2 = conv_block(
                in_channels=self.fea_channel_list[2],
                out_channels=self.cur_channel,
                kernel_size=3,
                stride=2
            )

            self.expand = conv_block(
                in_channels=self.cur_channel,
                out_channels=self.cur_channel,
                kernel_size=3,
                stride=1
            )

        compress_c = 8 if rfb else 16

        self.weight_0 = conv_block(
            in_channels=self.cur_channel,
            out_channels=compress_c,
            kernel_size=1,
            stride=1
        )
        self.weight_1 = conv_block(
            in_channels=self.cur_channel,
            out_channels=compress_c,
            kernel_size=1,
            stride=1
        )
        self.weight_2 = conv_block(
            in_channels=self.cur_channel,
            out_channels=compress_c,
            kernel_size=1,
            stride=1
        )

        self.weight_3 = conv_block(
            in_channels=self.cur_channel,
            out_channels=compress_c,
            kernel_size=1,
            stride=1
        )

        self.weight_levels = nn.Conv2d(
            in_channels=compress_c * 4,
            out_channels=4,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0
        )

    def forward(self, fpn_fea_list):

        x_level_0, x_level_1, x_level_2, x_level_3 = fpn_fea_list[0], fpn_fea_list[1], fpn_fea_list[2], fpn_fea_list[3]

        # level 0
        if self.cur_level == 0:

            level_0_resized = x_level_0

            level_1_compressed = self.channel_set_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')

            level_2_compressed = self.channel_set_2(x_level_2)
            level_2_resized = F.interpolate(level_2_compressed, scale_factor=4, mode='nearest')

            level_3_compressed = self.channel_set_3(x_level_3)
            level_3_resized = F.interpolate(level_3_compressed, scale_factor=8, mode='nearest')

        # level 1
        elif self.cur_level == 1:

            level_0_resized = self.down_sample_0(x_level_0)

            level_1_resized = x_level_1

            level_2_compressed = self.channel_set_2(x_level_2)
            level_2_resized = F.interpolate(level_2_compressed, scale_factor=2, mode='nearest')

            level_3_compressed = self.channel_set_3(x_level_3)
            level_3_resized = F.interpolate(level_3_compressed, scale_factor=4, mode='nearest')

        # level 2
        elif self.cur_level == 2:

            level_0_downsampled = F.max_pool2d(x_level_0, kernel_size=3, stride=2, padding=1)
            level_0_resized = self.down_sample_0(level_0_downsampled)

            level_1_resized = self.down_sample_1(x_level_1)

            level_2_resized = x_level_2

            level_3_compressed = self.channel_set_3(x_level_3)
            level_3_resized = F.interpolate(level_3_compressed, scale_factor=2, mode='nearest')

        # level 3
        elif self.cur_level == 3:

            level_0_downsampled = F.max_pool2d(x_level_0, kernel_size=3, stride=2, padding=1)
            level_0_resized = self.down_sample_0(level_0_downsampled)

            level_1_downsampled = F.max_pool2d(x_level_1, kernel_size=3, stride=2, padding=1)
            level_1_resized = self.down_sample_1(level_1_downsampled)

            level_2_resized = self.down_sample_2(x_level_2)

            level_3_resized = x_level_3

        level_0_weight_v = self.weight_0(level_0_resized)
        level_1_weight_v = self.weight_1(level_1_resized)
        level_2_weight_v = self.weight_2(level_2_resized)
        level_3_weight_v = self.weight_3(level_3_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:3, :, :] + \
                            level_3_resized * levels_weight[:, 3:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
