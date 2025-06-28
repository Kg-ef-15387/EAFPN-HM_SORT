import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=4, flag=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, out_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x if self.flag else self.sigmoid(out)


class PAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PAM, self).__init__()

        self.down1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_bn1 = nn.BatchNorm2d(out_channels)
        self.down2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_bn2 = nn.BatchNorm2d(out_channels)
        self.down3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_bn3 = nn.BatchNorm2d(out_channels)

        self.CA1 = ChannelAttention(in_planes=in_channels, out_planes=out_channels, flag=False)
        self.CA2 = ChannelAttention(in_planes=in_channels, out_planes=out_channels, flag=False)
        self.CA3 = ChannelAttention(in_planes=in_channels, out_planes=out_channels, flag=False)

        self.aug_feat1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.aug_bn1 = nn.BatchNorm2d(out_channels)
        self.aug_feat2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.aug_bn2 = nn.BatchNorm2d(out_channels)
        self.aug_feat3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.aug_bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        input_0, input_1, input_2, input_3 = inputs[0], inputs[1], inputs[2], inputs[3]
        output_0 = input_0

        # SFF1
        output_1 = F.relu(self.down_bn1(self.down1(input_0)))
        att1 = self.CA1(output_1)
        input_1 = input_1 * att1
        output_1 = output_1 + input_1
        output_1 = F.relu(self.aug_bn1(self.aug_feat1(output_1)))

        # SFF2
        output_2 = F.relu(self.down_bn1(self.down1(output_1)))
        att2 = self.CA1(output_2)
        input_2 = input_2 * att2
        output_2 = output_2 + input_2
        output_2 = F.relu(self.aug_bn2(self.aug_feat2(output_2)))

        # SFF3
        output_3 = F.relu(self.down_bn1(self.down1(output_2)))
        att3 = self.CA1(output_3)
        input_3 = input_3 * att3
        output_3 = output_3 + input_3
        output_3 = F.relu(self.aug_bn3(self.aug_feat3(output_3)))

        return [output_0, output_1, output_2, output_3]


class FPN_Path_Augmentation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN_Path_Augmentation, self).__init__()

        self.down1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_bn1 = nn.BatchNorm2d(out_channels)
        self.down2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_bn2 = nn.BatchNorm2d(out_channels)
        self.down3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_bn3 = nn.BatchNorm2d(out_channels)

        self.aug_feat1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.aug_bn1 = nn.BatchNorm2d(out_channels)
        self.aug_feat2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.aug_bn2 = nn.BatchNorm2d(out_channels)
        self.aug_feat3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.aug_bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        input_0, input_1, input_2, input_3 = inputs[0], inputs[1], inputs[2], inputs[3]

        output_0 = input_0
        output_1 = F.relu(self.down_bn1(self.down1(input_0))) + input_1
        output_1 = F.relu(self.aug_bn1(self.aug_feat1(output_1)))
        output_2 = F.relu(self.down_bn2(self.down2(output_1))) + input_2
        output_2 = F.relu(self.aug_bn2(self.aug_feat2(output_2)))
        output_3 = F.relu(self.down_bn3(self.down3(output_2))) + input_3
        output_3 = F.relu(self.aug_bn3(self.aug_feat3(output_3)))

        return [output_0, output_1, output_2, output_3]