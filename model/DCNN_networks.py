"""
Copyright (c) 2018 Maria Francesca Spadea.
- All Rights Reserved -

Unauthorized copying/distributing/editing/using/selling of this file (also partial), via any medium, is strictly prohibited.

The code is proprietary and confidential.

The software is just for research purpose, it is not intended for clinical application or for use in which the failure of the software
could lead to death, personal injury, or severe physical or environmental damage.
"""

import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, num_channels, initial_features, dropout):
        super(Generator, self).__init__()

        self.conv_down1 = SingleConv3x3(num_channels, initial_features, dropout=0)
        self.conv_down2 = SingleConv3x3(initial_features, initial_features, dropout=0)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv_down3_4 = DoubleConv3x3(initial_features, initial_features*2, dropout=0)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv_down5_6_7 = TripleConv3x3(initial_features*2, initial_features*(2**2), dropout=0)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv_down8_9_10 = TripleConv3x3(initial_features*(2**2), initial_features*(2**3), dropout)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv_down11_12_13 = TripleConv3x3(initial_features*(2**3), initial_features*(2**3), dropout)

        self.conv_down14_15_16 = TripleConv3x3(initial_features*(2**3), initial_features*(2**3), dropout)

        self.upconcat1 = UpConcat(initial_features*(2**3), initial_features*(2**3))

        self.conv_up1_2_3 = TripleConv3x3(initial_features*(2**3)+initial_features*(2**3), initial_features*(2**2), dropout)

        self.upconcat2 = UpConcat(initial_features*(2**3), initial_features*(2**2))

        self.conv_up4_5_6 = TripleConv3x3(initial_features*(2**2)+initial_features*(2**2), initial_features*2, dropout)

        self.upconcat3 = UpConcat(initial_features*(2**2), initial_features*2)

        self.conv_up7_8 = DoubleConv3x3(initial_features*2+initial_features*2, initial_features, dropout=0)

        self.upconcat4 = UpConcat(initial_features*2, initial_features)

        self.conv_up9_10 = DoubleConv3x3(initial_features+initial_features, initial_features, dropout=0)

        self.final = SingleConv1x1(initial_features, dropout=0)

    def forward(self, inputs):

        conv_down1_feat = self.conv_down1(inputs)
        conv_down2_feat = self.conv_down2(conv_down1_feat) ####
        maxpool1_feat = self.maxpool1(conv_down2_feat)

        conv_down3_4_feat = self.conv_down3_4(maxpool1_feat) ###
        maxpool2_feat = self.maxpool2(conv_down3_4_feat)

        conv_down5_6_7_feat = self.conv_down5_6_7(maxpool2_feat) ##
        maxpool3_feat = self.maxpool3(conv_down5_6_7_feat)

        conv_down8_9_10_feat = self.conv_down8_9_10(maxpool3_feat) #
        maxpool4_feat = self.maxpool4(conv_down8_9_10_feat)

        conv_down11_12_13_feat = self.conv_down11_12_13(maxpool4_feat)
        conv_down14_15_16_feat = self.conv_down14_15_16(conv_down11_12_13_feat)

        upconcat1_feat = self.upconcat1(conv_down14_15_16_feat, conv_down8_9_10_feat)
        conv_up1_2_3_feat = self.conv_up1_2_3(upconcat1_feat)

        upconcat2_feat = self.upconcat2(conv_up1_2_3_feat, conv_down5_6_7_feat)
        conv_up4_5_6_feat = self.conv_up4_5_6(upconcat2_feat)

        upconcat3_feat = self.upconcat3(conv_up4_5_6_feat, conv_down3_4_feat)
        conv_up7_8_feat = self.conv_up7_8(upconcat3_feat)

        upconcat4_feat = self.upconcat4(conv_up7_8_feat, conv_down2_feat)
        conv_up9_10_feat = self.conv_up9_10(upconcat4_feat)

        outputs = self.final(conv_up9_10_feat)

        return outputs

class SingleConv1x1(nn.Module):
    def __init__(self, in_feat, dropout):
        super(SingleConv1x1, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, 1,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0),
                                   nn.Dropout2d(dropout))

    def forward(self, inputs):
        outputs = self.conv1(inputs)

        return outputs


class SingleConv3x3(nn.Module):
    def __init__(self, in_feat, out_feat, dropout):
        super(SingleConv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.Dropout2d(dropout),
                                   nn.LeakyReLU(0.2, True))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class DoubleConv3x3(nn.Module):
    def __init__(self, in_feat, out_feat, dropout):
        super(DoubleConv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, in_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout2d(dropout),
                                   nn.BatchNorm2d(in_feat),
                                   nn.LeakyReLU(0.2, True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout2d(dropout),
                                   nn.BatchNorm2d(out_feat),
                                   nn.LeakyReLU(0.2, True))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class TripleConv3x3(nn.Module):
    def __init__(self, in_feat, out_feat, dropout):
        super(TripleConv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, in_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout2d(dropout),
                                   nn.BatchNorm2d(in_feat),
                                   nn.LeakyReLU(0.2, True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_feat, in_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout2d(dropout),
                                   nn.BatchNorm2d(in_feat),
                                   nn.LeakyReLU(0.2, True))

        self.conv3 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout2d(dropout),
                                   nn.BatchNorm2d(out_feat),
                                   nn.LeakyReLU(0.2, True))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs

class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.deconv = nn.ConvTranspose2d(out_feat, out_feat, kernel_size=4, stride=2, padding=1)

        #self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        #outputs = self.up(inputs)

        outputs = self.deconv(inputs)

        out = torch.cat([down_outputs, outputs], 1)
        return out


