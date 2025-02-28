import math
import torch
import torch.nn as nn
# from MobileNetV2 import MobileNetV2, InvertedResidual
# from dropblock import *
import math
import torch.nn.functional as F
# from dropblock import *
import torch
import torch.nn as nn
from network.Linknet_a.MobileNetV2 import MobileNetV2
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.distributions.uniform import Uniform

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class DropBlock3D(DropBlock2D):
    r"""Randomly zeroes 3D spatial blocks of the input tensor.

    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :],
                                  kernel_size=(self.block_size, self.block_size, self.block_size),
                                  stride=(1, 1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 3)


# class Drop(nn.Module):
#     # drop_rate : 1-keep_prob  (all droped feature points)
#     # block_size : drop掉的block大小
#     def __init__(self, drop_rate=0.1, block_size=7):
#         super(Drop, self).__init__()
#
#         self.drop_rate = drop_rate
#         self.block_size = block_size
#
#     def forward(self, x):
#         if self.drop_rate == 0:
#             return x
#         # 设置gamma,比gamma小的设置为1,大于gamma的为0（得到丢弃比率的随机点个数）算法第五步
#         # all droped feature center points
#         gamma = self.drop_rate / (self.block_size ** 2)
#         # torch.rand(*sizes, out=None) : 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
#         mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
#
#         mask = mask.to(x.device)
#
#         # compute block mask
#         block_mask = self._compute_block_mask(mask)
#         # apply block mask,为算法图的第六步
#         out = x * block_mask[:, None, :, :]
#         # Normalize the features,对应第七步
#         out = out * block_mask.numel() / block_mask.sum()
#         return out
#
#     def _compute_block_mask(self, mask):
#         # 取最大值,这样就能够取出一个block的块大小的1作为drop,当然需要翻转大小,使得1为0,0为1
#         block_mask = F.max_pool2d(input=mask[:, None, :, :],
#                                   kernel_size=(self.block_size,
#                                                self.block_size),
#                                   stride=(1, 1),
#                                   padding=self.block_size // 2)
#         if self.block_size % 2 == 0:
#             # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
#             block_mask = block_mask[:, :, :-1, :-1]
#         block_mask = 1 - block_mask.squeeze(1)
#         return block_mask


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PSAModule(nn.Module):
    #  k =3    5  7   9
    #  G = 1   4  8   16

    def __init__(self, inplans, planes, conv_kernels=[3, 9, 9, 9], stride=1, conv_groups=[1, 16, 16, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class EncodeBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodeBlock1, self).__init__()
        self.conv1_1 = ConvBlock(in_channels, out_channels, stride=2)
        self.conv1_2 = ConvBlock(out_channels, out_channels)
        self.conv2_1 = ConvBlock(out_channels, out_channels)
        self.conv2_2 = ConvBlock(out_channels, out_channels)
        self.shortcut = ConvBlock(in_channels, out_channels, stride=2)

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.conv2_1(out1)
        residue = self.shortcut(x)
        out2 = self.conv2_1(out1 + residue)
        out2 = self.conv2_2(out2)
        return out2 + out1

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 k_size=3,
                 stride=1,
                 pad=1):
        super(ConvBlock, self).__init__()
        self.conv_relu = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels,
                                      kernel_size=k_size,
                                      stride=stride,
                                      padding=pad),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.1),

        )
    def forward(self, x):
        x = self.conv_relu(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 k_size=3,
                 stride=2,
                 pad=1,
                 padding=1):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=k_size,
                                         stride=stride,
                                         padding=padding,
                                         output_padding=pad)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_act=True):
        x = self.deconv(x)
        if is_act:
            x = torch.relu(self.bn(x))
        return x

class DeconvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels,
                 k_size=3,
                 stride=2,
                 pad=1,
                 padding=1):
        super(DeconvBlock1, self).__init__()
        self.dconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.dconv2(x)
        return x


class EncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodeBlock, self).__init__()
        self.conv1_1 = ConvBlock(in_channels, out_channels, stride=2)
        self.conv1_2 = ConvBlock(out_channels, out_channels)

        self.conv2 = PSAModule(out_channels, out_channels, stride=1, conv_kernels=[3, 5, 7, 9], conv_groups=[1, 4, 8, 16])
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 替换掉conv2_2
        self.conv2_1 = ConvBlock(out_channels, out_channels)
        # self.conv2_2 = ConvBlock(out_channels, out_channels)
        self.shortcut = ConvBlock(in_channels, out_channels, stride=2)

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.conv1_2(out1)
        residue = self.shortcut(x)
        out2 = self.conv2_1(out1 + residue)
        # ---------------------------
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        # -----------------------------
        # out2 = self.conv2_2(out2)
        return out2 + out1

class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodeBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels//4,
                               k_size=1, pad=0)
        self.deconv = DeconvBlock(in_channels//4, in_channels//4)
        self.conv2 = ConvBlock(in_channels//4, out_channels,
                               k_size=1, pad=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x

class DecodeBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodeBlock1, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels//4,
                               k_size=1, pad=0)
        self.deconv = DeconvBlock1(in_channels//4, in_channels//4)
        self.conv2 = ConvBlock(in_channels//4, out_channels,
                               k_size=1, pad=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x

class Linknet(nn.Module):
    def __init__(self):
        super(Linknet, self).__init__()
        self.init_conv = ConvBlock(3, 64,
                                   k_size=7,
                                   stride=2,
                                   pad=3)
        self.init_maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.backbone = MobileNetV2()
        self.dropblock = DropBlock2D(drop_prob=0.2, block_size=7)
        self.Dropout = nn.Dropout2d(0.1)

        self.encode1 = EncodeBlock(64, 64)
        self.encode2 = EncodeBlock(64, 128)
        # self.encode3 = EncodeBlock1(128, 160)
        # self.encode4 = EncodeBlock1(160, 256)
        self.decode4 = DecodeBlock1(256, 160)
        self.decode3 = DecodeBlock1(160, 128)
        self.decode2 = DecodeBlock1(128, 64)
        self.decode1 = DecodeBlock(64, 64)
        self.backbone = MobileNetV2()
        self.deconv_last1 = DeconvBlock(64, 32)
        self.conv_last = ConvBlock(32, 16)
        self.deconv_last2 = DeconvBlock(16, 1,
                                        k_size=2,
                                        pad=0,
                                        padding=0)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)
    #  FeatureNoise
    def feature_based_noise(self, x):
        noise_vector = Uniform(-0.3, 0.3).sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.init_conv(x)  # (6, 128, 128, 64)
        x = self.init_maxpool(x)  # (6, 64, 64, 64)
        # x = self.Dropout(x)
        # x = self.dropblock(x)


        e1 = self.encode1(x)  # (6, 32, 32, 64)
        e2 = self.encode2(e1)  # (6, 16, 16, 128)
        e3 = self.backbone.features[1](e2)
        e3 = self.backbone.features[2](e3)
        e4 = self.backbone.features[3](e3)
        e4 = self.backbone.features[4](e4)
        # e3 = self.encode3(e2)  # (6, 8, 8, 256)
        # e4 = self.encode4(e3)  # (6, 4, 4, 512)
        # feature_dropout
        # e4 = self.feature_dropout(e4)
        # e4 = self.feature_based_noise(e4)
        # e4 = self.Dropout(e4)



        d4 = self.decode4(e4) + e3
        d3 = self.decode3(d4) + e2
        d2 = self.decode2(d3) + e1
        d1 = self.decode1(d2)

        f1 = self.deconv_last1(d1)
        f2 = self.conv_last(f1)
        f3 = self.deconv_last2(f2, is_act=False)

        return f3, d1

