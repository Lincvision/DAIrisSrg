import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from tsnecuda import TSNE

'''
    选择256/128要看，实际的expansion =？ expansion = 2则是128
'''
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256 # 128  256
        elif backbone == 'se_resnet_ibn':
            low_level_inplanes = 256
        elif backbone == 'resnet_ibn':
            low_level_inplanes = 256  # 128
        elif backbone == 'ibn_net':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        elif backbone == 'resnet34':
            low_level_inplanes = 64
        elif backbone == 'ibn_net34':
            low_level_inplanes = 64
        elif backbone == 'resnet_ibn_34':
            low_level_inplanes = 64
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(
                                       BatchNorm(304),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(304, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat): # x为aspp的输出结果，
        low_level_feat = self.conv1(low_level_feat)  # 浅层特征-进行卷积处理 采用1*1卷积
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)  # 对Encoder输出的结果经过一次1*1 C=48 torch.Size([32, 48, 64, 64])




        # 将aspp的输出结果进行上采样为了concat
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)  # torch.Size([32, 256, 64, 64])
        x_to_last = torch.cat((x, low_level_feat), dim=1)  # 与1*1Conv的结果进行concat X：C=256  输出为256+48=304  #  torch.Size([32, 304, 64, 64])
        # boundary = self.last_conv_boundary(x)  # 采用3*3Conv和最后的1*1卷积输出的通道数为1  输入的C=304 得到边界特征。  # torch.Size([32, 1, 64, 64])
        # x = torch.cat([x, boundary], 1)  # 304 +1 =305 torch.Size([32, 1, 64, 64])
        x_last = self.last_conv(x_to_last)  # 输出最后的结果numclass # torch.Size([32, 1, 64, 64])

        return x_last, x_to_last

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
