import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, use_feature=False, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

        # self.IN = nn.InstanceNorm2d(out_channel, affine=True) if use_feature else None
        self.IN = nn.InstanceNorm2d(out_channel, affine=False) if use_feature else None


    def forward(self, x_tuple):
        if len(x_tuple) == 2:
            x = x_tuple[0]
            w_arr = x_tuple[1]
        else:
            print("error!!!")
            return
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out_np = out.detach().cpu().numpy()
        if self.IN is not None:
            w_arr.append(self.IN(out))
            temp_np= self.IN(out).detach().cpu().numpy()

        out = self.relu(out)

        return [out, w_arr]








class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 use_feature=[False, False, False, False],
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64
                 ):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.IN_module1 = nn.InstanceNorm2d(self.in_channel, affine=False)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0], feature=use_feature[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2, feature=use_feature[1])
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2, feature=use_feature[2])
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2, feature=use_feature[3])
        # if self.include_top:
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1, feature=None):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group,
                            use_feature=feature))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                use_feature=feature))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature = []
        x = self.conv1(x)  # torch.Size([8, 64, 192, 144])
        feature.append(self.IN_module1(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # torch.Size([8, 64, 96, 72])

        x = self.layer1([x, feature]) # torch.Size([8, 64, 96, 72])
        low_level_feat = x[0]

        x = self.layer2([x[0], x[1]])
        x = self.layer3([x[0], x[1]])  # torch.Size([8, 256, 24, 18])
        x = self.layer4([x[0], x[1]])  # torch.Size([8, 512, 12, 9])


        return x[0], low_level_feat, x[1]


        # x = self.layer1(x)
        # low_level_feat = x
        #
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        #
        # return x, low_level_feat





def resnet34(num_classes=1000, include_top=False):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model = ResNet(BasicBlock, [3, 4, 6, 3], [True, True, False, False ], num_classes=num_classes, include_top=include_top)
    pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
    model.load_state_dict(pretrain_dict, strict= False)
    print('已经成功加载官方的restnet34权重')
    return model



def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


# if __name__ == "__main__":
#     import torch
#     model = resnet34()
#     pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
#     model.load_state_dict(pretrain_dict, strict= False)
#     print('ibn_net is Successfully Loaded from pretraining weights')
#
#     input = torch.rand(1, 3, 512, 512)
#     output, low_level_feat = model(input)
#     print(output.size())
#     print(low_level_feat.size())