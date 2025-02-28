from network.deeplab.backbone import resnet, xception, drn, mobilenet, resnet_ibn, se_resnet_ibn, ibn_net, resnet34

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'resnet_ibn':
        return resnet_ibn.resnet101_ibn_b()
    elif backbone == 'resnet_ibn_34':
        return resnet_ibn.resnet34_ibn_b()
    elif backbone == 'se_resnet_ibn':
        return se_resnet_ibn.se_resnet101_ibn_a()
    elif backbone == 'ibn_net':
        return ibn_net.resnet101_ibn_b(pretrained=True)
    elif backbone == 'resnet34':
        return resnet34.resnet34()
    elif backbone == 'ibn_net34':
        return ibn_net.resnet34_ibn_b(pretrained=True)

    else:
        raise NotImplementedError
