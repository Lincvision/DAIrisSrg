from torch.nn.functional import interpolate
import torch.nn as nn




def dice_loss(scale=None):
    def fn(input, target):
        smooth = 1.

        if scale is not None:
            scaled = interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)
            iflat = scaled.view(-1)
        else:
            iflat = input.view(-1)

        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        bce_loss = nn.BCELoss(size_average=True)
        bce_out = bce_loss(input, target)
        # print("bce_loss:", bce_out.data.cpu().numpy())


        return 0.4*(1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))) + 0.6*bce_out



    return fn

# 0.1*(1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))) + 0.9*bce_out