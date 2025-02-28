from skimage import measure
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, measure


class Connect_are(object):
    def __init__(self, SR):
        self.SR = SR

    def forward(self):
        SR_np_before = self.SR.detach().cpu().numpy()
        for i in range(self.SR.shape[0]):
            SR_temp = SR_np_before[i][0]
            SR_temp[SR_temp != 0] = 1
            SR_temp = measure.label(SR_temp, connectivity=2)
            props = measure.regionprops(SR_temp)

            # plt.imshow(SR_temp, cmap='hot')
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()

            max_area = 0
            max_index = 0
            for index, prop in enumerate(props, start=1):
                if prop.area > max_area:
                    max_area = prop.area
                    # index 代表每个联通区域内的像素值；prop.area代表相应连通区域内的像素个数
                    max_index = index

            SR_temp[SR_temp != max_index] = 0
            SR_temp[SR_temp == max_index] = 1
            SR_np_before[i][0] = SR_temp
        pred_u_w = torch.from_numpy(SR_np_before)

        return pred_u_w




