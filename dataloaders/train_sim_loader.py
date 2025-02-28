import os
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import copy

class ExtractAffinityLabelInRadius():
    """
        return :
        1. bg_pos_affinity_label : background area that remains unchanged before and after movement.
        2. fg_pos_affinity_label : Iris area that remains unchanged before and after movement
        3. neg_affinity_label : Part that changes before and after movement
    """
    def __init__(self, cropsize_w=64, cropsize_h=64, radius=4):
        self.radius = radius
        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))  #

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))  # len 54

        self.radius_floor = radius-1  # radius_floor：  radius：
        self.crop_height = cropsize_w - self.radius_floor  # 91
        self.crop_width = cropsize_h - 2 * self.radius_floor  # 62
        return

    def __call__(self, label):
        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]  #
        labels_from = np.reshape(labels_from, [-1])  # torch.Size([5642])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to_1 = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to_1, 255), np.less(labels_from, 255))
            # 754*1 255的区域是不确定性区域，将不确定的区域置为0，将确定性的区域置为1
            labels_to_list.append(labels_to_1)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)  # (1, 3186)
        concat_labels_to = np.stack(labels_to_list)  # (54, 3186)
        concat_valid_pair = np.stack(valid_pair_list)  # (54, 3186)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)  # (54, 3186) 移动前后不变的部分
        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)  # 取出移动前后保持不变的背景区域
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 1)).astype(np.float32) # 移动前后保持不变的虹膜区域
        # fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32) # 移动前后保持不变的虹膜区域
        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32) # 移动前后发生变化的部分，且该部分是虹膜区域或背景区域即确定性区域。

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)


class Train_Sim_dataset(data.DataLoader):
    def __init__(self, root, pseudo, transform, radius=4, mode='train_Sim',input_size_w=256,input_size_h=256):
        self.root = root
        self.transform = transform
        # sim_learn augmentation
        # self.transform_img = T.Compose([T.Resize((256, 256), interpolation=Image.NEAREST), T.ToTensor()])
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.mode = mode
        npfilename = pseudo
        npdata = np.load(npfilename, allow_pickle=True)
        self.pseudo_label_dic = npdata['arr_0'].item()
        self.uncertain_dic = npdata['arr_1'].item()
        self.proto_pseudo_dic = npdata['arr_2'].item()
        self.input_size_w = input_size_w
        self.input_size_h = input_size_h
        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize_w=input_size_w//4, cropsize_h=input_size_h//4, radius=radius)
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        _, filename = os.path.split(image_path)
        filename, _ = os.path.splitext(filename)

        pseudo_label = self.pseudo_label_dic.get(filename)  # (2, 512, 512)
        uncertain_map = self.uncertain_dic.get(filename)  # (2, 512, 512)
        proto_pseudo = self.proto_pseudo_dic.get(filename)

        pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float()
        uncertain_map = torch.from_numpy(np.asarray(uncertain_map)).float()
        proto_pseudo = torch.from_numpy(np.asarray(proto_pseudo)).float()

        # 创建mask模版：
        mask_0_obj = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])  # torch.Size([1, 512, 512])
        mask_0_bck = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])

        # 将不确定性小于0.05的置为1，其他置为0
        mask_0_obj[uncertain_map[0:1, ...] < 0.05] = 1.0  # torch.Size([1, 512, 512])
        mask_0_bck[uncertain_map[0:1, ...] < 0.05] = 1.0  # 对于边缘的预测是模棱两可的。
        mask = mask_0_obj * pseudo_label[0:1, ...] + mask_0_bck * (1.0 - pseudo_label[0:1, ...])
        # 在这里mask就是mask_0_obj

        mask_proto = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_proto[pseudo_label == proto_pseudo] = 1.0  # 当伪标签中的值1且距离前景的距离小于背景的距离时，将mask矩阵置为1.

        mask = mask * mask_proto  # 对应公式3中的mv
        pseudo_label[mask == 0] = 255  # 通过这上面的一系列计算为了得到mv，得到伪标签中更确信的部分。将不确定性的部分置为了255

        img = Image.open(image_path)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)  # resize_to (3, 384 288)
            pseudo_label = torch.nn.functional.interpolate(pseudo_label.unsqueeze(0), size=(self.input_size_w//4, self.input_size_h//4), mode='nearest')
            pseudo_label = pseudo_label.squeeze(0)

        label = self.extract_aff_lab_func(pseudo_label[0])
        return img, label, filename

    def __len__(self):
        return len(self.image_paths)




