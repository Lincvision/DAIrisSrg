
import os
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


class Train_pl_refine_dataset(data.DataLoader):
    def __init__(self, root, pseudo, transform, radius=4, mode='train_Sim',input_size_w=256,input_size_h=256):
        self.root = root
        self.transform = transform
        self.input_size_w = input_size_w
        self.input_size_h = input_size_h
        # sim_learn

        self.GT_paths = root[:-1] + '_GT/'

        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.mode = mode
        npfilename = pseudo
        npdata = np.load(npfilename, allow_pickle=True)
        self.pseudo_label_dic = npdata['arr_0'].item()  # 字典，存放在一个字典中，key为文件名字每一个values为(2, 512, 512)
        self.uncertain_dic = npdata['arr_1'].item()  # key：文件名 values：(2, 512, 512)
        self.proto_pseudo_dic = npdata['arr_2'].item()
        self.prob_dic = npdata['arr_3'].item()  # key：文件名 values：(2, 512, 512)
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        _, filename = os.path.split(image_path)
        filename, _ = os.path.splitext(filename)
        GT_path = self.GT_paths + filename + '.png'

        pseudo_label = self.pseudo_label_dic.get(filename)  # (2, 512, 512)
        uncertain_map = self.uncertain_dic.get(filename)  # (2, 512, 512)
        proto_pseudo = self.proto_pseudo_dic.get(filename)
        prob = self.prob_dic.get(filename)

        pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float()
        uncertain_map = torch.from_numpy(np.asarray(uncertain_map)).float()
        proto_pseudo = torch.from_numpy(np.asarray(proto_pseudo)).float()
        prob = torch.from_numpy(np.asarray(prob)).float()

        SR = prob.clone()

        # 创建mask模版：
        mask_0_obj = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])  # torch.Size([1, 512, 512])
        mask_0_bck = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])

        # 将不确定性小于0.05的置为1，其他置为0
        mask_0_obj[uncertain_map[0:1, ...] < 0.05] = 1.0  # torch.Size([1, 512, 512])
        mask_0_bck[uncertain_map[0:1, ...] < 0.05] = 1.0  # 对于边缘的预测是模棱两可的。
        mask = mask_0_obj * pseudo_label[0:1, ...] + mask_0_bck * (1.0 - pseudo_label[0:1, ...])

        mask_proto = torch.zeros([1, pseudo_label.shape[1], pseudo_label.shape[2]])
        mask_proto[pseudo_label == proto_pseudo] = 1.0  # 当伪标签中的值1且距离前景的距离小于背景的距离时，将mask矩阵置为1.

        mask = mask * mask_proto  # 对应公式3中的mv
        pseudo_label[mask == 0] = 255

        img = Image.open(image_path)
        img = img.convert("RGB")
        gt = Image.open(GT_path)

        if self.transform is not None:
            img = self.transform(img)
            gt = self.transform(gt)
            prob = torch.nn.functional.interpolate(prob.unsqueeze(0),  size=(self.input_size_w//4, self.input_size_h//4), mode='bilinear')
            prob = prob.squeeze(0)

        return img, filename, prob, gt, SR

    def __len__(self):
        return len(self.image_paths)


# pl_refine

