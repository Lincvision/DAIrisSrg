import os
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from copy import deepcopy
from PIL import Image, ImageOps, ImageFilter
import math
import cv2
import matplotlib.pyplot as plt

class Train_dataset(data.Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.root = root
        self.transform = transform
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.mode = mode
        print("Train image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        _, filename = os.path.split(image_path)
        filename, _ = os.path.splitext(filename)
        GT_path = self.GT_paths + filename + '.png'
        # image_equal = cv2.imread(image_path)

        image = Image.open(image_path)
        image = image.convert('RGB')
        GT = Image.open(GT_path)
        GT = GT.convert('L')

        # weak aug  resize/affine/flip:
        image, GT = self.tran(image, GT)  # hflip

        # equalizeHist:
        if random.random() < 0.9:
            opencv_image = cv2.imread(image_path, 0)
            opencv_image = cv2.equalizeHist(opencv_image)
            rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            image_s = Image.fromarray(rgb_image)
            image_s = image_s.convert('RGB')
        else:
            image_s = deepcopy(image)

        # image_s = deepcopy(image)
        # strong augmentation:
        if random.random() < 0.9:
            image_s = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(image_s)
        image_s = T.RandomGrayscale(p=0.5)(image_s)
        image_s = blur(image_s, p=0.5)

        # not use strong aug
        # image_s = deepcopy(image)
        if self.mode == 'train_l':
            return self.transform(image), self.transform(GT), self.transform(image_s)


    def __len__(self):
        return len(self.image_paths)

    def tran(self, image, GT):
        prob1 = random.random()
        if prob1 >= 0.5:
            image = F.hflip(image)
            GT = F.hflip(GT)
        '''
        prob2 = random.random()
        if prob2>=0.2:
            image=F.affine(image,45*prob2,translate=[0,0],scale=1,shear=0)
            GT=F.affine(GT,45*prob2,translate=[0,0],scale=1,shear=0)
        '''

        # prob3 = random.random()
        # if prob3 >= 0.7:
        #     image = F.affine(image, 0, translate=[0, 0], scale=prob3, shear=0)
        #     GT = F.affine(GT, 0, translate=[0, 0], scale=prob3, shear=0)

        return image, GT

    def De(self, image):  # 阴影
        x, y, _ = image.shape  # 获取图片大小
        radius = np.random.randint(10, int(min(x, y)//2), 1)  #
        pos_x = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心点坐标
        pos_y = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心坐标
        pos_x = int(pos_x[0])
        pos_y = int(pos_y[0])
        radius = int(radius[0])
        strength = 100
        for j in range(pos_y - radius, pos_y + radius):
            for i in range(pos_x - radius, pos_x + radius):

                distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
                distance = np.sqrt(distance)
                if distance < radius:
                    result = 1 - distance / radius
                    result = result * strength
                    # print(result)
                    image[i, j, 0] = max((image[i, j, 0] - result), 0)
                    image[i, j, 1] = max((image[i, j, 1] - result), 0)
                    image[i, j, 2] = max((image[i, j, 2] - result), 0)
        image = image.astype(np.uint8)
        return image

    def En(self, image): # light
        x, y, _ = image.shape  # 获取图片大小
        radius = np.random.randint(10, int(min(x, y) // 2), 1)  #
        pos_x = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心点坐标
        pos_y = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心坐标
        pos_x = int(pos_x[0])
        pos_y = int(pos_y[0])
        radius = int(radius[0])
        strength = 150
        for j in range(pos_y - radius, pos_y + radius):
            for i in range(pos_x - radius, pos_x + radius):

                distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
                distance = np.sqrt(distance)
                if distance < radius:
                    result = 1 - distance / radius
                    result = result * strength
                    # print(result)
                    image[i, j, 0] = min((image[i, j, 0] + result), 255)
                    image[i, j, 1] = min((image[i, j, 1] + result), 255)
                    image[i, j, 2] = min((image[i, j, 2] + result), 255)
        image = image.astype(np.uint8)
        return image



class Valid_dataset(data.Dataset):
    def __init__(self, root, transform=None, mode='test'):
        self.root = root
        self.transform = transform
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.mode = mode
        print("Vaild image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        _, filename = os.path.split(image_path)
        filename, _ = os.path.splitext(filename)
        GT_path = self.GT_paths + filename + '.png '

        image = Image.open(image_path)
        # image = image.convert("L")
        image = image.convert("RGB")
        GT = Image.open(GT_path)
        GT = GT.convert('L')  # 因为部分数据集0405是三通道因此加入此句话20231128

        width = image.size[0]
        length = image.size[1]

        image = self.transform(image)
        GT = self.transform(GT)

        return image, GT, filename, width, length

    def __len__(self):
        return len(self.image_paths)



class Test1_dataset(data.Dataset):
    def __init__(self, root, transform=None, mode='test'):
        self.root = root
        self.transform = transform
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.mode = mode
        print("Test1 image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        _, filename = os.path.split(image_path)
        filename, _ = os.path.splitext(filename)
        GT_path = self.GT_paths + filename + '.png '

        image = Image.open(image_path)
        # image = image.convert("L")
        image = image.convert("RGB")
        GT = Image.open(GT_path)
        GT = GT.convert('L')  # 因为部分数据集0405是三通道因此加入此句话20231128

        width = image.size[0]
        length = image.size[1]

        image = self.transform(image)
        GT = self.transform(GT)

        return image, GT, filename, width, length

    def __len__(self):
        return len(self.image_paths)


class Test2_dataset(data.Dataset):
    def __init__(self, root, transform=None, mode='test'):
        self.root = root
        self.transform = transform
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.mode = mode
        print("Test2 image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        _, filename = os.path.split(image_path)
        filename, _ = os.path.splitext(filename)

        image = Image.open(image_path)
        image = image.convert("RGB")

        width = image.size[0]
        length = image.size[1]

        image = self.transform(image)

        return image, filename, width, length

    def __len__(self):
        return len(self.image_paths)



def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def Use_data_augmentation(source_dataset, target_dataset):
    # Data_name_list = ['V4', 'IITD', 'syn', 'Lamp', '0405', 'UBIRIS']
    Data_name_list = ['CASIA-V4', 'IITD_2240', 'CASIA_iris_syn_10000', 'Lamp_16212', '0405_6475', 'UBIRIS_2249']
    "(256, 256) (320,280) (640,480) (320,240) (400,300)"

    # traindata_augmentation = T.Compose([T.Resize((256, 256), interpolation=Image.NEAREST), T.ToTensor()])
    # traindata_augmentation = T.Compose([T.Resize((300, 400), interpolation=Image.NEAREST), T.ToTensor()])
    traindata_augmentation = T.Compose([T.Resize((300, 400), interpolation=Image.NEAREST), T.ToTensor()])

    if target_dataset == Data_name_list[0]:
        testdata_augmentation = T.Compose([T.Resize((280, 320), interpolation=Image.NEAREST), T.ToTensor()])
    elif target_dataset == Data_name_list[1]:
        testdata_augmentation = T.Compose([T.Resize((240,320), interpolation=Image.NEAREST), T.ToTensor()])
    elif target_dataset == Data_name_list[2]:
        testdata_augmentation = T.Compose([T.Resize((480, 640), interpolation=Image.NEAREST), T.ToTensor()])
    elif target_dataset == Data_name_list[3]:
        testdata_augmentation = T.Compose([T.Resize((480, 640), interpolation=Image.NEAREST), T.ToTensor()])
        # traindata_augmentation = T.Compose([T.Resize((400, 300), interpolation=Image.NEAREST), T.ToTensor()])
    elif target_dataset == Data_name_list[4]:
        testdata_augmentation = T.Compose([T.Resize((240, 320), interpolation=Image.NEAREST), T.ToTensor()])
    elif target_dataset == Data_name_list[5]:
        # testdata_augmentation = T.Compose([T.Resize((256, 256), interpolation=Image.NEAREST), T.ToTensor()])
        testdata_augmentation = T.Compose([T.Resize((300, 400), interpolation=Image.NEAREST), T.ToTensor()])
        # testdata_augmentation = T.Compose([T.Resize((384, 288), interpolation=Image.NEAREST), T.ToTensor()])

    else:
        testdata_augmentation = T.Compose([T.Resize((256, 256), interpolation=Image.NEAREST), T.ToTensor()])

    return traindata_augmentation, testdata_augmentation