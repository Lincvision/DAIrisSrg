import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from PIL import Image
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data  
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms  
import os
import cv2
import glob
from PIL import Image
import copy
from network.Linknet_a.Linknet import *
from network.deeplab_a.deeplabv3 import *
from utils.metrics import *


class Solver(object):

    def __init__(self, config, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.net = None
        self.optimizer = None
        self.criterion = torch.nn.BCELoss(reduction='none') 
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.pseudo_path = config.pseudo_path
        self.origin_pseudo_path = config.origin_pseudo_path
        self.target_model_path = config.target_model_path
        self.origin_target_model_path = config.origin_target_model_path
        self.origin_target = config.origin_target
        self.mode = config.mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.best_mIoU = 0
        self.out_stride = config.out_stride
        self.origin_pseudo = config.origin_pseudo
        self.build_model()

    def build_model(self):
        if self.model_type == 'deeplab':
            self.net = DeepLab(num_classes=1, backbone='mobilenet', output_stride=self.out_stride, sync_bn=False,freeze_bn=False).to(self.device)
        elif self.model_type == 'Linknet':
            self.net = Linknet().to(self.device)

        self.optimizer = optim.Adam(list(self.net.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=32, eta_min=0)
        self.net.to(self.device)

    def train(self):

        net_path1 = self.model_path
        checkpoint = torch.load(net_path1)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        target_path = os.path.join(self.target_model_path, '%s-%d.pth' % (self.model_type, self.num_epochs))
        npdata = np.load(self.pseudo_path, allow_pickle=True)
        pseudo_label_dic = npdata['arr_0'].item()  
        uncertain_dic = npdata['arr_1'].item() 
        proto_pseudo_dic = npdata['arr_2'].item()  

        for epoch in tqdm.tqdm(range(self.num_epochs), ncols=70):
            self.net.train(True)
            epoch_loss = 0
            for i, (images, GT, filename) in enumerate(self.train_loader):
                images = images.to(self.device)
                prediction, feature = self.net(images)
                prediction = torch.sigmoid(prediction)
                pseudo_label = [pseudo_label_dic.get(key) for key in filename]
                uncertain_map = [uncertain_dic.get(key) for key in filename]
                proto_pseudo = [proto_pseudo_dic.get(key) for key in filename]
                pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()
                uncertain_map = torch.from_numpy(np.asarray(uncertain_map)).float().cuda()
                proto_pseudo = torch.from_numpy(np.asarray(proto_pseudo)).float().cuda()
                for param in self.net.parameters():
                    param.requires_grad = True
                self.optimizer.zero_grad()
                target_0_obj = F.interpolate(pseudo_label[:, 0:1, ...], size=feature.size()[2:], mode='nearest')
                target_0_bck = 1.0 - target_0_obj
                mask_0_obj = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
                mask_0_bck = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
                mask_0_obj[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
                mask_0_bck[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
                prediction1= []
                pseudo_label_np = pseudo_label.detach().cpu().numpy()
                prediction1.append(np.squeeze(pseudo_label_np[0]))
                mask_0_obj_np = mask_0_obj.detach().cpu().numpy()
                prediction1.append(np.squeeze(mask_0_obj_np[0]))
                mask1_np = (mask_0_obj * pseudo_label).detach().cpu().numpy()
                mask2_np = (mask_0_bck * (1.0 - pseudo_label)).detach().cpu().numpy()
                prediction1.append(np.squeeze( mask1_np[0]))
                prediction1.append(np.squeeze( mask2_np[0] ))
                mask = mask_0_obj * pseudo_label + mask_0_bck * (1.0 - pseudo_label)
                mask_np = mask.detach().cpu().numpy()
                prediction1.append(np.squeeze(mask_np[0]))
                mask_proto = torch.zeros([images.shape[0], 1, images.shape[2], images.shape[3]]).cuda()
                mask_proto[pseudo_label == proto_pseudo] = 1.0
                mask_proto_np = mask_proto.detach().cpu().numpy()
                prediction1.append(np.squeeze(mask_proto_np[0]))
                mask = mask * mask_proto
                mask_np = mask.detach().cpu().numpy()
                proto_pseudo_np =  proto_pseudo.detach().cpu().numpy()
                prediction1.append(np.squeeze(mask_np[0]))
                prediction1.append(np.squeeze(proto_pseudo_np[0]))

                fig = plt.figure(figsize=(1, 1))
                for i in range(8):
                    plt.imshow(prediction1[i], cmap='gray')
                    plt.show()

                train_loss = self.criterion(prediction, pseudo_label)
                train_loss_np1 = train_loss.detach().cpu().numpy()
                train_loss_np2 = (mask * train_loss).detach().cpu().numpy()
                train_loss_np3 = torch.sum(mask * train_loss)
                train_loss_np4 = torch.sum(mask)
                train_loss = torch.sum(mask * train_loss) / torch.sum(mask)


                epoch_loss += train_loss.item()
                train_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            epoch_loss = epoch_loss / len(self.train_loader)
            print('Epoch [%d/%d], Train Loss: %.8f' % (epoch + 1, self.num_epochs, epoch_loss))
            torch.save(
                {
                    'model_state_dict': self.net.state_dict(),
                }, target_path)


    def test_1(self):
        
        if self.origin_target =='origin':
            net_path = self.model_path
            checkpoint = torch.load(net_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
        elif self.origin_target =='target':
            target_model_path = self.target_model_path
            checkpoint = torch.load(target_model_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
        
        self.net.eval()
        epoch_mIoU = 0
        epoch_mIoU_p = 0

        target_path = os.path.join(self.target_model_path, '%s-%d.pth' % (self.model_type, self.num_epochs))
        npdata = np.load(self.pseudo_path, allow_pickle=True)
        pseudo_label_dic = npdata['arr_0'].item()  # (1, 256, 256)

        for i, (images, GT, filename, width, length) in enumerate(self.test_loader):
            images = images.to(self.device)
            GT = GT.to(self.device)
            prediction, feature = self.net(images)
            SR = torch.sigmoid(prediction)
            SR[SR >= 0.5] = 1
            SR[SR < 0.5] = 0
            iou_score = compute_mIoU(SR, GT)
            epoch_mIoU += iou_score

            pseudo_label = [pseudo_label_dic.get(key) for key in filename]
            pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()
            iou_score_p = compute_mIoU(pseudo_label, GT)
            epoch_mIoU_p += iou_score_p

        mIoU = epoch_mIoU / len(self.test_loader)
        mIoU_p = epoch_mIoU_p / len(self.test_loader)

        print('[Test] mIoU: %.4f' % (mIoU))
        print('[Test] pseudo_label_mIoU: %.4f' % (mIoU_p))


    def test_2(self):
        net_path = self.model_path
        self.net.load_state_dict(torch.load(net_path))
        self.net.eval()

        for i, (images, filename, width, length) in enumerate(self.test_loader):
            images = images.to(self.device)
            SR = torch.sigmoid(self.net(images)) 
            SR[SR >= 0.5] = 1
            SR[SR < 0.5] = 0
            SR = SR.cpu().data.numpy()
            SR = SR.reshape(256, 256)
            SR = SR * 255
            SR = np.uint8(SR)
            save_result = Image.fromarray(SR)
            save_result = save_result.resize((width, length))
            fn = os.path.join(self.result_path, str(*filename) + '.png')
            save_result.save(fn)





