from utils.metrics import *
from network.deeplab.deeplabv3 import *
import os
import tqdm
import timeit
from network.Linknet_a.Linknet import *
from network.sim_learn.net import *
from dataloaders.train_source_dataloaders import *
from dataloaders.all_dataloader import *
import torch.nn.functional as F
import uuid
import time

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def disenable_dropout(model):
    """ Function to disenable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()


class Solver_generate_pseudo(object):
    def __init__(self, config):


        # 定义网络
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.model_dis = BoundaryDiscriminator().to(self.device)
        self.model_dis2 = UncertaintyDiscriminator().to(self.device)
        self.model_type = config.model_type

        # 定义路径:
        self.model_path = None
        self.Bulid_model_path(config.model_path)

        self.source = config.source_dataset
        self.target = config.target_dataset
        self.pseudo_file = config.pseudo_file

        # pseudo result
        self.pseudo_label_dic = {}
        self.uncertain_dic = {}
        self.proto_pseudo_dic = {}
        self.prob_dic = {}

        self.input_size_w = None
        self.input_size_h = None
        self.Bulid_input_size(self.target)
        self.test_loader = All_dataloader(config,self.input_size_w,self.input_size_h).build_dataset_2()

        if config.model_type == 'deeplab':
            self.net = DeepLab(num_classes=1, backbone='resnet34', output_stride=config.out_stride, sync_bn=False,freeze_bn=False,input_size_w =self.input_size_w, input_size_h = self.input_size_h ).to(self.device)
        elif config.model_type == 'Linknet':
            self.net = Linknet().to(self.device)
        self.timestamp = config.timestamp

    def Bulid_input_size(self, target_dataset):
        Data_name_list = ['V4', 'IITD', 'syn', 'Lamp', '0405', 'UBIRIS']
        if target_dataset == Data_name_list[0]:
            self.input_size_w = 256
            self.input_size_h = 256
        elif target_dataset == Data_name_list[1]:
            self.input_size_w = 256
            self.input_size_h = 256
        elif target_dataset == Data_name_list[3]:
            self.input_size_w = 256
            self.input_size_h = 256
        elif target_dataset == Data_name_list[4]:
            self.input_size_w = 256
            self.input_size_h = 256
        elif target_dataset == Data_name_list[5]:
            self.input_size_w = 384
            self.input_size_h = 288
        else:
            self.input_size_w = 256
            self.input_size_h = 256


    def Bulid_model_path(self, model_path):
        for root, dirs, files in os.walk(model_path):
            for file in files:
                self.model_path = os.path.join(root, file)


    def generate_pseudo(self):
        start_time = timeit.default_timer()  # 定义起始时间

        net_path = self.model_path
        checkpoint = torch.load(net_path)
        self.net.load_state_dict(checkpoint, strict=False)
        self.net.eval()
        enable_dropout(self.net)
        epoch_mIoU = 0
        epoch_mIoU_persuado = 0
        epoch_origin = 0
        for i, (sample) in tqdm.tqdm(enumerate(self.test_loader), total=len(self.test_loader), ncols=80, leave=False):
            images = sample[0].to(self.device)
            GT = sample[1].to(self.device)
            filename = sample[2]

            'origin---------------------:'
            preds_origin, _, _ = self.net(images)
            preds_origin[preds_origin >= 0.5] = 1
            preds_origin[preds_origin < 0.5] = 0
            iou_score_origin = Compute_metrics(preds_origin, GT).compute_mIoU()
            epoch_origin += iou_score_origin
            'origin---------------------'

            preds = torch.zeros([10, images.shape[0], 1, self.input_size_w, self.input_size_h]).cuda() # torch.Size([10, 32, 1, 384, 288])
            if self.model_type == 'deeplab':
                features = torch.zeros([10, images.shape[0], 304, self.input_size_w//4, self.input_size_h//4]).cuda()
            elif self.model_type == 'Linknet':
                features = torch.zeros([10, images.shape[0], 64, self.input_size_w//4, self.input_size_h//4]).cuda()
            for i in range(10):  ##
                with torch.no_grad():
                    preds[i, ...], features[i, ...],  _  = self.net(images)

            preds1 = torch.sigmoid(preds)  # torch.Size([10, 32, 1, 256, 256])
            preds = torch.sigmoid(preds / 2.0)

            std_map = torch.std(preds, dim=0)  # (32, 1, 256, 256)
            prediction = torch.mean(preds1, dim=0)  # torch.Size([32, 1, 256, 256])

            prob = prediction.clone()
            pseudo_label = prediction.clone()  # torch.Size([32, 1, 256, 256])

            SR = prediction.clone()
            SR[SR >= 0.5] = 1
            SR[SR < 0.5] = 0

            pseudo_label[pseudo_label >= 0.5] = 1
            pseudo_label[pseudo_label < 0.5] = 0

            'test miou---------------------'
            SR = prediction.clone()
            SR[SR >= 0.5] = 1
            SR[SR < 0.5] = 0
            origin_pseudo = SR
            iou_score = Compute_metrics(SR, GT).compute_mIoU()
            epoch_mIoU += iou_score
            'test miou---------------------'
            feature = torch.mean(features, dim=0)
            target_0_obj = F.interpolate(pseudo_label[:, 0:1, ...], size=feature.size()[2:],mode='nearest')  # torch.Size([32, 1, 256, 256])  ----->  torch.Size([32, 1, 64, 64])
            prediction_small = F.interpolate(prediction, size=feature.size()[2:], mode='bilinear', align_corners=True)  # torch.Size([32, 1, 64, 64])
            std_map_small = F.interpolate(std_map, size=feature.size()[2:], mode='bilinear',align_corners=True)  # t(32, 1, 64, 64)
            target_0_bck = 1.0 - target_0_obj  # torch.Size([32, 1, 64, 64])

            mask_0_obj = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2],std_map_small.shape[3]]).cuda()
            mask_0_bck = torch.zeros([std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            mask_0_obj[std_map_small[:, 0:1, ...] < 0.05] = 1.0;mask_0_bck[std_map_small[:, 0:1, ...] < 0.05] = 1.0  # 0.07

            b_0_obj = target_0_obj * mask_0_obj;  b_0_bck = target_0_bck * mask_0_bck
            feature_0_obj = feature * b_0_obj;    feature_0_bck = feature * b_0_bck

            centroid_0_obj = torch.sum(feature_0_obj * prediction_small[:, 0:1, ...], dim=[0, 2, 3], keepdim=True)
            centroid_0_bck = torch.sum(feature_0_bck * (1.0 - prediction_small[:, 0:1, ...]), dim=[0, 2, 3],keepdim=True)

            target_0_obj_cnt = torch.sum(mask_0_obj * target_0_obj * prediction_small[:, 0:1, ...], dim=[0, 2, 3],keepdim=True)
            target_0_bck_cnt = torch.sum(mask_0_bck * target_0_bck * (1.0 - prediction_small[:, 0:1, ...]), dim=[0, 2, 3], keepdim=True)

            centroid_0_obj /= target_0_obj_cnt;   centroid_0_bck /= target_0_bck_cnt

            distance_0_obj = torch.sum(torch.pow(feature - centroid_0_obj, 2), dim=1, keepdim=True)
            distance_0_bck = torch.sum(torch.pow(feature - centroid_0_bck, 2), dim=1, keepdim=True)

            proto_pseudo_0 = torch.zeros([images.shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()
            proto_pseudo_0[distance_0_obj < distance_0_bck] = 1.0
            proto_pseudo = F.interpolate(proto_pseudo_0, size=images.size()[2:], mode='nearest')  # torch.Size([8, 2, 512, 512])

            pseudo_label_miou = pseudo_label.clone()
            std_map_miou = std_map.clone()
            proto_pseudo_miou = proto_pseudo.clone()
            mask_0_obj_miou = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).to(self.device)  # torch.Size([1, 512, 512])
            mask_0_bck_miou = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).to(self.device)
            mask_0_obj_miou[std_map_miou < 0.05] = 1.0; mask_0_bck_miou[std_map_miou < 0.05] = 1.0  # 对于边缘的预测是模棱两可的。
            mask_miou = mask_0_obj_miou * pseudo_label + mask_0_bck_miou * (1.0 - pseudo_label)
            mask_proto_miou = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).to(self.device)  # torch.Size([1, 1, 384, 288])
            mask_proto_miou[pseudo_label_miou == proto_pseudo_miou] = 1.0
            mask_miou = mask_miou * mask_proto_miou
            pseudo_label_miou[mask_miou == 0] = 255
            iou_score_pseudo = Compute_metrics(pseudo_label_miou, GT).compute_mIoU()
            epoch_mIoU_persuado += iou_score_pseudo

            pseudo_label = pseudo_label.detach().cpu().numpy()
            std_map = std_map.detach().cpu().numpy()
            proto_pseudo = proto_pseudo.detach().cpu().numpy()
            prob = prob.detach().cpu().numpy()

            for i in range(prediction.shape[0]):
                self.pseudo_label_dic[filename[i]] = pseudo_label[i]  # (1, 384, 288)
                self.uncertain_dic[filename[i]] = std_map[i]  # (1, 384, 288)
                self.proto_pseudo_dic[filename[i]] = proto_pseudo[i]  # (1, 384, 288)
                self.prob_dic[filename[i]] = prob[i]  # (1, 384, 288)

        mIoU = epoch_mIoU / len(self.test_loader)
        persudo_miou = epoch_mIoU_persuado / len(self.test_loader)
        pre_miou = epoch_origin / len(self.test_loader)
        print('[pre_miou] mIoU: %.4f' % (pre_miou))
        print('[prediction] mIoU: %.4f' % (mIoU))
        print('[persudo_miou] mIoU', persudo_miou)


        pseudo_label_path = os.path.join(self.pseudo_file, '%s-%s' % (self.source, self.target), str(self.timestamp))


        np.savez(pseudo_label_path,self.pseudo_label_dic, self.uncertain_dic, self.proto_pseudo_dic, self.prob_dic,)
        stop_time = timeit.default_timer()

        print('Execution time: %.5f' % (stop_time - start_time))
        print("Stage2.1 has been completed")







