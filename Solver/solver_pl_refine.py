from dataloaders.train_pl_refine import *
import os
from PIL import Image
from network.Linknet_a.Linknet import *
from network.deeplab.deeplabv3 import *
from utils.metrics import *
from dataloaders.all_dataloader import *
from utils.metrics import *
import torch.nn.functional as F
import os.path as osp
import tqdm
from utils.File_create import *

class Solver_pl_refine(object):

    def __init__(self, config):
        self.timestamp = config.timestamp

        self.train_loader = None
        self.pseudo_file = config.pseudo_file
        self.source = config.source_dataset
        self.target = config.target_dataset
        self.pseudo_path = None
        self.test_path = config.test_path
        self.input_size_w = None
        self.input_size_h = None
        self.Bulid_input_size(self.target)

        self.net = None
        self.optimizer = None
        self.batch_size = config.refine_batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.out_stride = config.out_stride
        self.radius = config.radius

        self.train_loader = All_dataloader(config,self.input_size_w,self.input_size_h).build_dataset_4()

        self.build_model()

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


    def build_model(self):
        if self.model_type == 'deeplab':
            self.net = DeepLab(num_classes=1, backbone='resnet34', output_stride=self.out_stride, sync_bn=False, freeze_bn=False, use_sim=True, input_size_w =self.input_size_w, input_size_h=self.input_size_h,radius=self.radius).to(self.device)

        elif self.model_type == 'Linknet':
            self.net = Linknet().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.get_scratch_parameters(), lr=3e-2, betas=(0.9, 0.99))
        self.net.to(self.device)


    def train(self):

        sim_learn_weights_path ='./log/' + ('%s-%s' % (self.source, self.target)) + '/' + str(self.timestamp) + '.pth.tar'

        file_path_1 = os.path.join('./result/', ('%s-%s' % (self.source, self.target)) + '/')
        file_path_2 = file_path_1 + str(self.timestamp) + '/'
        Create_file_2(file_path_1, file_path_2).build_file()

        checkpoint = torch.load(sim_learn_weights_path)
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.net.train(False)
        self.net.eval()
        dice_before = 0
        dice_after = 0
        dice_origin = 0
        prob_dic = {}
        pseudo_label_dic = {}

        for i, (images, filename, prob, gt, SR) in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=80, leave=False):
            images = images.to(self.device)

            prob_upsample = F.interpolate(prob, size=(images.shape[2], images.shape[3]), mode='bilinear')
            prob_upsample = (prob_upsample > 0.5).float()

            "compute before refine MIOU"
            iou_score = Compute_metrics(prob_upsample[:, 0], gt[:, 0]).compute_mIoU()
            dice_before += iou_score

            dheight = int(np.ceil(images.shape[2] / 4))  # 96
            dwidth = int(np.ceil(images.shape[3] / 4))   # 72
            cam = prob  # torch.Size([16, 1, 96, 72])

            "compute origin MIOU"
            SR_revise = SR.clone()
            SR_revise[SR_revise >= 0.5] = 1
            SR_revise[SR_revise < 0.5] = 0
            iou_score_3 = Compute_metrics(SR_revise, gt).compute_mIoU()
            dice_origin += iou_score_3

            with torch.no_grad():
                _, _,  aff_cup = self.net.forward(images, to_dense=True)
                aff_mat_cup = torch.pow(aff_cup, 2) 
                trans_mat_cup = aff_mat_cup / torch.sum(aff_mat_cup, dim=0, keepdim=True) 
                for _ in range(2):
                    trans_mat_cup = torch.matmul(trans_mat_cup, trans_mat_cup)

                cam_vec_cup = cam[:, 0].view(1, -1) # pj
                cam_rw_cup = torch.matmul(cam_vec_cup.cuda(), trans_mat_cup) 
                cam_rw_cup = cam_rw_cup.view(1, 1, dheight, dwidth)
                cam_rw_save_cup = torch.nn.Upsample((self.input_size_w, self.input_size_h), mode='bilinear')(cam_rw_cup)
                cam_rw = cam_rw_save_cup[0, 0] / torch.max(cam_rw_save_cup[0, 0])
                prob_dic[filename] = cam_rw.detach().cpu().numpy()
                pseudo_label_dic[filename] = (cam_rw > 0.5).long().detach().cpu().numpy()

                cam_rw_save_cup = torch.nn.Upsample((images.shape[2], images.shape[3]), mode='bilinear')(cam_rw_cup)
                cam_rw = cam_rw_save_cup[0, 0] / torch.max(cam_rw_save_cup[0, 0])
                pseudo_label_rw = (cam_rw > 0.5).long().detach().cpu().numpy() 
                "Save after refine result"
                pseudo_label_rw_np = pseudo_label_rw * 255
                pseudo_label_rw_np = np.uint8(pseudo_label_rw_np)
                save_result = Image.fromarray(pseudo_label_rw_np)
                save_result = save_result.resize((images.shape[2], images.shape[3]))
                fn = os.path.join(file_path_2, filename[0] + '.png')
                save_result.save(fn)
                gt = gt.to(self.device)
                pseudo_label_rw = torch.unsqueeze(torch.from_numpy(pseudo_label_rw), 0)
                pseudo_label_rw = pseudo_label_rw.to(self.device)
                "Compute after refine MIOU"
                dice_cam_rw_cup = Compute_metrics(pseudo_label_rw, torch.squeeze(gt, 0)).compute_mIoU()
                dice_after += dice_cam_rw_cup
        dice_before = dice_before / len(self.train_loader)
        dice_after = dice_after / len(self.train_loader)
        dice_origin = dice_origin / len(self.train_loader)















