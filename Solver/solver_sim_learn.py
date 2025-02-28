import tqdm
import os.path as osp
from network.Linknet_a.Linknet import *
from network.deeplab.deeplabv3 import *
from utils.metrics import *
from dataloaders.train_sim_loader import *
from dataloaders.all_dataloader import *

class Solver_sim_learn(object):

    def __init__(self, config):
        self.num_epochs = config.sim_learn_num_epochs
        self.radius = config.radius

        "other param"
        self.timestamp = config.timestamp

        "for path and dataset"
        self.source = config.source_dataset
        self.target = config.target_dataset
        self.batch_size = config.sim_learn_batch_size
        self.num_workers = config.num_workers

        "All path"
        self.pseudo_file = config.pseudo_file
        self.model_path = None
        self.Bulid_model_path(config.model_path)

        self.test_path = config.test_path
        self.pseudo_path = None

        "model"
        self.net = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.out_stride = config.out_stride

        self.input_size_w = None
        self.input_size_h = None
        self.Bulid_input_size(self.target)
        self.train_loader = All_dataloader(config,self.input_size_w,self.input_size_h).build_dataset_3()
        self.build_model()


    def Bulid_model_path(self, model_path):
        for root, dirs, files in os.walk(model_path):
            for file in files:
                self.model_path = os.path.join(root, file)

    def build_model(self):
        if self.model_type == 'deeplab':
            self.net = DeepLab(num_classes=1, backbone='resnet34', output_stride=self.out_stride, sync_bn=False, freeze_bn=False, use_sim=True, input_size_w =self.input_size_w, input_size_h=self.input_size_h,radius=self.radius).to(self.device)
        elif self.model_type == 'Linknet':
            self.net = Linknet().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.get_scratch_parameters(), lr=3e-2, betas=(0.9, 0.99))
        self.net.to(self.device)

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

    def train(self):
        # trainloss = []

        net_path = self.model_path
        checkpoint = torch.load(net_path)
        self.net.load_state_dict(checkpoint, strict=False)
        trainloss = []
        for epoch in tqdm.tqdm(range(self.num_epochs), ncols=70):
            self.net.train(True)
            epoch_loss = 0
            for i, (sample) in enumerate(self.train_loader):
                images, label_cup, filename = sample  # img, label, filename
                images= images.to(self.device)
                prediction, _, aff_cup = self.net(images)
                self.optimizer.zero_grad()

                bg_label = label_cup[0].cuda(non_blocking=True)  
                fg_label = label_cup[1].cuda(non_blocking=True)  
                neg_label = label_cup[2].cuda(non_blocking=True) 

                bg_count = torch.sum(bg_label) + 1e-5  
                fg_count = torch.sum(fg_label) + 1e-5  
                neg_count = torch.sum(neg_label) + 1e-5  

                bg_loss = torch.sum(- bg_label * torch.log(aff_cup + 1e-5)) / bg_count
                fg_loss = torch.sum(- fg_label * torch.log(aff_cup + 1e-5)) / fg_count
                neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff_cup)) / neg_count

                train_loss = bg_loss / 4 + fg_loss / 4 + neg_loss / 2
                epoch_loss += train_loss.item()
                train_loss.backward()
                self.optimizer.step()

            epoch_loss = epoch_loss / len(self.train_loader)
            trainloss.append(epoch_loss)
            print('Epoch [%d/%d], Train Loss: %.8f' % (epoch + 1, self.num_epochs, epoch_loss))


        pseudo_label_path = './log/' + ('%s-%s' % (self.source, self.target)) + '/' + str(self.timestamp) + '.pth.tar'
        file_path = './log/' + ('%s-%s' % (self.source, self.target))
        if not osp.exists(file_path):
            os.mkdir(file_path)
        torch.save({'model_state_dict': self.net.state_dict(), }, pseudo_label_path)








