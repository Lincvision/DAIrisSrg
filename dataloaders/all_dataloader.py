from dataloaders.train_source_dataloaders import *
from dataloaders.train_sim_loader import *
from dataloaders.train_pl_refine import *

"""
    train source : 1
    generate persuado: 2
    sim_learn :3 
    pl_refine : 4 

"""
class All_dataloader(object):
    def __init__(self, config,input_size_w=256,input_size_h=256):
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.input_size_w = input_size_w
        self.input_size_h = input_size_h

        self.test_mode = config.test_mode
        self.timestamp = config.timestamp

        self.train_path = config.train_path
        self.valid_path = config.valid_path
        self.test_path = config.test_path
        self.pseudo_path = None

        self.source = config.source_dataset
        self.target = config.target_dataset

        self.num_workers = config.num_workers

        self.batch_size = config.batch_size
        self.sim_batch_size = config.sim_learn_batch_size
        self.refine_batch_size = config.refine_batch_size
        self.pseudo_file = config.pseudo_file

        self.radius = config.radius
        self.traindata_augmentation, self.testdata_augmentation = Use_data_augmentation(self.source, self.target)
        self.test = None

    def build_dataset_1(self):
        "train source"
        self.train_loader = data.DataLoader(Train_dataset(root=self.train_path, transform=self.traindata_augmentation, mode='train_l'),
                                         batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.valid_loader = data.DataLoader(Valid_dataset(root=self.valid_path, transform=self.traindata_augmentation, mode='valid'),
                                         batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        if self.test_mode == 1:
            self.test_loader = data.DataLoader(Test1_dataset(root=self.test_path, transform=self.testdata_augmentation, mode='test'),
                batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        else:
            self.test_loader = data.DataLoader(Test2_dataset(root=self.test_path, transform=self.testdata_augmentation, mode='test'),
                batch_size=1, shuffle=False, num_workers=self.num_workers)

        return self.train_loader, self.valid_loader, self.test_loader

    def build_dataset_2(self):
        "generate persuado"
        # traindata_augmentation = T.Compose([T.Resize((256, 256), interpolation=Image.NEAREST), T.ToTensor()])
        traindata_augmentation = T.Compose([T.Resize((self.input_size_w, self.input_size_h), interpolation=Image.NEAREST), T.ToTensor()])

        self.test_loader = data.DataLoader(
            Test1_dataset(root=self.test_path, transform=traindata_augmentation, mode='test'),
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return self.test_loader

    def build_dataset_3(self):
        "sim_learn"
        # self.pseudo_path = os.path.join(self.pseudo_file, '%s-%s' % (self.source, self.target), str(self.timestamp))
        pseudo_path = os.path.join(self.pseudo_file, '%s-%s' % (self.source, self.target))
        for root, dirs, files in os.walk(pseudo_path):
            for file in files:
                file_name = file.split('.')[0]
                if int(file_name) == self.timestamp:
                    self.pseudo_path = os.path.join(root, file)
        traindata_augmentation = T.Compose([T.Resize((self.input_size_w, self.input_size_h), interpolation=Image.NEAREST), T.ToTensor()])
        # traindata_augmentation = T.Compose([T.Resize((256, 256), interpolation=Image.NEAREST), T.ToTensor()])
        self.train_loader = data.DataLoader(Train_Sim_dataset(root=self.test_path, pseudo=self.pseudo_path,transform=traindata_augmentation,mode='train',radius = self.radius,input_size_w=self.input_size_w,input_size_h=self.input_size_h),
                                            batch_size=self.sim_batch_size, shuffle=True, num_workers=self.num_workers)
        return self.train_loader

    def build_dataset_4(self):
        "pl_refine"
        pseudo_path = os.path.join(self.pseudo_file, '%s-%s' % (self.source, self.target))
        for root, dirs, files in os.walk(pseudo_path):
            for file in files:
                file_name = file.split('.')[0]
                if int(file_name) == self.timestamp:
                    self.pseudo_path = os.path.join(root, file)
        traindata_augmentation = T.Compose([T.Resize((self.input_size_w, self.input_size_h), interpolation=Image.NEAREST), T.ToTensor()])
        # traindata_augmentation = T.Compose([T.Resize((256, 256), interpolation=Image.NEAREST), T.ToTensor()])
        self.train_loader = data.DataLoader(Train_pl_refine_dataset(root=self.test_path,pseudo=self.pseudo_path,transform=traindata_augmentation,mode='train',input_size_w=self.input_size_w,input_size_h=self.input_size_h),
        batch_size=self.refine_batch_size, shuffle=False, num_workers=self.num_workers)
        return self.train_loader



