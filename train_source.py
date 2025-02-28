from datetime import datetime
import os
import os.path as osp
from torch.utils import data
from dataloaders.train_source_dataloaders import *
from Solver.train_source import Solver
# PyTorch includes
import argparse
import yaml
from torch.backends import cudnn

# Custom includes
from network.deeplab_a.deeplabv3 import *

here = osp.dirname(osp.abspath(__file__))

import winsound

def main():
    # cudnn.benchmark = True #GPU加速
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)

    # Optimizer超参数
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)

    parser.add_argument('--lr-gen', type=float, default=1e-3, help='learning rate',)
    parser.add_argument('--lr-dis', type=float, default=2.5e-5, help='learning rate', )
    parser.add_argument('--lr-decrease-rate', type=float, default=0.1, help='ratio multiplied to initial lr',)
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay',)
    parser.add_argument('--momentum', type=float, default=0.99, help='momentum',)
    # 其他设置
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_type', type=str, default='deeplab')
    parser.add_argument('--model_path', type=str, default='./models/1022_baseline_3/deeplab-100.pth')  # deeplab-100.pth
    parser.add_argument('--train_path', type=str, default='./dataset/1-16/train/')
    parser.add_argument('--valid_path', type=str, default='./dataset/1-16/valid/')
    parser.add_argument('--test_path', type=str, default='./dataset/1-16/UBIRIS_2249/test/')
    # UBIRIS_2249 400 * 300
    # 0405_6475   320 * 240
    # IITD_2240   320 * 240
    # CASIA-V4    320 * 280
    parser.add_argument('--source_dataset', type=str, default='casia_v4')
    parser.add_argument('--target_dataset', type=str, default='iitd')
    parser.add_argument('--test_mode', type=int, default=1, help='1 or 2')
    parser.add_argument('--out_stride', type=int, default=16, help='out-stride of deeplabv3+',)
    parser.add_argument('--result_path', type=str, default='./result2_1123/')

    traindata_augmentation = T.Compose([T.Resize((300, 400), interpolation=Image.NEAREST), T.ToTensor()])
    testdata_augmentation = T.Compose([T.Resize((300, 400), interpolation=Image.NEAREST), T.ToTensor() ])

    # testdata_augmentation = T.Compose([T.Resize((240, 320), interpolation=Image.NEAREST), T.ToTensor() ])
    # testdata_augmentation = T.Compose([T.Resize((280, 320), interpolation=Image.NEAREST), T.ToTensor() ])

    args = parser.parse_args()
    args.model = 'FCN8s'

    now = datetime.now()

    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    args.result_path = os.path.join(args.result_path,args.model_type)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # 定义dataset和dataloader


    train_loader_l = data.DataLoader(Train_dataset(root=args.train_path,
                                   transform=traindata_augmentation,
                                   mode='train_l'),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)


    valid_loader=data.DataLoader(Test1_dataset(root=args.valid_path,transform=testdata_augmentation,mode='valid'),
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers)

    if args.test_mode == 1:
        test1_loader = data.DataLoader(Test1_dataset(root=args.test_path, transform=testdata_augmentation, mode='test'),
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=args.num_workers)
    elif args.test_mode == 2:
        test2_loader=data.DataLoader(Test2_dataset(root=args.test_path,transform=testdata_augmentation,mode='test'),
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers)
    if args.test_mode == 1:
        solver = Solver(args, train_loader_l, valid_loader, test1_loader)
    elif args.test_mode == 2:
        solver = Solver(args, train_loader_l, valid_loader, test2_loader)

    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test' and args.test_mode == 1:
        solver.test_1()
    elif args.mode == 'test' and args.test_mode == 2:
        solver.test_2()

    duration = 2000  # 持续时间/ms
    frequency = 500  # 频率/Hz
    # winsound.Beep(frequency, duration)


if __name__ == '__main__':
    main()