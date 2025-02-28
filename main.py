import os.path as osp
from dataloaders.train_source_dataloaders import *
from Solver.solver_train_source import Solver
from Solver.solver_pl_refine import Solver_pl_refine
from Solver.solver_sim_learn import Solver_sim_learn
from Solver.solver_generate_pseudo import Solver_generate_pseudo
from utils.File_create import *
import argparse
from network.deeplab.deeplabv3 import *
here = osp.dirname(osp.abspath(__file__))
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time

def main():
    # cudnn.benchmark = True #GPU加速
    timestamp = int(time.time())

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument('--timestamp', type=int, default=82601)  # 82601

    "training param"
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)

    "Optimizer param"
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lr-gen', type=float, default=1e-3, help='learning rate',)
    parser.add_argument('--lr-dis', type=float, default=2.5e-5, help='learning rate', )
    parser.add_argument('--lr-decrease-rate', type=float, default=0.1, help='ratio multiplied to initial lr',)
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay',)
    parser.add_argument('--momentum', type=float, default=0.99, help='momentum',)

    """  ---------------------------------------stage1: train source------------------------------------------- """
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--loss_weight', type=float, default=0.5)


    parser.add_argument('--mode', type=str, default='test')  # train test
    parser.add_argument('--model_type', type=str, default='deeplab')
    parser.add_argument('--model_path', type=str, default='./models/1022_baseline_5/deeplab-100.pth') # deeplab-100.pth 1023_baseline_640_1  1022_baseline_3
    parser.add_argument('--train_path', type=str, default='./dataset/1-2400/train/')  # 1-16
    parser.add_argument('--valid_path', type=str, default='./dataset/1-2400/valid/')
    parser.add_argument('--test_path', type=str, default='./dataset/1-16/CASIA-V4/test/')
    parser.add_argument('--target_dataset', type=str, default='CASIA-V4')
    parser.add_argument('--result_path', type=str, default='./result/visulize_CASIA-V4_weak/')

    # Data_name_list = ['CASIA-V4', 'IITD_2240', 'CASIA_iris_syn_10000', 'Lamp_16212', '0405_6475', 'UBIRIS_2249']
    parser.add_argument('--source_dataset', type=str, default='Lamp')

    parser.add_argument('--test_mode', type=int, default=1, help='1 or 2')

    parser.add_argument('--out_stride', type=int, default=16, help='out-stride of deeplabv3+',)
    '是否使用可视化特征图/是否采用连通域优化'
    parser.add_argument('--use_vis_feature', type=bool, default=True)
    parser.add_argument('--use_connect', type=bool, default=True)

    """  -------------------------------------- stage2: domain adaptive-------------------------------------  """
    '2.1 generate_pseudo:'
    parser.add_argument('--pseudo_file', type=str, default='./generate_pseudo/')  # 4. 检查伪标签生成的地址

    '2.2 sim_learn:'
    parser.add_argument('--sim_learn_num_epochs', type=int, default=10)
    parser.add_argument('--sim_learn_batch_size', type=int, default=8)
    parser.add_argument('--radius', type=int, default=6, help='radius')  # 6/8都不错， 4是标准

    '2.3 pl_refine:'
    parser.add_argument('--refine_batch_size', type=int, default=32)

    """   -----------------------------------stage3: Parameterized iris boundary：image/GT/------------------   """
    parser.add_argument('--use_parameter_iris_boundary', type=bool, default=True)
    parser.add_argument('--inner_path', type=str, default='./result/normalize/inner/')
    parser.add_argument('--output_path', type=str, default='./result/normalize/output/')

    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    Create_file(args).build_file()

    if args.mode == 'train':
        Solver(args).train()  # stage 1
        Solver_generate_pseudo(args).generate_pseudo()  # stage 2.1
        Solver_sim_learn(args).train()  # stage 2.2
        Solver_pl_refine(args).train()  # stage 2.3
        # Normalized_iris(args).train()  # stage 3
    elif args.mode == 'test' and args.test_mode == 1:
        Solver(args).tsne()
        # Solver(args).test_1()
    elif args.mode == 'test' and args.test_mode == 2:
        Solver(args).test_2()


if __name__ == '__main__':
    main()