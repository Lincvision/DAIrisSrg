from torch import optim
from torch.optim import lr_scheduler
from utils.metrics import *
from utils.Connect_region import  Connect_are
from network.deeplab.deeplabv3 import *
import tqdm
from network.Linknet_a.Linknet import *
from dataloaders.all_dataloader import *
from Solver.instance_whitening import get_matrix, instance_whitening_loss
import matplotlib
import torch.nn.functional as F
matplotlib.use('TkAgg')
import time
from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
from scipy import ndimage
from TSNE import RunTsne
class Solver(object):
    def __init__(self, config):

        # 定义网络
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if config.model_type == 'deeplab':
            # self.net = DeepLab(num_classes=1, backbone='resnet34', output_stride=config.out_stride, sync_bn=False,freeze_bn=False).to(self.device)
            self.net = DeepLab(num_classes=1, backbone='resnet34', output_stride=config.out_stride, sync_bn=False,freeze_bn=False).to(self.device)

        elif config.model_type == 'Linknet':
            self.net = Linknet().to(self.device)
        self.model_type = config.model_type
        # 定义优化器
        self.feat_vecs = torch.tensor([]).cuda()            # 特征向量
        self.feat_vec_labels = torch.tensor([]).cuda()      # 特征向量的类别
        self.feat_vec_domlabels = torch.tensor([]).cuda()   # 特征向量的域信息
        self.mem_vecs = None                                # 聚类中心的向量
        self.mem_vec_labels = None
        self.trainId2name = {0: 'iris', 1: 'background'}
        self.selected_cls = ['iris', 'background']
        self.max_pointnum = 9000  # 最大特征向量的数量
        self.perplexity = 30 # 未知
        self.learning_rate = 100  # t-SNE的学习率
        self.n_iter = 3500  # t-SNE迭代步数
        self.num_neighbors = 128  # 未知，以上几个参数是针对t-SNE比较重要的参数，可以根据自己的需要进行调整
        self.TSNE = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
                         metric='manhattan', n_iter=self.n_iter, verbose=1)
        # 聚类中心的类别

        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr_gen, betas=(0.9, 0.99))
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=32, eta_min=0)
        # 定义损失函数
        self.criterion = torch.nn.BCELoss(size_average=True)  # 选择二分类交叉熵作为损失函数
        # 定义学习率超参数
        self.lr = config.lr
        self.num_epochs = config.num_epochs
        # 定义路径:
        self.model_path = config.model_path
        self.model_path_final = None
        self.result_path = config.result_path
        # 定义dataset和dataloader
        self.train_loader, self.valid_loader, self.test_loader = All_dataloader(config).build_dataset_1()
        # 定义变量
        self.best_mIoU = 0

        self.Bulid_model_path(config.model_path)

        self.num_clusters = config.num_clusters
        self.start_epoch = config.start_epoch
        self.loss_weight = config.loss_weight

        self.use_vis_feature = config.use_vis_feature
        self.use_connect = config.use_connect


    def Bulid_model_path(self, model_path):
        for root, dirs, files in os.walk(model_path):
            for file in files:
                self.model_path_final = os.path.join(root, file)


    def train(self):
        trainloss = []
        valloss = []
        valmIoU = []
        r = []
        p = []
        acc = []

        # 判断是否载入模型继续训练
        net_path = os.path.join(self.model_path, '%s-%d.pth' % (self.model_type, self.num_epochs))
        if os.path.isfile(net_path):
            self.net.load_state_dict(torch.load(net_path),strict=False)
            print('%s is Successfully Loaded from %s' % (self.model_type, net_path))
        vis_map = []
        count = 0
        for epoch in range(self.num_epochs):

            self.net.train(True)
            epoch_loss = 0
            for i, (images, GT, image_s) in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader),desc='Train epoch=%d' % epoch, ncols=80, leave=False):
                img_x = images.to(self.device)
                mask_x = GT.to(self.device)
                image_s = image_s.to(self.device)
                index_inter = i
                if random.random() < 0.65:  # cutout
                    # img_cutout = image_s.clone()
                    img_cutout, mask_cutout = self.cut_out(image_s.clone(), mask_x.clone())
                else:
                    img_cutout = image_s.clone()

                _, _, features = self.net(img_x, need_fp=False)
                _, _, features_s = self.net(image_s, need_fp=False)
                # cutout
                pred_x, _ ,_ = self.net(img_cutout, need_fp=False)
                if epoch >= self.start_epoch:
                    wt_loss = torch.FloatTensor([0]).cuda()  # 初始化
                    for i, (feature, feature_s) in enumerate(zip(features, features_s)):
                        B, C, H, W = feature.shape
                        eye, mask_matrix, num_remove_cov = get_matrix(C)
                        loss, f_cor_np, variance_map_batch_np = instance_whitening_loss(feature, feature_s, eye, mask_matrix, num_remove_cov, epoch, i, num_clusters=self.num_clusters)
                        if i == 0 and index_inter == 0 and epoch % 10 == 0:
                            vis_map.append([])
                            vis_map[count].append(f_cor_np)
                            vis_map[count].append(variance_map_batch_np)
                            count = count + 1
                        wt_loss = wt_loss + loss
                    wt_loss = wt_loss / len(features)
                pred_x = torch.sigmoid(pred_x)
                loss_x = self.criterion(pred_x, mask_x.detach())

                if epoch >= self.start_epoch:
                    train_loss = loss_x + self.loss_weight * wt_loss
                else:
                    train_loss = loss_x

                self.optimizer.zero_grad()
                epoch_loss += train_loss.item()
                train_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            epoch_loss = epoch_loss / len(self.train_loader)
            trainloss.append(epoch_loss)
            print('Epoch [%d/%d], Train Loss: %.8f' % (epoch + 1, self.num_epochs, epoch_loss))

            # ===================================== Validation ====================================#
            self.net.train(False)
            self.net.eval()
            epoch_mIoU = 0
            epoch_loss = 0
            epoch_R = 0
            epoch_P = 0
            epoch_ACC = 0

            for i, (images, GT, filenname, _, _) in enumerate(self.valid_loader):
                images = images.to(self.device)
                GT = GT.to(self.device)
                pred_x, _, _ = self.net(images)

                SR = torch.sigmoid(pred_x)
                valid_loss = self.criterion(SR, GT)
                epoch_loss += valid_loss.item()
                SR[SR >= 0.5] = 1
                SR[SR < 0.5] = 0

                iou_score = Compute_metrics(SR, GT).compute_mIoU()
                epoch_mIoU += iou_score

                R = Compute_metrics(SR, GT).compute_R()
                epoch_R += R

                P = Compute_metrics(SR, GT).compute_P()
                epoch_P += P

                ACC = Compute_metrics(SR, GT).compute_accuracy()
                epoch_ACC += ACC

            mIoU = epoch_mIoU / len(self.valid_loader)
            epoch_loss = epoch_loss / len(self.valid_loader)

            epoch_R = epoch_R / len(self.valid_loader)
            epoch_P = epoch_P / len(self.valid_loader)
            epoch_ACC = epoch_ACC / len(self.valid_loader)

            r.append(epoch_R)
            p.append(epoch_P)
            acc.append(epoch_ACC)
            valloss.append(epoch_loss)
            valmIoU.append(mIoU)

            print('[Validation] Valid Loss: %.8f' % epoch_loss)
            print('[Validation] mIoU: %.8f' % (mIoU))

            # 根据验证集上mIoU的大小判断是否保存模型
            if mIoU > self.best_mIoU:
                self.best_mIoU = mIoU
                best_epoch = epoch + 1
                best_net = self.net.state_dict()
                print('The Best epoch:%d,Best %s model mIoU : %.8f' % (best_epoch, self.model_type, self.best_mIoU))
                torch.save(best_net, net_path)

        self.Vis_feature(vis_map, count, self.use_vis_feature)


    def Vis_feature(self, vis_feature_np, count, Judge):
        count_1 = 0
        if Judge:
            for i in range(count):
                plt.figure(figsize=(10, 10))
                plt.subplot(2, 2, 1)
                plt.title('feature_np_all')
                plt.imshow(vis_feature_np[i][0][0])
                count_1 = count_1 + 1

                plt.subplot(2, 2, 2)
                plt.title('fvariance_map_batch_np')
                plt.imshow(vis_feature_np[i][1][0])
                count_1 = count_1 + 1
                plt.show()


    def tsne(self):

        net_path = self.model_path
        checkpoint = torch.load(net_path)
        self.net.load_state_dict(checkpoint, strict=False)
        self.net.eval()

        with torch.no_grad():
            temp_count =0
            for i, (images, GT, filename, width, length) in enumerate(self.test_loader):
                images = images.to(self.device)
                # GT_1 = GT.to(self.device).to(torch.int64)
                x_last, x_to_last, _ = self.net(images, filename, GT)
                selected_clsid = [0, 1]
                sequence_of_colors = ["tab:red", "tab:blue"]
                x_last = torch.sigmoid(x_last)
                x_last[x_last >= 0.5] = 1
                x_last[x_last< 0.5] = 0
                GT = F.interpolate(x_last.float(), size=(x_to_last.size(2), x_to_last.size(3)), mode='bilinear',align_corners=False)  # torch.Size([32, 1, 75, 100])
                GT = GT.to(torch.int64)
                for index in range(images.size(0)):
                    gt_cuda = GT[index]  # torch.Size([1, 75, 100])
                    gt_cuda = F.one_hot(gt_cuda, num_classes=2)  # torch.Size([1, 75, 100, 2])
                    gt = gt_cuda.view(1, -1, 2)  # torch.Size([1, 7500, 2])
                    denominator = gt.sum(1).unsqueeze(dim=1)  # [1,1,2]
                    denominator = denominator.sum(0)  # batchwise sum
                    denominator = denominator.squeeze()  # torch.Size([2]) tensor([6799,  701], device='cuda:0')

                    features = x_to_last[index].view(1, x_to_last.size(1), -1)
                    nominator = torch.matmul(features, gt.type(torch.float32))
                    nominator = torch.t(nominator.sum(0))  # batchwise sum

                    for slot in selected_clsid:
                        if denominator[slot] != 0:
                            count = -1 - slot
                            temp = denominator[count]
                            cls_vec = nominator[slot] / denominator[count]  # mean vector
                            cls_label = (torch.zeros(1, 1) + slot).cuda()
                            self.feat_vecs = torch.cat((self.feat_vecs, cls_vec.unsqueeze(dim=0)), dim=0)
                            temp_count = temp_count + 1
                            self.feat_vec_labels = torch.cat((self.feat_vec_labels, cls_label), dim=0)

            # feat_vecs_temp = F.normalize(self.feat_vecs.clone(), dim=1).cpu().numpy()
            feat_vecs_temp = self.feat_vecs.cpu().numpy()
            feat_vec_labels_temp = self.feat_vec_labels.clone().to(torch.int64).squeeze().cpu().numpy()  # (4480,)
            vecs2tsne = feat_vecs_temp  # (4480, 304)
            assert len(feat_vecs_temp.shape) == 2

            for tries in range(5):
                X_embedded = self.TSNE.fit_transform(vecs2tsne)  # (540, 2)
                print('\ntsne done')
                # X_embedded[:, 0] = (X_embedded[:, 0] - X_embedded[:, 0].min()) / (
                #         X_embedded[:, 0].max() - X_embedded[:, 0].min())
                # X_embedded[:, 1] = (X_embedded[:, 1] - X_embedded[:, 1].min()) / (
                #         X_embedded[:, 1].max() - X_embedded[:, 1].min())
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                feat_coords = X_embedded

                for cls_i in [0, 1]:
                    temp_coords = feat_coords[(feat_vec_labels_temp == cls_i), :]
                    ax.scatter(temp_coords[:, 0], temp_coords[:, 1],
                               color=sequence_of_colors[cls_i],
                               label=sequence_of_colors[cls_i] + '_' + self.trainId2name[cls_i], s=30, marker='o')

                # lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1))
                # ax.set_xlim(-0.05, 1.05)
                # ax.set_ylim(-0.05, 1.05)
                plt.show()

                t = 1

                # draw


    def test_2(self):

        net_path = self.model_path
        checkpoint = torch.load(net_path)
        self.net.load_state_dict(checkpoint, strict=False)
        self.net.eval()

        for i, (images, filename, width, length) in enumerate(self.test_loader):
            images = images.to(self.device)

            # 保存分割结果为png文件
            prediction, _, _ = self.net(images)
            SR = torch.sigmoid(prediction)



            # SR = torch.sigmoid(self.net(images)) #SR(Segmentation Result)
            SR[SR >= 0.5] = 1
            SR[SR < 0.5] = 0

            if self.use_connect:
                "使用连通域优化"
                pred_u_w = Connect_are(SR).forward()
                SR = pred_u_w.float().to(self.device)


            SR = SR.cpu().data.numpy()
            SR = SR.reshape(240, 320)
            SR = SR * 255

            # GT_np = np.array(SR)
            # mask = np.zeros([GT_np.shape[0], GT_np.shape[1]])
            # mask[GT_np == 255] = 0
            # mask[GT_np == 0] = 1
            #
            # dila_mask = ndimage.binary_dilation(mask, iterations=1).astype(mask.dtype)
            # eros_mask = ndimage.binary_erosion(mask, iterations=1).astype(mask.dtype)
            # boundary_mask = dila_mask + eros_mask
            #
            # plt.imshow(boundary_mask)
            # plt.title('boundary_mask')
            # plt.show()
            #
            # boundary_mask[boundary_mask == 1] = 255
            # boundary = boundary_mask > 128
            # boundary = boundary.astype(np.uint8)
            #
            # plt.imshow(boundary, cmap='gray')
            # plt.title('boundary')
            # plt.show()

            center_0, radius_0 = (167, 123), 25
            mask = np.ones_like(SR)
            cv2.circle(mask, center_0, radius_0, 0, -1)
            SR[mask == 0] = 0

            plt.imshow(SR)
            plt.title('SR_after')
            plt.show()




            SR = np.uint8(SR)
            save_result = Image.fromarray(SR)
            save_result = save_result.resize((width, length))
            fn = os.path.join(self.result_path, str(*filename) + '.png')
            save_result.save(fn)

        # print(1)

    def test_1(self):
        net_path = self.model_path
        checkpoint = torch.load(net_path)
        # self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.net.load_state_dict(checkpoint, strict=False)

        # net_path =self.model_path_final
        # checkpoint = torch.load(net_path)
        # self.net.load_state_dict(checkpoint, strict=False)
        self.net.eval()
        r = []
        p = []
        acc = []
        F1_score = []
        miou = []
        NICE1 = []
        NICE2 = []

        epoch_mIoU = 0
        epoch_mean_F1_score = 0
        epoch_R = 0
        epoch_P = 0
        epoch_ACC = 0
        epoch_nice1 = 0
        epoch_nice2 = 0

        for i, (images, GT, filename, width, length) in enumerate(self.test_loader):
            images = images.to(self.device)
            GT = GT.to(self.device).to(torch.int64)

            # GT = GT.detach().cpu().numpy()

            img_x_np = images.detach().cpu().numpy()
            # image_np = img_x_np[0].transpose((1, 2, 0))  # 转换为 (H, W, C) 格式
            # plt.imshow(image_np)
            # plt.title('image_np')
            # plt.show()
            # GT[GT >= 0.5] = 1
            # SR[SR < 0.5] = 0

            prediction, _, _ = self.net(images, filename, GT)
            SR = prediction.clone()
            SR = torch.sigmoid(SR)

            SR[SR >= 0.5] = 1
            SR[SR < 0.5] = 0

            if self.use_connect:
                "使用连通域优化"
                pred_u_w = Connect_are(SR).forward()
                SR = pred_u_w.float().to(self.device)

            iou_score = Compute_metrics(SR, GT).compute_mIoU()
            f1_score = Compute_metrics(SR, GT).compute_mean_F1_score()

            epoch_mIoU += iou_score
            epoch_mean_F1_score += f1_score

            R = Compute_metrics(SR, GT).compute_R()
            epoch_R += R

            P = Compute_metrics(SR, GT).compute_P()
            epoch_P += P

            ACC = Compute_metrics(SR, GT).compute_accuracy()
            epoch_ACC += ACC

            nice1 = Compute_metrics(SR, GT).compute_NICE1()
            epoch_nice1 += nice1

            nice2 = Compute_metrics(SR, GT).compute_NICE2()
            epoch_nice2 += nice2

            r.append(R)
            p.append(P)
            acc.append(ACC)
            F1_score.append(f1_score)
            miou.append(iou_score)
            NICE1.append(epoch_nice1)
            NICE2.append(epoch_nice2)


        mIoU = epoch_mIoU / len(self.test_loader)
        mean_F1_score = epoch_mean_F1_score / len(self.test_loader)
        epoch_R = epoch_R / len(self.test_loader)
        epoch_P = epoch_P / len(self.test_loader)
        epoch_ACC = epoch_ACC / len(self.test_loader)
        epoch_nice1 = epoch_nice1 / (len(self.test_loader) * 256 * 256)  # 这里的256替换成 实际的图像的大小
        epoch_nice2 = epoch_nice2 / len(self.test_loader)

        print('[Test] mIoU: %.4f' % (mIoU))
        print('[Test] mean_F1_score: %.4f' % (mean_F1_score))
        print('[Test] epoch_R: %.4f' % (epoch_R))
        print('[Test] epoch_P: %.4f' % (epoch_P))
        print('[Test] epoch_ACC: %.4f' % (epoch_ACC))
        print('[Test] epoch_nice1: %.4f' % (epoch_nice1))
        print('[Test] epoch_nice2: %.4f' % (epoch_nice2))



    # def test_2(self):
    #
    #     net_path = self.model_path
    #     checkpoint = torch.load(net_path)
    #     # self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    #     self.net.load_state_dict(checkpoint, strict=False)
    #
    #     # net_path = self.model_path_final
    #     # checkpoint = torch.load(net_path)
    #     # self.net.load_state_dict(checkpoint, strict=False)
    #     self.net.eval()
    #
    #     for i, (images, filename, width, length) in enumerate(self.test_loader):
    #         images = images.to(self.device)
    #
    #         # 保存分割结果为png文件
    #         prediction, _, _ = self.net(images)
    #         SR = torch.sigmoid(prediction)
    #
    #         # if self.use_connect:
    #         #     "使用连通域优化"
    #         #     pred_u_w = Connect_are(SR).forward()
    #         #     SR = pred_u_w.float().to(self.device)
    #
    #         # SR = torch.sigmoid(self.net(images)) #SR(Segmentation Result)
    #         SR[SR >= 0.5] = 1
    #         SR[SR < 0.5] = 0
    #         SR = SR.cpu().data.numpy()
    #         SR = SR.reshape(240, 320)
    #         SR = SR * 255
    #         SR = np.uint8(SR)
    #         save_result = Image.fromarray(SR)
    #         save_result = save_result.resize((width, length))
    #         fn = os.path.join(self.result_path, str(*filename) + '.png')
    #         save_result.save(fn)

    def rand_bbox_region(self, coord):
        # past implementation

        bbx1= []
        bby1= []
        bbx2 = []
        bby2= []
        for i in range(len(coord)):
            length = len(coord[i])
            k = random.randrange(length)
            cut_len = np.random.randint(50,100)
            bbx1.append(max(coord[i][k][0]-cut_len, 0))
            bby1.append(max(coord[i][k][1]-cut_len,0))
            bbx2.append(min(coord[i][k][0]+cut_len,255))
            bby2.append(min(coord[i][k][1]+cut_len,255))
        return bbx1, bby1, bbx2, bby2



    def region_cut_out(self, data, target, coord):
        # target = target.unsqueeze(dim=1)
        mix_data = data.clone()
        mix_target = target.clone()
        u_bbx1, u_bby1, u_bbx2, u_bby2 = self.rand_bbox_region(coord)
        for i in range(0, mix_data.shape[0]):
            mix_data[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = 0
            mix_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = 0

        del data, target
        torch.cuda.empty_cache()
        return mix_data, mix_target


    def rand_bbox_cut_out_and_region(self, mix_data, coord):
        # past implementation
        # past implementation
        W = mix_data.shape[2]
        H = mix_data.shape[3]
        B = mix_data.shape[0]
        lam = np.random.beta(4, 4)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = []
        cy = []

        for i in range(len(coord)):
            length = len(coord[i])
            k = random.randrange(length)
            cx.append(coord[i][k][0])
            cy.append(coord[i][k][1])

        cx = np.array(cx)
        cy = np.array(cy)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)

        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


    def cut_out_and_region(self, data, target, coord):
        # target = target.unsqueeze(dim=1)
        mix_data = data.clone()
        mix_target = target.clone()
        u_bbx1, u_bby1, u_bbx2, u_bby2 = self.rand_bbox_cut_out_and_region(mix_data, coord)
        for i in range(0, mix_data.shape[0]):
            mix_data[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = 0
            mix_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = 0

        del data, target
        torch.cuda.empty_cache()
        return mix_data, mix_target



    def rand_bbox_1(self, size, lam=None):
        # past implementation
        W = size[2]
        H = size[3]
        B = size[0]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
        cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)

        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cut_mixer(self, data, target):
        # target = target.unsqueeze(dim=1)
        mix_data = data.clone()
        mix_target = target.clone()
        u_rand_index = torch.randperm(data.size()[0])[:data.size()[0]].cuda()
        u_bbx1, u_bby1, u_bbx2, u_bby2 = self.rand_bbox_1(data.size(), lam=np.random.beta(4, 4))

        for i in range(0, mix_data.shape[0]):
            mix_data[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                data[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

            mix_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                target[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        del data, target
        torch.cuda.empty_cache()
        return mix_data, mix_target

    def cut_out(self, data, target):
        mix_data = data.clone()
        mix_target = target.clone()
        u_rand_index = torch.randperm(data.size()[0])[:data.size()[0]].cuda()
        u_bbx1, u_bby1, u_bbx2, u_bby2 = self.rand_bbox_1(data.size(), lam=np.random.beta(4, 4))

        for i in range(0, mix_data.shape[0]):
            mix_data[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = 0
            mix_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = 0

        del data, target
        torch.cuda.empty_cache()
        return mix_data, mix_target

    def outline(self, data, target):
        coord = []

        mix_target = target.clone()
        B = data.shape[0]
        for k in range(0,B):
            single_coord = []
            for i in range(8, 248):
                for j in range(8, 248):
                    if target[k][0][i][j] == 1 :
                        if target[k][0][i-1][j] == 0 or target[k][0][i+1][j] == 0 or target[k][0][i][j+1] == 0 or target[k][0][i][j-1] == 0 :
                            mix_target[k][0][i][j] = 2
                            single_coord.append((i,j))

            coord.append(single_coord)

        return mix_target, coord


