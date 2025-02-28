import torch
import torch.nn as nn
import torch.nn.functional as F
from network.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from network.deeplab.aspp import build_aspp
from network.deeplab.decoder import build_decoder
from network.deeplab.backbone import build_backbone
import numpy as np
import sys
import torch.sparse as sparse
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
import cv2
import os
def get_indices_of_pairs(radius, size):

    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    radius_floor = radius - 1  # radius_floor 3
    # 32*32 = 1024
    full_indices = np.reshape(np.arange(0, size[0]*size[1], dtype=np.int64),
                                   (size[0], size[1]))  #   96*72

    cropped_height = size[0] - radius_floor  #
    cropped_width = size[1] - 2 * radius_floor #

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor],
                              [-1])  # 6138

    indices_to_list = []

    for dy, dx in search_dist:
        indices_to = full_indices[dy:dy + cropped_height,
                     radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_to_list.append(indices_to)

    concat_indices_to = np.concatenate(indices_to_list, axis=0)
    # 172044 = （59*54=3186）* 54
    # 6138 * 22

    return indices_from, concat_indices_to



class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False,use_sim= False, input_size_w=256, input_size_h=256, radius=4):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.Dropout = nn.Dropout2d(0.1)

        "use 1*1 conv for sim_learn and pl_refine"
        self.use_sim = use_sim
        self.aff_cup = torch.nn.Conv2d(304, 256, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.aff_cup.weight, gain=4)
        self.bn_cup = BatchNorm(256)
        self.bn_cup.weight.data.fill_(1)
        self.bn_cup.bias.data.zero_()
        self.from_scratch_layers = [self.aff_cup, self.bn_cup]

        self.predefined_featuresize_w = int(input_size_w//4)  # 在这里预定义64*64的  96
        self.predefined_featuresize_h = int(input_size_h//4)  # 在这里预定义64*64的  72

        self.ind_from, self.ind_to = get_indices_of_pairs(radius=radius, size=(self.predefined_featuresize_w, self.predefined_featuresize_h))
        self.ind_from = torch.from_numpy(self.ind_from)
        self.ind_to = torch.from_numpy(self.ind_to)



        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, filename=None, GT=None, need_fp=False, to_dense=False):
        x, low_level_feat, feature_list = self.backbone(input)  # X: torch.Size([32, 512, 13, 10])  low: torch.Size([32, 64, 100, 75])
        x = self.aspp(x)  # torch.Size([8, 256, 12, 9])
        feature = x
        if need_fp == True:
            x = self.Dropout(x)
            low_level_feat = self.Dropout(low_level_feat)

        x_last, x_to_last = self.decoder(x, low_level_feat)  # x_last: torch.Size([32, 1, 100, 75])  x_to_last: torch.Size([32, 304, 100, 75])
        x_last= F.interpolate(x_last, size=input.size()[2:], mode='bilinear', align_corners=True)  # torch.Size([32, 1, 256, 256])

        # x_visualize = x_to_last  # torch.Size([32, 256, 10, 13])
        # x_visualize = F.interpolate(x_visualize, size=(input.shape[2], input.shape[3]), mode='bilinear')
        # x_visualize = x_visualize.detach().cpu().numpy()  # 用Numpy处理返回的[1,256,513,513]特征图
        # x_visualize = np.mean(x_visualize, axis=1).reshape(input.shape[2], input.shape[3]) # shape为[513,513]，二维
        # x_visualize = (((x_visualize - np.min(x_visualize)) / (np.max(x_visualize) - np.min(x_visualize))) * 255).astype(np.uint8)
        # x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)
        # cv2.imwrite('image1.jpg', x_visualize)

        T =1


        t =1
        " use sim_learin or pl_refine"
        if self.use_sim:
            f_cup = F.relu(self.bn_cup(self.aff_cup(x_to_last)))  # torch.Size([16, 304, 96, 72])--> torch.Size([16, 256, 96, 72])

            # feature_np = feature.detach().cpu().numpy()
            # plt.imshow(feature_np[0][0])
            # plt.title('feature:after aspp')
            # plt.show()
            #
            # x_to_last_np = x_to_last.detach().cpu().numpy()
            # plt.imshow(x_to_last_np[0][0])
            # plt.title('feature:beforer_aff_cup:x_to_last_np')
            # plt.show()
            #
            # f_cup_np = f_cup.detach().cpu().numpy()
            # plt.imshow(f_cup_np[0][0])
            # plt.title('feature:after_aff_cup(1*1_Conv)')
            # plt.show()
            #
            # x_last_np = x_last.detach().cpu().numpy()
            # plt.imshow(x_last_np[0][0])
            # plt.title('feature:x_last')
            # plt.show()

            if f_cup.size(2) == self.predefined_featuresize_w and f_cup.size(3) == self.predefined_featuresize_h:
                ind_from = self.ind_from  # 5642
                ind_to = self.ind_to  # torch.Size([304668])
            else:
                print('featuresize error')
                sys.exit()

            f_cup = f_cup.view(f_cup.size(0), f_cup.size(1), -1)  # torch.Size([8, 256, 6912])
            ff = torch.index_select(f_cup, dim=2, index=ind_from.cuda(non_blocking=True))  # torch.Size([8, 256, 5642])
            ft = torch.index_select(f_cup, dim=2, index=ind_to.cuda(non_blocking=True))  # torch.Size([8, 256, 304668])
            ff = torch.unsqueeze(ff, dim=2)  # torch.Size([8, 256, 1, 5642])
            ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))  # torch.Size([8, 256, 54, 5642])

            aff_cup = torch.exp(-torch.mean(torch.abs(ft - ff), dim=1))  # 对应公式1 aff_cup为Sij  torch.Size([8, 54, 5642])
            aff_cup_np = aff_cup.detach().cpu().numpy()
            # for i in range(0,54):
            #     image_x = aff_cup_np[0][i].reshape(91, 62)
            #
            #     plt.imshow(image_x)
            #     plt.title('feature:x_last')
            #     plt.show()

            if to_dense:

                aff_cup = aff_cup.view(-1).cpu()  # torch.Size([304668])
                ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)  # torch.Size([304668])
                indices = torch.stack([ind_from_exp, ind_to])  # torch.Size([2, 304668])
                indices_tp = torch.stack([ind_to, ind_from_exp])  # torch.Size([2, 304668])

                area = f_cup.size(2)  # 6912 = 96 * 72
                indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])  # torch.Size([2, 6912])

                # aff_cup = torch.sparse_coo_tensor(torch.cat([indices, indices_id, indices_tp], dim=1), torch.cat([aff_cup, torch.ones([area]), aff_cup]), (6912,6912)).cuda()

                aff_cup_1_np = torch.cat([indices, indices_id, indices_tp], dim=1).detach().cpu().numpy()  # (2, 616248)
                aff_cup_2_np = torch.cat([aff_cup, torch.ones([area]), aff_cup]).detach().cpu().numpy()  # 616248
                aff_cup = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                             torch.cat([aff_cup, torch.ones([area]), aff_cup])).to_dense().cuda() # torch.Size([6912, 6912])
                # aff_cup = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1), torch.cat([aff_cup, torch.ones([area]), aff_cup]), torch.Size([6912, 6912])).to_dense().cuda()  # torch.Size([6912, 6912])
            return x_last, x_to_last,  aff_cup

        else:
            return x_last, x_to_last, feature_list



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_scratch_parameters(self):
        # 只训练[self.aff_cup, self.aff_disc, self.bn_cup, self.bn_disc]，其他部分冻结不训练
        groups = []
        for param in self.parameters():
            param.requires_grad = False

        for m in self.modules():
            if m in self.from_scratch_layers:
                groups.append(m.weight)
                m.weight.requires_grad = True
                if isinstance(m, SynchronizedBatchNorm2d):
                    groups.append(m.bias)
                    m.bias.requires_grad = True
        return groups


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


