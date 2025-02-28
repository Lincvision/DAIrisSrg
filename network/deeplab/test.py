class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, image_res=256, radius=4):
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
        self.aff_cup = torch.nn.Conv2d(256, 256, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.aff_cup.weight, gain=4)
        self.bn_cup = BatchNorm(256)
        self.bn_cup.weight.data.fill_(1)
        self.bn_cup.bias.data.zero_()
        self.from_scratch_layers = [self.aff_cup, self.bn_cup]
        self.predefined_featuresize = int(image_res//4)  # 在这里预定义64*64的
        self.ind_from, self.ind_to = get_indices_of_pairs(radius=radius, size=(self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from)
        self.ind_to = torch.from_numpy(self.ind_to)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, need_fp=False, to_dense=False):
        x, low_level_feat, feature = self.backbone(input)
        x = self.aspp(x)
        feature = x
        if need_fp == True:
            x = self.Dropout(x)
            low_level_feat = self.Dropout(low_level_feat)

        x_last, x_to_last = self.decoder(x, low_level_feat)
        x_last= F.interpolate(x_last, size=input.size()[2:], mode='bilinear', align_corners=True)  # torch.Size([32, 1, 256, 256])

        'sim_learn'
        f_cup = F.relu(self.bn_cup(self.aff_cup(feature)))  # torch.Size([1, 256, 64, 64])
        if f_cup.size(2) == self.predefined_featuresize and f_cup.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from  # 3186
            ind_to = self.ind_to   # 172044
        else:
            print('featuresize error')
            sys.exit()

        f_cup = f_cup.view(f_cup.size(0), f_cup.size(1), -1)  # torch.Size([1, 256, 4096])
        ff = torch.index_select(f_cup, dim=2, index=ind_from.cuda(non_blocking=True))  # torch.Size([1, 256, 3186])
        ft = torch.index_select(f_cup, dim=2, index=ind_to.cuda(non_blocking=True))  # torch.Size([1, 256, 172044])

        ff = torch.unsqueeze(ff, dim=2)  # torch.Size([1, 256, 1, 3186])
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))  # torch.Size([1, 256, 54, 3186])

        aff_cup = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))  # 对应公式1 aff_cup为Sij  torch.Size([1, 54, 3186])


        if to_dense:
            aff_cup = aff_cup.view(-1).cpu()  #  torch.Size([1, 54, 3186]) ---> torch.Size([172044])
            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)  # torch.Size([172044])
            indices = torch.stack([ind_from_exp, ind_to])  # torch.Size([2, 172044])
            indices_tp = torch.stack([ind_to, ind_from_exp])  #

            area = f_cup.size(2)  # torch.Size([1, 256, 4096])  area: 4096
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])  # torch.Size([2, 4096])
            aff_cup = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                      torch.cat([aff_cup, torch.ones([area]), aff_cup])).to_dense().cuda()  #  torch.Size([256, 256])



        return x_last, x_to_last, feature, aff_cup