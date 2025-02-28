import torch
import torch.nn as nn
import kmeans1d
import timeit

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# matplotlib.use('WebAgg')
import os
'''
    本节超参数
    1.     clusters, centroids = kmeans1d.cluster(var_flatten, 5) 聚类数量的选择。
    

'''

def instance_whitening_loss(f_map, feature_s, eye, mask_matrix,  num_remove_cov, epoch, inter, num_clusters):
    # 特征图可视化部分

    f_cor, B, C, H, W = get_covariance_matrix(f_map, eye=eye)      # # torch.Size([16, 64, 64])
    f_cor_s, B, C, H, W = get_covariance_matrix(feature_s, eye=eye)

    f_cor_np = f_cor.detach().cpu().numpy()
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3,  1)
    plt.imshow(f_cor_np[0] )
    plt.title('f_cor_np')

    f_cor_s_np = f_cor_s.detach().cpu().numpy()
    plt.subplot(2, 3,  2)
    plt.imshow(f_cor_s_np[0])
    plt.title('f_cor_s_np')


    # mean_map = torch.eye(C).cuda()
    mean_map = (f_cor + f_cor_s)/2
    variance_map = ((f_cor - mean_map) ** 2 + (f_cor_s - mean_map) ** 2) / 2  # torch.Size([16, 64, 64])
    variance_map_batch = torch.mean(variance_map, dim=0)  # 64 * 64

    variance_map_np_np = variance_map.detach().cpu().numpy()
    plt.subplot(2, 3,  3)
    plt.imshow(variance_map_np_np[0])
    plt.title( 'variance_map_np_np')

    variance_map_batch_np = variance_map_batch.detach().cpu().numpy()
    plt.subplot(2, 3,  3)
    plt.imshow(variance_map_batch_np)
    plt.title('variance_map_batch_np')
    plt.show()

    if  inter == 0:
        f_cor_np = f_cor.detach().cpu().numpy()
        variance_map_batch_np = f_cor.detach().cpu().numpy()

    # 2. 对方差进行聚类。
    matrix_1 = torch.ones(C, C).cuda()  # 主对角元素为0
    matrix_1.fill_diagonal_(0)
    matrix_0 = torch.zeros(C, C).cuda()  # 主对角元素为1
    matrix_0.fill_diagonal_(1)

    # 计算对角线上元素的聚类。聚类数 = 2
    var_diag_flatten = torch.flatten(variance_map_batch * matrix_0)
    clusters_diag, centroids_diag = kmeans1d.cluster(var_diag_flatten, 2)
    num_diag_sensitive = clusters_diag.count(1) # 将最敏感的部分进行保留

    # 计算除了对角线上元素的聚类。
    var_flatten = torch.flatten(variance_map_batch * matrix_1)
    clusters, centroids = kmeans1d.cluster(var_flatten, num_clusters)
    max_cluster = num_clusters - 1

    

    #  3. 对聚类后的结果，乘以mask，计算损失。
    loss_diag = torch.clamp(torch.div(num_diag_sensitive, torch.sum(matrix_0)), min=0)
    loss_off_diag = torch.clamp(torch.div(num_sensitive, num_remove_cov), min=0)
    loss = torch.sum(loss_diag + loss_off_diag)
    if inter == 0:
        return loss, f_cor_np, variance_map_batch_np
    else:
        return loss, mean_map, mean_map


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda() # torch.eye用于生成单位矩阵，这里是 C* C矩阵。

    f_map_np = f_map.detach().cpu().numpy()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW
    return f_cor, B, C, H , W



def get_matrix(dim):
    i = torch.eye(dim, dim).cuda()
    reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda()
    # reversal_i_np = reversal_i.detach().cpu().numpy()
    num_off_diagonal = torch.sum(reversal_i)
    return i, reversal_i, num_off_diagonal