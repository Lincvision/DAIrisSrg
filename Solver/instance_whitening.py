import torch
import torch.nn as nn
import kmeans1d
import timeit
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

def instance_whitening_loss(f_map, feature_s, eye, mask_matrix,  num_remove_cov, epoch, inter, num_clusters):

    f_cor, B, C, H, W = get_covariance_matrix(f_map, eye=eye)      
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

    mean_map = (f_cor + f_cor_s)/2
    variance_map = ((f_cor - mean_map) ** 2 + (f_cor_s - mean_map) ** 2) / 2 
    variance_map_batch = torch.mean(variance_map, dim=0) 

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

    matrix_1 = torch.ones(C, C).cuda()  
    matrix_1.fill_diagonal_(0)
    matrix_0 = torch.zeros(C, C).cuda()  
    matrix_0.fill_diagonal_(1)


    var_diag_flatten = torch.flatten(variance_map_batch * matrix_0)
    clusters_diag, centroids_diag = kmeans1d.cluster(var_diag_flatten, 2)
    num_diag_sensitive = clusters_diag.count(1) 

    var_flatten = torch.flatten(variance_map_batch * matrix_1)
    clusters, centroids = kmeans1d.cluster(var_flatten, num_clusters)
    max_cluster = num_clusters - 1

    loss_diag = torch.clamp(torch.div(num_diag_sensitive, torch.sum(matrix_0)), min=0)
    loss_off_diag = torch.clamp(torch.div(num_sensitive, num_remove_cov), min=0)
    loss = torch.sum(loss_diag + loss_off_diag)
    if inter == 0:
        return loss, f_cor_np, variance_map_batch_np
    else:
        return loss, mean_map, mean_map


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape 
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda() 

    f_map_np = f_map.detach().cpu().numpy()
    f_map = f_map.contiguous().view(B, C, -1) 
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)
    return f_cor, B, C, H , W


def get_matrix(dim):
    i = torch.eye(dim, dim).cuda()
    reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda()
    num_off_diagonal = torch.sum(reversal_i)
    return i, reversal_i, num_off_diagonal
