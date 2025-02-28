import torch
import numpy

class Compute_metrics(object):
    def __init__(self, SR, GT):
        self.SR = SR
        self.GT = GT

    def compute_mIoU(self):
        '''
        计算SR(Segmentation Result)和GT(Ground Truth)的mIoU(平均交并比)
        '''
        TP = ((self.SR == 1) & (self.GT == 1)).cpu().data.numpy().sum()
        TN = ((self.SR == 0) & (self.GT == 0)).cpu().data.numpy().sum()
        FP = ((self.SR == 1) & (self.GT == 0)).cpu().data.numpy().sum()
        FN = ((self.SR == 0) & (self.GT == 1)).cpu().data.numpy().sum()

        mIoU = (TP / (TP + FP + FN) + TN / (TN + FP + FN)) * 1 / 2

        return mIoU

    def compute_mean_F1_score(self):
        '''
        计算SR(Segmentation Result)和GT(Ground Truth)的mIoU(平均交并比)
        '''
        TP = ((self.SR == 1) & (self.GT == 1)).cpu().data.numpy().sum()
        TN = ((self.SR == 0) & (self.GT == 0)).cpu().data.numpy().sum()
        FP = ((self.SR == 1) & (self.GT == 0)).cpu().data.numpy().sum()
        FN = ((self.SR == 0) & (self.GT == 1)).cpu().data.numpy().sum()

        mean_F1_score = TP / (2 * TP + FP + FN) + TN / (2 * TN + FP + FN)  # 平均F1分数化简后的公式

        return mean_F1_score

    def compute_R(self):
        '''
        计算SR(Segmentation Result)和GT(Ground Truth)的mIoU(平均交并比)
        '''
        TP = ((self.SR == 1) & (self.GT == 1)).cpu().data.numpy().sum()
        TN = ((self.SR == 0) & (self.GT == 0)).cpu().data.numpy().sum()
        FP = ((self.SR == 1) & (self.GT == 0)).cpu().data.numpy().sum()
        FN = ((self.SR == 0) & (self.GT == 1)).cpu().data.numpy().sum()

        R = TP / (TP + FN)

        return R

    def compute_P(self):
        '''
        计算SR(Segmentation Result)和GT(Ground Truth)的mIoU(平均交并比)
        '''
        TP = ((self.SR == 1) & (self.GT == 1)).cpu().data.numpy().sum()
        TN = ((self.SR == 0) & (self.GT == 0)).cpu().data.numpy().sum()
        FP = ((self.SR == 1) & (self.GT == 0)).cpu().data.numpy().sum()
        FN = ((self.SR == 0) & (self.GT == 1)).cpu().data.numpy().sum()

        P = TP / (TP + FP)

        return P

    def compute_accuracy(self):
        '''
        计算SR(Segmentation Result)和GT(Ground Truth)的mIoU(平均交并比)
        '''
        TP = ((self.SR == 1) & (self.GT == 1)).cpu().data.numpy().sum()
        TN = ((self.SR == 0) & (self.GT == 0)).cpu().data.numpy().sum()
        FP = ((self.SR == 1) & (self.GT == 0)).cpu().data.numpy().sum()
        FN = ((self.SR == 0) & (self.GT == 1)).cpu().data.numpy().sum()

        accuracy = (TP + TN) / (TP + TN + FN + FP)

        return accuracy

    def compute_NICE1(self):
        '''
        计算SR(Segmentation Result)和GT(Ground Truth)的mIoU(平均交并比)
        '''
        TP = ((self.SR == 1) & (self.GT == 1)).cpu().data.numpy().sum()
        TN = ((self.SR == 0) & (self.GT == 0)).cpu().data.numpy().sum()
        FP = ((self.SR == 1) & (self.GT == 0)).cpu().data.numpy().sum()
        FN = ((self.SR == 0) & (self.GT == 1)).cpu().data.numpy().sum()

        nice1 = FP + FN

        return nice1

    def compute_NICE2(self):
        '''
        计算SR(Segmentation Result)和GT(Ground Truth)的mIoU(平均交并比)
        '''
        TP = ((self.SR == 1) & (self.GT == 1)).cpu().data.numpy().sum()
        TN = ((self.SR == 0) & (self.GT == 0)).cpu().data.numpy().sum()
        FP = ((self.SR == 1) & (self.GT == 0)).cpu().data.numpy().sum()
        FN = ((self.SR == 0) & (self.GT == 1)).cpu().data.numpy().sum()
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        nice2 = (FPR + FNR) / 2

        return nice2











