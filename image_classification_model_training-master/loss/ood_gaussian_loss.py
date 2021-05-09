import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import random
import collections
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import json
import torch.nn.functional as F

# rectified Sigmoid
def ReSigmoid(x):
    if x < 0:
        return 0
    return 1 / (1 + torch.exp(-x)) - 0.5

# rectified Huber
def ReHuber(x, delta=2):
    if x < 0:
        return 0
    if x <= delta:
        return 1/2 * x**2
    else:
        return delta*(x-1/2*delta)

# exp and log loss
def ExpLogLoss(x, delta=2):
    if x < 0:
        return 0
    if x < delta:
        return 1/2*x**2
    else:
        return torch.log(x+np.exp(1/2*delta**2)-delta)

# arXiv 1701.03077, A General and Adaptive Robust Loss Function, Jonathan T. Barron, CVPR 2019
def ReGenAdaRobustLoss(x, alpha, c):
    if x < 0:
        return 0
    return np.abs(alpha-2) / alpha * (((x/c)**2 / np.abs(alpha-2) + 1)**(alpha/2)-1)


class ood_gaussian_loss(nn.Module):
    def __init__(self, kernel):
        super(ood_gaussian_loss, self).__init__()
        self.kernels = kernel

    def forward(self, e):
        pdfs = self.kernels.log_prob(e)
        return torch.sum(torch.exp(pdfs))


class ood_mahalanobis_loss(nn.Module):
    def __init__(self, gaussians):
        super(ood_mahalanobis_loss, self).__init__()
        self.gaussians = gaussians

    def forward(self, batch_embeddings):
        batch_size = batch_embeddings.shape[0]
        val = 0
        for idx in range(batch_size):
            e = batch_embeddings[idx, :]
            min_dist = float('inf')
            for key in self.gaussians:
                mu, cov = self.gaussians[key]
                dist = torch.matmul(torch.matmul(e-mu, torch.inverse(cov)), (e-mu).T)
                dist = F.relu(dist)
                if dist < min_dist:
                    min_dist = dist
            val += min_dist
        return (-val / batch_size) / 10000



if __name__ == '__main__':
    gaussians = {}
    for i in range(8):
        gaussians[i] = (np.random.random(64), np.eye(64))
    ood_criterion = ood_mahalanobis_loss(gaussians)
    print(ood_criterion(torch.from_numpy(np.random.random((16, 64))).to(device_id)))