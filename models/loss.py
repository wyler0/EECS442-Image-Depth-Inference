
import torch

from pytorch_msssim import ssim
from torch import tensor

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ThreePartLoss(nn.Module):
    def __init__(self, use_cuda):
        super().__init__()
        self.use_cuda = use_cuda

    # given a batch of single channel images of shape (N, C, H, W), returns G_x, G_y
    def gradient(self, img):
        # use sobel filter
        kernel_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        #todo
        kernel_x = kernel_x.view((1,1,3,3))
        kernel_y = kernel_y.view((1,1,3,3))

        if (self.use_cuda):
            kernel_x = kernel_x.to(device="cuda")
            kernel_y = kernel_y.to(device="cuda")
     
        return F.conv2d(img, kernel_x), F.conv2d(img, kernel_y)

    def forward(self, y_pred, y_ground, depthScale=0.1):
        # reshape inputs to be (N, C, H, W)
        if(y_ground.shape[1] != 1):
            y_ground = y_ground.permute(0,3,1,2)
        if(y_pred.shape[1] != 1):
            y_pred = y_pred.permute(0,3,1,2)
        # L1 loss of pixel-wise diff
        l_depth = torch.sum(torch.abs(y_pred - y_ground)) / torch.numel(y_pred)
        # differentiable ssim loss from external package
        l_ssim = (1 - ssim(y_pred, y_ground)) / 2

        xgrad_y_pred, ygrad_y_pred = self.gradient(y_pred)
        xgrad_y_ground, ygrad_y_ground = self.gradient(y_ground)

        # L1 loss of pixel-wise diff betweem x and y image gradients
        l_grad = torch.sum(torch.abs(xgrad_y_pred - xgrad_y_ground) / torch.numel(xgrad_y_pred) + 
            torch.abs(ygrad_y_pred - ygrad_y_ground) / torch.numel(ygrad_y_pred))

        return depthScale * l_depth + l_grad + l_ssim