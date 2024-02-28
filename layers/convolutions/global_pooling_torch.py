import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAvgPool1d(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)

class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

class GlobalAvgPool3d(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)

class GlobalMaxPool1d(nn.Module):
    def forward(self, x):
        return F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)

class GlobalMaxPool2d(nn.Module):
    def forward(self, x):
        return F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

class GlobalMaxPool3d(nn.Module):
    def forward(self, x):
        return F.adaptive_max_pool3d(x, (1, 1, 1)).view(x.size(0), -1)
