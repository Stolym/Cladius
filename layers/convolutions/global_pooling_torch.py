import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

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

class GlobalChannelCovariancePooling(nn.Module):

    def __init__(self, attention: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rearrange = Rearrange("b c h w -> b (h w) c")
        self.rearrange_attention = Rearrange("b d -> b d 1 1")
        self.softmax = nn.Softmax(dim=1)
        self.attention = attention


    def forward(self, x: torch.Tensor):
        _x = x
        x = self.rearrange(x)
        B, N, D = x.size()
        xmean = x.mean(dim=1, keepdim=True)
        xdiff = (x - xmean).reshape(B * N, D)
        xprods = torch.bmm(xdiff.unsqueeze(2), xdiff.unsqueeze(1)).reshape(B, N, D, D)
        xbcov = xprods.sum(dim=1) / (N - 1)
        if self.attention:
            return self.softmax(self.attention(xbcov.mean(dim=-1)))
        return xbcov