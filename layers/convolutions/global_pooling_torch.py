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

class GlobalLearnablePooling(nn.Module):
    # Paper
    def __init__(self, num_channels: int):
        super(GlobalLearnablePooling, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.randn(num_channels, 1, 1))

    def forward(self, x):
        mean_pooled = F.adaptive_avg_pool2d(x, (1, 1))
        out = mean_pooled * self.weights + self.bias
        return out.squeeze(-1).squeeze(-1)

class GlobalLearnablePoolingV2(nn.Module):
    # Homemade
    def __init__(self, num_channels: int, spatial_size: int):
        super(GlobalLearnablePooling, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, spatial_size, spatial_size))
        self.bias = nn.Parameter(torch.randn(num_channels, 1, 1))

    def forward(self, x):
        out = x * self.weights
        out = out.sum(dim=[2, 3], keepdim=True)
        out = out + self.bias
        return out.squeeze(-1).squeeze(-1)

class GlobalChannelCovariancePooling(nn.Module):
    # Paper
    def __init__(self, attention: bool = False, residual: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rearrange = Rearrange("b c h w -> b (h w) c")
        self.rearrange_attention = Rearrange("b d -> b d 1 1")
        self.softmax = nn.Softmax(dim=1)
        self.attention = attention
        self.residual = residual


    def forward(self, x: torch.Tensor):
        _x = x
        x = self.rearrange(x)
        B, N, D = x.size()
        xmean = x.mean(dim=1, keepdim=True)
        xdiff = (x - xmean).reshape(B * N, D)
        xprods = torch.bmm(xdiff.unsqueeze(2), xdiff.unsqueeze(1)).reshape(B, N, D, D)
        xbcov = xprods.sum(dim=1) / (N - 1)
        if self.attention and self.residual:
            return _x + self.softmax(self.rearrange_attention(xbcov.mean(dim=-1)))
        elif self.attention:
            return self.softmax(self.rearrange_attention(xbcov.mean(dim=-1)))
        return xbcov

class GlobalSpatialCovariancePooling(nn.Module):
    # Homemade
    def __init__(self, attention: bool = False, residual: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rearrange = Rearrange("b c h w -> b c (h w)")
        self.rearrange_attention = Rearrange("b d -> b d 1 1")
        self.softmax = nn.Softmax(dim=1)
        self.attention = attention
        self.residual = residual

    def forward(self, x: torch.Tensor):
        _x = x
        x = self.rearrange(x)  # Now x is of shape (B, C, H*W)
        B, C, N = x.size()  # N is now H*W, spatial dimensions flattened
        xmean = x.mean(dim=-1, keepdim=True)
        xdiff = (x - xmean).transpose(-1, -2)  # Transpose to make it (B, N, C) for spatial cov
        xprods = torch.bmm(xdiff, xdiff.transpose(-1, -2)) / (N - 1)  # Now computing spatial covariance
        xscov = xprods  # This is the spatial covariance matrix

        if self.attention and self.residual:
            return _x + self.softmax(self.rearrange_attention(xscov.mean(dim=1)))  # Apply attention on mean spatial covariance
        elif self.attention:
            return self.softmax(self.rearrange_attention(xscov.mean(dim=1)))  # Only attention applied
        return xscov  # Return spatial covariance

