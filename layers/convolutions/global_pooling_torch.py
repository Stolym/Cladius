import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

# Torch, Mandatory
class GlobalAvgPool1d(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)

# Torch, Mandatory
class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

# Torch, Mandatory
class GlobalAvgPool3d(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)

# Torch, Mandatory
class GlobalMaxPool1d(nn.Module):
    def forward(self, x):
        return F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)

# Torch, Mandatory
class GlobalMaxPool2d(nn.Module):
    def forward(self, x):
        return F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

# Torch, Mandatory
class GlobalMaxPool3d(nn.Module):
    def forward(self, x):
        return F.adaptive_max_pool3d(x, (1, 1, 1)).view(x.size(0), -1)

# Torch, Mandatory
class GlobalMinPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMinPooling1D, self).__init__()

    def forward(self, x):
        return torch.min(x, dim=-1)[0]

# Torch, Mandatory
class GlobalMinPooling2D(nn.Module):
    def __init__(self):
        super(GlobalMinPooling2D, self).__init__()

    def forward(self, x):
        x = torch.min(x, dim=-1)[0]
        x = torch.min(x, dim=-1)[0]
        return x

# Torch, Mandatory
class GlobalMinPooling3D(nn.Module):
    def __init__(self):
        super(GlobalMinPooling3D, self).__init__()

    def forward(self, x):
        x = torch.min(x, dim=-1)[0]
        x = torch.min(x, dim=-1)[0]
        x = torch.min(x, dim=-1)[0]
        return x

# Homemade, the paper will be released soon
# May be improved
class GlobalAvgLearnablePooling1D(nn.Module):
    def __init__(self, num_channels: int):
        super(GlobalAvgLearnablePooling1D, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, 1))
        self.bias = nn.Parameter(torch.randn(num_channels, 1))

    def forward(self, x):
        mean_pooled = F.adaptive_avg_pool1d(x, 1)
        out = mean_pooled * self.weights + self.bias
        return out.squeeze(-1)

# Homemade, the paper will be released soon
# May be improved
class GlobalAvgLearnablePooling2D(nn.Module):
    def __init__(self, num_channels: int):
        super(GlobalAvgLearnablePooling2D, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.randn(num_channels, 1, 1))

    def forward(self, x):
        mean_pooled = F.adaptive_avg_pool2d(x, (1, 1))
        out = mean_pooled * self.weights + self.bias
        return out.squeeze(-1).squeeze(-1)

# Homemade, the paper will be released soon
# May be improved
class GlobalAvgLearnablePooling3D(nn.Module):
    def __init__(self, num_channels: int):
        super(GlobalAvgLearnablePooling3D, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.randn(num_channels, 1, 1, 1))

    def forward(self, x):
        mean_pooled = F.adaptive_avg_pool3d(x, (1, 1, 1))
        out = mean_pooled * self.weights + self.bias
        return out.squeeze(-1).squeeze(-1).squeeze(-1)

# Homemade, the paper will be released soon
# May be improved
class GlobalMaxLearnablePooling1D(nn.Module):
    def __init__(self, num_channels: int):
        super(GlobalMaxLearnablePooling1D, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, 1))
        self.bias = nn.Parameter(torch.randn(num_channels, 1))

    def forward(self, x):
        max_pooled = F.adaptive_max_pool1d(x, 1)
        out = max_pooled * self.weights + self.bias
        return out.squeeze(-1)

# Homemade, the paper will be released soon
# May be improved
class GlobalMaxLearnablePooling2D(nn.Module):
    def __init__(self, num_channels: int):
        super(GlobalMaxLearnablePooling2D, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.randn(num_channels, 1, 1))

    def forward(self, x):
        max_pooled = F.adaptive_max_pool2d(x, (1, 1))
        out = max_pooled * self.weights + self.bias
        return out.squeeze(-1).squeeze(-1)

# Homemade, the paper will be released soon
# May be improved
class GlobalMaxLearnablePooling3D(nn.Module):
    def __init__(self, num_channels: int):
        super(GlobalMaxLearnablePooling3D, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.randn(num_channels, 1, 1, 1))

    def forward(self, x):
        max_pooled = F.adaptive_max_pool3d(x, (1, 1, 1))
        out = max_pooled * self.weights + self.bias
        return out.squeeze(-1).squeeze(-1).squeeze(-1)

# Homemade, the paper will be released soon
# May be improved
class GlobalMinLearnablePooling1D(nn.Module):
    def __init__(self, num_channels: int):
        super(GlobalMinLearnablePooling1D, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, 1))
        self.bias = nn.Parameter(torch.randn(num_channels, 1))
        self.gminp = GlobalMinPooling1D()

    def forward(self, x):
        min_pooled = self.gminp(x)
        out = min_pooled * self.weights + self.bias
        return out.squeeze(-1)

# Homemade, the paper will be released soon
# May be improved
class GlobalMinLearnablePooling2D(nn.Module):
    def __init__(self, num_channels: int):
        super(GlobalMinLearnablePooling2D, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.randn(num_channels, 1, 1))
        self.gminp = GlobalMinPooling2D()

    def forward(self, x):
        min_pooled = self.gminp(x)
        out = min_pooled * self.weights + self.bias
        return out.squeeze(-1).squeeze(-1)

# Homemade, the paper will be released soon
# May be improved
class GlobalMinLearnablePooling3D(nn.Module):
    def __init__(self, num_channels: int):
        super(GlobalMinLearnablePooling3D, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.randn(num_channels, 1, 1, 1))
        self.gminp = GlobalMinPooling3D()

    def forward(self, x):
        min_pooled = self.gminp(x)
        out = min_pooled * self.weights + self.bias
        return out.squeeze(-1).squeeze(-1).squeeze(-1)

# Paper, Deprecated need to be improved.
class GlobalLearnablePooling(nn.Module):
    def __init__(self, num_channels: int):
        super(GlobalLearnablePooling, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.randn(num_channels, 1, 1))

    def forward(self, x):
        mean_pooled = F.adaptive_avg_pool2d(x, (1, 1))
        out = mean_pooled * self.weights + self.bias
        return out.squeeze(-1).squeeze(-1)

# Homemade, the paper will be released soon
# May be improved
class GlobalLearnablePoolingV2(nn.Module):
    def __init__(self, num_channels: int, spatial_size: int):
        super(GlobalLearnablePooling, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_channels, spatial_size, spatial_size))
        self.bias = nn.Parameter(torch.randn(num_channels, 1, 1))

    def forward(self, x):
        out = x * self.weights
        out = out.sum(dim=[2, 3], keepdim=True)
        out = out + self.bias
        return out.squeeze(-1).squeeze(-1)

# Paper, Deprecated need to be improved.
class GlobalChannelCovariancePooling3D(nn.Module):
    def __init__(
            self, 
            bias=True,
            *args,
            **kwargs
        ) -> None:
        super(GlobalChannelCovariancePooling3D, self).__init__(*args, **kwargs)
        self.rearrange = Rearrange("b g c h w -> b (h w c) g")
        # self.rearrange_attention = Rearrange("b d -> b d 1 1 1")
        self.softmax = nn.Softmax(dim=1)
        self.bias = bias

    def forward(self, x):
        x = self.rearrange(x)
        B, N, D = x.size()
        xmean = x.mean(dim=1, keepdim=True)
        xdiff = (x - xmean).reshape(B * N, D)
        xprods = torch.bmm(xdiff.unsqueeze(2), xdiff.unsqueeze(1)).reshape(B, N, D, D)
        xbcov = xprods.sum(dim=1, keepdim=False) / (N if self.bias else (N-1))
        return xbcov

# Paper, Deprecated need to be improved.
class GlobalChannelCovariancePooling(nn.Module):
    def __init__(self, attention: bool = False, residual: bool = False, bias: bool = False, *args, **kwargs) -> None:
        super(GlobalChannelCovariancePooling, self).__init__(*args, **kwargs)
        self.rearrange = Rearrange("b c h w -> b (h w) c")
        self.rearrange_attention = Rearrange("b d -> b d 1 1")
        self.softmax = nn.Softmax(dim=1)
        self.attention = attention
        self.residual = residual
        self.bias = bias


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        x = self.rearrange(x)
        B, N, D = x.size()
        xmean = x.mean(dim=1, keepdim=True)
        xdiff = (x - xmean).reshape(B * N, D)
        xprods = torch.bmm(xdiff.unsqueeze(2), xdiff.unsqueeze(1)).reshape(B, N, D, D)
        xbcov = xprods.sum(dim=1) / (N if self.bias else (N-1))
        if self.attention and self.residual:
            return _x + self.softmax(self.rearrange_attention(xbcov.mean(dim=-1)))
        elif self.attention:
            return self.softmax(self.rearrange_attention(xbcov.mean(dim=-1)))
        return xbcov


# Homemade, Deprecated need to be finished.
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