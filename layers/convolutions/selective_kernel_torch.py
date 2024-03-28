import torch
import torch.nn as nn
import torch.nn.functional as F

from global_pooling_torch import GlobalAvgPool2d

# https://paperswithcode.com/method/selective-kernel
# https://arxiv.org/pdf/1903.06586v2.pdf
class SelectiveKernel(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            M: int = 2, # Number of branches
            D: int = 0, # Number of dilations
            R: int = 16, # Reduction ratio
            L: int = 32, # Latent dimension
            *args,
            **kwargs
        ) -> None:
        super(SelectiveKernel, self).__init__(*args, **kwargs)
        
        self.M = M

        d = max(in_channels // R, L)

        self.split = nn.ModuleList()
        for i in range(M):
            k_side = 3 + 2 * i
            padding = ((k_side - 1) * (D * i + 1)) // 2
            self.split.append(nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=(k_side, k_side),
                    dilation=(D * i + 1),
                    padding=padding, 
                    groups=in_channels, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.gap = GlobalAvgPool2d()
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(
            out_channels,
            d,
            bias=False
        )
        # self.bn = nn.InstanceNorm1d(d, affine=True)
        # self.bn = nn.BatchNorm1d(d)
        # self.ln = nn.LayerNorm(d)
        self.bn = nn.GroupNorm(1, d)
        self.select = nn.Linear(
            d,
            out_channels * M
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        # Split
        U = [branch(x) for branch in self.split]
        # Fuse
        U_fused = sum(U)
        S = self.gap(U_fused)#.view(batch_size, -1)

        # Selection guidance
        Z = F.relu(self.bn(self.fc(S)))

        attention_scores = self.select(Z).view(batch_size, self.M, -1)
        attention_scores = F.softmax(attention_scores, dim=1)
        
        # Select and merge branches
        U_final = sum([attention_scores[:, i].unsqueeze(2).unsqueeze(3) * U[i] for i in range(self.M)])
        return U_final


# https://paperswithcode.com/method/selective-kernel
# https://arxiv.org/pdf/1903.06586v2.pdf
# + Homemade
class CustomSelectiveKernel(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            K: tuple[tuple[int, int]],
            D: tuple[tuple[int, int]],
            R: int = 16, # Reduction ratio
            L: int = 32, # Latent dimension
            *args, 
            **kwargs
        ) -> None:
        super(CustomSelectiveKernel, self).__init__(*args, **kwargs)

        assert len(K) == len(D), "K and D must have the same length"
        self.M = len(K)

        d = max(in_channels // R, L)

        self.split = nn.ModuleList()
        for i in range(self.M):
            k_side = K[i]
            padding = ((k_side - 1) * D[i]) // 2
            self.split.append(nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=(k_side, k_side),
                    dilation=D[i],
                    padding=padding, 
                    groups=in_channels, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.gap = GlobalAvgPool2d()
        self.fc = nn.Linear(
            out_channels,
            d,
            bias=False
        )
        self.bn = nn.GroupNorm(1, d)
        self.select = nn.Linear(
            d,
            out_channels * self.M
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        # Split
        U = [branch(x) for branch in self.split]
        # Fuse
        U_fused = sum(U)
        S = self.gap(U_fused)

        # Selection guidance
        Z = F.relu(self.bn(self.fc(S)))

        attention_scores = self.select(Z).view(batch_size, self.M, -1)
        attention_scores = F.softmax(attention_scores, dim=1)
        
        # Select and merge branches
        U_final = sum([attention_scores[:, i].unsqueeze(2).unsqueeze(3) * U[i] for i in range(self.M)])
        return U_final


def unit_test_selective_kernel():
    x = torch.rand(1, 3, 224, 224)
    sk = SelectiveKernel(
        in_channels=3,
        out_channels=3,
        M=2,
        D=0,
        R=16,
        L=32
    )
    y = sk(x)
    print(y.shape)
    assert y.shape == (1, 3, 224, 224)
    print("Unit test for SelectiveKernel passed")

if __name__ == "__main__":
    unit_test_selective_kernel()