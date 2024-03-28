import torch
import torch.nn as nn

# https://paperswithcode.com/method/bottleneck-residual-block
# https://arxiv.org/pdf/1512.03385v1.pdf
class BottleneckResidualBlock(nn.Module):

    def __init__(self, bottleneck: nn.ModuleList, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bottleneck = bottleneck
    
    def forward(self, x):
        r = x
        for layer in self.bottleneck:
            x = layer(x)
        return x + r


# https://paperswithcode.com/method/bottleneck-residual-block
# https://arxiv.org/pdf/1512.03385v1.pdf
# + Homemade
class CombinedBottleneckResidualBlock(nn.Module):

    def __init__(
            self, 
            bottlenecks: tuple[nn.ModuleList], 
            residual: bool = True,
            *args, 
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual = residual
        self.bottlenecks = nn.ModuleList(bottlenecks)

    def forward(self, x):
        r = x
        combiner = []
        for bottleneck in self.bottlenecks:
            x = r
            for layer in bottleneck:
                x = layer(x)
            combiner.append(x)
        x = torch.concat(combiner, dim=1)
        if self.residual:
            return x + r
        return x
    