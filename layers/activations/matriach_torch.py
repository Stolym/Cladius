import torch
import torch.nn as nn 

# Homemade
# Fusion/Mix between the performance of Gish and Logish
# Capricious
class Matriach(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, theta: float = 1.0, meta: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.meta = meta

    def forward(self, x):
        return x*self.alpha*(1/(self.theta*1+torch.exp(-x*self.beta)*torch.log(2-torch.exp(-torch.exp(-x*self.meta)))))