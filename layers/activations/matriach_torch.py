import torch
import torch.nn as nn 

class Matriach(nn.Module):
    # Homemade

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, theta: float = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def forward(self, x):
        return x * self.alpha * (1/(self.theta+torch.log(1+torch.exp(-x*self.beta))))