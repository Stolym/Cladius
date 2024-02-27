import torch
import torch.nn as nn 

# https://www.researchgate.net/publication/352721812_Logish_A_New_Nonlinear_Nonmonotonic_Activation_Function_for_Convolutional_Neural_Network
class Logish(nn.Module):

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return x * self.alpha * torch.log(1 + torch.sigmoid(self.beta * x))