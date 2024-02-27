import torch
import torch.nn as nn 

# https://www.researchgate.net/publication/374231011_Gish_a_novel_activation_function_for_image_classification
class Gish(nn.Module):

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return x * self.alpha * torch.log(2 - torch.exp(-torch.exp(x * self.beta)))