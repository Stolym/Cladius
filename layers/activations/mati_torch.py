import torch
import torch.nn as nn
import torch.nn.functional as F

# Homemade, the paper will be released soon
class MagicAutomateTrendInteract(nn.Module):
    def __init__(self):
        super(MagicAutomateTrendInteract, self).__init__()
        self.params = nn.Parameter(torch.randn(12) * 0.1 + 0.001)
        self.params_bias = nn.Parameter(torch.zeros(6))

    def forward(self, inputs):
        alpha, alpham, beta, betam, gamma, gammam, gammad, delta, deltam, epsilon, epsilonm, zeta = self.params.unbind()
        balpha, bbeta, bgamma, bdelta, bepsilon, bzeta = self.params_bias.unbind()
        
        gelu_part = alpham * (inputs * torch.sigmoid(alpha * (inputs * 1.702))) + balpha
        soft_part = betam * F.softmax(beta * inputs, dim=-1) + bbeta
        daa_part = gamma * inputs + gammam * torch.exp(gammad * inputs) + bgamma
        naaa_part = deltam * torch.tanh(delta * (2 * inputs)) + bdelta
        paaa_part = epsilonm * torch.log(1 + 0.5 * torch.abs(epsilon * inputs)) + bepsilon
        aaa_part = torch.where(inputs < 0, naaa_part, paaa_part)
        linear_part = zeta * inputs + bzeta
        combined_activation = (gelu_part * delta + soft_part * betam + daa_part * gamma + aaa_part * epsilon + linear_part * zeta) / (delta + betam + gamma + epsilon + zeta)
        
        return combined_activation

# Homemade, the paper will be released soon
class MagicAutomateTrendInteractV2(nn.Module):
    def __init__(self):
        super(MagicAutomateTrendInteractV2, self).__init__()
        self.params = nn.Parameter(torch.rand(12) * (0.1 - 0.01) + 0.01)  # Uniform initialization
        self.params_bias = nn.Parameter(torch.zeros(6))

    def forward(self, inputs):
        alpha, alpham, beta, betam, gamma, gammam, gammad, delta, deltam, epsilon, epsilonm, zeta = self.params.unbind()
        balpha, bbeta, bgamma, bdelta, bepsilon, bzeta = self.params_bias.unbind()
        
        gelu_part = alpham * (inputs * torch.sigmoid(alpha * (inputs * 1.702))) + balpha
        soft_part = betam * F.softmax(beta * inputs, dim=-1) + bbeta
        daa_part = gamma * inputs + gammam * torch.exp(gammad * inputs) + bgamma
        naaa_part = deltam * torch.tanh(delta * (2 * inputs)) + bdelta
        paaa_part = epsilonm * torch.log(1 + 0.5 * torch.abs(epsilon * inputs)) + bepsilon
        aaa_part = torch.where(inputs < 0, naaa_part, paaa_part)
        linear_part = zeta * inputs + bzeta

        combined_activation = gelu_part + soft_part + daa_part + aaa_part + linear_part
        
        return combined_activation
