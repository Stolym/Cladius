from auto.model.i_model import IModel

from layers.activations.mati_torch import MagicAutomateTrendInteractV2

from torch import nn
from torch.nn import functional as F

import torch

from einops.layers.torch import Rearrange, Reduce

class Estona_1(IModel):
    config = {
        "epochs": 100,
        "batch_size": 32,
        "eval_split": 0.2,
        "plots": True,
        "L": 64,
        "LN_eps": 1e-4,
        "L_Dropout": 0.5,
        "LR": 1e-4,
    }

    def __init__(self, shared_config: dict = {}) -> None:
        super().__init__(type="pytorch", shared_config=shared_config)
    
    def load_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, self.config["L"]),
            nn.LayerNorm(self.config["L"], eps=1e-4),
            MagicAutomateTrendInteractV2(),
            nn.Dropout(self.config["L_Dropout"]),
            nn.Linear(self.config["L"], 10),
        )
    
    def load_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.config["LR"])
    
    def load_loss(self) -> nn.Module:
        return nn.CrossEntropyLoss()
    
    def compile_model(self) -> nn.Module:
        # model.compile(optimizer=optimizer, loss=loss)
        return self.model


class ArithmeticLayer(nn.Module):

    def __init__(
            self,
            epsilon: float = 1e-4,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def forward(self, x):
        multiplied = x.unsqueeze(-1) * x.unsqueeze(-2)
        divided = x.unsqueeze(-1) / (x.unsqueeze(-2) + self.epsilon)
        added = x.unsqueeze(-1) + x.unsqueeze(-2)
        subtracted = x.unsqueeze(-1) - x.unsqueeze(-2)
        result = torch.stack([multiplied, divided, added, subtracted], dim=1)
        return result


class Estona_Core_Basic(nn.Module):

    def __init__(
            self, 
            *args, 
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)

        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, 7),
            nn.BatchNorm2d(16),
            # MagicAutomateTrendInteractV2(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7),
            nn.BatchNorm2d(32),
            # MagicAutomateTrendInteractV2(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LayerNorm(64, eps=1e-4),
            # nn.ReLU(),
            # MagicAutomateTrendInteractV2(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.convs(x)


class Estona_Core(nn.Module):
    def __init__(
            self,
            config,
            *args, 
            **kwargs
        ) -> None:
        super(Estona_Core, self).__init__(*args, **kwargs)
        self.config = config

        # self.reshape = Rearrange('b c h w -> b (c h) w')
        self.arithmetic = ArithmeticLayer()
        self.rearrange_arithmetic = Rearrange('b f1 f2 h w -> b (f1 f2) h w')


        self.encoder = nn.Sequential( # Batch, 113, 28, 28
            nn.Conv2d(
                in_channels=113,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            # nn.ReLU(),
            nn.GELU(),
            nn.BatchNorm2d(64),
            # MagicAutomateTrendInteractV2(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.GELU(),
            nn.BatchNorm2d(128),
            # MagicAutomateTrendInteractV2(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # self.encoder = nn.Conv2d(
        #     in_channels=4*28+1,
        #     out_channels=1,
        #     kernel_size=3,
        # )

        # self.encoder_norm = nn.LayerNorm(normalized_shape=(1, 26, 26), eps=self.config["LN_eps"])
        # self.encoder_act = MagicAutomateTrendInteractV2()

        self.flatten = nn.Flatten()

        # self.latent = nn.Linear(26*26, self.config["L"])
        # self.latent = nn.Linear(28*28*28*4+28*28, self.config["L"])
        self.latent = nn.Linear(128*7*7, self.config["L"])
        self.latent_norm = nn.LayerNorm(self.config["L"], eps=self.config["LN_eps"])
        # self.latent_act = MagicAutomateTrendInteractV2()
        self.latent_act = nn.GELU()
        self.latent_dropout = nn.Dropout(self.config["L_Dropout"])
        self.output = nn.Linear(self.config["L"], 10)


    def forward(self, x):
        xa = torch.reshape(x, (x.shape[0], 28, 28))
        xa = self.arithmetic(xa)
        xa = self.rearrange_arithmetic(xa)
        x = torch.cat([x, xa], dim=1)

        x = self.encoder(x)
        # x = self.encoder_norm(x)
        # x = self.encoder_act(x)

        x = self.flatten(x)
        x = self.latent(x)
        x = self.latent_act(x)
        x = self.latent_norm(x)
        # x = self.latent_act(x)
        # x = self.latent_dropout(x)
        x = self.output(x)
        return x


class Estona_2(IModel):
    config = {
        "epochs": 100,
        "batch_size": 32,
        "eval_split": 0.2,
        "plots": True,
        "L": 64,
        "LN_eps": 1e-4,
        "L_Dropout": 0.5,
        "LR": 1e-4,
    }

    def __init__(self, shared_config: dict = {}) -> None:
        super().__init__(type="pytorch", shared_config=shared_config)
    
    def load_model(self) -> nn.Module:
        return Estona_Core(self.config)
        # return Estona_Core_Basic()
    
    def load_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.config["LR"])
    
    def load_loss(self) -> nn.Module:
        return nn.CrossEntropyLoss()
    
    def compile_model(self) -> nn.Module:
        # model.compile(optimizer=optimizer, loss=loss)
        return self.model
    
