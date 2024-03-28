import layers.layers_torch as cladius
import layers.utils_torch as cladius_utils

from torchinfo import summary

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import time


from einops.layers.torch import Rearrange, Reduce 

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

class UnitTestModelBottleneckResidual(nn.Module):

    def __init__(
            self,
            custom: bool = False,
            *args, 
            **kwargs
        ) -> None:
        super(UnitTestModelBottleneckResidual, self).__init__(*args, **kwargs)

        L = 24

        if custom:
            self.bottleneck = cladius.CombinedBottleneckResidualBlock(
                bottlenecks=(
                    nn.ModuleList([
                        nn.Conv2d(1, L, kernel_size=3, padding=1),
                        nn.BatchNorm2d(L),
                        # nn.ReLU(),
                        cladius.MagicAutomateTrendInteractV2(),
                        nn.Conv2d(L, L, kernel_size=3, padding=1),
                        nn.BatchNorm2d(L),
                        # nn.ReLU(),
                        cladius.MagicAutomateTrendInteractV2(),
                    ]),
                    nn.ModuleList([
                        nn.Conv2d(1, L, kernel_size=3, padding=1),
                        nn.BatchNorm2d(L),
                        # nn.ReLU(),
                        cladius.MagicAutomateTrendInteractV2(),
                        nn.Conv2d(L, L, kernel_size=3, padding=1),
                        nn.BatchNorm2d(L),
                        # nn.ReLU(),
                        cladius.MagicAutomateTrendInteractV2(),
                    ]),
                )
            )
        else:
            self.bottleneck = cladius.BottleneckResidualBlock(
                bottleneck=nn.ModuleList([
                    nn.Conv2d(1, L, kernel_size=3, padding=1),
                    nn.BatchNorm2d(L),
                    # nn.ReLU(),
                    cladius.MagicAutomateTrendInteractV2(),
                    nn.Conv2d(L, L, kernel_size=3, padding=1),
                    nn.BatchNorm2d(L),
                    # nn.ReLU(),
                    cladius.MagicAutomateTrendInteractV2(),
                ])
            )
        if custom:
            self.glap = cladius.GlobalAvgLearnablePooling2D(2*L)
            self.fc = nn.Linear(
                in_features=2*L,
                out_features=10
            )
        else:
            self.glap = cladius.GlobalAvgLearnablePooling2D(L)
            self.fc = nn.Linear(
                in_features=L,
                out_features=10
            )

        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.bottleneck(x)
        x = self.glap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
def calculate_padding(height: int, width: int, kernel_size: int, stride: int) -> tuple[int, int]:
    pad_height = max((height - 1) * stride + kernel_size - height, 0)
    pad_width = max((width - 1) * stride + kernel_size - width, 0)

    pad_height = pad_height // 2 if pad_height % 2 == 0 else (pad_height + 1) // 2
    pad_width = pad_width // 2 if pad_width % 2 == 0 else (pad_width + 1) // 2
    
    return pad_height, pad_width

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class UnitTestModelMixedSpatialChannelAttention(nn.Module):

    def __init__(
            self, 
            shape: tuple[int, int, int] = (1, 64, 64), 
            L: int = 24,
            *args, 
            **kwargs
        ) -> None:
        super(UnitTestModelMixedSpatialChannelAttention, self).__init__(*args, **kwargs)

        channels, width, height = shape

        self.mha_a = nn.MultiheadAttention(
            embed_dim=4,
            num_heads=1,
        )
        self.mha_aln = nn.LayerNorm(4, eps=1e-6)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=channels*height*width*4,
                out_features=L
            ),
            cladius.MagicAutomateTrendInteractV2(),
            nn.Linear(
                in_features=L,
                out_features=10
            ),
        )

        # self.main = nn.Conv2d(
        #     in_channels=channels,
        #     out_channels=3,
        #     kernel_size=28,
        #     padding="same" #calculate_padding(height, width, 64, 1)
        # )
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=3,
                kernel_size=3,
                padding="same" #calculate_padding(height, width, 64, 1)
            ),
            cladius.MagicAutomateTrendInteractV2(),
        )

        self.combined_bottleneck = cladius.CombinedBottleneckResidualBlock(
            residual=False,
            bottlenecks=(
                nn.ModuleList([
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=1,
                        kernel_size=5,
                        stride=1,
                        padding="same"#calculate_padding(height, width, 5, 1)
                    ),
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=24,
                        kernel_size=6,
                        stride=1,
                        padding="same"#calculate_padding(height, width, 6, 2)
                    ),
                    cladius.MixSpatialChannelAttention(),
                    nn.Conv2d(
                        in_channels=24,
                        out_channels=3,
                        kernel_size=1,
                        stride=1,
                        padding="same"#calculate_padding(height, width, 1, 1)
                    ),
                    cladius.MagicAutomateTrendInteractV2(),
                ]),
                nn.ModuleList([
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=1,
                        kernel_size=3,
                        stride=1,
                        padding="same"#calculate_padding(height, width, 3, 2)
                    ),
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=24,
                        kernel_size=5,
                        stride=1,
                        padding="same"#calculate_padding(height, width, 5, 1)
                    ),
                    cladius.MixSpatialChannelAttention(),
                    nn.Conv2d(
                        in_channels=24,
                        out_channels=3,
                        kernel_size=1,
                        stride=1,
                        padding="same"#calculate_padding(height, width, 1, 1)
                    ),
                    cladius.MagicAutomateTrendInteractV2(),
                ]),
            )
        )
        self.rearange = Rearrange('b c h w -> b (h w) c')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, C, H, W = x.shape
        _x = self.main(x)
        _x = torch.cat([_x, x], dim=1)
        # _x = torch.cat([self.combined_bottleneck(x), x], dim=1)
        _x = self.rearange(_x)

        # Add enoips rearrange because mha don't support 4D tensor + create PositionalEncoding
        # _x = _x.flatten(start_dim=2).permute(0, 2, 1)  # Now (B, H*W, Channels*2)
        # pos_encoder = PositionalEncoding(d_model=9, max_len=H*W)
        # _x = pos_encoder(_x)
        # x = x.permute(0, 2, 1).view(batch_size, -1, H, W)

        x, _ = self.mha_a(_x, _x, _x)
        # Rearrange back to 4D tensor
        x = self.mha_aln(_x + x)
        x = self.fc(x)
        return x

# BottleneckResidualBlock 
def unit_test_model_bottleneck_residual(device: torch.device):
    epochs = 10
    batch_size = 250
    train_dataloader, test_dataloader = cladius_utils.download_fashion_mnist(batch_size)


    for x, y in test_dataloader:
        print(f"Shape of x [N, C, H, W]: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = UnitTestModelBottleneckResidual(custom=False).to(device)
    summary(model, input_size=(batch_size, 1, 28, 28))

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer_l = torch.optim.Adam(model.parameters(), lr=3e-3)


    losses = []
    accuracies = []

    size = len(train_dataloader.dataset)

    for epoch in range(epochs):
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            

            outputs = model(x)
            loss = loss_fn(outputs, y)

            loss.backward()
            optimizer_l.step()
            optimizer_l.zero_grad()

            losses.append(loss.detach().item())

            if batch % 10 == 0:
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y).sum()
                accuracy = 100 * (correct.item() / len(outputs))
                accuracies.append(accuracy)
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"losses: {np.array(losses).mean():>7f} [{len(losses):>5d}]")
                print(f"accuracy: {accuracy:>3f}\n")
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].plot(accuracies)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Batch")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(losses)
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Batch")
    ax[1].set_ylabel("Loss")

    # plt.show()

def unit_test_model_combined_bottleneck_residual(device: torch.device):
    epochs = 10
    batch_size = 250
    train_dataloader, test_dataloader = cladius_utils.download_fashion_mnist(batch_size)


    for x, y in test_dataloader:
        print(f"Shape of x [N, C, H, W]: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = UnitTestModelBottleneckResidual(custom=True).to(device)
    summary(model, input_size=(batch_size, 1, 28, 28))

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer_l = torch.optim.Adam(model.parameters(), lr=3e-3)


    losses = []
    accuracies = []

    size = len(train_dataloader.dataset)

    for epoch in range(epochs):
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            

            outputs = model(x)
            loss = loss_fn(outputs, y)

            loss.backward()
            optimizer_l.step()
            optimizer_l.zero_grad()

            losses.append(loss.detach().item())

            if batch % 10 == 0:
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y).sum()
                accuracy = 100 * (correct.item() / len(outputs))
                accuracies.append(accuracy)
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"losses: {np.array(losses).mean():>7f} [{len(losses):>5d}]")
                print(f"accuracy: {accuracy:>3f}\n")
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].plot(accuracies)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Batch")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(losses)
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Batch")
    ax[1].set_ylabel("Loss")

    plt.show()

def unit_test_mix_spatial_channel_attention(device: torch.device):
    epochs = 10
    batch_size = 250
    train_dataloader, test_dataloader = cladius_utils.download_fashion_mnist(batch_size)


    for x, y in test_dataloader:
        print(f"Shape of x [N, C, H, W]: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = UnitTestModelMixedSpatialChannelAttention(
        shape=(1, 28, 28),
        L=24
    ).to(device)
    summary(model, input_size=(batch_size, 1, 28, 28))
    # exit()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer_l = torch.optim.Adam(model.parameters(), lr=1e-3)


    losses = []
    accuracies = []

    size = len(train_dataloader.dataset)

    for epoch in range(epochs):
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            

            outputs = model(x)
            loss = loss_fn(outputs, y)

            loss.backward()
            optimizer_l.step()
            optimizer_l.zero_grad()

            losses.append(loss.detach().item())

            if batch % 10 == 0:
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y).sum()
                accuracy = 100 * (correct.item() / len(outputs))
                accuracies.append(accuracy)
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"losses: {np.array(losses).mean():>7f} [{len(losses):>5d}]")
                print(f"accuracy: {accuracy:>3f}\n")
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].plot(accuracies)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Batch")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(losses)
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Batch")
    ax[1].set_ylabel("Loss")

    plt.show()
    pass

class AdaptiveDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AdaptiveDropout, self).__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        if self.training:
            dropout_rates = self.dropout_rate * torch.sigmoid(x)
            return x * torch.bernoulli(1. - dropout_rates)
        return x

class DynamicAdaptiveDropout(nn.Module):
    def __init__(self, base_dropout_rate=0.5, early_dropout_rate=0.8, early_epochs=50):
        super(DynamicAdaptiveDropout, self).__init__()
        self.base_dropout_rate = base_dropout_rate
        self.early_dropout_rate = early_dropout_rate
        self.early_epochs = early_epochs
        self.epoch_counter = 0
    
    def forward(self, x, epoch):
        if self.training:
            if epoch < self.early_epochs:
                dropout_rate = self.early_dropout_rate * torch.sigmoid(x)
            else:
                dropout_rate = self.base_dropout_rate * torch.sigmoid(x)
            return x * torch.bernoulli(1. - dropout_rate).to(x.device)
        return x

class InnovationModel(nn.Module):

    def __init__(
            self,
            in_L: int = 64,
            L: int = 64,
            *args, 
            **kwargs
        ) -> None:
        super(InnovationModel, self).__init__(*args, **kwargs)
        self.f = nn.Flatten()
        self.na = nn.Linear(in_L, L)
        self.na.weight.data.fill_(0.001)
        self.mati = cladius.MagicAutomateTrendInteractV2()
        self.n = nn.Linear(L, 1)
        self.n.weight.data.fill_(0.001)

        self.ada = AdaptiveDropout(dropout_rate=0.5)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f(x)
        x = self.na(x)
        x = self.mati(x)
        x = self.ada(x)
        x = self.n(x)
        return x
    
    def reset_weights(self, reset_rate=0.1):
        with torch.no_grad():
            for param in self.parameters():
                random_mask = (torch.rand(param.size()) > reset_rate).float()
                param.mul_(random_mask)

class InnovationModel2(nn.Module):

    def __init__(
            self,
            in_L: int = 64,
            L: int = 64,
            TE: int = 100,
            *args, 
            **kwargs
        ) -> None:
        super(InnovationModel2, self).__init__(*args, **kwargs)
        self.f = nn.Flatten()
        self.na = nn.Linear(in_L, L)
        self.na.weight.data.fill_(0.001)
        self.mati = cladius.MagicAutomateTrendInteractV2()
        self.n = nn.Linear(L, 1)
        self.n.weight.data.fill_(0.001)

        self.dada = DynamicAdaptiveDropout(base_dropout_rate=0.3, early_dropout_rate=0.15, early_epochs=TE//2)
        self.epoch = 0


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f(x)
        x = self.na(x)
        x = self.mati(x)
        x = self.dada(x, self.epoch)
        x = self.n(x)
        return x
    
    def reset_weights(self, reset_rate=0.1):
        with torch.no_grad():
            for param in self.parameters():
                random_mask = (torch.rand(param.size()) > reset_rate).float().to(param.device)
                param.mul_(random_mask)
    
    def increment_epoch(self):
        self.epoch += 1

class InnovationModel3(nn.Module):

    def __init__(
            self,
            in_L: int = 64,
            L: int = 64,
            TE: int = 100,
            *args, 
            **kwargs
        ) -> None:
        super(InnovationModel3, self).__init__(*args, **kwargs)
        self.f = nn.Flatten()
        self.na = nn.Linear(in_L, L)
        self.na.weight.data.fill_(0.001)
        self.mati = cladius.MagicAutomateTrendInteractV2()
        self.n = nn.Linear(L, 1)
        self.n.weight.data.fill_(0.001)

        self.dada = DynamicAdaptiveDropout(base_dropout_rate=0.3, early_dropout_rate=0.15, early_epochs=TE//2)
        self.epoch = 0
        self.store_initial_weights()

    def store_initial_weights(self):
        self.initial_weights = {}
        for name, param in self.named_parameters():
            self.initial_weights[name] = param.data.clone()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f(x)
        x = self.na(x)
        x = self.mati(x)
        x = self.dada(x, self.epoch)
        x = self.n(x)
        return x
    
    def reset_weights(self, reset_rate=0.1, max_reset_rate=0.2):
        with torch.no_grad():
            for name, param in self.named_parameters():
                random_mask = (torch.rand(param.size()) > reset_rate).float()
                random_mask = torch.clamp(random_mask, max=max_reset_rate)
                param.data = param.data * (1 - random_mask) + self.initial_weights[name] * random_mask
            self.store_initial_weights()

    def increment_epoch(self):
        self.epoch += 1

def unit_test_innovation():
    # torch.set_printoptions(precision=4, sci_mode=False, profile="full")
    size = (20, 20, 20)
    latent = 50
    mf = 1
    batch = 256

    channels, height, width = size
    data_x = torch.rand(batch, *size) * mf
    data_y = torch.rand(batch, 1) * mf

    eval_data_x = torch.rand(batch, *size) * mf
    eval_data_y = torch.rand(batch, 1) * mf

    # na = nn.Linear(
    #     channels * height * width,
    #     latent,
    # )
    # na.weight.data.fill_(0.001)
    # na.bias.data.fill_(0.0)
    # n = nn.Linear(
    #     latent, 
    #     1,
    # )
    # n.weight.data.fill_(0.001)
    # n.bias.data.fill_(0.0)
    # model = nn.Sequential(
    #     nn.Flatten(),
    #     na,
    #     # nn.GELU(),
    #     cladius.MagicAutomateTrendInteractV2(),
    #     nn.Dropout(0.5),
    #     n
    # )
    
    model = InnovationModel3(
        in_L=channels * height * width,
        L=latent
    )


    # na = nn.Linear(
    #     channels * height * width,
    #     latent,
    # )
    # na.weight.data.fill_(0.001)
    # na.bias.data.fill_(0.0)
    # n = nn.Linear(
    #     latent, 
    #     1,
    # )
    # n.weight.data.fill_(0.001)
    # n.bias.data.fill_(0.0)
    # darkness_model = nn.Sequential(
    #     nn.Flatten(),
    #     na,
    #     # nn.GELU(),
    #     cladius.MagicAutomateTrendInteractV2(),
    #     AdaptiveDropout(dropout_rate=0.5),
    #     n
    # )

    darkness_model = InnovationModel2(
        in_L=channels * height * width,
        L=latent
    )

    # na = nn.Linear(
    #     channels * height * width,
    #     latent,
    # )
    # na.weight.data.fill_(0.001)
    # na.bias.data.fill_(0.0)
    # n = nn.Linear(
    #     latent, 
    #     1,
    # )
    # n.weight.data.fill_(0.001)
    # n.bias.data.fill_(0.0)
    # outsider_model = nn.Sequential(
    #     nn.Flatten(),
    #     na,
    #     cladius.MagicAutomateTrendInteractV2(),
    #     # nn.Dropout(0.5),
    #     n
    # )

    outsider_model = InnovationModel(
        in_L=channels * height * width,
        L=latent
    )

    lr=1e-5
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    darkness_criterion = nn.MSELoss()
    darkness_optimizer = torch.optim.Adam(darkness_model.parameters(), lr=lr)

    outsider_criterion = nn.MSELoss()
    outsider_optimizer = torch.optim.Adam(outsider_model.parameters(), lr=lr)

    losses = []
    darkness_losses = []
    outsider_losses = []

    eval_losses = []
    darkness_eval_losses = []
    outsider_eval_losses = []


    epochs = 5000
    weights_reset_interval = epochs // 10
    change_data = True

    for epoch in range(epochs):
        if change_data:
            data_x = torch.rand(batch, *size) * mf
            data_y = torch.rand(batch, 1) * mf

            eval_data_x = torch.rand(batch, *size) * mf
            eval_data_y = torch.rand(batch, 1) * mf

        outputs = model(data_x)
        loss = criterion(outputs, data_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch [{epoch + 1:>4d}/{epochs:>4d}], Loss: {loss.item():.4f}")

        darkness_outputs = darkness_model(data_x)
        darkness_loss = darkness_criterion(darkness_outputs, data_y)
        darkness_loss.backward()
        darkness_optimizer.step()
        darkness_optimizer.zero_grad()
        print(f"Epoch [{epoch + 1:>4d}/{epochs:>4d}], Darkness Loss: {darkness_loss.item():.4f}")

        outsider_outputs = outsider_model(data_x)
        outsider_loss = outsider_criterion(outsider_outputs, data_y)
        outsider_loss.backward()
        outsider_optimizer.step()
        outsider_optimizer.zero_grad()
        print(f"Epoch [{epoch + 1:>4d}/{epochs:>4d}], Outsider Loss: {outsider_loss.item():.4f}")

        losses.append(loss.detach().numpy())
        darkness_losses.append(darkness_loss.detach().numpy())
        outsider_losses.append(outsider_loss.detach().numpy())

        eval_outputs = model(eval_data_x)
        eval_loss = criterion(eval_outputs, eval_data_y)
        eval_losses.append(eval_loss.detach().numpy())

        darkness_eval_outputs = darkness_model(eval_data_x)
        darkness_eval_loss = darkness_criterion(darkness_eval_outputs, eval_data_y)
        darkness_eval_losses.append(darkness_eval_loss.detach().numpy())

        outsider_eval_outputs = outsider_model(eval_data_x)
        outsider_eval_loss = outsider_criterion(outsider_eval_outputs, eval_data_y)
        outsider_eval_losses.append(outsider_eval_loss.detach().numpy())

        if epoch % weights_reset_interval == 0 and epoch != 0:
            model.reset_weights(0.5, 0.49995)
            darkness_model.reset_weights(0.01)
            outsider_model.reset_weights(0.01)
        darkness_model.increment_epoch()
        model.increment_epoch()
    
    remove_first_data = True
    nbr_remove = 100
    if remove_first_data:
        losses = losses[nbr_remove:]
        darkness_losses = darkness_losses[nbr_remove:]
        outsider_losses = outsider_losses[nbr_remove:]
        eval_losses = eval_losses[nbr_remove:]
        darkness_eval_losses = darkness_eval_losses[nbr_remove:]
        outsider_eval_losses = outsider_eval_losses[nbr_remove:]

    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(10, 10))
    ax[0][0].plot(losses)
    ax[0][0].set_title("Loss")
    ax[0][0].set_xlabel("Epoch")
    ax[0][0].set_ylabel("Loss")

    ax[0][1].plot(darkness_losses)
    ax[0][1].set_title("Darkness Loss")
    ax[0][1].set_xlabel("Epoch")
    ax[0][1].set_ylabel("Loss")

    ax[0][2].plot(outsider_losses)
    ax[0][2].set_title("Outsider Loss")
    ax[0][2].set_xlabel("Epoch")
    ax[0][2].set_ylabel("Loss")

    delta_loss_nd = [losses[i] - darkness_losses[i] for i in range(len(losses))]
    delta_loss_no = [losses[i] - outsider_losses[i] for i in range(len(losses))]
    delta_loss_do = [darkness_losses[i] - outsider_losses[i] for i in range(len(losses))]

    ax[1][0].plot(delta_loss_nd)
    ax[1][0].set_title("Delta Loss ND")
    ax[1][0].set_xlabel("Epoch")
    ax[1][0].set_ylabel("Loss")

    ax[1][1].plot(delta_loss_no)
    ax[1][1].set_title("Delta Loss NO")
    ax[1][1].set_xlabel("Epoch")
    ax[1][1].set_ylabel("Loss")

    ax[1][2].plot(delta_loss_do)
    ax[1][2].set_title("Delta Loss DO")
    ax[1][2].set_xlabel("Epoch")
    ax[1][2].set_ylabel("Loss")

    ax[2][0].plot(eval_losses)
    ax[2][0].set_title("Eval Loss")
    ax[2][0].set_xlabel("Epoch")
    ax[2][0].set_ylabel("Loss")

    ax[2][1].plot(darkness_eval_losses)
    ax[2][1].set_title("Darkness Eval Loss")
    ax[2][1].set_xlabel("Epoch")
    ax[2][1].set_ylabel("Loss")

    ax[2][2].plot(outsider_eval_losses)
    ax[2][2].set_title("Outsider Eval Loss")
    ax[2][2].set_xlabel("Epoch")
    ax[2][2].set_ylabel("Loss")

    plt.show()

def unit_test_new_optimizer():
    
    def node(x: torch.Tensor, func, regions_stack: list[dict[str, float]]):
        stack_results = []
        for region in regions_stack:
            start, end = region["start"], region["end"]
            mean = (start + end) / 2
            condition = (x > start) & (x < end)

            a_slope_condition = (x > start) & (x < mean)
            b_slope_condition = (x > mean) & (x < end)

            peak_value = region["peak_value"]        
            range = end - start
            slope = 2 * peak_value / range

            result = torch.where(a_slope_condition, slope * (x - start), torch.tensor(0.0))
            result = torch.where(b_slope_condition, slope * (end - x), result)
            stack_results.append(result)
        return func(x) + torch.sum(torch.stack(stack_results), dim=0)

    def func(x):
        _2pi = 2 * torch.pi
        return ((((torch.sin(x%_2pi)+x/4)-(torch.pi*torch.sin(x)+_2pi*torch.cos(x/4))-torch.cos(x%_2pi))-x/3)+torch.exp(torch.sin(x*torch.pi)))/(torch.log(torch.exp(2*torch.cos(x/_2pi))+torch.exp(torch.sin(x/_2pi)))+torch.exp(torch.cos(x/_2pi)-2*torch.sin(x/_2pi)))


    regions_stack_sgd = [
        {
            "start": -0.1,
            "end": 0.1,
            "peak_value": 0.1
        }
    ]

    regions_stack_adam = [
        {
            "start": -0.1,
            "end": 0.1,
            "peak_value": 0.1
        }
    ]

    # def func(x: torch.Tensor, regions_stack: list[dict[str, float]]):
    #     stack_results = []
    #     for region in regions_stack:
    #         start, end = region["start"], region["end"]
    #         mean = (start + end) / 2
    #         condition = (x > start) & (x < end)

    #         a_slope_condition = (x > start) & (x < mean)
    #         b_slope_condition = (x > mean) & (x < end)

    #         peak_value = region["peak_value"]        
    #         range = end - start
    #         slope = 2 * peak_value / range

    #         result = torch.where(a_slope_condition, slope * (x - start), torch.tensor(0.0))
    #         result = torch.where(b_slope_condition, slope * (end - x), result)
    #         stack_results.append(result)
    #     return torch.sin(x+(torch.pi/2)) + 1 + torch.sum(torch.stack(stack_results), dim=0)

    # def func(x: torch.Tensor, regions_stack: list[dict[str, float]]):
    #     eps = 1
    #     start, end = -0.2, 0.2
    #     mean = (start + end) / 2
    #     condition = (x > start) & (x < end)

    #     a_slope_condition = (x > start) & (x < mean)
    #     b_slope_condition = (x > mean) & (x < end)

    #     peak_value = 1        
    #     range = end - start
    #     slope = 2 * peak_value / range

    #     result = torch.where(a_slope_condition, slope * (x - start), torch.tensor(0.0))
    #     result = torch.where(b_slope_condition, slope * (end - x), result)
    #     return torch.sin(x+(torch.pi/2)) + 1 + result

    # def func(x: torch.Tensor):
    #     return (torch.sin(x+(torch.pi/2)) + 1)

    sgd_x = torch.tensor([0.0], requires_grad=True)
    adam_x = torch.tensor([0.0], requires_grad=True)

    sgd_optimizer = torch.optim.SGD([sgd_x], lr=0.03, momentum=0.9)
    adam_optimizer = torch.optim.Adam([adam_x], lr=0.03, betas=(0.2, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    sgd_history_x = []
    sgd_history_f = []

    adam_history_x = []
    adam_history_f = []


    M1 = 99.00 # /100%
    M2 = 101.00 # /100%
    EM = 100

    R1 = 0.3
    R2 = 0.2

    P = 100

    LI1 = 4

    past_epoch_local_optimum_sgd = 0
    past_epoch_local_optimum_adam = 0
    for _ in range(10000):
        print(f"Epoch {_}")
        sgd_optimizer.zero_grad()
        f = node(sgd_x, func, regions_stack_sgd)
        f.backward()
        sgd_optimizer.step()

        sgd_history_x.append(sgd_x.item())
        sgd_history_f.append(f.item())

        adam_optimizer.zero_grad()
        f = node(adam_x, func, regions_stack_adam)
        f.backward()
        adam_optimizer.step()

        adam_history_x.append(adam_x.item())
        adam_history_f.append(f.item())

        # Check if we are in optimum local
        if len(sgd_history_f) > EM:
            working_sgd = np.array(sgd_history_f[-EM:])

            # print(f"Working sgd: {working_sgd}")

            min_working_sgd = working_sgd.min()
            max_working_sgd = working_sgd.max()
            
            min_pourcentage_working_sgd: np.ndarray = (working_sgd/min_working_sgd)*100.0
            max_pourcentage_working_sgd: np.ndarray = (working_sgd/max_working_sgd)*100.0

            min_pourcentage_working_sgd_mean = min_pourcentage_working_sgd.mean()
            max_pourcentage_working_sgd_mean = max_pourcentage_working_sgd.mean()

            # print(f"Max pourcentage working sgd mean: {max_pourcentage_working_sgd_mean:.2f}%")
            # print(f"Min pourcentage working sgd mean: {min_pourcentage_working_sgd_mean:.2f}%")

            if min_pourcentage_working_sgd_mean > M1 and min_pourcentage_working_sgd_mean < M2 and past_epoch_local_optimum_sgd + LI1 < _:
                past_epoch_local_optimum_sgd = _
                regions_stack_sgd.append({
                    "start": sgd_history_x[-1] - R1, #if len(regions_stack) % 2 == 0 else sgd_history_x[-1] - 0.51,
                    "end": sgd_history_x[-1] + R2,
                    "peak_value": P
                })
                print(f"Optimum local found with SGD {min_working_sgd}")
                # break
        
        if len(adam_history_f) > EM:
            working_adam = np.array(adam_history_f[-EM:])

            # print(f"Working adam: {working_adam}")

            min_working_adam = working_adam.min()
            max_working_adam = working_adam.max()
            
            min_pourcentage_working_adam: np.ndarray = (working_adam/min_working_adam)*100.0
            max_pourcentage_working_adam: np.ndarray = (working_adam/max_working_adam)*100.0

            min_pourcentage_working_adam_mean = min_pourcentage_working_adam.mean()
            max_pourcentage_working_adam_mean = max_pourcentage_working_adam.mean()

            # print(f"Max pourcentage working adam mean: {max_pourcentage_working_adam_mean:.2f}%")
            # print(f"Min pourcentage working adam mean: {min_pourcentage_working_adam_mean:.2f}%")

            if min_pourcentage_working_adam_mean > M1 and min_pourcentage_working_adam_mean < M2 and past_epoch_local_optimum_adam + LI1 < _:
                past_epoch_local_optimum_adam = _
                regions_stack_adam.append({
                    "start": adam_history_x[-1] - R1, #if len(regions_stack) % 2 == 0 else sgd_history_x[-1] - 0.51,
                    "end": adam_history_x[-1] + R2,
                    "peak_value": P
                })
                print(f"Optimum local found with Adam {min_working_adam}")
                # break
            


    print(f"Great Global Optimum found with SGD {np.array(sgd_history_f).min()}")
    print(f"Great Global Optimum found with Adam {np.array(adam_history_f).min()}")     

    plt.figure(figsize=[10,5])
    xs = torch.linspace(-500, 500, 250000)
    ys_sgd = node(xs, func, regions_stack_sgd).detach()
    ys_adam = node(xs, func, regions_stack_adam).detach()
    plt.plot(xs, ys_sgd, '-r', label='f(x)')
    plt.plot(xs, ys_adam, '-b', label='f(x)')
    plt.scatter(sgd_history_x, sgd_history_f, c='#4848F888', label='SGD')
    plt.scatter(adam_history_x, adam_history_f, c='#48F84888', label='Adam')
    plt.title('Optimisation avec SGD et Adam')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()
    
    # plt.figure(figsize=[10,5])
    # xs = torch.linspace(-10, 10, 400)
    # ys = func(xs).detach()
    # plt.plot(xs, ys, '-r', label='f(x)')
    # plt.scatter(history_x, history_f, c='blue', label='Trajectoire de l\'optimiseur')
    # plt.title('Optimisation avec SGD')
    # plt.xlabel('x')
    # plt.ylabel('f(x)')

    # # Définir les valeurs de x
    # x = np.linspace(-10, 10, 400)
    # # Calculer les valeurs de y
    # y = x**2

    # # Tracer la fonction
    # plt.figure(figsize=[10,5])
    # plt.plot(x, y, '-r')
    # plt.title('Fonction f(x) = x^2')
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.grid(True)
    # plt.show()

    # x = np.linspace(-6, 6, 400)
    # y = np.linspace(-6, 6, 400)
    # x, y = np.meshgrid(x, y)
    # z = np.sin(np.sqrt(x**2 + y**2))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x, y, z, cmap='viridis')
    # ax.set_title('Surface Plot')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('f(x, y)')
    # plt.show()

        
    # # Définir la fonction à optimiser
    # def func(x):
    #     return (x - 3)**2

    # # Initialiser la variable x
    # x = torch.tensor([0.0], requires_grad=True)

    # # Choisir un optimiseur
    # optimizer = torch.optim.SGD([x], lr=0.1)
    # # Pour stocker les points de la trajectoire
    # history_x = []
    # history_f = []

    # # Boucle d'optimisation
    # for _ in range(25):
    #     optimizer.zero_grad()   # Réinitialiser les gradients
    #     f = func(x)             # Calculer la valeur de la fonction
    #     f.backward()            # Calculer les gradients
    #     optimizer.step()        # Mise à jour des paramètres

    #     # Enregistrer les valeurs pour la visualisation
    #     history_x.append(x.item())
    #     history_f.append(f.item())

    # # Tracer la fonction et le chemin de l'optimiseur
    # plt.figure(figsize=[10,5])
    # xs = torch.linspace(-1, 7, 100)
    # ys = func(xs).detach()
    # plt.plot(xs, ys, '-r', label='f(x) = (x-3)^2')
    # plt.scatter(history_x, history_f, c='blue', label='Trajectoire de l\'optimiseur')
    # plt.title('Optimisation avec PyTorch')
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.legend()
    # plt.show()

    # def func(x, y):
    #     return torch.sin(torch.sqrt(x**2 + y**2)) + (x/5)**2 + (y/5)**2

    # # Initialiser les variables x et y
    # x = torch.tensor([2.0], requires_grad=True)
    # y = torch.tensor([2.0], requires_grad=True)

    # # Choisir un optimiseur
    # optimizer = torch.optim.Adam([x, y], lr=0.1)

    # # Pour stocker les points de la trajectoire
    # history_x = []
    # history_y = []
    # history_f = []

    # # Boucle d'optimisation
    # for _ in range(200):
    #     optimizer.zero_grad()   # Réinitialiser les gradients
    #     f = func(x, y)          # Calculer la valeur de la fonction
    #     f.backward()            # Calculer les gradients
    #     optimizer.step()        # Mise à jour des paramètres

    #     # Enregistrer les valeurs pour la visualisation
    #     history_x.append(x.item())
    #     history_y.append(y.item())
    #     history_f.append(f.item())

    # # Tracer la fonction et le chemin de l'optimiseur
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X = np.linspace(-10, 10, 100)
    # Y = np.linspace(-10, 10, 100)
    # X, Y = np.meshgrid(X, Y)
    # Z = func(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    # Z = Z.detach().numpy()
    # ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100, cmap='viridis', edgecolor='none')
    # ax.scatter(history_x, history_y, history_f, color='r', s=10) # Chemin de l'optimiseur
    # ax.set_title('Optimisation d\'une fonction complexe avec PyTorch')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('f(x, y)')
    # plt.show()

class NodeOptimizer(nn.Module):

    def __init__(
            self, 
            *args, 
            **kwargs
        ) -> None:
        super(NodeOptimizer, self).__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        
        L = 64

        self.weights = nn.Parameter(torch.ones((784, L)), requires_grad=True)
        self.weights.data.fill_(0.001)
        self.bias = nn.Parameter(torch.zeros(L), requires_grad=True)

        self.weights_output = nn.Parameter(torch.ones((L, 10)), requires_grad=True)
        self.weights_output.data.fill_(0.001)
        self.bias_output = nn.Parameter(torch.zeros(10), requires_grad=True)

        self.mati = cladius.MagicAutomateTrendInteractV2()

        self.region_stack = [
            {
                "start": -0.1,
                "end": 0.1,
                "peak_value": 0.1
            }
        ]

        self.regions_stack = { 
            "latent": [
                {
                    "start": -0.1,
                    "end": 0.1,
                    "peak_value": 0.1
                }
            ],
            "mati": [
                {
                    "start": -0.1,
                    "end": 0.1,
                    "peak_value": 0.1
                }
            ],
            "output": [
                {
                    "start": -0.1,
                    "end": 0.1,
                    "peak_value": 0.1
                }
            ]
        }

    def node(self, x: torch.Tensor, regions_stack: list[dict[str, float]]):
        stack_results = []
        for region in regions_stack:
            start, end = region["start"], region["end"]
            mean = (start + end) / 2
            condition = (x > start) & (x < end)

            a_slope_condition = (x > start) & (x < mean)
            b_slope_condition = (x > mean) & (x < end)

            peak_value = region["peak_value"]        
            range = end - start
            slope = 2 * peak_value / range

            result = torch.where(a_slope_condition, slope * (x - start), torch.tensor(0.0))
            result = torch.where(b_slope_condition, slope * (end - x), result)
            stack_results.append(result)
        # torch.set_printoptions(precision=4, sci_mode=False, profile="full")
        # print(torch.sum(torch.stack(stack_results), dim=0))
        return x + torch.sum(torch.stack(stack_results))

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = x @ self.node(self.weights, self.region_stack) + self.bias
        # x = x @ self.weights + self.bias
        # x = self.node(F.gelu(x), self.region_stack)
        # x = self.mati(x)
        x = F.gelu(x)
        x = x @ self.node(self.weights_output, self.region_stack) + self.bias_output
        # x = self.node(x @ self.weights_output + self.bias_output, self.region_stack)
        # x = x @ self.weights_output + self.bias_output
        return x
    
    # def node_gradient(self, x: torch.Tensor, y: torch.Tensor, loss_fn: torch.nn.modules.loss._Loss):
    #     outputs = self(x)
    #     loss = loss_fn(outputs, y)
    #     loss.backward()
    #     return loss.item()
        
def unit_test_mnist_node():
    epochs = 100
    batch_size = 32
    train_dataloader, test_dataloader = cladius_utils.download_fashion_mnist(batch_size)


    for x, y in test_dataloader:
        print(f"Shape of x [N, C, H, W]: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = NodeOptimizer().to(device)
    summary(model, input_size=(batch_size, 1, 28, 28))

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer_l = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0)


    losses = []
    accuracies = []

    size = len(train_dataloader.dataset)

    M1 = 99.00 # /100%
    M2 = 101.00 # /100%
    EM = 100

    R1 = 0.01
    R2 = 0.01

    P = 1

    LI1 = 4

    past_epoch_local_optimum = 0
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            

            outputs = model(x)
            loss = loss_fn(outputs, y)

            loss.backward()
            optimizer_l.step()
            optimizer_l.zero_grad()

            losses.append(loss.detach().item())

            if batch % EM == 0 and batch != 0:
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y).sum()
                accuracy = 100 * (correct.item() / len(outputs))
                accuracies.append(accuracy)
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"losses: {np.array(losses).mean():>7f} [{len(losses):>5d}]")
                print(f"accuracy: {accuracy:>3f}\n")
            
            if batch % EM == 0 and batch != 0:
                working = np.array(losses[-EM:])
                min_working = working.min()
                max_working = working.max()
                eps = 1e-6

                min_pourcentage_working: np.ndarray = (working/(min_working+eps))*100.0
                max_pourcentage_working: np.ndarray = (working/(max_working+eps))*100.0

                min_pourcentage_working_mean = min_pourcentage_working.mean()
                max_pourcentage_working_mean = max_pourcentage_working.mean()

                # torch.set_printoptions(precision=4, sci_mode=False, profile="full")
                # print(working)
                print(f"Max pourcentage working mean: {max_pourcentage_working_mean:.2f}%")
                # exit()

                if min_pourcentage_working_mean > M1 and min_pourcentage_working_mean < M2 and past_epoch_local_optimum + LI1 < batch:
                    past_epoch_local_optimum = batch
                    model.region_stack.append({
                        "start": -R1,
                        "end": R2,
                        "peak_value": P
                    })                    
                    print(f"Optimum local found {min_working} {min_pourcentage_working_mean:.2f}%")

                # model.regions_stack["latent"].append({
                #     "start": -R1,
                #     "end": R2,
                #     "peak_value": P
                # })
                # model.regions_stack["mati"].append({
                #     "start": -R1,
                #     "end": R2,
                #     "peak_value": P
                # })
                # model.regions_stack["output"].append({
                #     "start": -R1,
                #     "end": R2,
                #     "peak_value": P
                # })

    print("GGO with SGD: ", np.array(losses).min())
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].plot(accuracies)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Batch")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(losses)
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Batch")
    ax[1].set_ylabel("Loss")

class DiagonalKernelAverage(nn.Module):
    def __init__(
            self,
            size: int | tuple[int, int] = 28,
            with_channels: bool = False,
            *args, 
            **kwargs) -> None:
        super(DiagonalKernelAverage, self).__init__(*args, **kwargs)
        self.size = size
        self.with_channels = with_channels

    def forward(self, x: torch.Tensor):
        diag_kernel_average_tl = []
        diag_kernel_average_tr = []
        diag_kernel_average_bl = []
        diag_kernel_average_br = []

        for i in range(1, self.size + 1):
            diag_kernel_average_tl.append(torch.mean(x[:, :, :i, :i], dim=(2, 3)))
            diag_kernel_average_tr.append(torch.mean(x[:, :, :i, -i:], dim=(2, 3)))
            diag_kernel_average_bl.append(torch.mean(x[:, :, -i:, :i], dim=(2, 3)))
            diag_kernel_average_br.append(torch.mean(x[:, :, -i:, -i:], dim=(2, 3)))

        diag_kernel_average_tl = torch.stack(diag_kernel_average_tl, dim=1)
        diag_kernel_average_tr = torch.stack(diag_kernel_average_tr, dim=1)
        diag_kernel_average_bl = torch.stack(diag_kernel_average_bl, dim=1)
        diag_kernel_average_br = torch.stack(diag_kernel_average_br, dim=1)

        diag_kernel = torch.cat([diag_kernel_average_tl, diag_kernel_average_tr, diag_kernel_average_bl, diag_kernel_average_br], dim=2)
        if self.with_channels:
            diag_kernel = diag_kernel.unsqueeze(1)
        return diag_kernel

class DiagonalKernelAverageV2(nn.Module):
    def __init__(
            self,
            size: int | tuple[int, int] = 28,
            with_channels: bool = False,
            *args, 
            **kwargs) -> None:
        super(DiagonalKernelAverageV2, self).__init__(*args, **kwargs)
        self.size = size
        self.with_channels = with_channels

    def forward(self, x: torch.Tensor):
        diag_kernel_average_tl = []
        diag_kernel_average_tr = []
        diag_kernel_average_bl = []
        diag_kernel_average_br = []

        for i in range(1, self.size + 1):
            if i == 1:
                diag_kernel_average_tl.append(torch.mean(x[:, :, :i, :i], dim=(2, 3)))
                diag_kernel_average_tr.append(torch.mean(x[:, :, :i, -i:], dim=(2, 3)))
                diag_kernel_average_bl.append(torch.mean(x[:, :, -i:, :i], dim=(2, 3)))
                diag_kernel_average_br.append(torch.mean(x[:, :, -i:, -i:], dim=(2, 3)))
            else:
                # padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                # padding[:, :, :i-1, :i-1] = x[:, :, :i-1, :i-1]
                # diag_kernel_average_tl.append(torch.mean((x[:, :, :i, :i]+i) - padding, dim=(2, 3)))

                # padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                # padding[:, :, :i-1, -i+1:] = x[:, :, :i-1, -i+1:]
                # diag_kernel_average_tr.append(torch.mean((x[:, :, :i, -i:]+i) - padding, dim=(2, 3)))

                # padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                # padding[:, :, -i+1:, :i-1] = x[:, :, -i+1:, :i-1]
                # diag_kernel_average_bl.append(torch.mean((x[:, :, -i:, :i]+i) - padding, dim=(2, 3)))

                # padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                # padding[:, :, -i+1:, -i+1:] = x[:, :, -i+1:, -i+1:]
                # diag_kernel_average_br.append(torch.mean((x[:, :, -i:, -i:]+i) - padding, dim=(2, 3)))

                # padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                # padding[:, :, :i-1, :i-1] = x[:, :, :i-1, :i-1]
                # diag_kernel_average_tl.append(torch.mean((x[:, :, :i, :i]) - padding, dim=(2, 3)))

                # padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                # padding[:, :, :i-1, -i+1:] = x[:, :, :i-1, -i+1:]
                # diag_kernel_average_tr.append(torch.mean((x[:, :, :i, -i:]) - padding, dim=(2, 3)))

                # padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                # padding[:, :, -i+1:, :i-1] = x[:, :, -i+1:, :i-1]
                # diag_kernel_average_bl.append(torch.mean((x[:, :, -i:, :i]) - padding, dim=(2, 3)))

                # padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                # padding[:, :, -i+1:, -i+1:] = x[:, :, -i+1:, -i+1:]
                # diag_kernel_average_br.append(torch.mean((x[:, :, -i:, -i:]) - padding, dim=(2, 3)))

                mean_without_0 = ((i*i)-((i-1)*(i-1)))

                padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                padding[:, :, :i-1, :i-1] = x[:, :, :i-1, :i-1]
                diag_kernel_average_tl.append(torch.sum((x[:, :, :i, :i]) - padding, dim=(2, 3)) / mean_without_0)

                padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                padding[:, :, :i-1, -i+1:] = x[:, :, :i-1, -i+1:]
                diag_kernel_average_tr.append(torch.sum((x[:, :, :i, -i:]) - padding, dim=(2, 3)) / mean_without_0)

                padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                padding[:, :, -i+1:, :i-1] = x[:, :, -i+1:, :i-1]
                diag_kernel_average_bl.append(torch.sum((x[:, :, -i:, :i]) - padding, dim=(2, 3)) / mean_without_0)

                padding = torch.zeros((x.shape[0], x.shape[1], i, i), device=x.device, dtype=x.dtype)
                padding[:, :, -i+1:, -i+1:] = x[:, :, -i+1:, -i+1:]
                diag_kernel_average_br.append(torch.sum((x[:, :, -i:, -i:]) - padding, dim=(2, 3)) / mean_without_0)


        diag_kernel_average_tl = torch.stack(diag_kernel_average_tl, dim=1)
        diag_kernel_average_tr = torch.stack(diag_kernel_average_tr, dim=1)
        diag_kernel_average_bl = torch.stack(diag_kernel_average_bl, dim=1)
        diag_kernel_average_br = torch.stack(diag_kernel_average_br, dim=1)

        diag_kernel = torch.cat([diag_kernel_average_tl, diag_kernel_average_tr, diag_kernel_average_bl, diag_kernel_average_br], dim=2)
        if self.with_channels:
            diag_kernel = diag_kernel.unsqueeze(1)
        return diag_kernel



# class DepthwiseMultiKernelAttention(nn.Module):
#     """
#     In this layer we must do same things than a normal depthwise convolution.
#     We need to iniatialize out_channels, kernel_size, stride, padding, dilation (Parameters).
#     When we do the forward pass we must to apply the attention mechanism for each kernel, i will explain more.

#     1: Kernel forward must respect the normal depthwise convolution so the stride, padding, dilation etc etc...
#     2: When one kernel is applied we must to apply the attention mechanism from the input so (Input -> Attention -> Residual Kernel Attention).
#     3: We must repeat the step 1, 2 for each kernel.
#     4: Pointwise convolution for the outputs.

#     After we must to do a global attention from the inputs and outputs -> to the residual outputs.
#     """
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: int,
#             stride: int,
#             padding: int,
#             dilation: int,
#             *args, 
#             **kwargs
#         ) -> None:
#         super(DepthwiseMultiKernelAttention, self).__init__(*args, **kwargs)

class DepthwiseMultiKernelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(DepthwiseMultiKernelAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Manual weight initialization for depthwise convolution
        self.depthwise_weights = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        
        # Manual weight initialization for pointwise convolution
        self.pointwise_weights = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        
        # Attention weights
        self.attention_weights = nn.Parameter(torch.randn(in_channels, in_channels))
        
        # Global attention weight
        self.global_attention_weight = nn.Parameter(torch.randn(in_channels, in_channels))

    def forward(self, x):
        # Manually applying padding
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        
        # Implementing depthwise convolution from scratch
        B, C, H, W = x_padded.shape
        out_height = (H - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_width = (W - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        depthwise_out = torch.zeros((B, C, out_height, out_width), device=x.device, dtype=x.dtype)
        
        for b in range(B):  # Batch size
            for c in range(C):  # In-channels
                for h in range(out_height):
                    for w in range(out_width):
                        h_start, w_start = h * self.stride, w * self.stride
                        h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size
                         # Extracting the kernel's receptive field
                        receptive_field = x_padded[b, c, h_start:h_end, w_start:w_end]
                        
                        # Flatten the receptive field and the corresponding channel for computing attention
                        flatten_rf = receptive_field.flatten().unsqueeze(1)
                        flatten_channel = x_padded[b, c].flatten().unsqueeze(1)
                        attention_scores = flatten_channel @ flatten_rf.T
                        attention_scores = F.softmax(attention_scores, dim=-1)  # Normalizing scores

                        fig, ax = plt.subplots(2, 4, figsize=(10, 10))
                        manager = plt.get_current_fig_manager()
                        manager.full_screen_toggle()

                        ax[0, 0].imshow(F.softmax(attention_scores, dim=0), cmap="magma", aspect="auto")
                        ax[0, 0].set_title("Attention Scores Channel x RF DIM 0")
                        ax[0, 1].imshow(F.softmax(attention_scores, dim=1), cmap="magma", aspect="auto")
                        ax[0, 1].set_title("Attention Scores Channel x RF DIM 1")
                        
                        ax[0, 2].imshow(F.softmax(attention_scores, dim=0) * F.softmax(attention_scores, dim=1), cmap="magma", aspect="auto")
                        ax[0, 2].set_title("Attention Scores Channel x RF - DIM 0 x 1")

                        ax[0, 3].imshow(F.softmax(attention_scores, dim=0) @ F.softmax(attention_scores, dim=1).T, cmap="magma", aspect="auto")
                        ax[0, 3].set_title("Attention Scores Channel x RF - DIM 0 @ 1.T")
                        
                        ax[1, 0].imshow(F.softmax(attention_scores, dim=0).T @ F.softmax(attention_scores, dim=1), cmap="magma", aspect="auto")
                        ax[1, 0].set_title("Attention Scores Channel x RF - DIM 0.T @ 1")

                        ax[1, 1].imshow(F.softmax(attention_scores, dim=1).T @ F.softmax(attention_scores, dim=0), cmap="magma", aspect="auto")
                        ax[1, 1].set_title("Attention Scores Channel x RF - DIM 1.T @ 0")

                        ax[1, 2].imshow(F.softmax(attention_scores, dim=1) @ F.softmax(attention_scores, dim=0).T, cmap="magma", aspect="auto")
                        ax[1, 2].set_title("Attention Scores Channel x RF - DIM 1 @ 0.T")
                        plt.show()
                        # fig, ax = plt.subplots(2, 4, figsize=(10, 10))
                        # manager = plt.get_current_fig_manager()
                        # manager.full_screen_toggle()

                        # # ax[0, 0].imshow(receptive_field, cmap="magma")
                        # # ax[0, 0].set_title("Receptive Field")
                        # # ax[0, 1].imshow(x_padded[b, c], cmap="magma")
                        # # ax[0, 1].set_title("Channel")

                        # # ax[2].imshow(receptive_field.flatten().unsqueeze(1))
                        # # ax[2].set_title("Flatten RF")
                        # # ax[3].imshow(x_padded[b, c].flatten().unsqueeze(1))
                        # # ax[3].set_title("Flatten Channel")

                        # flatten_rf = receptive_field.flatten().unsqueeze(1)
                        # flatten_channel = x_padded[b, c].flatten().unsqueeze(1)

                        # print(flatten_rf.shape, flatten_channel.shape)
                        
                        # # Computing attention scores between receptive field and the entire channel
                        # # attention_scores = torch.matmul(flatten_rf, flatten_channel.T)
                        # attention_scores = flatten_channel @ flatten_rf.T

                        # ax[0, 0].imshow(attention_scores, cmap="magma", aspect="auto")
                        # ax[0, 0].set_title("Attention Scores Channel x RF")

                        # attention_scores = F.softmax(attention_scores, dim=-1)  # Normalizing scores

                        # ax[0, 1].imshow(attention_scores, cmap="magma", aspect="auto")
                        # ax[0, 1].set_title("Softmax Attention Scores Channel x RF")

                        # attention_scores_b = flatten_rf @ flatten_channel.T

                        # ax[0, 2].imshow(attention_scores_b, cmap="magma", aspect="auto")
                        # ax[0, 2].set_title("Attention Scores RF x Channel")

                        # attention_scores_b = F.softmax(attention_scores_b, dim=-1)  # Normalizing scores

                        # ax[0, 3].imshow(attention_scores_b, cmap="magma", aspect="auto")
                        # ax[0, 3].set_title("Softmax Attention Scores RF x Channel")

                        # # plt.show()

                        # print(attention_scores.shape)

                        # sum_attention_scores = torch.sum(attention_scores, dim=0, keepdim=True)
                        # ax[1, 0].imshow(sum_attention_scores, cmap="magma", aspect="auto")
                        # ax[1, 0].set_title("Sum Attention Scores Dim 0")

                        # sum_attention_scores = torch.sum(attention_scores, dim=1, keepdim=True)
                        # ax[1, 1].imshow(sum_attention_scores, cmap="magma", aspect="auto")
                        # ax[1, 1].set_title("Sum Attention Scores Dim 1")

                        
                        # sum_attention_scores_b = torch.sum(attention_scores_b, dim=0, keepdim=True)
                        # ax[1, 2].imshow(sum_attention_scores_b, cmap="magma", aspect="auto")
                        # ax[1, 2].set_title("Sum Attention Scores Dim 0")

                        # sum_attention_scores_b = torch.sum(attention_scores_b, dim=1, keepdim=True)
                        # ax[1, 3].imshow(sum_attention_scores_b, cmap="magma", aspect="auto")
                        # ax[1, 3].set_title("Sum Attention Scores Dim 1")

                        # # plt.show()

                        # # print(torch.sum(attention_scores, dim=0, keepdim=True).shape, sum_attention_scores_b.shape)
                        # # exit(0)

                        # fig, ax = plt.subplots(2, 4, figsize=(10, 10))
                        # manager = plt.get_current_fig_manager()
                        # manager.full_screen_toggle()


                        # print(flatten_rf)
                        # print(torch.sum(attention_scores, dim=0, keepdim=True).T)
                        # print(self.depthwise_weights[c, 0])

                        # ax[0, 0].imshow((flatten_rf * torch.sum(attention_scores, dim=0, keepdim=True).T).view_as(receptive_field) * self.depthwise_weights[c, 0], cmap="magma", aspect="auto")
                        # ax[0, 0].set_title("Weighted Channel x RF")

                        # ax[0, 1].imshow((flatten_rf * sum_attention_scores_b).view_as(receptive_field) * self.depthwise_weights[c, 0], cmap="magma", aspect="auto")
                        # ax[0, 1].set_title("Weighted RF x Channel")
                        
                        # ax[0, 2].imshow(self.depthwise_weights[c, 0], cmap="magma", aspect="auto")
                        # ax[0, 2].set_title("Depthwise Kernel")

                        # ax[0, 3].imshow(receptive_field * self.depthwise_weights[c, 0], cmap="magma", aspect="auto")
                        # ax[0, 3].set_title("Depthwise Kernel Convolution")


                        # ax[1, 0].imshow((flatten_rf * torch.sum(attention_scores, dim=0, keepdim=True).T).view_as(receptive_field), cmap="magma", aspect="auto")
                        # ax[1, 0].set_title("Applied Attention Channel x RF")

                        # ax[1, 1].imshow((flatten_rf * sum_attention_scores_b).view_as(receptive_field), cmap="magma", aspect="auto")
                        # ax[1, 1].set_title("Applied Attention RF x Channel")


                        # ax[1, 3].imshow(receptive_field, cmap="magma", aspect="auto")
                        # ax[1, 3].set_title("Receptive Field")

                        # plt.show()

                        # exit(0)
                        # ax[1, 0].imshow(flatten_rf * attention_scores[:flatten_rf.size(0)], cmap="magma", aspect="auto")
                        # ax[1, 0].set_title("Weighted RF")
                        # weighted_rf = torch.sum(flatten_rf * attention_scores[:flatten_rf.size(0)], dim=1)  # Ensure shapes are aligned

                        # ax[1, 1].imshow(weighted_rf.view_as(receptive_field) * self.depthwise_weights[c, 0], cmap="magma", aspect="auto")
                        # ax[1, 1].set_title("Depthwise Kernel Attention Convolution")

                        # ax[1, 2].imshow(x_padded[b, c, h_start:h_end, w_start:w_end] * self.depthwise_weights[c, 0], cmap="magma", aspect="auto")
                        # ax[1, 2].set_title("Depthwise Convolution")

                        # print(attention_scores.shape)
                        # print(attention_scores[0])
                        # plt.show()
                        # exit(0)
                        # # Apply attention before element-wise multiplication
                        # # Here we're simplifying; normally, you'd reshape or process this further.
                        # weighted_rf = flatten_rf * attention_scores[:flatten_rf.size(0)]  # Ensure shapes are aligned
                        
                        # # Applying the depthwise weight and summing up
                        # depthwise_out[b, c, h, w] = torch.sum(weighted_rf.view_as(receptive_field) * self.depthwise_weights[c, 0])
                        # # Element-wise multiplication followed by sum (convolution)

                        # flatten_x = x_padded[b, c, h_start:h_end, w_start:w_end].flatten() # MY TEST DONT WORRY
                        # flatten_actual_channel = x_padded[b, c].flatten() # MY TEST DONT WORRY

                        # print(flatten_x.shape, flatten_actual_channel.shape) # MY TEST DONT WORRY
                        # exit(0) # MY TEST DONT WORRY

                        # print(x_padded[b, c, h_start:h_end, w_start:w_end] * self.depthwise_weights[c, 0]) # MY TEST DONT WORRY
                        # exit(0) # MY TEST DONT WORRY

                        # # For GPT: Look here that here i wanted to implement the attention mechanism for each kernel from the actual channels.
                        # # For GPT: So how could we do that ? in the sum or external ?
                        # # For GPT: I think that we will be better to do that in the sum.
                        # # For GPT: but we need to flatten or no ?  
                        # depthwise_out[b, c, h, w] = torch.sum(
                        #     x_padded[b, c, h_start:h_end, w_start:w_end] * self.depthwise_weights[c, 0]
                        # )
        exit(0)
        # Applying self-attention for each channel
        attention_applied = torch.zeros_like(depthwise_out)
        for c in range(C):  # In-channels (also equals to out-channels for depthwise)
            attention_applied[:, c] = F.softmax(self.attention_weights[c], dim=1) @ depthwise_out[:, c].view(B, -1).T
            attention_applied[:, c] = attention_applied[:, c].view(B, out_height, out_width)
        
        # Implementing pointwise convolution from scratch
        pointwise_out = torch.zeros((B, self.out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
        for b in range(B):
            for k in range(self.out_channels):
                for c in range(C):
                    pointwise_out[b, k] += attention_applied[b, c] * self.pointwise_weights[k, c, 0, 0]
        
        # Applying global attention (simplified)
        global_attention_out = torch.zeros_like(pointwise_out)
        for c in range(self.out_channels):  # Out-channels
            global_attention_out[:, c] = F.softmax(self.global_attention_weight, dim=1) @ pointwise_out[:, c].view(B, -1).T
            global_attention_out[:, c] = global_attention_out[:, c].view(B, out_height, out_width)
        
        return global_attention_out

class Conscious2D(nn.Module):
    def __init__(self, input_dim, generated_features_dim, attn_heads):
        super(Conscious2D, self).__init__()
        self.input_dim = input_dim
        self.generated_features_dim = generated_features_dim
        self.attn_heads = attn_heads

        self.mean = nn.Parameter(torch.randn(generated_features_dim))
        self.std = nn.Parameter(torch.abs(torch.randn(generated_features_dim)))

        self.attention = nn.MultiheadAttention(embed_dim=self.input_dim + self.generated_features_dim, num_heads=attn_heads)

    def forward(self, x):
        batch_size = x.shape[0]
        generated_features = torch.randn(batch_size, self.generated_features_dim) * self.std + self.mean
        combined_features = torch.cat((x, generated_features), dim=1).unsqueeze(0)
        attended_features, _ = self.attention(combined_features, combined_features, combined_features)
        return attended_features.squeeze(0)

class Galileo(nn.Module):

    def __init__(
            self,
            with_weights: bool = False,
            depth: int = 1,
            *args, 
            **kwargs
        ) -> None:
        super(Galileo, self).__init__(*args, **kwargs)
        self.depth = depth
        
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        all_depth_attention_scores = []

        for b in range(B):  # Iterate over the batch
            # Initial processing of the image
            image = x[b].unsqueeze(0)  # Add batch dimension back for compatibility
            
            # Initial attention calculations
            side_scores = [image]  # Start with the input image as the initial 'side'
            inter_scores = []  # Initialize the list for inter scores

            all_side_scores = []

            # Hierarchical expansion based on depth
            for d in range(self.depth):
                new_side_scores = []
                for side in side_scores:
                    # Calculate new sides based on each existing side
                    first_side = F.softmax(side @ side.transpose(-1, -2), dim=-1)
                    second_side = F.softmax(side.transpose(-1, -2) @ side, dim=-1)
                    
                    # Add new sides to the list
                    new_side_scores.extend([first_side, second_side])
                    all_side_scores.extend([first_side, second_side])

                    # Create new inter scores from the new sides
                    inter_scores.append(first_side @ second_side)

                # Update the list of side scores for the next depth level
                side_scores = new_side_scores
            # Concatenate all side and inter scores across all depths for this batch item
            total_side_scores = torch.cat(all_side_scores, dim=1)
            total_inter_scores = torch.cat(inter_scores, dim=1)
            total_attention_scores = torch.cat([total_side_scores, total_inter_scores, image], dim=1)
            all_depth_attention_scores.append(total_attention_scores)

        # Concatenate all the attention scores for all batch items
        output = torch.cat(all_depth_attention_scores, dim=0)
        return output

    def _forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        recursive_depth = []


        for b in range(B):
            image = x[b].squeeze(0)

            # For GPT: It's here where you must add the functionnality for the attention mechanism.
            print(image.shape)

            first_side_attention_scores = image @ image.T
            second_side_attention_scores = image.T @ image

            first_side_attention_scores = F.softmax(first_side_attention_scores, dim=0)
            second_side_attention_scores = F.softmax(second_side_attention_scores, dim=0)

            inter_side_attention_scores = first_side_attention_scores @ second_side_attention_scores

            recursive_depth.append(first_side_attention_scores)
            recursive_depth.append(second_side_attention_scores)
            recursive_depth.append(inter_side_attention_scores)

            # For GPT: Here we must to apply the attention mechanism for each depth.
            # For GPT: So you could see the commentary under.
            # For GPT: In the loop depth we will have 2 first side attention scores, 2 second side attention scores and 3 inter side attention scores. if depth is 2.

            # first_first_side_attention_scores = first_side_attention_scores @ first_side_attention_scores.T
            # first_second_side_attention_scores = first_side_attention_scores.T @ first_side_attention_scores

            # inter_first_side_attention_scores = first_first_side_attention_scores @ second_first_side_attention_scores

            # second_first_side_attention_scores = second_side_attention_scores @ second_side_attention_scores.T
            # second_second_side_attention_scores = second_side_attention_scores.T @ second_side_attention_scores

            # inter_second_side_attention_scores = first_second_side_attention_scores @ second_second_side_attention_scores



            # print(first_side_attention_scores.shape, second_side_attention_scores.shape)


            # print(inter_side_attention_scores.shape)

            # fig, ax = plt.subplots(3, 3, figsize=(10, 10))

            # ax[0, 0].imshow(first_side_attention_scores, cmap="magma", aspect="auto")
            # ax[0, 0].set_title("First Side Attention Scores")

            # ax[0, 1].imshow(second_side_attention_scores, cmap="magma", aspect="auto")
            # ax[0, 1].set_title("Second Side Attention Scores")

            # ax[0, 2].imshow(inter_side_attention_scores, cmap="magma", aspect="auto")
            # ax[0, 2].set_title("Inter Side Attention Scores")

            # ax[1, 0].imshow(first_first_side_attention_scores, cmap="magma", aspect="auto")
            # ax[1, 0].set_title("First First Side Attention Scores")

            # ax[1, 1].imshow(second_first_side_attention_scores, cmap="magma", aspect="auto")
            # ax[1, 1].set_title("Second First Side Attention Scores")

            # ax[1, 2].imshow(inter_first_side_attention_scores, cmap="magma", aspect="auto")
            # ax[1, 2].set_title("Inter First Side Attention Scores")

            # ax[2, 0].imshow(first_second_side_attention_scores, cmap="magma", aspect="auto")
            # ax[2, 0].set_title("First Second Side Attention Scores")

            # ax[2, 1].imshow(second_second_side_attention_scores, cmap="magma", aspect="auto")
            # ax[2, 1].set_title("Second Second Side Attention Scores")

            # ax[2, 2].imshow(inter_second_side_attention_scores, cmap="magma", aspect="auto")
            # ax[2, 2].set_title("Inter Second Side Attention Scores")



            # plt.show()

            # exit(0)

            # flatten_x = x[b].flatten(1)

            # print(flatten_x.shape)
            
            # first_side_attention_scores =  flatten_x @ flatten_x.T
            # second_side_attention_scores = flatten_x.T @ flatten_x

            # print(first_side_attention_scores.shape, second_side_attention_scores.shape)

            # inter_side_attention_scores = first_side_attention_scores @ second_side_attention_scores

            # print(inter_side_attention_scores.shape)
            # exit(0)

        # For GPT: Now we will concatenate all the attention scores of each depth in the channels.
        # For GPT: Example: B, 1, 28, 28 -> B, 4, 28, 28 with depth = 1 || B, 10, 28, 28 with depth = 2
        pass

class GalileoV2(nn.Module):

    def __init__(self, size: tuple = (1, 28, 28), with_weights: bool = False, depth: int = 1, *args, **kwargs) -> None:
        C, W, H = size
        super(GalileoV2, self).__init__(*args, **kwargs)
        self.depth = depth
        self.with_weights = with_weights
        # If with_weights is True, initialize weight matrices

        res = ((2 ** np.arange(0, depth)) * 3).sum() + 1

        if self.with_weights:
            self.weights = nn.ParameterList([nn.Parameter(torch.randn(H, W)) for _ in range(res)])  # Adjust size according to the number of layers and attention matrices
    
    def apply_weights(self, side_scores, weights):
        # Apply weights to the side scores
        weighted_scores = []
        for score, weight in zip(side_scores, weights):
            weighted_score = score @ weight
            weighted_scores.append(weighted_score)
        return weighted_scores
    
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        all_depth_attention_scores = []

        for b in range(B):  # Iterate over the batch
            # Initial processing of the image
            image = x[b].unsqueeze(0)  # Add batch dimension back for compatibility
            side_scores = [image]  # Start with the input image as the initial 'side'
            inter_scores = []  # Initialize the list for inter scores
            all_side_scores = []

            # Hierarchical expansion based on depth
            for d in range(self.depth):
                new_side_scores = []
                for side in side_scores:
                    # Calculate new sides based on each existing side
                    first_side = F.softmax(side @ side.transpose(-1, -2), dim=-1)
                    second_side = F.softmax(side.transpose(-1, -2) @ side, dim=-1)
                    new_side_scores.extend([first_side, second_side])
                    all_side_scores.extend([first_side, second_side])
                    inter_scores.append(first_side @ second_side)

                side_scores = new_side_scores
            
            # If weights are to be applied, modify the side and inter scores accordingly
            # print(len(inter_scores), len(all_side_scores))
            if self.with_weights:
                all_side_scores = self.apply_weights(all_side_scores, self.weights[:len(all_side_scores) + 1])
                inter_scores = self.apply_weights(inter_scores, self.weights[len(all_side_scores) + 1:(len(all_side_scores) + len(inter_scores) + 1)])
            # print(len(inter_scores), len(all_side_scores))
            # exit(0)

            # Concatenate all side and inter scores across all depths for this batch item
            total_side_scores = torch.cat(all_side_scores, dim=1)
            total_inter_scores = torch.cat(inter_scores, dim=1)
            total_attention_scores = torch.cat([total_side_scores, total_inter_scores, image], dim=1)
            all_depth_attention_scores.append(total_attention_scores)

        # Concatenate all the attention scores for all batch items
        output = torch.cat(all_depth_attention_scores, dim=0)
        return output


# class Conscious2D(nn.Module):
#     """
#     This Layer must be used to generate correlation between the input features and some generated features from a gaussian distribution (From 2D inputs).
#     Each generated feature must be a gaussian distribution with a mean and a standard deviation.

#     After we must to use the attention mechanism for extract the important features from both input and generated features.

#     That's allow the model to generate some specific case after find the local optimum. so make him more flexible, more robust (Conscious).

#     I don't know how the input can be.
#     """
    
#     def __init__(
#             self, 
#             *args, 
#             **kwargs
#         ) -> None:
#         super(Conscious2D, self).__init__(*args, **kwargs)

class DiagonalKernelAverageLinear(nn.Module):
    def __init__(
            self,
            activation: nn.Module = None,
            out_features: int = 1,
            size: int | tuple[int, int] = 28,
            keep_dims: bool = False,
            extra_bias: bool = False,
            test: bool = False,
            *args, 
            **kwargs
        ) -> None:
        super(DiagonalKernelAverageLinear, self).__init__(*args, **kwargs)
        self.diag_avg = DiagonalKernelAverageV2(
            size=size,
            with_channels=False
        )
        input_features = 4
        self.size = size
        self.keep_dims = keep_dims
        self.activation = activation
        self.extra_bias = extra_bias

        self.activations = nn.ModuleList([activation() for _ in range(size)])

        self.weights = nn.Parameter(torch.randn(self.size, input_features, out_features))
        self.biases = nn.Parameter(torch.randn(self.size, out_features))
        if extra_bias:
            self.extra_biases = nn.Parameter(torch.randn(self.size, out_features))
        if test:
            self.weights.data.fill_(0.001)
            self.biases.data.fill_(0.0)
            if extra_bias:
                self.extra_biases.data.fill_(0.0)


    def forward(self, x: torch.Tensor):
        x = self.diag_avg(x)
        outputs = []
        for i in range(self.size):
            out = F.linear(x[:, i, :], self.weights[i].T, self.biases[i])
            if self.activation is not None:
                out = self.activations[i](out)
            if self.extra_bias:
                out = out + self.extra_biases[i]
            outputs.append(out)
        x = torch.cat(outputs, dim=1)
        if self.keep_dims:
            return x.unsqueeze(1)
        return x.squeeze(-1)

class ModelKirby(nn.Module):
    def __init__(
            self,
            output_features: int = 10,
            custom: bool = False,
            test: bool = False,
            *args, 
            **kwargs
        ) -> None:
        super(ModelKirby, self).__init__(*args, **kwargs)
        # L=1
        # LA=28
        # self.diag_avg_linear = DiagonalKernelAverageLinear(
        #     activation=cladius.MagicAutomateTrendInteractV2,
        #     out_features=L,
        #     size=28,
        #     keep_dims=False,
        #     extra_bias=True,
        #     test=test
        # )
        

        # self.diag_avg_linear = DiagonalKernelAverage(
        #     size=28,
        #     with_channels=False
        # )

        self.dropout = nn.Dropout(0.2)

        if custom:
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(28*28, output_features)
        else:
            DG = 4
            self.flatten = nn.Flatten()
            self.galileo = GalileoV2(
                with_weights=True,
                depth=DG
            )
            # C=4
            # self.node_modules = nn.ModuleList()
            # in_channels = 22
            # for c in range(1, C + 1):
            #     self.node_modules.append(nn.Sequential(nn.Conv2d(
            #         in_channels=in_channels,
            #         out_channels=10 // (c * 2),
            #         kernel_size=3,
            #         bias=False
            #     ), cladius.MagicAutomateTrendInteractV2()))
            #     in_channels = 10 // (c * 2)

            C=4
            self.node_modules = nn.ModuleList()
            in_channels = ((2 ** np.arange(0, DG)) * 3).sum() + 1

            # for c in range(1, C + 1):
            #     self.node_modules.append(nn.AvgPool2d(2))


            self.linear = nn.Linear(28*28*in_channels, output_features, bias=False)
            # self.dmka = DepthwiseMultiKernelAttention(
            #     in_channels=1,
            #     out_channels=1,
            #     kernel_size=3,
            #     stride=1,
            #     padding=0,
            #     dilation=1           
            # )
            # self.conscious = Conscious2D(28*28, 28*28, 4)
            # self.linear = nn.Linear(28*28+28*28, output_features)
            # self.linear = nn.Linear(28*L, output_features)
            self.mati = cladius.MagicAutomateTrendInteractV2()
            # self.pre_linear = nn.Linear(28*L, LA)
            # self.linear = nn.Linear(LA, output_features)

        self.custom = custom
        if test:
            self.linear.weight.data.fill_(0.001)
            self.linear.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor):
        if self.custom:
            x = self.flatten(x)
        else:            
            # x = self.dmka(x)

            # x = self.diag_avg_linear(x)
            x = self.galileo(x)
            # for conv in self.node_modules:
            #     x = conv(x)
            # return x
            x = self.flatten(x)
            # x = self.conscious(x)
            x = self.mati(x)
            # x = self.pre_linear(x)
        # x = self.dropout(x)
        x = self.linear(x)
        return x



def unit_test_kirby(device: torch.device):
    epochs = 10
    batch_size = 100
    train_dataloader, test_dataloader = cladius_utils.download_fashion_mnist(batch_size)


    for x, y in test_dataloader:
        print(f"Shape of x [N, C, H, W]: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = ModelKirby(
        output_features=10,
        custom=False,
        test=False
    ).to(device)
    
    model_b = ModelKirby(
        output_features=10,
        custom=True,
        test=False
    ).to(device)
    summary(model, input_size=(batch_size, 1, 28, 28))
    summary(model_b, input_size=(batch_size, 1, 28, 28))
    # exit(0)

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer_l = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss_fn_b = nn.CrossEntropyLoss(reduction="mean")
    optimizer_l_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

    losses = []
    accuracies = []

    losses_b = []
    accuracies_b = []

    size = len(train_dataloader.dataset)

    for epoch in range(epochs):
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            # loss = loss_fn(outputs, y.float().unsqueeze(1))
            loss = loss_fn(outputs, y)

            loss.backward()
            optimizer_l.step()
            optimizer_l.zero_grad()

            losses.append(loss.detach().item())

            if batch % 10 == 0:
                predicted = torch.argmax(outputs.data, dim=1)
                correct = (predicted == y).sum()
                accuracy = 100 * (correct.item() / len(outputs))
                accuracies.append(accuracy)
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"losses: {np.array(losses).mean():>7f} [{len(losses):>5d}]")
                print(f"accuracy: {accuracy:>3f}\n")

            outputs_b = model_b(x)
            loss_b = loss_fn_b(outputs_b, y)

            loss_b.backward()
            optimizer_l_b.step()
            optimizer_l_b.zero_grad()

            losses_b.append(loss_b.detach().item())

            if batch % 10 == 0:
                predicted = torch.argmax(outputs_b.data, dim=1)
                correct = (predicted == y).sum()
                accuracy = 100 * (correct.item() / len(outputs_b))
                accuracies_b.append(accuracy)
                loss, current = loss_b.item(), (batch + 1) * len(x)
                print(f"loss b: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"losses b: {np.array(losses_b).mean():>7f} [{len(losses_b):>5d}]")
                print(f"accuracy b: {accuracy:>3f}\n")

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].plot(accuracies)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Batch")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(losses)
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Batch")
    ax[1].set_ylabel("Loss")

    ax[0].plot(accuracies_b)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Batch")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(losses_b)
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Batch")
    ax[1].set_ylabel("Accuracy")

    ax[0].legend(["Model A", "Model B"])
    ax[1].legend(["Model A", "Model B"])

    plt.show()

if __name__ == "__main__":
    device = cladius.using_gpu_or_cpu()
    unit_test_kirby(device)
    # unit_test_mnist_node()
    # unit_test_new_optimizer()
    # unit_test_innovation()
    # unit_test_model_bottleneck_residual(device)
    # unit_test_model_combined_bottleneck_residual(device)
    # unit_test_mix_spatial_channel_attention(device)