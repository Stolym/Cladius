import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.convolutions.global_pooling_torch import GlobalAvgPool2d, \
GlobalMaxPool2d, \
GlobalMinPooling2D, \
GlobalChannelCovariancePooling3D \

# from activations.mati_torch import MagicAutomateTrendInteractV2

# from torchinfo import summary

# import numpy as np

# import matplotlib.pyplot as plt


# class MagicAutomateTrendInteractV2(nn.Module):
#     def __init__(self):
#         super(MagicAutomateTrendInteractV2, self).__init__()
#         self.params = nn.Parameter(torch.rand(12) * (0.001 - 0.0001) + 0.0001)  # Uniform initialization
#         self.params_bias = nn.Parameter(torch.zeros(6))

#     def forward(self, inputs):
#         alpha, alpham, beta, betam, gamma, gammam, gammad, delta, deltam, epsilon, epsilonm, zeta = self.params.unbind()
#         balpha, bbeta, bgamma, bdelta, bepsilon, bzeta = self.params_bias.unbind()
        
#         gelu_part = alpham * (inputs * torch.sigmoid(alpha * (inputs * 1.702))) + balpha
#         soft_part = betam * F.softmax(beta * inputs, dim=-1) + bbeta
#         daa_part = gamma * inputs + gammam * torch.exp(gammad * inputs) + bgamma
#         naaa_part = deltam * torch.tanh(delta * (2 * inputs)) + bdelta
#         paaa_part = epsilonm * torch.log(1 + 0.5 * torch.abs(epsilon * inputs)) + bepsilon
#         aaa_part = torch.where(inputs < 0, naaa_part, paaa_part)
#         linear_part = zeta * inputs + bzeta

#         combined_activation = gelu_part + soft_part + daa_part + aaa_part + linear_part
        
#         return combined_activation

# class AdaptiveAsymActivation(nn.Module):

#     def __init__(self, *args, **kwargs) -> None:
#         super(AdaptiveAsymActivation, self).__init__(*args, **kwargs)

#     def forward(self, x):
#         negative_part = torch.tanh(x * 2)
#         positive_part = torch.log(1 + 0.5 * torch.abs(x))
#         return torch.where(x < 0, negative_part, positive_part)


class MixSpatialChannelAttention(nn.Module):

    def __init__(
            self,
            softmax_norm: bool = False,
            residual: bool = True,
            *args, 
            **kwargs) -> None:
        super(MixSpatialChannelAttention, self).__init__(*args, **kwargs)

        self.global_avg_pool    =   GlobalAvgPool2d()
        self.global_max_pool    =   GlobalMaxPool2d()
        self.global_min_pool    =   GlobalMinPooling2D()

        self.global_channel_covariance_pool =   GlobalChannelCovariancePooling3D()
        self.residual= residual
        self.softmax_norm = softmax_norm


    def forward(self, x):
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        min_pool = self.global_min_pool(x)

        # Mix
        mix = torch.stack([min_pool, avg_pool, max_pool], dim=1).unsqueeze(-1).unsqueeze(-1)
        cov_mix = self.global_channel_covariance_pool(mix)
        mix = mix.squeeze(-1).squeeze(-1)
        if self.softmax_norm:
            mix = mix * F.softmax(torch.sum(cov_mix, dim=1, keepdim=False).unsqueeze(-1))
        else:
            mix = mix * torch.sum(cov_mix, dim=1, keepdim=False).unsqueeze(-1)
    
        mix = mix.unsqueeze(-1).unsqueeze(-1)
        mix = torch.sum(mix, dim=1, keepdim=False) / mix.shape[1]

        spatial_average = torch.mean(x, dim=(1), keepdim=True)
        spatial_max, _ = torch.max(x, dim=(1), keepdim=True)
        spatial_min, _ = torch.min(x, dim=(1), keepdim=True)

        smix = torch.stack([spatial_min, spatial_average, spatial_max], dim=1)

        cov_smix = self.global_channel_covariance_pool(smix)
        if self.softmax_norm:
            smix = smix * F.softmax(torch.sum(cov_smix, dim=2, keepdim=True).unsqueeze(-1).unsqueeze(-1))
        else:
            smix = smix * torch.sum(cov_smix, dim=2, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        smix = torch.sum(smix, dim=(1), keepdim=False) / smix.shape[1]
        return (x + mix + smix if self.residual else torch.cat([x, mix, smix], dim=(1, 2, 3)))    

# def unit_test_mix_spatial_channel_attention():
#     size = (6, 70, 35)
#     latent = 64
#     channels, height, width = size
#     data_x = torch.rand(50, *size) * 100
#     data_y = torch.rand(50, 1) * 100

#     na = nn.Linear(
#         channels * height * width,
#         latent,
#     )
#     na.weight.data.fill_(0.001)
#     na.bias.data.fill_(0.0)
#     n = nn.Linear(
#         latent, 
#         1,
#     )
#     n.weight.data.fill_(0.001)
#     n.bias.data.fill_(0.0)
#     model = nn.Sequential(
#         MixSpatialChannelAttention(),
#         nn.Flatten(),
#         na,
#         # nn.GELU(),
#         MagicAutomateTrendInteractV2(),
#         nn.Dropout(0.5),
#         n
#     )
    
#     na = nn.Linear(
#         channels * height * width,
#         latent,
#     )
#     na.weight.data.fill_(0.001)
#     na.bias.data.fill_(0.0)
#     n = nn.Linear(
#         latent, 
#         1,
#     )
#     n.weight.data.fill_(0.001)
#     n.bias.data.fill_(0.0)
#     darkness_model = nn.Sequential(
#         MixSpatialChannelAttention(),
#         nn.Flatten(),
#         na,
#         nn.GELU(),
#         nn.Dropout(0.5),
#         n
#     )


#     na = nn.Linear(
#         channels * height * width,
#         latent,
#     )
#     na.weight.data.fill_(0.001)
#     na.bias.data.fill_(0.0)
#     n = nn.Linear(
#         latent, 
#         1,
#     )
#     n.weight.data.fill_(0.001)
#     n.bias.data.fill_(0.0)
#     outsider_model = nn.Sequential(
#         # MixSpatialChannelAttention(),
#         nn.Flatten(),
#         na,
#         # AdaptiveAsymActivation(),
#         nn.GELU(),
#         nn.Dropout(0.5),
#         n
#     )


#     lr=1e-3
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     darkness_criterion = nn.MSELoss()
#     darkness_optimizer = torch.optim.Adam(darkness_model.parameters(), lr=lr)

#     outsider_criterion = nn.MSELoss()
#     outsider_optimizer = torch.optim.Adam(outsider_model.parameters(), lr=lr)

#     losses = []
#     darkness_losses = []
#     outsider_losses = []

#     epochs = 2500
#     for epoch in range(epochs):
#         outputs = model(data_x)
#         loss = criterion(outputs, data_y)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         print(f"Epoch [{epoch + 1:>4d}/{epochs:>4d}], Loss: {loss.item():.4f}")

#         darkness_outputs = darkness_model(data_x)
#         darkness_loss = darkness_criterion(darkness_outputs, data_y)
#         darkness_loss.backward()
#         darkness_optimizer.step()
#         darkness_optimizer.zero_grad()
#         print(f"Epoch [{epoch + 1:>4d}/{epochs:>4d}], Darkness Loss: {darkness_loss.item():.4f}")

#         outsider_outputs = outsider_model(data_x)
#         outsider_loss = outsider_criterion(outsider_outputs, data_y)
#         outsider_loss.backward()
#         outsider_optimizer.step()
#         outsider_optimizer.zero_grad()
#         print(f"Epoch [{epoch + 1:>4d}/{epochs:>4d}], Outsider Loss: {outsider_loss.item():.4f}")

#         losses.append(loss.detach().numpy())
#         darkness_losses.append(darkness_loss.detach().numpy())
#         outsider_losses.append(outsider_loss.detach().numpy())
    
#     fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 5))
#     ax[0][0].plot(losses)
#     ax[0][0].set_title("Loss")
#     ax[0][0].set_xlabel("Epoch")
#     ax[0][0].set_ylabel("Loss")

#     ax[0][1].plot(darkness_losses)
#     ax[0][1].set_title("Darkness Loss")
#     ax[0][1].set_xlabel("Epoch")
#     ax[0][1].set_ylabel("Loss")

#     ax[0][2].plot(outsider_losses)
#     ax[0][2].set_title("Outsider Loss")
#     ax[0][2].set_xlabel("Epoch")
#     ax[0][2].set_ylabel("Loss")

#     delta_loss_nd = [losses[i] - darkness_losses[i] for i in range(len(losses))]
#     delta_loss_no = [losses[i] - outsider_losses[i] for i in range(len(losses))]
#     delta_loss_do = [darkness_losses[i] - outsider_losses[i] for i in range(len(losses))]

#     ax[1][0].plot(delta_loss_nd)
#     ax[1][0].set_title("Delta Loss ND")
#     ax[1][0].set_xlabel("Epoch")
#     ax[1][0].set_ylabel("Loss")

#     ax[1][1].plot(delta_loss_no)
#     ax[1][1].set_title("Delta Loss NO")
#     ax[1][1].set_xlabel("Epoch")
#     ax[1][1].set_ylabel("Loss")

#     ax[1][2].plot(delta_loss_do)
#     ax[1][2].set_title("Delta Loss DO")
#     ax[1][2].set_xlabel("Epoch")
#     ax[1][2].set_ylabel("Loss")

#     plt.show()




# def voronoi_instance_generator(
#         map_size: tuple = (100, 100),
#         n_points: int = 10,
#         distance: int = 5,
#     ):
#     map = np.zeros(map_size)
#     points = np.random.randint(0, map_size[0], (n_points, 2))
#     for i in range(map_size[0]):
#         for j in range(map_size[1]):
#             distances = np.linalg.norm(points - [i, j], axis=1)
#             if np.min(distances) < distance:
#                 map[i, j] = np.argmin(distances)
#             else:
#                 map[i, j] = -1
#     return map
    

# def fake_data_generator(
#         n_samples: int = 100,
#         map_size: tuple = (100, 100),
#         n_points: int = 10,
#         distance: int = 5,
#     ):
#     data = np.array([voronoi_instance_generator(
#         map_size=map_size,
#         n_points=n_points,
#         distance=distance
#     ) for _ in range(n_samples)])
#     return data



# def generate_voronoi_instance(
#         size: int = 10,
#         n_points: int = 20, 
#         map_size: tuple = (10, 10),
#         distance: float = 2.5,
#         multiplier: float = 5
#     ):
#     data_x = []
#     data_y = []

#     size = size
#     map_size = map_size
#     distance = distance
#     multiplier = multiplier
#     for i in range(size):
#         # print(f"Generating [{i + 1:>3d}/{size:>3d}] data !")
#         r_points = np.random.randint(1, n_points)
#         n_samples = 10 #np.random.randint(1, 5)
#         data = fake_data_generator(
#             map_size=map_size,
#             n_samples=n_samples,
#             distance=distance,
#             n_points=r_points
#         ) * multiplier
#         data_x.append(data)

#         [data_y.append(r_points) for x in range(n_samples)]
#     data_x = np.array(data_x).reshape(-1, 1, *map_size)
#     data_y = np.array(data_y).reshape(-1, 1)
#     return data_x, data_y


# def unit_test_mix_spatial_channel_attention_voronoi(data_x, data_y, map_size):
#     size = map_size

#     latent = 512
#     channels, height, width = size

#     data_x = torch.tensor(data_x, dtype=torch.float32)
#     data_y = torch.tensor(data_y, dtype=torch.float32)

    
#     na = nn.Linear(
#         channels * height * width,
#         latent,
#     )
#     na.weight.data.fill_(0.0001)
#     na.bias.data.fill_(0.0)
#     n = nn.Linear(
#         latent, 
#         1,
#     )
#     n.weight.data.fill_(0.0001)
#     n.bias.data.fill_(0.0)
#     model = nn.Sequential(
#         MixSpatialChannelAttention(),
#         nn.Flatten(),
#         na,
#         nn.LayerNorm(latent),
#         MagicAutomateTrendInteractV2(),
#         # AdaptiveAsymActivation(),
#         nn.Dropout(0.5),
#         n
#     )
    
#     na = nn.Linear(
#         channels * height * width,
#         latent,
#     )
#     na.weight.data.fill_(0.0001)
#     na.bias.data.fill_(0.0)
#     n = nn.Linear(
#         latent, 
#         1,
#     )
#     n.weight.data.fill_(0.0001)
#     n.bias.data.fill_(0.0)
#     darkness_model = nn.Sequential(
#         # MixSpatialChannelAttention(),
#         nn.Flatten(),
#         na,
#         nn.LayerNorm(latent),
#         MagicAutomateTrendInteractV2(),
#         # nn.GELU(),
#         nn.Dropout(0.5),
#         n
#     )


#     na = nn.Linear(
#         channels * height * width,
#         latent,
#     )
#     na.weight.data.fill_(0.0001)
#     na.bias.data.fill_(0.0)
#     n = nn.Linear(
#         latent, 
#         1,
#     )
#     n.weight.data.fill_(0.0001)
#     n.bias.data.fill_(0.0)
#     outsider_model = nn.Sequential(
#         nn.Flatten(),
#         na,
#         nn.LayerNorm(latent),
#         nn.GELU(),
#         # AdaptiveAsymActivation(),
#         nn.Dropout(0.5),
#         n
#     )


#     # summary(model, input_size=(1, *size))
#     # summary(darkness_model, input_size=(1, *size))
#     # summary(outsider_model, input_size=(1, *size))
#     # exit()
#     lr=1e-3
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     darkness_criterion = nn.MSELoss()
#     darkness_optimizer = torch.optim.Adam(darkness_model.parameters(), lr=lr)

#     outsider_criterion = nn.MSELoss()
#     outsider_optimizer = torch.optim.Adam(outsider_model.parameters(), lr=lr)

#     losses = []
#     darkness_losses = []
#     outsider_losses = []



#     epochs = 1000
#     for epoch in range(epochs):
#         data_x, data_y = generate_voronoi_instance()
#         data_x = torch.tensor(data_x, dtype=torch.float32)
#         data_y = torch.tensor(data_y, dtype=torch.float32)
#         outputs = model(data_x)
#         loss = criterion(outputs, data_y)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         print(f"Epoch [{epoch + 1:>4d}/{epochs:>4d}], Loss: {loss.item():.4f}")

#         darkness_outputs = darkness_model(data_x)
#         darkness_loss = darkness_criterion(darkness_outputs, data_y)
#         darkness_loss.backward()
#         darkness_optimizer.step()
#         darkness_optimizer.zero_grad()
#         print(f"Epoch [{epoch + 1:>4d}/{epochs:>4d}], Darkness Loss: {darkness_loss.item():.4f}")

#         outsider_outputs = outsider_model(data_x)
#         outsider_loss = outsider_criterion(outsider_outputs, data_y)
#         outsider_loss.backward()
#         outsider_optimizer.step()
#         outsider_optimizer.zero_grad()
#         print(f"Epoch [{epoch + 1:>4d}/{epochs:>4d}], Outsider Loss: {outsider_loss.item():.4f}")

#         losses.append(loss.detach().numpy())
#         darkness_losses.append(darkness_loss.detach().numpy())
#         outsider_losses.append(outsider_loss.detach().numpy())
    
#     fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 5))
#     ax[0][0].plot(losses)
#     ax[0][0].set_title("Loss")
#     ax[0][0].set_xlabel("Epoch")
#     ax[0][0].set_ylabel("Loss")

#     ax[0][1].plot(darkness_losses)
#     ax[0][1].set_title("Darkness Loss")
#     ax[0][1].set_xlabel("Epoch")
#     ax[0][1].set_ylabel("Loss")

#     ax[0][2].plot(outsider_losses)
#     ax[0][2].set_title("Outsider Loss")
#     ax[0][2].set_xlabel("Epoch")
#     ax[0][2].set_ylabel("Loss")

#     delta_loss_nd = [losses[i] - darkness_losses[i] for i in range(len(losses))]
#     delta_loss_no = [losses[i] - outsider_losses[i] for i in range(len(losses))]
#     delta_loss_do = [darkness_losses[i] - outsider_losses[i] for i in range(len(losses))]

#     ax[1][0].plot(delta_loss_nd)
#     ax[1][0].set_title("Delta Loss ND")
#     ax[1][0].set_xlabel("Epoch")
#     ax[1][0].set_ylabel("Loss")

#     ax[1][1].plot(delta_loss_no)
#     ax[1][1].set_title("Delta Loss NO")
#     ax[1][1].set_xlabel("Epoch")
#     ax[1][1].set_ylabel("Loss")

#     ax[1][2].plot(delta_loss_do)
#     ax[1][2].set_title("Delta Loss DO")
#     ax[1][2].set_xlabel("Epoch")
#     ax[1][2].set_ylabel("Loss")

#     plt.show()

# if __name__ == "__main__":
#     data_x = []
#     data_y = []

#     size = 1000
#     map_size = (10, 10)
#     distance = 2.5
#     multiplier = 2
#     for i in range(size):
#         # print(f"Generating [{i + 1:>3d}/{size:>3d}] data !")
#         r_points = np.random.randint(1, 20)
#         n_samples = 10 #np.random.randint(1, 5)
#         data = fake_data_generator(
#             map_size=map_size,
#             n_samples=n_samples,
#             distance=distance,
#             n_points=r_points
#         ) * multiplier
#         data_x.append(data)

#         [data_y.append(r_points) for x in range(n_samples)]
#     data_x = np.array(data_x).reshape(-1, 1, *map_size)
#     data_y = np.array(data_y).reshape(-1, 1)

#     # print(data_x[0])
#     # print(data_y[0])
#     # print(data_x.shape, data_y.shape)

#     # map = data_x[0][0]
#     # plt.imshow(map)
#     # plt.title("Voronoi Instance Map")
#     # plt.xlabel("X")
#     # plt.ylabel("Y")
#     # plt.colorbar(label="Cluster ID")
#     # plt.show()

#     unit_test_mix_spatial_channel_attention_voronoi(data_x, data_y, (1, *map_size))

#     # map = data[0]
#     # map = voronoi_instance_generator(n_points=50)
#     # plt.imshow(map)
#     # plt.title("Voronoi Instance Map")
#     # plt.xlabel("X")
#     # plt.ylabel("Y")
#     # plt.colorbar(label="Cluster ID")
#     # plt.show()
#     # unit_test_mix_spatial_channel_attention()