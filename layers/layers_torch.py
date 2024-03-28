import torch

from layers.activations.logish_torch import Logish
from layers.activations.gish_torch import Gish
from layers.activations.matriach_torch import Matriach
from layers.activations.mati_torch import MagicAutomateTrendInteract, \
MagicAutomateTrendInteractV2


from layers.convolutions.global_pooling_torch import GlobalAvgPool1d, \
GlobalAvgPool2d, \
GlobalAvgPool3d, \
GlobalMaxPool1d, \
GlobalMaxPool2d, \
GlobalMaxPool3d, \
GlobalMinPooling1D, \
GlobalMinPooling2D, \
GlobalMinPooling3D, \
GlobalAvgLearnablePooling1D, \
GlobalAvgLearnablePooling2D, \
GlobalAvgLearnablePooling3D, \
GlobalMaxLearnablePooling1D, \
GlobalMaxLearnablePooling2D, \
GlobalMaxLearnablePooling3D, \
GlobalMinLearnablePooling1D, \
GlobalMinLearnablePooling2D, \
GlobalMinLearnablePooling3D, \
GlobalChannelCovariancePooling, \
GlobalSpatialCovariancePooling, \
GlobalLearnablePooling, \
GlobalLearnablePoolingV2

from layers.convolutions.bottleneck_residual_torch import BottleneckResidualBlock, CombinedBottleneckResidualBlock

from layers.convolutions.mix_spatial_channel_attention import MixSpatialChannelAttention

def using_gpu_or_cpu():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device