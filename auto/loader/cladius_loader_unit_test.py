from layers.utils_torch import download_fashion_mnist
from auto.loader.i_loader import ILoader

from auto.utils.cladius_serializer import base64_to_ndarray, ndarray_to_base64

from auto.externals.colors import Colors, print_color

import torch

class FashionMNISTLoader(ILoader):
    def __init__(self) -> None:
        super().__init__()

    def load_data(self) -> tuple:
        dataloader_train, dataloader_test = download_fashion_mnist()

        data_x = []
        data_y = []

        for x, y in dataloader_train:
            data_x.append(x)
            data_y.append(y)

        for x, y in dataloader_test:
            data_x.append(x)
            data_y.append(y)

        # convert to numpy
            
        data_x = torch.cat(data_x, 0).numpy()
        data_y = torch.cat(data_y, 0).numpy()
        
        data_x_b64 = ndarray_to_base64(data_x)
        data_y_b64 = ndarray_to_base64(data_y)

        print_color(f"Data X Shape: {data_x.shape}", Colors.DARK_YELLOW)
        print_color(f"Data Y Shape: {data_y.shape}", Colors.DARK_YELLOW)

        return data_x_b64, data_y_b64, data_x.shape, data_y.shape, data_x.dtype, data_y.dtype
