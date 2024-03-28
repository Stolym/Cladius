from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

def download_fashion_mnist(batch_size: int = 32):
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return DataLoader(training_data, batch_size=batch_size), DataLoader(test_data, batch_size=batch_size)

