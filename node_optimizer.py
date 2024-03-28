
import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn import datasets

class ExampleModel(nn.Module):
    def __init__(
            self, 
            in_features: int,
            out_features: int,
            *args, 
            **kwargs
        ) -> None:
        super(ExampleModel, self).__init__(*args, **kwargs)

        self.weights = nn.Parameter(torch.rand((in_features, out_features)), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros((out_features, 1)), requires_grad=True)

    def forward(self, inputs):
        return (inputs @ self.weights) + self.bias


class NodeOptimizer(nn.Module):
    def __init__(
            self,
            parameters,
            *args, 
            **kwargs
        ) -> None:
        super(NodeOptimizer, self).__init__(*args, **kwargs)
        self._parameters = parameters

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        print(self._parameters)
        return None


def unit_test():

    iris = datasets.load_iris()

    X = torch.tensor(iris.data[:, 0: 2], dtype=torch.float32)
    Y = torch.tensor(iris.target, dtype=torch.int64).unsqueeze(1)

    model = ExampleModel(2, 1)
    # optimizer = NodeOptimizer(model.parameters())
    # optimizer(None)
    

if __name__ == "__main__":
    unit_test()