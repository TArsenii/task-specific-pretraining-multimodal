import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList


class MaxOut(Module):
    def __init__(self, input_dim: int, output_dim: int, num_units: int = 2, use_bias: bool = True):
        super(MaxOut, self).__init__()
        self.layers = ModuleList([Linear(input_dim, output_dim, bias=use_bias) for _ in range(num_units)])

    def forward(self, x: Tensor) -> Tensor:
        max_output = self.layers[0](x)
        for layer in self.layers[1:]:
            max_output = torch.max(max_output, layer(x))
        return max_output
