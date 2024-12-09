import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList


class MaxOut(Module):
    """
    MaxOut layer implementation for neural networks.

    This layer applies multiple linear transformations to the input and
    computes the element-wise maximum across the outputs of these transformations.
    """

    def __init__(self, input_dim: int, output_dim: int, num_units: int = 2, use_bias: bool = True) -> None:
        """
        Initialize the MaxOut layer.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            num_units (int): Number of linear transformations (default: 2).
            use_bias (bool): Whether to include bias in the linear transformations (default: True).
        """
        super().__init__()
        self.layers = ModuleList([Linear(input_dim, output_dim, bias=use_bias) for _ in range(num_units)])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the MaxOut layer.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, input_dim)`.

        Returns:
            Tensor: Output tensor of shape `(batch_size, output_dim)`, where the maximum values
            are computed element-wise across the outputs of the linear transformations.
        """
        max_output = self.layers[0](x)  # Initialize with the output of the first layer
        for layer in self.layers[1:]:
            max_output = torch.max(max_output, layer(x))  # Element-wise max across layers
        return max_output
