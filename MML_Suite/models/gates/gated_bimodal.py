import torch
from torch import Tensor
from torch.nn import Linear, Module, Sigmoid, Tanh


class GatedBiModalNetwork(Module):
    """
    A gated network that combines two modalities with learned gating.

    Args:
        input_one_dim (int): Input dimension for the first modality.
        input_two_dim (int): Input dimension for the second modality.
        output_one_size (int): Output size for the first modality.
        output_two_size (int): Output size for the second modality.
        use_bias (bool, optional): Whether to use biases in the linear layers. Default is False.
    """

    def __init__(
        self,
        input_one_dim: int,
        input_two_dim: int,
        output_one_size: int,
        output_two_size: int,
        *,
        use_bias: bool = False,
    ):
        super().__init__()

        # Linear layers for each modality
        self.fc_one = Linear(input_one_dim, output_one_size, bias=use_bias)
        self.fc_two = Linear(input_two_dim, output_two_size, bias=use_bias)

        # Gate computation layer
        self.hidden_sigmoid = Linear(output_one_size + output_two_size, 1, bias=use_bias)

        # Activations
        self.activation = Tanh()
        self.gate_activation = Sigmoid()

    def forward(self, modality_one: Tensor, modality_two: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            modality_one (Tensor): Input tensor for the first modality, shape [batch_size, input_one_dim].
            modality_two (Tensor): Input tensor for the second modality, shape [batch_size, input_two_dim].

        Returns:
            Tensor: Combined gated output, shape [batch_size, output_one_size].
        """
        # Process each modality
        output_one = self.activation(self.fc_one(modality_one))
        output_two = self.activation(self.fc_two(modality_two))

        # Combine and compute gating
        combined_features = torch.cat([output_one, output_two], dim=1)
        gate = self.gate_activation(self.hidden_sigmoid(combined_features))

        # Apply gating mechanism
        gated_output = gate.unsqueeze(-1) * output_one + (1 - gate).unsqueeze(-1) * output_two
        return gated_output
