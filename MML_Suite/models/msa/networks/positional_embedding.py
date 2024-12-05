import math
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import Module


def create_position_indices(tensor: Tensor, padding_idx: int, left_pad: bool) -> Tensor:
    """Create position indices for input tensor, handling padding appropriately.

    Replaces non-padding symbols with their position numbers, starting from
    padding_idx + 1. Handles both left and right padding cases.

    Args:
        tensor: Input tensor of shape (batch_size, sequence_length)
        padding_idx: Index used for padding
        left_pad: Whether padding is on the left side

    Returns:
        Tensor of same shape as input with positions instead of symbol indices
    """
    # Calculate maximum position needed
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()

    # Create or retrieve position buffer for this device
    buffer_name = f"range_buffer_{device}"
    if not hasattr(create_position_indices, buffer_name):
        setattr(create_position_indices, buffer_name, tensor.new())

    # Ensure buffer has correct type
    position_buffer = getattr(create_position_indices, buffer_name)
    position_buffer = position_buffer.type_as(tensor)
    setattr(create_position_indices, buffer_name, position_buffer)

    # Expand buffer if needed
    if position_buffer.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=position_buffer)

    # Create mask for non-padding positions
    mask = tensor.ne(padding_idx)

    # Get positions and expand to tensor size
    positions = position_buffer[: tensor.size(1)].expand_as(tensor)

    # Adjust positions for left padding
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)

    # Create output tensor with positions
    result = tensor.clone()
    return result.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(Module):
    """Sinusoidal positional embeddings module.

    This module produces sinusoidal positional embeddings of any length as described
    in "Attention Is All You Need". It handles padding and supports both left and
    right padding configurations.

    The implementation uses cached weights for efficiency and supports dynamic
    expansion when longer sequences are encountered.

    Args:
        embedding_dim: Dimension of the positional embeddings
        padding_idx: Index used for padding tokens (default: 0)
        left_pad: Whether padding is applied on the left (default: False)
        init_size: Initial size of the embedding cache (default: 128)

    Attributes:
        embedding_dim: Dimension of the positional embeddings
        padding_idx: Index used for padding
        left_pad: Whether padding is on the left side
        weights: Dictionary mapping devices to embedding weights
    """

    def __init__(self, embedding_dim: int, padding_idx: int = 0, left_pad: bool = False, init_size: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights: Dict[int, Tensor] = {}
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None) -> Tensor:
        """Generate sinusoidal positional embeddings.

        Creates positional embeddings using sine and cosine functions of different
        frequencies, as described in the Transformer paper. This allows the model to
        easily learn relative positions.

        Args:
            num_embeddings: Number of positions to embed
            embedding_dim: Dimension of each positional embedding
            padding_idx: Optional index to zero out for padding

        Returns:
            Tensor of shape (num_embeddings, embedding_dim) containing positional embeddings
        """
        half_dim = embedding_dim // 2

        # Create frequency scale
        emb_scale = math.log(10000) / (half_dim - 1)

        # Create dimension indices
        dim_indices = torch.arange(embedding_dim, dtype=torch.int32)

        # Calculate frequency multipliers
        frequencies = torch.exp((dim_indices // 2).to(torch.float) * -emb_scale)

        # Create position-frequency matrix
        pos_frequencies = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * frequencies.unsqueeze(0)

        # Apply sine to even indices and cosine to odd indices
        emb = pos_frequencies.clone()
        emb[:, dim_indices % 2 == 0] = torch.sin(pos_frequencies[:, dim_indices % 2 == 0])
        emb[:, dim_indices % 2 == 1] = torch.cos(pos_frequencies[:, dim_indices % 2 == 1])

        # Handle odd embedding dimensions
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)

        # Zero out padding token position
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Generate positional embeddings for input tensor.

        Args:
            input_tensor: Input tensor of shape (batch_size, sequence_length)

        Returns:
            Tensor of shape (batch_size, sequence_length, embedding_dim) containing
            positional embeddings
        """
        batch_size, seq_len = input_tensor.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input_tensor.get_device()

        # Recompute/expand embeddings if needed
        if device not in self.weights or max_pos > self.weights[device].size(0):
            self.weights[device] = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)

        # Ensure correct tensor type
        self.weights[device] = self.weights[device].type_as(self._float_tensor)

        # Get position indices
        positions = create_position_indices(input_tensor, self.padding_idx, self.left_pad)

        # Select and reshape embeddings
        return self.weights[device].index_select(0, positions.view(-1)).view(batch_size, seq_len, -1).detach()

    def max_positions(self) -> int:
        """Get maximum number of supported positions.

        Returns:
            Maximum sequence length supported by the embedding
        """
        return int(1e7)  # Arbitrary large number for practical purposes
