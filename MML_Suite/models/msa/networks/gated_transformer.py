import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ModuleList, Sequential, Sigmoid, init

from .div_encoder import DIVEncoder
from .multihead_attention import MultiheadAttention
from .positional_embedding import SinusoidalPositionalEmbedding


def create_future_mask(tensor_one: Tensor, tensor_two: Optional[Tensor] = None) -> Tensor:
    """Creates a future mask for transformer attention.

    Args:
        tensor_one: First tensor to determine mask dimension
        tensor_two: Optional second tensor to determine second mask dimension

    Returns:
        Future mask tensor filled with -inf where attention should be masked
    """
    dim_one = dim_two = tensor_one.size(0)
    if tensor_two is not None:
        dim_two = tensor_two.size(0)

    future_mask = torch.triu(_fill_negative_infinity(torch.ones(dim_one, dim_two)), 1 + abs(dim_two - dim_one))

    if tensor_one.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim_one, :dim_two]


def _fill_negative_infinity(tensor: Tensor) -> Tensor:
    """Fills a tensor with negative infinity in a FP16-compatible way.

    Args:
        tensor: Input tensor to fill

    Returns:
        Tensor filled with -inf values
    """
    return tensor.float().fill_(float("-inf")).type_as(tensor)


def create_linear_layer(in_features: int, out_features: int, bias: bool = True) -> Linear:
    """Creates a linear layer with Xavier initialization.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term

    Returns:
        Initialized linear layer
    """
    layer = Linear(in_features, out_features, bias)
    init.xavier_uniform_(layer.weight)
    if bias:
        init.constant_(layer.bias, 0.0)
    return layer


def create_layer_norm(embedding_dim: int) -> LayerNorm:
    """Creates a layer normalization module.

    Args:
        embedding_dim: Dimension to normalize over

    Returns:
        Layer normalization module
    """
    return LayerNorm(embedding_dim)


class TransformerEncoderLayer(Module):
    """Transformer encoder layer with gated attention mechanism.

    This implements a single transformer encoder layer with additional gating
    mechanisms for memory and attention control. The layer follows the architecture
    described in "Attention is All You Need" with some modifications.

    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        attn_dropout: Dropout rate for attention weights
        relu_dropout: Dropout rate for ReLU activations
        res_dropout: Dropout rate for residual connections
        attn_mask: Whether to use attention masking
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        relu_dropout: float = 0.1,
        res_dropout: float = 0.1,
        attn_mask: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        # Dropout rates
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        # Control gates
        self.memory_projection = Sequential(create_linear_layer(2 * embed_dim, embed_dim), Sigmoid())
        self.attention_projection = Sequential(create_linear_layer(2 * embed_dim, embed_dim), Sigmoid())

        # Feed-forward network
        self.feed_forward_one = create_linear_layer(self.embed_dim, 4 * self.embed_dim)
        self.feed_forward_two = create_linear_layer(4 * self.embed_dim, self.embed_dim)

        # Layer normalization
        self.layer_norms = ModuleList([create_layer_norm(self.embed_dim) for _ in range(2)])

    def _create_attention_mask(
        self, batch_size: int, length_one: int, length_two: int, lengths: Tensor, is_language_projection: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Creates attention masks for the transformer.

        Args:
            batch_size: Size of batch
            length_one: Length of first sequence
            length_two: Length of second sequence
            lengths: Tensor of actual sequence lengths
            is_language_projection: Whether projecting to language modality

        Returns:
            Tuple of (additive mask, multiplicative mask)
        """
        assert length_one == length_two

        bool_mask_one = torch.cuda.BoolTensor(batch_size, length_one, length_two)
        bool_mask_two = torch.cuda.BoolTensor(batch_size, length_one, length_two)

        for idx, length in enumerate(lengths):
            bool_mask_two[idx, :, :] = False
            if length < length_one:
                bool_mask_one[idx, length:, :] = True
                bool_mask_one[idx, :, length:] = True
                bool_mask_two[idx, :, length:] = True
            bool_mask_one[idx, :length, :length] = False

        add_mask = torch.masked_fill(torch.zeros(bool_mask_two.size()).cuda(), bool_mask_two, float("-inf"))
        mul_mask = torch.masked_fill(torch.ones(bool_mask_one.size()).cuda(), bool_mask_one, 0.0)

        add_mask.detach_()
        mul_mask.detach_()

        return add_mask, mul_mask

    def _apply_layer_norm(self, index: int, tensor: Tensor, before: bool = False, after: bool = False) -> Tensor:
        """Applies layer normalization conditionally.

        Args:
            index: Index of layer norm to use
            tensor: Input tensor
            before: Apply norm before
            after: Apply norm after

        Returns:
            Normalized tensor if conditions are met
        """
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[index](tensor)
        return tensor

    def forward(
        self,
        x: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        control_vector: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        mode: str = "l2o",
    ) -> Tensor:
        """Forward pass for transformer encoder layer.

        Args:
            x: Input tensor of shape (seq_len, batch, embed_dim)
            key: Optional key tensor for attention
            value: Optional value tensor for attention
            control_vector: Optional control vector from DIV encoder
            lengths: Sequence lengths
            mode: Projection mode ("l2o" or "o2l")

        Returns:
            Encoded output tensor
        """
        residual = x
        x = self._apply_layer_norm(0, x, before=True)

        # Create attention masks
        if mode == "l2o":
            add_mask, mul_mask = self._create_attention_mask(
                x.size(1), x.size(0), value.size(0), lengths=lengths, is_language_projection=False
            )
        elif mode == "o2l":
            add_mask, mul_mask = self._create_attention_mask(
                x.size(1), value.size(0), x.size(0), lengths=lengths, is_language_projection=True
            )

        # Apply attention
        if key is None and value is None:
            x, _ = self.self_attn(query=x, key=x, value=x, add_mask=add_mask, mul_mask=mul_mask)
        else:
            key = self._apply_layer_norm(0, key, before=True)
            value = self._apply_layer_norm(0, value, before=True)
            x, _ = self.self_attn(query=x, key=key, value=value, add_mask=add_mask, mul_mask=mul_mask)

        x = F.dropout(x, p=self.res_dropout, training=self.training)

        # Apply gating
        if control_vector is not None:
            memory_gate = self.memory_projection(control_vector)
            fusion_gate = self.attention_projection(control_vector)
            x = x * fusion_gate + residual * memory_gate
        else:
            x = residual + x

        x = self._apply_layer_norm(0, x, after=True)

        # Feed-forward network
        residual = x
        x = self._apply_layer_norm(1, x, before=True)
        x = F.relu(self.feed_forward_one(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.feed_forward_two(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self._apply_layer_norm(1, x, after=True)

        return x


class GatedTransformer(Module):
    """Gated transformer encoder for multimodal fusion.

    This transformer implements a gated architecture for fusing multiple modalities,
    particularly designed for language and another modality. It uses domain-invariant
    encoding and cross-modal attention mechanisms.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        layers: Number of transformer layers
        attn_dropout: Dropout rate for attention
        relu_dropout: Dropout rate for ReLU activations
        res_dropout: Dropout rate for residual connections
        embed_dropout: Dropout rate for embeddings
        div_dropout: Dropout rate for DIV encoder
        attn_mask: Whether to use attention masking
        use_disc: Whether to use discriminator in DIV encoder
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layers: int,
        attn_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        res_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        div_dropout: float = 0.0,
        attn_mask: bool = False,
        use_disc: bool = True,
    ) -> None:
        super().__init__()

        # Embedding parameters
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask

        # Initialize transformer layers
        self.text_to_other_layers = ModuleList()
        self.other_to_text_layers = ModuleList()
        self.div_encoders = ModuleList()

        for layer in range(layers):
            # Create cross-modal transformer layers
            text_to_other = TransformerEncoderLayer(
                embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask,
            )
            other_to_text = TransformerEncoderLayer(
                embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask,
            )

            # Create DIV encoder with different configuration for first layer
            if layer == 0:
                div_layer = DIVEncoder(
                    embed_dim, embed_dim, prj_type="linear", use_disc=use_disc, p_t=div_dropout, p_o=div_dropout
                )
            else:
                div_layer = DIVEncoder(
                    embed_dim,
                    embed_dim,
                    prj_type="rnn",
                    rnn_type="gru",
                    rdc_type="avg",
                    use_disc=use_disc,
                    p_t=div_dropout,
                    p_o=div_dropout,
                )

            self.text_to_other_layers.append(text_to_other)
            self.other_to_text_layers.append(other_to_text)
            self.div_encoders.append(div_layer)

        # Initialize normalization
        self.register_buffer("version", torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = create_layer_norm(embed_dim)

    def forward(
        self,
        seq_t: Tensor,
        seq_other: Tensor,
        h_l: Tensor,
        h_other: Tensor,
        lengths: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass through the gated transformer.

        Args:
            seq_l: Language sequence input
            seq_other: Other modality sequence input
            h_l: Language hidden states
            h_other: Other modality hidden states
            lengths: Sequence lengths
            mask: Attention mask

        Returns:
            Tuple containing:
                - Encoded language output
                - Encoded other modality output
                - Discriminator outputs
                - Discriminator labels
        """
        assert lengths is not None or mask is not None

        # Create mask if not provided
        if mask is None:
            batch_size = lengths.size(0)
            mask = torch.arange(lengths.max()).repeat(batch_size, 1).cuda() < lengths.unsqueeze(-1)
            mask = mask.unsqueeze(-1).to(torch.float)
        elif lengths is None:
            lengths = mask.squeeze().sum(1)

        # Initialize inputs
        input_t, input_other = seq_t, seq_other
        disc_outputs = []
        disc_labels = []

        # Process through layers
        for div_encoder, trans_l2other, trans_other2l in zip(
            self.div_encoders, self.text_to_other_layers, self.other_to_text_layers
        ):
            # Get domain-invariant encodings
            enc_l, enc_other, disc_out, disc_labels = div_encoder(h_l, h_other, lengths, mask)

            # Create control vector for gating
            control_vector = torch.cat([enc_l, enc_other], dim=-1)

            # Collect discriminator outputs
            disc_outputs.append(disc_out)
            disc_labels.append(disc_labels)

            # Cross-modal projections
            lang_to_other = trans_other2l(
                input_other, key=input_t, value=input_t, control_vector=control_vector, lengths=lengths, mode="l2o"
            )

            other_to_lang = trans_l2other(
                input_t, key=input_other, value=input_other, control_vector=control_vector, lengths=lengths, mode="o2l"
            )

            # Update inputs for next layer
            input_t = other_to_lang
            input_other = lang_to_other
            h_l = other_to_lang
            h_other = lang_to_other

        # Combine discriminator outputs
        combined_disc_out = torch.cat(disc_outputs)
        combined_disc_labels = torch.cat(disc_labels)

        return (
            other_to_lang,  # Final language representation
            lang_to_other,  # Final other modality representation
            combined_disc_out,
            combined_disc_labels,
        )

    def forward_transformer(
        self, x_in: Tensor, x_in_k: Optional[Tensor] = None, x_in_v: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass through vanilla transformer (without gating).

        Args:
            x_in: Input tensor of shape (seq_len, batch, embed_dim)
            x_in_k: Optional key tensor
            x_in_v: Optional value tensor

        Returns:
            Encoded output tensor
        """
        # Embed tokens and positions
        x = self.embed_scale * x_in

        # Add positional embeddings
        if self.embed_positions is not None:
            pos_emb = self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x = x + pos_emb

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Process key and value if provided
        if x_in_k is not None and x_in_v is not None:
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v

            if self.embed_positions is not None:
                x_k = x_k + self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)
                x_v = x_v + self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)

            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # Store intermediate states
        intermediates = [x]

        # Process through layers
        for layer in self.text_to_other_layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        # Apply final normalization if needed
        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self) -> int:
        """Get maximum supported input length.

        Returns:
            Maximum sequence length supported by the encoder
        """
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())
