from typing import Dict, List, Literal, Tuple

import torch
from torch import Tensor
from torch.nn import GRU, LSTM, Conv1d, LayerNorm, Linear, Module, Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from modalities import Modality

class SeqEncoder(Module):
    """Multimodal sequence encoder that processes audio, text, and video inputs.

    This encoder supports different network architectures (linear, CNN, LSTM, or GRU) to encode
    multiple modalities (audio, text, video) into a common representation space. Each modality
    is processed independently and projected to a specified attention dimension.

    Attributes:
        proj_type: The type of projection network to use ('linear', 'cnn', 'lstm', 'gru')
        dim_a: Output dimension for audio modality
        dim_t: Output dimension for text modality
        dim_v: Output dimension for video modality
    """

    def __init__(
        self,
        orig_dim_a: int,
        orig_dim_t: int,
        orig_dim_v: int,
        attention_dim: int,
        num_enc_layers: int,
        proj_type: Literal["linear", "cnn", "lstm", "gru"],
        a_ksize: int,
        t_ksize: int,
        v_ksize: int,
    ) -> None:
        """Initialize the sequence encoder.

        Args:
            orig_dim_a: Original dimension of audio features
            orig_dim_t: Original dimension of text features
            orig_dim_v: Original dimension of video features
            attention_dim: Target dimension for all modalities after encoding
            num_enc_layers: Number of encoding layers for RNN-based models
            proj_type: Type of projection network to use
            a_ksize: Kernel size for audio CNN
            t_ksize: Kernel size for text CNN
            v_ksize: Kernel size for video CNN
        """
        super().__init__()

        # Store original and target dimensions
        self.orig_dim_a = orig_dim_a
        self.orig_dim_t = orig_dim_t
        self.orig_dim_v = orig_dim_v
        self.dim_a = attention_dim
        self.dim_t = attention_dim
        self.dim_v = attention_dim
        self.proj_type = proj_type.lower()

        # Initialize the appropriate projection network based on proj_type
        if proj_type == "linear":
            self._init_linear_projections()
        elif proj_type == "cnn":
            self._init_cnn_projections(a_ksize, t_ksize, v_ksize)
        elif proj_type in ["lstm", "gru"]:
            self._init_rnn_projections(num_enc_layers, proj_type)
        else:
            raise ValueError("proj_type must be one of: 'linear', 'cnn', 'lstm', 'gru'")

    def _init_linear_projections(self) -> None:
        """Initialize linear projection layers with layer normalization."""
        # Linear projections for each modality
        self.proj_a = Linear(self.orig_dim_a, self.dim_a)
        self.proj_t = Linear(self.orig_dim_t, self.dim_t)
        self.proj_v = Linear(self.orig_dim_v, self.dim_v)

        # Layer normalization for each projection
        self.layer_norm_a = LayerNorm(self.dim_a)
        self.layer_norm_t = LayerNorm(self.dim_t)
        self.layer_norm_v = LayerNorm(self.dim_v)

    def _init_cnn_projections(self, a_ksize: int, t_ksize: int, v_ksize: int) -> None:
        """Initialize CNN projection layers.

        Args:
            a_ksize: Kernel size for audio CNN
            t_ksize: Kernel size for text CNN
            v_ksize: Kernel size for video CNN
        """
        # Calculate padding for each modality to maintain sequence length
        pad_a = int((a_ksize - 1) / 2)
        pad_t = int((t_ksize - 1) / 2)
        pad_v = int((v_ksize - 1) / 2)

        # 1D convolution layers for each modality
        self.proj_a = Conv1d(self.orig_dim_a, self.dim_a, kernel_size=a_ksize, padding=pad_a, bias=False)
        self.proj_t = Conv1d(self.orig_dim_t, self.dim_t, kernel_size=t_ksize, padding=pad_t, bias=False)
        self.proj_v = Conv1d(self.orig_dim_v, self.dim_v, kernel_size=v_ksize, padding=pad_v, bias=False)

    def _init_rnn_projections(self, num_layers: int, rnn_type: str) -> None:
        """Initialize RNN projection layers with associated linear projections and layer norms.

        Args:
            num_layers: Number of RNN layers
            rnn_type: Type of RNN ('lstm' or 'gru')
        """
        # Select RNN type
        rnn_class = LSTM if rnn_type == "lstm" else GRU

        # Initialize bidirectional RNNs for each modality
        self.rnn_t = rnn_class(self.orig_dim_t, self.orig_dim_t, num_layers, bidirectional=True)
        self.rnn_v = rnn_class(self.orig_dim_v, self.orig_dim_v, num_layers, bidirectional=True)
        self.rnn_a = rnn_class(self.orig_dim_a, self.orig_dim_a, num_layers, bidirectional=True)

        # Create mapping dictionary for easy access
        self.rnn_dict = {Modality.TEXT: self.rnn_t, Modality.VIDEO: self.rnn_v, Modality.AUDIO: self.rnn_a}

        # Initialize linear projections for hidden states
        self._init_rnn_linear_projections()

    def _init_rnn_linear_projections(self) -> None:
        """Initialize linear projections and layer norms for RNN outputs."""
        # Hidden state projections
        self.linear_proj_t_h = Linear(2 * self.orig_dim_t, self.dim_t)
        self.linear_proj_v_h = Linear(2 * self.orig_dim_v, self.dim_v)
        self.linear_proj_a_h = Linear(2 * self.orig_dim_a, self.dim_a)

        # Sequence projections
        self.linear_proj_t_seq = Linear(2 * self.orig_dim_t, self.dim_t)
        self.linear_proj_v_seq = Linear(2 * self.orig_dim_v, self.dim_v)
        self.linear_proj_a_seq = Linear(2 * self.orig_dim_a, self.dim_a)

        # Layer normalization
        self.layer_norm_t = LayerNorm(self.dim_t)
        self.layer_norm_v = LayerNorm(self.dim_v)
        self.layer_norm_a = LayerNorm(self.dim_a)

        # Create sequential models combining projections and layer norms
        self.proj_a_h = Sequential(self.linear_proj_a_h, self.layer_norm_a)
        self.proj_t_h = Sequential(self.linear_proj_t_h, self.layer_norm_t)
        self.proj_v_h = Sequential(self.linear_proj_v_h, self.layer_norm_v)

        # Sequential projections without layer norm
        self.proj_a_seq = Sequential(self.linear_proj_a_seq)
        self.proj_t_seq = Sequential(self.linear_proj_t_seq)
        self.proj_v_seq = Sequential(self.linear_proj_v_seq)

        # Create mapping dictionaries for easy access
        self.proj_dict_h = {"text": self.proj_t_h, "video": self.proj_v_h, "audio": self.proj_a_h}
        self.proj_dict_seq = {"text": self.proj_t_seq, "video": self.proj_v_seq, "audio": self.proj_a_seq}

    def forward_rnn_prj(self, input_tensor: Tensor, lengths: Tensor, modality: Modality) -> Tuple[Tensor, Tensor]:
        """Process input through RNN and project outputs.

        Args:
            input_tensor: Input sequence tensor of shape (seq_len, batch_size, embed_size)
            lengths: Sequence lengths tensor of shape (batch_size,)
            modality: Modality of the input (TEXT, VIDEO, or AUDIO)

        Returns:
            Tuple containing:
                - Projected sequence outputs of shape (seq_len, batch_size, attention_dim)
                - Projected final hidden state of shape (batch_size, attention_dim)
        """
        # Convert lengths to CPU and correct dtype for pack_padded_sequence
        lengths = lengths.to("cpu").to(torch.int64)

        # Pack sequence for efficient RNN processing
        packed_sequence = pack_padded_sequence(input_tensor, lengths)
        packed_h, h_out = self.rnn_dict[modality](packed_sequence)

        # Unpack sequence
        padded_h, _ = pad_packed_sequence(packed_h)

        if self.proj_type == "lstm":
            h_out = h_out[0]  # Extract hidden state, ignore cell state for LSTM

        # Concatenate bidirectional hidden states
        h_out = torch.cat((h_out[0], h_out[1]), dim=-1)

        # Project hidden states and sequence
        h_out = self.proj_dict_h[modality](h_out)
        h_out_seq = self.proj_dict_seq[modality](padded_h)

        return h_out_seq, h_out

    def _masked_avg_pool(self, lengths: Tensor, mask: Tensor, *inputs: Tensor) -> List[Tensor]:
        """Perform masked average pooling over sequence dimension.

        Args:
            lengths: Sequence lengths of shape (batch_size,)
            mask: Boolean mask of shape (batch_size, seq_len, 1)
            inputs: Variable number of tensors to pool, each of shape (batch_size, seq_len, embed_size)

        Returns:
            List of pooled tensors, each of shape (batch_size, embed_size)
        """
        return [(t * mask).sum(1) / lengths.unsqueeze(-1) for t in inputs]

    def forward(
        self, input_t: Tensor, input_v: Tensor, input_a: Tensor, lengths: Tensor
    ) -> Dict[Modality, Tuple[Tensor, Tensor]]:
        """Process inputs from all modalities through the encoder.

        Args:
            input_t: Text input of shape (seq_len, batch_size, text_embed_size)
            input_v: Video input of shape (seq_len, batch_size, video_embed_size)
            input_a: Audio input of shape (seq_len, batch_size, audio_embed_size)
            lengths: Sequence lengths of shape (batch_size,)

        Returns:
            Dictionary containing encoded representations for each modality:
            {
                Modality.TEXT: (sequence_encoding, hidden_encoding),
                Modality.VIDEO: (sequence_encoding, hidden_encoding),
                Modality.AUDIO: (sequence_encoding, hidden_encoding)
            }
            where sequence_encoding is of shape (seq_len, batch_size, attention_dim)
            and hidden_encoding is of shape (batch_size, attention_dim)
        """
        # Create attention mask based on sequence lengths
        batch_size = lengths.size(0)
        mask = torch.arange(lengths.max()).repeat(batch_size, 1).cuda() < lengths.unsqueeze(-1)
        mask = mask.unsqueeze(-1).to(torch.float)

        # Process inputs through appropriate encoder type
        if self.proj_type == "linear":
            return self._forward_linear(input_t, input_v, input_a, lengths, mask)
        elif self.proj_type == "cnn":
            return self._forward_cnn(input_t, input_v, input_a, lengths, mask)
        else:  # lstm or gru
            return self._forward_rnn(input_t, input_v, input_a, lengths)

    def _forward_linear(
        self, input_t: Tensor, input_v: Tensor, input_a: Tensor, lengths: Tensor, mask: Tensor
    ) -> Dict[Modality, Tuple[Tensor, Tensor]]:
        """Forward pass for linear projection encoder."""
        # Rearrange dimensions for linear projection
        perm = (1, 0, 2)
        t_seq = self.proj_t(input_t.permute(*perm))
        v_seq = self.proj_v(input_v.permute(*perm))
        a_seq = self.proj_a(input_a.permute(*perm))

        # Apply masked average pooling
        t_h, v_h, a_h = self._masked_avg_pool(lengths, mask, t_seq, v_seq, a_seq)

        # Restore original dimension ordering
        t_seq = t_seq.permute(*perm)
        v_seq = v_seq.permute(*perm)
        a_seq = a_seq.permute(*perm)

        return {Modality.TEXT: (t_seq, t_h), Modality.VIDEO: (v_seq, v_h), Modality.AUDIO: (a_seq, a_h)}

    def _forward_cnn(
        self, input_t: Tensor, input_v: Tensor, input_a: Tensor, lengths: Tensor, mask: Tensor
    ) -> Dict[Modality, Tuple[Tensor, Tensor]]:
        """Forward pass for CNN encoder."""
        # Rearrange dimensions for CNN
        perm1 = (1, 2, 0)
        perm2 = (0, 2, 1)
        perm3 = (1, 0, 2)

        # Apply CNN projections
        t_seq = self.proj_t(input_t.permute(*perm1)).permute(*perm2)
        v_seq = self.proj_v(input_v.permute(*perm1)).permute(*perm2)
        a_seq = self.proj_a(input_a.permute(*perm1)).permute(*perm2)

        # Apply masked average pooling
        t_h, v_h, a_h = self._masked_avg_pool(lengths, mask, t_seq, v_seq, a_seq)

        # Restore original dimension ordering
        t_seq = t_seq.permute(*perm3)
        v_seq = v_seq.permute(*perm3)
        a_seq = a_seq.permute(*perm3)

        return {Modality.TEXT: (t_seq, t_h), Modality.VIDEO: (v_seq, v_h), Modality.AUDIO: (a_seq, a_h)}

    def _forward_rnn(
        self, input_t: Tensor, input_v: Tensor, input_a: Tensor, lengths: Tensor
    ) -> Dict[Modality, Tuple[Tensor, Tensor]]:
        """Forward pass for RNN (LSTM/GRU) encoder.

        Args:
            input_t: Text input of shape (seq_len, batch_size, text_embed_size)
            input_v: Video input of shape (seq_len, batch_size, video_embed_size)
            input_a: Audio input of shape (seq_len, batch_size, audio_embed_size)
            lengths: Sequence lengths of shape (batch_size,)

        Returns:
            Dictionary containing encoded representations for each modality
        """
        # Process each modality through its respective RNN
        t_seq, t_h = self.forward_rnn_prj(input_t, lengths, modality=Modality.TEXT)
        v_seq, v_h = self.forward_rnn_prj(input_v, lengths, modality=Modality.VIDEO)
        a_seq, a_h = self.forward_rnn_prj(input_a, lengths, modality=Modality.AUDIO)

        return {Modality.TEXT: (t_seq, t_h), Modality.VIDEO: (v_seq, v_h), Modality.AUDIO: (a_seq, a_h)}
