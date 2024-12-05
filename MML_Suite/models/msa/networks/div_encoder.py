from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, Linear, Dropout, Sequential, ReLU, Sigmoid
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DIVEncoder(Module):
    """Domain-invariant encoder for multimodal inputs.
    
    This encoder processes pairs of modalities to create domain-invariant encodings,
    optionally computing similarity and reconstruction losses. It supports both
    linear and RNN-based encoding strategies.
    
    Args:
        in_size: Input dimension of the feature vectors
        out_size: Output dimension of the encoded representations
        prj_type: Projection type, either 'linear' or 'rnn'
        use_disc: Whether to use a discriminator for domain adaptation
        rnn_type: Type of RNN to use if prj_type='rnn' ('LSTM' or 'GRU')
        rdc_type: Reduction method ('avg', 'last', or None)
        p_t: Dropout probability for first modality
        p_o: Dropout probability for second modality
        
    Raises:
        ValueError: If invalid projection type or reduction method is specified
    """
    
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        prj_type: Literal['linear', 'rnn'] = 'linear',
        use_disc: bool = False,
        rnn_type: Optional[Literal['lstm', 'gru']] = None,
        rdc_type: Optional[Literal['avg', 'last']] = None,
        p_t: float = 0.0,
        p_o: float = 0.0
    ) -> None:
        super().__init__()
        
        self.prj_type = prj_type
        self.reduce = rdc_type
        self.use_disc = use_disc
        self.in_size = in_size
        self.out_size = out_size
        
        # Initialize encoders based on projection type
        if prj_type == 'linear':
            self._init_linear_encoders()
        elif prj_type == 'rnn':
            if rnn_type is None:
                raise ValueError("rnn_type must be specified when using RNN projection")
            self._init_rnn_encoders(rnn_type.upper(), p_t, p_o)
        else:
            raise ValueError("prj_type must be either 'linear' or 'rnn'")
        
        # Initialize discriminator if requested
        if use_disc:
            self._init_discriminator()
        
        # Initialize dropouts
        self.dropout_l = Dropout(p_t)
        self.dropout_o = Dropout(p_o)

    def _init_linear_encoders(self) -> None:
        """Initialize linear projection layers for both modalities."""
        self.encode_l = Linear(self.in_size, self.out_size)
        self.encode_o = Linear(self.in_size, self.out_size)
        
    def _init_rnn_encoders(self, rnn_type: str, p_t: float, p_o: float) -> None:
        """Initialize RNN encoders for both modalities.
        
        Args:
            rnn_type: Type of RNN to use ('LSTM' or 'GRU')
            p_t: Dropout probability for first modality
            p_o: Dropout probability for second modality
        """
        self.rnn_type = rnn_type
        rnn = getattr(torch.nn, rnn_type)
        
        self.encode_l = rnn(
            input_size=self.in_size,
            hidden_size=self.out_size,
            num_layers=1,
            dropout=p_t,
            bidirectional=True
        )
        
        self.encode_o = rnn(
            input_size=self.in_size,
            hidden_size=self.out_size,
            num_layers=1,
            dropout=p_o,
            bidirectional=True
        )
        
    def _init_discriminator(self) -> None:
        """Initialize the discriminator network."""
        self.discriminator = Sequential(
            Linear(self.out_size, 4 * self.out_size),
            ReLU(),
            Linear(4 * self.out_size, 1),
            Sigmoid()
        )
        
    def _masked_avg_pool(
        self,
        lengths: Tensor,
        mask: Tensor,
        *inputs: Tensor
    ) -> list[Tensor]:
        """Perform masked average pooling over sequence dimension.
        
        Args:
            lengths: Sequence lengths of shape (batch_size,)
            mask: Attention mask of shape (batch_size, seq_len) or (batch_size, seq_len, 1)
            inputs: Variable number of tensors to pool, each of shape (seq_len, batch_size, embed_size)
            
        Returns:
            List of pooled tensors, each of shape (batch_size, embed_size)
        """
        # Ensure mask has 3 dimensions
        if len(mask.size()) == 2:
            mask = mask.unsqueeze(-1)
            
        return [
            (t.permute(1, 0, 2) * mask).sum(1) / lengths.unsqueeze(-1)
            for t in inputs
        ]
    
    def _forward_rnn(
        self,
        rnn: Module,
        input_tensor: Tensor,
        lengths: Tensor
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Process input through RNN encoder.
        
        Args:
            rnn: The RNN module to use
            input_tensor: Input tensor of shape (seq_len, batch_size, input_size)
            lengths: Sequence lengths of shape (batch_size,)
            
        Returns:
            Tuple containing:
                - Padded sequence output of shape (seq_len, batch_size, 2*hidden_size)
                - Final hidden state(s)
        """
        packed_sequence = pack_padded_sequence(input_tensor, lengths.cpu())
        packed_h, h_out = rnn(packed_sequence)
        padded_h, _ = pad_packed_sequence(packed_h)
        return padded_h, h_out

    def forward(
        self,
        input_t: Tensor,
        input_o: Tensor,
        lengths: Tensor,
        mask: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """Forward pass through the domain-invariant encoder.
        
        Args:
            input_t: First modality input of shape (seq_len, batch_size, in_size)
            input_o: Second modality input of shape (seq_len, batch_size, in_size)
            lengths: Sequence lengths of shape (batch_size,)
            mask: Attention mask of shape (batch_size, seq_len) or (batch_size, seq_len, 1)
            
        Returns:
            Tuple containing:
                - Encoded first modality of shape (batch_size, out_size)
                - Encoded second modality of shape (batch_size, out_size)
                - Discriminator outputs if use_disc=True, else None
                - Discriminator labels if use_disc=True, else None
                
        Raises:
            ValueError: If invalid reduction method is specified for the chosen projection type
        """
        if self.prj_type == 'linear':
            enc_l, enc_o = self._forward_linear(input_t, input_o, lengths, mask)
        elif self.prj_type == 'rnn':
            enc_l, enc_o = self._forward_rnn_encoder(input_t, input_o, lengths, mask)
            
        # Apply dropout
        enc_l = self.dropout_l(enc_l)
        enc_o = self.dropout_o(enc_o)
        
        # Process through discriminator if requested
        disc_out = None
        disc_labels = None
        if self.use_disc:
            disc_out, disc_labels = self._forward_discriminator(enc_l, enc_o)
            
        return enc_l, enc_o, disc_out, disc_labels
    
    def _forward_linear(
        self,
        input_t: Tensor,
        input_o: Tensor,
        lengths: Tensor,
        mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for linear projection encoder.
        
        Args:
            input_t: First modality input
            input_o: Second modality input
            lengths: Sequence lengths
            mask: Attention mask
            
        Returns:
            Tuple of encoded representations for both modalities
            
        Raises:
            ValueError: If invalid reduction method is specified
        """
        if self.reduce == 'avg':
            avg_l, avg_o = self._masked_avg_pool(lengths, mask, input_t, input_o)
        elif self.reduce is None:
            avg_l, avg_o = input_t, input_o
        else:
            raise ValueError("Reduce method must be 'avg' or None for linear projection")
            
        return self.encode_l(avg_l), self.encode_o(avg_o)
    
    def _forward_rnn_encoder(
        self,
        input_t: Tensor,
        input_o: Tensor,
        lengths: Tensor,
        mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for RNN encoder.
        
        Args:
            input_t: First modality input
            input_o: Second modality input
            lengths: Sequence lengths
            mask: Attention mask
            
        Returns:
            Tuple of encoded representations for both modalities
            
        Raises:
            ValueError: If invalid reduction method is specified
        """
        out_l, h_l = self._forward_rnn(self.encode_l, input_t, lengths)
        out_o, h_o = self._forward_rnn(self.encode_o, input_o, lengths)
        
        if self.reduce == 'last':
            h_l_last = h_l[0] if isinstance(h_l, tuple) else h_l
            h_o_last = h_o[0] if isinstance(h_o, tuple) else h_o
            enc_l = (h_l_last[0] + h_l_last[1]) / 2
            enc_o = (h_o_last[0] + h_o_last[1]) / 2
        elif self.reduce == 'avg':
            enc_l, enc_o = self._masked_avg_pool(lengths, mask, out_l, out_o)
            enc_l = (enc_l[:, :enc_l.size(1) // 2] + enc_l[:, enc_l.size(1) // 2:]) / 2
            enc_o = (enc_o[:, :enc_o.size(1) // 2] + enc_o[:, enc_o.size(1) // 2:]) / 2
        else:
            raise ValueError("Reduce method must be 'last' or 'avg' for RNN projection")
            
        return enc_l, enc_o
    
    def _forward_discriminator(
        self,
        enc_l: Tensor,
        enc_o: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through the discriminator.
        
        Args:
            enc_l: Encoded first modality
            enc_o: Encoded second modality
            
        Returns:
            Tuple containing:
                - Discriminator outputs
                - Discriminator labels
        """
        batch_size = enc_l.size(0)
        
        # Generate discriminator output
        disc_out = self.discriminator(
            torch.cat((enc_l, enc_o), dim=0)
        ).squeeze()
        
        # Generate discriminator labels
        disc_labels = torch.cat([
            torch.zeros(batch_size),
            torch.ones(batch_size)
        ], dim=0).squeeze().to(enc_l.device)
        
        return disc_out, disc_labels