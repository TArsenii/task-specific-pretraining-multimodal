import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalPooling(nn.Module):
    """
    Multimodal pooling module supporting different pooling strategies.
    Strategies:
    - "max": Max pooling across modalities
    - "avg" or "average": Average pooling across modalities
    - "sum": Sum pooling across modalities
    - "attention": Attention-based pooling across modalities
    - "gated": Gated fusion similar to GatedBiModalNetwork
    """

    def __init__(
        self,
        input_dim_a: int,
        input_dim_b: int,
        output_dim: int,
        pooling_type: str = "gated",
        hidden_dim: int = None,
        dropout: float = 0.0,
    ):
        """
        Initialize the multimodal pooling module.

        Args:
            input_dim_a (int): Dimension of the first modality input
            input_dim_b (int): Dimension of the second modality input
            output_dim (int): Dimension of the output
            pooling_type (str): Type of pooling to use ('max', 'avg'/'average', 'sum', 'attention', 'gated')
            hidden_dim (int, optional): Hidden dimension for attention and gated mechanisms
            dropout (float, optional): Dropout probability
        """
        super().__init__()
        self.pooling_type = pooling_type.lower()
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or max(input_dim_a, input_dim_b)
        self.dropout = dropout

        # Projection layers for each modality to ensure same dimensions
        self.proj_a = nn.Linear(input_dim_a, output_dim)
        self.proj_b = nn.Linear(input_dim_b, output_dim)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Activation function
        self.activation = nn.Tanh()
        
        # Additional layers for attention and gated mechanisms
        if self.pooling_type == 'attention':
            # Attention mechanism
            self.attention_layer = nn.Sequential(
                nn.Linear(output_dim * 2, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, 2),  # 2 attention scores, one for each modality
                nn.Softmax(dim=1)
            )
        
        elif self.pooling_type == 'gated':
            # Gated mechanism similar to GatedBiModalNetwork
            self.gate_layer = nn.Sequential(
                nn.Linear(output_dim * 2, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multimodal pooling.

        Args:
            x_a (torch.Tensor): Input tensor for the first modality
            x_b (torch.Tensor): Input tensor for the second modality

        Returns:
            torch.Tensor: Pooled multimodal representation
        """
        # Project inputs to the same dimension
        a = self.activation(self.proj_a(x_a))
        b = self.activation(self.proj_b(x_b))
        
        # Apply dropout
        a = self.dropout_layer(a)
        b = self.dropout_layer(b)

        # Perform pooling based on the specified type
        if self.pooling_type == 'max':
            # Element-wise max
            return torch.max(a, b)
            
        elif self.pooling_type in ['avg', 'average']:
            # Element-wise average
            return (a + b) / 2
            
        elif self.pooling_type == 'sum':
            # Element-wise sum
            return a + b
            
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            combined = torch.cat([a, b], dim=1)
            attention_scores = self.attention_layer(combined)
            
            # Reshape attention scores to match feature dimensions
            att_a = attention_scores[:, 0].unsqueeze(1).expand_as(a)
            att_b = attention_scores[:, 1].unsqueeze(1).expand_as(b)
            
            # Apply attention weights
            return att_a * a + att_b * b
            
        elif self.pooling_type == 'gated':
            # Gated mechanism
            combined = torch.cat([a, b], dim=1)
            gate = self.gate_layer(combined)
            
            # Apply gate
            return gate * a + (1 - gate) * b
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}") 