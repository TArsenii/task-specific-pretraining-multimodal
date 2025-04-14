import torch
import torch.nn as nn


class FcEncoder(nn.Module):
    def __init__(self, input_dim, layers, dropout=0.5, use_bn=False):
        """Fully Connect classifier
        fc+relu+bn+dropout， 最后分类128-4层是直接fc的
        Parameters:
        --------------------------
        input_dim: input feature dim
        layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
        dropout: dropout rate
        use_bn: use batchnorm or not
        """
        super().__init__()
        self.all_layers = []
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            if use_bn:
                self.all_layers.append(nn.BatchNorm1d(layers[i]))
            if dropout > 0:
                self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]

        self.module = nn.Sequential(*self.all_layers)
        self.input_dim = input_dim

    def forward(self, x):
        # Handle different input dimensions
        original_shape = x.shape
        
        # Flatten input to 2D if it's not already
        if len(original_shape) > 2:
            # For batched data: reshape to (batch_size, -1)
            batch_size = original_shape[0]
            x = x.reshape(batch_size, -1)
            
        # If tensor width doesn't match input_dim, add additional reshaping
        if x.shape[1] != self.all_layers[0].in_features:
            # Try to reshape based on expected input dimension
            if x.shape[1] % self.all_layers[0].in_features == 0:
                # If divisible, we can reshape
                factor = x.shape[1] // self.all_layers[0].in_features
                x = x[:, :self.all_layers[0].in_features * factor].reshape(x.shape[0], factor, self.all_layers[0].in_features)
                # Average along the middle dimension
                x = x.mean(dim=1)
            elif x.shape[1] > self.all_layers[0].in_features:
                # If larger, truncate
                x = x[:, :self.all_layers[0].in_features]
            else:
                # If smaller, pad with zeros
                padding = torch.zeros(x.shape[0], self.all_layers[0].in_features - x.shape[1], device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        # Apply the network
        feat = self.module(x)
        return feat
