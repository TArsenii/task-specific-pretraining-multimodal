import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Type, Union, Tuple


class LeNetEncoder(nn.Module):
    """
    LeNet-5 encoder architecture for image processing.
    Adapted from the original LeNet-5 architecture by Yann LeCun.
    
    Parameters:
    --------------------------
    in_channels: Number of input channels (1 for grayscale, 3 for RGB)
    hidden_dim: Dimension of the output feature vector
    feature_maps: Number of feature maps in conv layers [default: [6, 16]]
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 84,
        feature_maps: list = [6, 16],
        norm_layer: Optional[Type[nn.Module]] = None
    ) -> None:
        super(LeNetEncoder, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # First conv block: Conv -> ReLU -> Pool
        self.conv1 = nn.Conv2d(in_channels, feature_maps[0], kernel_size=5, stride=1, padding=2)
        self.bn1 = norm_layer(feature_maps[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second conv block: Conv -> ReLU -> Pool
        self.conv2 = nn.Conv2d(feature_maps[0], feature_maps[1], kernel_size=5, stride=1, padding=0)
        self.bn2 = norm_layer(feature_maps[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculating output size of conv layers to determine FC input size
        # Assuming input is 28x28 (like MNIST)
        # After conv1 (5x5, padding=2): 28x28 -> 28x28
        # After pool1 (2x2, stride=2): 28x28 -> 14x14
        # After conv2 (5x5, padding=0): 14x14 -> 10x10
        # After pool2 (2x2, stride=2): 10x10 -> 5x5
        # So flattened size = 5*5*16 = 400
        self.fc1 = nn.Linear(feature_maps[1] * 5 * 5, 120)
        self.fc2 = nn.Linear(120, hidden_dim)
        
        # Save dimensions for later
        self.input_channels = in_channels
        self.hidden_dim = hidden_dim
        self.feature_maps = feature_maps
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        """Initialize weights using standard initialization techniques"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def get_embedding_size(self) -> int:
        """Return the output dimension of the encoder"""
        return self.hidden_dim
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function of LeNet.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        # Handle different input shapes
        if len(x.shape) == 2:
            # Add batch and channel dimensions if missing
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            # Handle audio input [batch_size, channels, timesteps]
            if x.shape[2] > 32:  # Assuming this is audio (with timesteps)
                # Reshape to 2D by treating timesteps as a 2D grid
                batch_size, channels, timesteps = x.shape
                
                # Ensure minimum size for network to work properly
                # We need at least 24x24 after first conv+pool for LeNet to work properly
                # (which becomes 10x10 after second conv, 5x5 after pool)
                min_side = 24  # минимальный размер стороны для корректной работы после первого слоя
                
                # Calculate side dimension that's sufficiently large
                # Aim for a square side that's at least min_side
                side = max(min_side, int(timesteps ** 0.5) + 1)
                
                # Padding to make it square
                padded_length = side * side
                padding = torch.zeros(batch_size, channels, padded_length - timesteps, device=x.device)
                x_padded = torch.cat([x, padding], dim=2)
                
                # Reshape to [batch_size, channels, side, side]
                x = x_padded.view(batch_size, channels, side, side)
                
                # Исправляем проблему с каналами - берем только первый канал или объединяем каналы в один
                # если в модели ожидается только один входной канал
                if self.conv1.in_channels == 1 and channels > 1:
                    # Вариант 1: Берем только первый канал
                    # x = x[:, 0:1, :, :]
                    
                    # Вариант 2: Объединяем все каналы в один через среднее значение
                    x = x.mean(dim=1, keepdim=True)
            else:
                # Add spatial dimension if missing (convert to 2D)
                if x.shape[0] > 3:  # Assuming batch comes first
                    x = x.unsqueeze(1)
                else:  # Assuming channel comes first
                    x = x.unsqueeze(0)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Handle different input sizes with adaptive pooling in feature space
        if x.size(1) != self.fc1.in_features:
            # If the flattened size doesn't match expected input, reshape
            if x.size(1) > self.fc1.in_features:
                # If larger, use adaptive pooling
                x = F.adaptive_avg_pool1d(x.unsqueeze(1), self.fc1.in_features).squeeze(1)
            else:
                # If smaller, pad with zeros
                padding = torch.zeros(x.size(0), self.fc1.in_features - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def LeNet5(in_channels: int = 1, hidden_dim: int = 84) -> LeNetEncoder:
    """
    Create a LeNet-5 encoder with specified parameters
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        hidden_dim: Dimension of the output feature vector
        
    Returns:
        Initialized LeNet-5 encoder
    """
    return LeNetEncoder(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        feature_maps=[6, 16]
    )


def LeNet5Enhanced(in_channels: int = 1, hidden_dim: int = 128) -> LeNetEncoder:
    """
    Create an enhanced LeNet-5 encoder with more feature maps
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        hidden_dim: Dimension of the output feature vector
        
    Returns:
        Initialized enhanced LeNet-5 encoder
    """
    return LeNetEncoder(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        feature_maps=[16, 32]
    ) 