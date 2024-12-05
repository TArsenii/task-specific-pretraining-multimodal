from dataclasses import dataclass, field
from typing import List, Optional

from .base_config import BaseConfig


@dataclass
class MonitorConfig(BaseConfig):
    """Configuration for experiment monitoring."""

    enabled: bool = True

    # Tracking frequencies (in steps)
    gradient_interval: int = 100
    activation_interval: int = 100
    weight_interval: int = 200

    # Storage settings
    buffer_size: int = 1000  # Number of items to buffer before writing
    compression: str = "gzip"  # HDF5 compression type
    compression_opts: int = 4  # Compression level

    # Feature toggles
    enable_gradient_tracking: bool = True
    enable_activation_tracking: bool = True
    enable_weight_tracking: bool = True
    enable_layer_convergence: bool = True
    enable_information_flow: bool = False

    # Layer monitoring
    include_layers: Optional[List[str]] = None  # None means all layers
    exclude_layers: List[str] = field(default_factory=lambda: ["BatchNorm", "Dropout"])

    # Buffer settings
    flush_interval: int = 1000  # Steps between forced buffer flushes
