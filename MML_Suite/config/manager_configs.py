from typing import Dict, List
from dataclasses import dataclass
from modalities import Modality
from .base_config import BaseConfig

@dataclass
class FeatureManagerConfig(BaseConfig):
    """
    Configuration for FeatureManager.

    Args:
        modality_dims: Mapping of modalities to their feature dimensions
        num_samples: Number of training samples to track
        device: Device to store tensors on
    """

    modality_dims: Dict[Modality, int]
    num_samples: int
    device: str = "cuda"

    def __post_init__(self):
        # Validate modalities
        if Modality.MULTIMODAL not in self.modality_dims:
            raise ValueError("FeatureManager requires MULTIMODAL dimension")


@dataclass
class CenterManagerConfig(BaseConfig):
    """
    Configuration for CenterManager.

    Args:
        modality_dims: Mapping of modalities to their feature dimensions
        device: Device to store tensors on
        exclude_zero: Whether to exclude zero labels when computing positive centers
    """

    modality_dims: Dict[Modality, int]
    device: str = "cuda"
    exclude_zero: bool = True

    def __post_init__(self):
        if Modality.MULTIMODAL not in self.modality_dims:
            raise ValueError("CenterManager requires MULTIMODAL dimension")


@dataclass
class LabelManagerConfig(BaseConfig):
    """
    Configuration for LabelManager.

    Args:
        modalities: List of modalities to track labels for
        num_samples: Number of samples to track labels for
        device: Device to store tensors on
    """

    modalities: List[Modality]
    num_samples: int = -1
    device: str = "cuda"

    def __post_init__(self):
        if Modality.MULTIMODAL not in self.modalities:
            raise ValueError("LabelManager requires MULTIMODAL modality")
