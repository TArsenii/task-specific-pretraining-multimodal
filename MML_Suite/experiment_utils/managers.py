from typing import Dict, Optional
import torch
from modalities import Modality

from experiment_utils import get_console, safe_detach

console = get_console()


class FeatureManager:
    """
    Manages feature representations for multiple modalities across training samples.
    Tracks embeddings/features for each modality and their fusion.

    Args:
        modality_dims (Dict[Modality, int]): Dictionary mapping modalities to their feature dimensions
        device (torch.device): Device to store the features on
    """

    def __init__(self, modality_dims: Dict[str, int], device: torch.device):
        self.device = device
        self.modality_dims = modality_dims
        self.fully_init = False
        self.feature_maps = None
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            console.print("[bold yellow]Warning!:[/] CUDA not available, switching to CPU device.")

    def is_initialized(self) -> bool:
        """
        Check if the FeatureManager has been fully initialized.

        Returns:
            bool: True if fully initialized, False otherwise
        """
        return self.fully_init

    def set_num_samples(self, num_samples: int) -> None:
        """
        Initialize feature maps with the specified number of samples.

        Args:
            num_samples (int): Number of samples to initialize features for
        """
        self.feature_maps = {
            modality: torch.zeros(num_samples, dim, requires_grad=False).to(self.device)
            for modality, dim in self.modality_dims.items()
        }
        self.fully_init = True

    def update(self, features: Dict[Modality, torch.Tensor], indexes: torch.Tensor) -> None:
        """
        Update feature maps with new features at specified indexes.

        Args:
            features (Dict[Modality, torch.Tensor]): Dictionary mapping modalities
                to feature tensors of shape (batch_size, feature_dim)
            indexes (torch.Tensor): Tensor of indexes indicating which samples to update

        Raises:
            ValueError: If FeatureManager is not initialized
            KeyError: If a modality in features doesn't exist in feature_maps
            ValueError: If feature dimensions don't match expected dimensions
        """
        if not self.fully_init:
            raise ValueError("FeatureManager must be fully initialized before updating features")

        for modality, feature in features.items():
            if modality not in self.feature_maps:
                raise KeyError(f"Unknown modality: {modality}")
            if feature.shape[1] != self.modality_dims[modality]:
                raise ValueError(
                    f"Feature dimension mismatch for {modality}. "
                    f"Expected {self.modality_dims[modality]}, got {feature.shape[1]}"
                )
            self.feature_maps[modality][indexes] = safe_detach(feature, to_np=False)

    def get_features(self, modality: Modality, indexes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get features for a specific modality.

        Args:
            modality (Modality): Modality to retrieve features for
            indexes (Optional[torch.Tensor]): Optional tensor of indexes to retrieve.
                If None, returns all features for the modality.

        Returns:
            torch.Tensor: Feature tensor for specified modality and indexes

        Raises:
            ValueError: If FeatureManager is not initialized
            KeyError: If modality doesn't exist in feature_maps
        """
        if not self.fully_init:
            raise ValueError("FeatureManager must be fully initialized before getting features")

        if modality not in self.feature_maps:
            raise KeyError(f"Unknown modality: {modality}")

        if indexes is None:
            return self.feature_maps[modality]
        return self.feature_maps[modality][indexes]

    def __getitem__(self, k: Modality):
        return self.get_features(k, None)


class CenterManager:
    """
    Manages prototype/center representations for different classes across modalities.
    For sentiment analysis, maintains positive and negative centers for each modality.

    Args:
        config (CenterManagerConfig): Config containing initialization parameters
    """

    def __init__(self, device: torch.device, modality_dims: Dict[Modality, int], exclude_zero: bool = True):
        self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            console.print("[bold yellow]Warning!:[/] CUDA not available, switching to CPU device.")

        self.modality_dims = modality_dims
        self.exclude_zero = exclude_zero

        # Initialize center maps for each modality
        self.center_maps = {
            modality: {
                "pos": torch.zeros(dim, requires_grad=False).to(self.device),
                "neg": torch.zeros(dim, requires_grad=False).to(self.device),
            }
            for modality, dim in modality_dims.items()
        }

    def update(self, features: Dict[Modality, torch.Tensor], labels: torch.Tensor) -> None:
        """
        Update center representations based on current features and labels.

        Args:
            features (Dict[Modality, torch.Tensor]): Dictionary mapping modalities
                to feature tensors of shape (batch_size, feature_dim)
            labels (torch.Tensor): Labels tensor for determining positive/negative samples
        """
        neg_mask = labels < 0
        if self.exclude_zero:
            pos_mask = labels > 0
        else:
            pos_mask = labels >= 0

        for modality, feature in features.items():
            if modality not in self.center_maps:
                raise KeyError(f"Unknown modality: {modality}")

            if pos_mask.any():
                self.center_maps[modality]["pos"] = safe_detach(feature[pos_mask].mean(dim=0), to_np=False)
            if neg_mask.any():
                self.center_maps[modality]["neg"] = safe_detach(feature[neg_mask].mean(dim=0), to_np=False)

    def get_center(self, modality: Modality, polarity: str) -> torch.Tensor:
        """
        Get center representation for a specific modality and polarity.

        Args:
            modality (Modality): Modality to retrieve center for
            polarity (str): Either 'pos' or 'neg'

        Returns:
            torch.Tensor: Center tensor for specified modality and polarity
        """
        if modality not in self.center_maps:
            raise KeyError(f"Unknown modality: {modality}")
        if polarity not in ["pos", "neg"]:
            raise KeyError("Polarity must be 'pos' or 'neg'")

        return self.center_maps[modality][polarity]


class LabelManager:
    """
    Manages label information across different modalities.
    Supports maintaining and updating labels for each modality separately.

    Args:
        config (LabelManagerConfig): Config containing initialization parameters
    """

    def __init__(self, device: torch.device, modalities: Dict[Modality, int], exclude_zero: bool = True):
        self.device = device
        self.modalities = modalities
        self.num_samples = None
        self.exclude_zero = exclude_zero
        self.fully_init = False
        self.label_maps = None

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            console.print("[bold yellow]Warning!:[/] CUDA not available, switching to CPU device.")

    def is_initialized(self) -> bool:
        """
        Check if the LabelManager has been fully initialized.

        Returns:
            bool: True if fully initialized, False otherwise
        """
        return self.fully_init

    def set_num_samples(self, num_samples: int) -> None:
        """
        Update the number of samples in the label maps.

        Args:
            num_samples (int): New number of samples
        """
        self.num_samples = num_samples
        label_maps = {}
        for modality in self.modalities:
            label_maps[modality] = torch.zeros(num_samples, requires_grad=False).to(self.device)
        self.label_maps = label_maps
        self.fully_init = True

    def init_labels(self, indexes: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Initialize labels for all modalities at specified indexes.
        Initially all modalities get the same labels as fusion.

        Args:
            indexes (torch.Tensor): Tensor of indexes to initialize
            labels (torch.Tensor): Label values to set
        """
        if not self.fully_init:
            raise ValueError("LabelManager must be fully initialized before setting labels")
        for modality in self.label_maps:
            self.label_maps[modality][indexes] = labels.to(self.device).float()

    def update_labels(self, modality: Modality, indexes: torch.Tensor, new_labels: torch.Tensor) -> None:
        """
        Update labels for a specific modality at specified indexes.

        Args:
            modality (Modality): Modality to update labels for
            indexes (torch.Tensor): Tensor of indexes to update
            new_labels (torch.Tensor): New label values to set

        """
        if not self.fully_init:
            raise ValueError("LabelManager must be fully initialized before updating labels")

        if modality not in self.label_maps:
            raise KeyError(f"Unknown modality: {modality}")

        self.label_maps[modality][indexes] = new_labels

    def get_labels(self, modality: Modality, indexes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get labels for a specific modality.

        Args:
            modality (Modality): Modality to retrieve labels for
            indexes (Optional[torch.Tensor]): Optional tensor of indexes to retrieve.
                If None, returns all labels for the modality.

        Returns:
            torch.Tensor: Label tensor for specified modality and indexes
        """
        if not self.fully_init:
            raise ValueError("LabelManager must be fully initialized before getting labels")

        if modality not in self.label_maps:
            raise KeyError(f"Unknown modality: {modality}")

        if indexes is None:
            return self.label_maps[modality]
        return self.label_maps[modality][indexes]

    def __getitem__(self, k: Modality):
        return self.get_labels(k, None)
