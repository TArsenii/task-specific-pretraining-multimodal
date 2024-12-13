from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np


class DataType(Enum):
    """Types of data stored in monitoring file."""

    WEIGHTS = "weights"
    ACTIVATIONS = "activations"
    GRADIENTS = "gradients"
    METADATA = "metadata"


@dataclass
class MonitoringConfig:
    """Configuration for data loading and processing."""

    start_epoch: Optional[int] = None
    end_epoch: Optional[int] = None
    modalities: List[str] = None
    layer_patterns: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ["audio", "video", "text"]

        if self.layer_patterns is None:
            # Default patterns for finding modality-specific layers
            self.layer_patterns = {
                "audio": ["audio_encoder", "audio_layer"],
                "video": ["video_encoder", "video_layer"],
                "text": ["text_encoder", "text_layer"],
                "fusion": ["fusion_layer", "decoder"],
            }


class MonitoringDataLoader:
    """Handles loading and basic processing of neural network monitoring data."""

    def __init__(self, data_path: Path, config: Optional[MonitoringConfig] = None):
        """
        Initialize the data loader.

        Args:
            data_path: Path to HDF5 file containing monitoring data
            config: Configuration for data loading and processing
        """
        self.data_path = Path(data_path)
        self.config = config or MonitoringConfig()
        self.data: Dict[str, Any] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load data from HDF5 file into memory with basic preprocessing."""

        try:
            with h5py.File(self.data_path, "r") as f:
                # Load each data type
                for data_type in DataType:
                    if data_type.value in f:
                        self.data[data_type.value] = self._process_group(f[data_type.value], data_type)

                # Load metadata if available
                if "metadata" in f:
                    self.metadata = dict(f["metadata"].attrs)

            print("Data loading complete")

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _process_group(self, group: h5py.Group, data_type: DataType) -> Dict[str, Any]:
        """
        Process a group of monitoring data.

        Args:
            group: HDF5 group containing monitoring data
            data_type: Type of data being processed

        Returns:
            Processed data organized by epoch and layer
        """
        processed_data = {}

        # Get epoch range
        epochs = sorted([int(k.split("_")[1]) for k in group.keys() if k.startswith("epoch_")])

        # Filter epochs based on config
        if self.config.start_epoch is not None:
            epochs = [e for e in epochs if e >= self.config.start_epoch]
        if self.config.end_epoch is not None:
            epochs = [e for e in epochs if e <= self.config.end_epoch]

        for epoch in epochs:
            epoch_key = f"epoch_{epoch}"
            if epoch_key in group:
                epoch_data = {}
                epoch_group = group[epoch_key]

                # Process each layer in the epoch
                for layer_name, layer_data in epoch_group.items():
                    # Check if layer belongs to any modality
                    modality = self._get_layer_modality(layer_name)
                    if modality:
                        epoch_data[layer_name] = {
                            "data": layer_data[:],
                            "modality": modality,
                            "attrs": dict(layer_data.attrs),
                        }

                processed_data[epoch] = epoch_data

        return processed_data

    def _get_layer_modality(self, layer_name: str) -> Optional[str]:
        """
        Determine which modality a layer belongs to based on its name.

        Args:
            layer_name: Name of the layer

        Returns:
            Modality name or None if no match
        """
        for modality, patterns in self.config.layer_patterns.items():
            if any(pattern in layer_name for pattern in patterns):
                return modality
        return None

    def get_modality_data(
        self, modality: str, data_type: DataType, epoch: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get data for a specific modality.

        Args:
            modality: Name of modality to get data for
            data_type: Type of data to retrieve
            epoch: Optional specific epoch to get data for

        Returns:
            Dictionary mapping layer names to their data
        """
        if data_type.value not in self.data:
            raise KeyError(f"No {data_type.value} data available")

        modality_data = {}
        data_group = self.data[data_type.value]

        # Filter epochs if specified
        epochs = [epoch] if epoch is not None else sorted(data_group.keys())

        for e in epochs:
            epoch_data = data_group[e]
            for layer_name, layer_info in epoch_data.items():
                if layer_info["modality"] == modality:
                    if epoch is not None:
                        modality_data[layer_name] = layer_info["data"]
                    else:
                        if layer_name not in modality_data:
                            modality_data[layer_name] = []
                        modality_data[layer_name].append(layer_info["data"])

        # Stack temporal data if no specific epoch
        if epoch is None:
            modality_data = {k: np.stack(v) for k, v in modality_data.items()}

        return modality_data

    def get_layer_data(
        self, layer_pattern: str, data_type: DataType, epoch: Optional[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Get data for layers matching a pattern.

        Args:
            layer_pattern: Pattern to match layer names
            data_type: Type of data to retrieve
            epoch: Optional specific epoch to get data for

        Returns:
            Dictionary mapping epochs to layer data
        """
        if data_type.value not in self.data:
            raise KeyError(f"No {data_type.value} data available")

        layer_data = {}
        data_group = self.data[data_type.value]

        # Filter epochs if specified
        epochs = [epoch] if epoch is not None else sorted(data_group.keys())

        for e in epochs:
            epoch_data = data_group[e]
            for layer_name, layer_info in epoch_data.items():
                if layer_pattern in layer_name:
                    layer_data[e] = layer_info["data"]

        return layer_data

    def get_epochs(self) -> List[int]:
        """Get list of available epochs."""
        # Use weights data as reference for available epochs
        if DataType.WEIGHTS.value in self.data:
            return sorted(self.data[DataType.WEIGHTS.value].keys())
        return []

    def get_layers(self, modality: Optional[str] = None) -> List[str]:
        """
        Get list of available layers, optionally filtered by modality.

        Args:
            modality: Optional modality to filter layers by

        Returns:
            List of layer names
        """
        layers = set()
        # Use weights data as reference for available layers
        if DataType.WEIGHTS.value in self.data:
            for epoch_data in self.data[DataType.WEIGHTS.value].values():
                for layer_name, layer_info in epoch_data.items():
                    if modality is None or layer_info["modality"] == modality:
                        layers.add(layer_name)
        return sorted(layers)
