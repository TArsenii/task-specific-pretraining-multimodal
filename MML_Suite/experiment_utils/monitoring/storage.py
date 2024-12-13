from pathlib import Path
from typing import Any, Dict, Union

import h5py
import numpy as np
import torch
from config.monitor_config import MonitorConfig
from experiment_utils.printing import get_console
from experiment_utils.utils import safe_detach

console = get_console()


class MonitorStorage:
    """Handles efficient storage of monitoring data."""

    def __init__(self, path: Path, config: MonitorConfig, mode: str = "w"):
        self.path = path
        self.config = config
        self.mode = mode
        self.file = None
        self.buffers = {}

        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize HDF5 file and create basic structure."""
        self.file = h5py.File(self.path / "monitor_data.h5", self.mode)

        # Create main groups only if they don't exist
        for group_name in ["metadata", "gradients", "activations", "weights", "convergence"]:
            if group_name not in self.file:
                self.file.create_group(group_name)

        # Store config
        self.file["metadata"].attrs["config"] = str(self.config)

    def _flatten_tensor_data(self, data, base_name: str) -> list[tuple[str, np.ndarray]]:
        """
        Recursively flatten nested tensor data structures into a list of (name, array) pairs.

        Args:
            data: Input data (tensor, array, tuple, or nested tuple)
            base_name: Base name for the dataset

        Returns:
            List of (name, array) pairs
        """
        result = []

        if isinstance(data, (torch.Tensor, np.ndarray)):
            return [(base_name, safe_detach(data, to_np=True))]
        elif isinstance(data, tuple):
            for i, item in enumerate(data):
                item_name = f"{base_name}_{i}"
                if isinstance(item, (torch.Tensor, np.ndarray)):
                    result.append((item_name, safe_detach(item, to_np=True)))
                elif isinstance(item, tuple):
                    # Recursively handle nested tuples
                    result.extend(self._flatten_tensor_data(item, item_name))
                elif item is None:
                    continue
                else:
                    raise ValueError(f"Unsupported data type: {type(item)}")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        return result

    def _safe_create_or_update_dataset(
        self, name: str, data: Union[np.ndarray, torch.Tensor, tuple], compression=None, compression_opts=None
    ):
        """
        Safely create or update a dataset, handling existing datasets appropriately.
        Supports single tensors, numpy arrays, and nested tuples of tensors (e.g., LSTM outputs).
        Properly handles CUDA tensors by moving them to CPU before conversion.
        """
        try:
            # Flatten the data structure into (name, array) pairs
            flattened_data = self._flatten_tensor_data(data, name)
            created_datasets = []  # Track successfully created datasets

            # Process each array in the flattened data
            for sub_name, sub_data in flattened_data:
                try:
                    if sub_name in self.file:
                        # Update existing dataset
                        if self.file[sub_name].shape == sub_data.shape:
                            self.file[sub_name][:] = sub_data
                            created_datasets.append(sub_name)
                        else:
                            # Recreate if shapes don't match
                            del self.file[sub_name]
                            self.file.create_dataset(
                                sub_name,
                                data=sub_data,
                                compression=compression,
                                compression_opts=compression_opts,
                            )
                            created_datasets.append(sub_name)
                    else:
                        # Create new dataset
                        self.file.create_dataset(
                            sub_name,
                            data=sub_data,
                            compression=compression,
                            compression_opts=compression_opts,
                        )
                        created_datasets.append(sub_name)
                except Exception as e:
                    console.print(f"Failed to write dataset {sub_name}: {str(e)}")
                    continue

            return created_datasets  # Return list of successfully created datasets

        except Exception as e:
            console.print(f"[error_prefix]Error handling dataset {name}: {str(e)}[/]")
            raise

    def flush_buffer(self, group: str):
        """Write buffered data to disk."""
        if not self.buffers.get(group):
            return

        for item in self.buffers[group]:
            try:
                # Create/update datasets and get list of created dataset names
                dataset_names = self._safe_create_or_update_dataset(
                    f"{group}/{item['name']}",
                    item["data"],
                    compression=self.config.compression,
                    compression_opts=self.config.compression_opts,
                )

                # Add metadata to successfully created datasets
                if dataset_names and item["metadata"]:
                    for dataset_name in dataset_names:
                        for k, v in item["metadata"].items():
                            self.file[dataset_name].attrs[k] = v

            except Exception as e:
                console.print(f"Failed to write dataset {item['name']}: {str(e)}")
                continue

        self.buffers[group] = []
        self.file.flush()

    def add_to_buffer(
        self, group: str, name: str, data: Union[np.ndarray, torch.Tensor, tuple], metadata: Dict[str, Any] = None
    ):
        """Add data to buffer for given group."""
        if group not in self.buffers:
            self.buffers[group] = []

        self.buffers[group].append({"name": name, "data": data, "metadata": metadata or {}})

        if len(self.buffers[group]) >= self.config.buffer_size:
            self.flush_buffer(group)

    def flush_all(self):
        """Flush all buffers to disk."""
        console.print("Flushing buffers...")
        for group in list(self.buffers.keys()):
            self.flush_buffer(group)
        console.print("Buffers flushed.")

    def close(self):
        """Close storage and clean up."""
        self.flush_all()
        if self.file is not None:
            self.file.close()
            self.file = None
