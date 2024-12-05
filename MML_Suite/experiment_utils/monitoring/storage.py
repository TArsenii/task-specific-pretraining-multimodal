from pathlib import Path
from typing import Any, Dict, Union

import h5py
import numpy as np
import torch
from config.monitor_config import MonitorConfig
from experiment_utils.printing import get_console

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

    def _safe_create_or_update_dataset(self, name: str, data: np.ndarray, compression=None, compression_opts=None):
        """Safely create or update a dataset, handling existing datasets appropriately."""
        try:
            if name in self.file:
                # If dataset exists, check if shapes match
                if self.file[name].shape == data.shape:
                    # Update existing dataset
                    self.file[name][:] = data
                else:
                    # Delete and recreate if shapes don't match
                    del self.file[name]
                    self.file.create_dataset(
                        name, data=data, compression=compression, compression_opts=compression_opts
                    )
            else:
                # Create new dataset
                self.file.create_dataset(name, data=data, compression=compression, compression_opts=compression_opts)
        except Exception as e:
            console.print(f"[error_prefix]Error handling dataset {name}: {str(e)}[/]")
            raise

    def add_to_buffer(
        self, group: str, name: str, data: Union[np.ndarray, torch.Tensor], metadata: Dict[str, Any] = None
    ):
        """Add data to buffer for given group."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if group not in self.buffers:
            self.buffers[group] = []

        self.buffers[group].append({"name": name, "data": data, "metadata": metadata or {}})

        if len(self.buffers[group]) >= self.config.buffer_size:
            self.flush_buffer(group)

    def flush_buffer(self, group: str):
        """Write buffered data to disk."""
        if not self.buffers.get(group):
            return

        for item in self.buffers[group]:
            dataset_name = f"{group}/{item['name']}"
            try:
                self._safe_create_or_update_dataset(
                    dataset_name,
                    item["data"],
                    compression=self.config.compression,
                    compression_opts=self.config.compression_opts,
                )

                # Update metadata attributes
                for k, v in item["metadata"].items():
                    self.file[dataset_name].attrs[k] = v

            except Exception as e:
                console.print(f"[error_prefix]Failed to write dataset {dataset_name}: {str(e)}[/]")
                raise

        self.buffers[group] = []
        self.file.flush()

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
