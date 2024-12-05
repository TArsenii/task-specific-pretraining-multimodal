import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from modalities import Modality
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import PILToTensor, ToDtype
from experiment_utils import get_logger
from .base_dataset import MultimodalBaseDataset

logger = get_logger()


class AVMNIST(MultimodalBaseDataset):
    """Dataset class for audio-visual MNIST with enhanced missing modality support."""

    NUM_CLASSES = 10
    VALID_SPLITS = ["train", "valid", "test"]

    AVAILABLE_MODALITIES = {"audio": Modality.AUDIO, "image": Modality.IMAGE}

    def __init__(
        self,
        data_fp: Union[str, Path],
        split: str,
        target_modality: Union[Modality, str] = Modality.MULTIMODAL,
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
        force_binary: bool = False,
        selected_patterns: Optional[List[str]] = None,
        audio_column: str = "audio",
        image_column: str = "image",
        labels_column: str = "label",
        split_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            data_fp: Path to the data CSV file
            split: Dataset split ('train', 'valid', or 'test')
            target_modality: Target modality for the dataset
            missing_patterns: Dict of pattern configurations
            force_binary: Whether to force binary masking
            selected_patterns: List of patterns to use
            audio_column: Name of the audio column in the CSV
            image_column: Name of the image column in the CSV
            labels_column: Name of the labels column in the CSV
            split_indices: Optional indices for dataset splitting
        """
        super().__init__()

        # Validate split
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split: {split}. Must be one of {self.VALID_SPLITS}")

        self.data_fp = Path(data_fp)
        if not self.data_fp.exists():
            raise FileNotFoundError(f"Data file not found: {data_fp}")

        self.split = split
        self.audio_column = audio_column
        self.image_column = image_column
        self.labels_column = labels_column

        # Set up transforms
        self.transforms = {
            "pil_to_tensor": PILToTensor(),
            "scale": ToDtype(torch.float32, scale=True),
        }

        # Set up missing patterns
        self.missing_patterns = missing_patterns or {
            "ai": {"audio": 1.0, "image": 1.0},  # both present
            "a": {"audio": 1.0, "image": 0.0},  # audio only
            "i": {"audio": 0.0, "image": 1.0},  # image only
        }
        self.force_binary = force_binary

        # Handle pattern selection
        if selected_patterns is not None:
            self.selected_patterns = self.validate_patterns(selected_patterns)
        else:
            self.selected_patterns = self.get_all_possible_patterns()

        # Load and process data
        self._load_data(split_indices)
        self.num_samples = len(self.data)

        # Set up target modality
        if isinstance(target_modality, str):
            target_modality = Modality.from_str(target_modality)
        if target_modality not in [Modality.AUDIO, Modality.IMAGE, Modality.MULTIMODAL]:
            raise ValueError(f"Invalid target_modality: {target_modality}")
        self.target_modality = target_modality

        # For validation/test, organize samples by pattern
        if split != "train":
            self.pattern_indices = {pattern: list(range(self.num_samples)) for pattern in self.selected_patterns}

        logger.info(
            f"Initialized AVMNIST dataset:"
            f"\n  Split: {split}"
            f"\n  Target Modality: {target_modality}"
            f"\n  Samples: {self.num_samples}"
            f"\n  Patterns: {', '.join(self.selected_patterns)}"
        )

    def _load_data(self, split_indices: Optional[List[int]] = None) -> None:
        """Load and validate data."""
        self.data = pd.read_csv(self.data_fp)
        if split_indices is not None:
            self.data = self.data.iloc[split_indices].reset_index(drop=True)

        # Validate required columns
        required_columns = [self.audio_column, self.image_column, self.labels_column]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def __len__(self) -> int:
        """Return the total length of the dataset."""
        if self.split == "train":
            return self.num_samples
        else:
            return self.num_samples * len(self.selected_patterns)

    @lru_cache(maxsize=1000)
    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio data with caching for improved performance."""
        return torch.load(path, weights_only=True)

    @lru_cache(maxsize=1000)
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and process image data with caching for improved performance."""
        img_data = np.array(torch.load(path, weights_only=False))
        img = Image.fromarray(np.uint8(cm.gist_earth(img_data) * 255)).convert("L")
        img_tensor = self.transforms["pil_to_tensor"](img)
        return self.transforms["scale"](img_tensor)

    def _get_pattern_and_sample_idx(self, idx: int) -> Tuple[str, int]:
        """Get pattern and original sample index for a given dataset index."""
        if self.split == "train":
            return random.choice(self.selected_patterns), idx
        else:
            pattern_idx = idx // self.num_samples
            sample_idx = idx % self.num_samples
            return self.selected_patterns[pattern_idx], sample_idx

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item with missing pattern applied."""
        # Get pattern and sample index
        pattern_name, sample_idx = self._get_pattern_and_sample_idx(idx)
        pattern = self.missing_patterns[pattern_name]
        row = self.data.iloc[sample_idx]

        # Prepare base sample
        sample = {
            "label": torch.tensor(row[self.labels_column]),
            "pattern_name": pattern_name,
            "missing_mask": {},
            "sample_idx": sample_idx,
        }

        # Load and apply masking to each modality
        modality_loaders = {
            "audio": (self._load_audio, self.audio_column, Modality.AUDIO),
            "image": (self._load_image, self.image_column, Modality.IMAGE),
        }

        for mod_name, (loader_fn, column, mod_enum) in modality_loaders.items():
            if self.target_modality == Modality.MULTIMODAL or self.target_modality == mod_enum:
                # Load data
                data = loader_fn(row[column])

                # Apply masking
                if mod_name in pattern:
                    prob = pattern[mod_name]
                    if self.force_binary:
                        mask = float(prob > 0.5)
                    else:
                        mask = float(random.random() < prob) if self.split == "train" else prob
                else:
                    mask = 0.0

                sample[mod_enum] = data * mask
                sample["missing_mask"][mod_enum] = mask

        return sample

    def get_pattern_batches(self, batch_size: int, **dataloader_kwargs) -> Dict[str, DataLoader]:
        """Get separate dataloaders for each pattern (validation/test only)."""
        if self.split == "train":
            raise ValueError("Pattern-specific batches only available for validation/test")

        pattern_loaders = {}
        for pattern in self.selected_patterns:
            # Create pattern-specific dataset view
            pattern_dataset = PatternSpecificDataset(self, pattern)

            # Create dataloader
            pattern_loaders[pattern] = DataLoader(
                pattern_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn, **dataloader_kwargs
            )

        return pattern_loaders

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of samples."""
        device = batch[0]["label"].device

        # Basic items
        collated = {
            "label": torch.stack([b["label"] for b in batch]),
            "pattern_names": [b["pattern_name"] for b in batch],
            "missing_masks": {
                mod: torch.tensor([b["missing_mask"][mod] for b in batch], device=device)
                for mod in [Modality.AUDIO, Modality.IMAGE]
                if mod in batch[0]["missing_mask"]
            },
        }

        # Handle modalities based on target
        if self.target_modality == Modality.MULTIMODAL:
            for mod in [Modality.AUDIO, Modality.IMAGE]:
                if mod in batch[0]:
                    collated[mod] = torch.stack([b[mod] for b in batch])
        else:
            collated[self.target_modality] = torch.stack([b[self.target_modality] for b in batch])

        return collated


class PatternSpecificDataset(Dataset):
    """View of the main dataset that only shows samples for a specific pattern."""

    def __init__(self, parent_dataset: AVMNIST, pattern: str):
        self.parent = parent_dataset
        self.pattern = pattern
        self.sample_indices = self.parent.pattern_indices[pattern]

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = idx + (self.parent.selected_patterns.index(self.pattern) * self.parent.num_samples)
        return self.parent[real_idx]
