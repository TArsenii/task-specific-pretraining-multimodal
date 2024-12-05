import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from .base_dataset import MultimodalBaseDataset
from collections import OrderedDict
from experiment_utils import get_logger, get_console
from modalities import Modality, add_modality
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

logger = get_logger()
console = get_console()

add_modality("video")


class MultimodalSentimentDataset(MultimodalBaseDataset):
    """Base class for CMU-MOSI and CMU-MOSEI datasets with missing modality support."""

    AVAILABLE_MODALITIES = OrderedDict(
        [
            ("audio", Modality.AUDIO),
            ("video", Modality.VIDEO),
            ("text", Modality.TEXT),
        ]
    )

    @staticmethod
    def get_full_modality() -> str:
        m = [k[0] for k in MultimodalSentimentDataset.AVAILABLE_MODALITIES.keys()]
        m.sort()
        return "".join(m)

    def __init__(
        self,
        data_fp: Union[str, Path],
        split: str,
        target_modality: Union[str, Modality] = Modality.MULTIMODAL,
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
        force_binary: bool = False,
        selected_patterns: Optional[List[str]] = None,
        labels_key: str = "regression_labels",
        aligned: bool = False,
        length: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            data_fp: Path to data file
            split: Dataset split ('train', 'valid', 'test')
            target_modality: Target modality for model
            missing_patterns: Dict of pattern configurations
            force_binary: Whether to force binary masking
            selected_patterns: List of patterns to use
            labels_key: Key for accessing labels in data
        """
        super(MultimodalSentimentDataset, self).__init__()
        self.data_fp = Path(data_fp)
        self.split = split
        self.aligned = aligned

        if self.aligned:
            self.length = length
        self.labels_key = labels_key
        # Process target modality
        if isinstance(target_modality, str):
            target_modality = Modality.from_str(target_modality)
        self.target_modality = target_modality

        # Set up missing patterns
        self.missing_patterns = missing_patterns or {}
        self.force_binary = force_binary

        # Load and validate data
        self.data = self._load_data(labels_key)
        self.num_samples = len(self.data["label"])

        # Set up pattern selection
        if selected_patterns is not None:
            self.selected_patterns = self.validate_patterns(selected_patterns)
        else:
            self.selected_patterns = self.get_all_possible_patterns()
            if "m" in self.missing_patterns:
                full_condition = "".join([k for k in self.AVAILABLE_MODALITIES.keys()])
                self.missing_patterns[full_condition] = self.missing_patterns["m"]
                del self.missing_patterns["m"]

        # For validation/test, organize samples by pattern
        if split != "train":
            self.pattern_indices = {pattern: list(range(self.num_samples)) for pattern in self.selected_patterns}

        logger.info(
            f"Initialized {self.__class__.__name__} dataset:"
            f"\n  Split: {split}"
            f"\n  Target Modality: {target_modality}"
            f"\n  Samples: {self.num_samples}"
            f"\n  Patterns: {', '.join(self.selected_patterns)}"
        )

    def _load_data(self, labels_key: str) -> Dict[str, torch.Tensor]:
        """Load and preprocess data from file."""
        if not self.data_fp.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_fp}")

        with open(self.data_fp, "rb") as f:
            raw_data = pickle.load(f)

        if self.split not in raw_data:
            raise KeyError(f"Split '{self.split}' not found in data")

        split_data = raw_data[self.split]

        if labels_key not in split_data:
            raise KeyError(f"Labels key '{labels_key}' not found in data")

        # Convert data to tensors
        core_data = {
            "audio": torch.tensor(split_data["audio"]).float(),
            "video": torch.tensor(split_data["vision"]).float(),
            "text": torch.tensor(split_data["text"]).float(),
            "label": torch.tensor(
                split_data[labels_key], dtype=torch.float32 if "regression" in self.labels_key else torch.long
            ),
        }

        return (
            core_data
            if self.aligned
            else core_data
            | {
                "audio_lengths": torch.tensor(split_data["audio_lengths"]).float(),
                "video_lengths": torch.tensor(split_data["vision_lengths"]).float(),
            }
        )

    def __len__(self) -> int:
        if self.split == "train":
            return self.num_samples
        else:
            return self.num_samples * len(self.selected_patterns)

    def _get_pattern_and_sample_idx(self, idx: int) -> tuple[str, int]:
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

        # Prepare base sample
        sample = {
            "label": self.data["label"][sample_idx],
            "pattern_name": pattern_name,
            "missing_mask": {},
            "sample_idx": sample_idx,
        }

        if not self.aligned:
            sample["audio_length"] = self.data["audio_lengths"][sample_idx]
            sample["video_length"] = self.data["video_lengths"][sample_idx]

        # Apply masking to each modality
        for mod_name, mod_enum in self.AVAILABLE_MODALITIES.items():
            data = self.data[mod_name][sample_idx]

            if mod_name in pattern:
                prob = pattern[mod_name]
                if self.force_binary:
                    mask = float(prob > 0.5)
                else:
                    mask = float(random.random() < prob) if self.split == "train" else prob
            else:
                mask = 0.0

            sample[str(mod_enum) + "_original"] = data
            sample[mod_enum] = data * mask
            sample[str(mod_enum) + "_reverse"] = data * -1 * (mask - 1)

            sample["missing_mask"][mod_enum] = mask

        # shape_info = {
        #     mod_name: sample[mod_enum].shape
        #     for mod_name, mod_enum in self.AVAILABLE_MODALITIES.items()
        #     if mod_enum in sample
        # }

        # console.print(f"Shapes: {shape_info}")

        # Handle target modality selection
        if self.target_modality != Modality.MULTIMODAL:
            return (
                {
                    self.target_modality: sample[self.target_modality],
                    "label": sample["label"],
                    "length": sample["length"],
                    "pattern_name": sample["pattern_name"],
                    "missing_mask": {self.target_modality: sample["missing_mask"][self.target_modality]},
                    "sample_idx": sample["sample_idx"],
                }
                if self.aligned
                else {
                    self.target_modality: sample[self.target_modality],
                    "label": sample["label"],
                    "audio_length": sample["audio_length"],
                    "video_length": sample["video_length"],
                    "pattern_name": sample["pattern_name"],
                    "missing_mask": {self.target_modality: sample["missing_mask"][self.target_modality]},
                    "sample_idx": sample["sample_idx"],
                }
            )
        return sample

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of samples with pattern-aware batching."""
        if self.split == "train":
            return self._collate_train_batch(batch)
        else:
            return self._collate_eval_batch(batch)

    def _collate_train_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate training batch with mixed patterns."""

        # Basic items
        collated = {
            "label": torch.stack([b["label"] for b in batch]),
            "lengths": torch.stack([b["length"] for b in batch]),
            "pattern_names": [b["pattern_name"] for b in batch],
            # "missing_masks": {
            #     mod: torch.tensor([b["missing_mask"][mod] for b in batch], device=device)
            #     for mod in self.AVAILABLE_MODALITIES.values()
            # },
        }

        # Handle modalities based on target
        if self.target_modality == Modality.MULTIMODAL:
            for mod_enum in self.AVAILABLE_MODALITIES.values():
                sequences = [b[mod_enum] for b in batch]
                padded = pad_sequence(sequences, batch_first=True, padding_value=0)
                collated[mod_enum] = padded
        else:
            sequences = [b[self.target_modality] for b in batch]
            padded = pad_sequence(sequences, batch_first=True, padding_value=0)
            collated[self.target_modality] = padded

        return collated

    def _collate_eval_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate evaluation batch with pattern-specific grouping."""
        # Group samples by pattern
        pattern_groups = {}
        for b in batch:
            pattern = b["pattern_name"]
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(b)

        # Collate each pattern group
        collated_groups = {}
        for pattern, group in pattern_groups.items():
            collated_groups[pattern] = self._collate_train_batch(group)

        return collated_groups if len(pattern_groups) > 1 else list(collated_groups.values())[0]

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
                pattern_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate_train_batch,
                **dataloader_kwargs,
            )

        return pattern_loaders

    @staticmethod
    def normalize_features(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize features along time dimension."""
        mean = torch.mean(features, dim=0, keepdim=True)
        std = torch.std(features, dim=0, keepdim=True)
        std = torch.clamp(std, min=eps)
        return (features - mean) / std


class PatternSpecificDataset(Dataset):
    """View of the main dataset that only shows samples for a specific pattern."""

    def __init__(self, parent_dataset: MultimodalSentimentDataset, pattern: str):
        self.parent = parent_dataset
        self.pattern = pattern
        self.sample_indices = self.parent.pattern_indices[pattern]

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = idx + (self.parent.selected_patterns.index(self.pattern) * self.parent.num_samples)
        return self.parent[real_idx]


class MOSEI(MultimodalSentimentDataset):
    """CMU-MOSEI dataset implementation."""

    @staticmethod
    def get_num_classes(is_classification: bool = True) -> int:
        """Get number of classes for the task."""
        return 3 if is_classification else 1


class MOSI(MultimodalSentimentDataset):
    """CMU-MOSI dataset implementation."""

    @staticmethod
    def get_num_classes(is_classification: bool = True) -> int:
        """Get number of classes for the task."""
        return 3 if is_classification else 1
