from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from data.base_dataset import MultimodalBaseDataset
from data.pattern import PatternSpecificDataset
from experiment_utils.logging import get_logger
from matplotlib import cm
from modalities import Modality
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import PILToTensor, ToDtype

logger = get_logger()


class AVMNIST(MultimodalBaseDataset):
    """
    Dataset class for the audio-visual MNIST dataset with support for missing modality patterns.

    This dataset supports both training and evaluation splits with customizable missing patterns and
    dynamic masking for audio and image modalities.
    """

    NUM_CLASSES: int = 10
    VALID_SPLITS: List[Literal["train", "valid", "test"]] = ["train", "valid", "test"]
    AVAILABLE_MODALITIES: Dict[str, Modality] = {"audio": Modality.AUDIO, "image": Modality.IMAGE}

    @staticmethod
    def get_full_modality() -> str:
        """
        Get the concatenated string representation of all available modalities.

        Returns:
            str: Sorted concatenation of the first letters of all available modality keys.
        """
        modality_keys = [k[0] for k in AVMNIST.AVAILABLE_MODALITIES.keys()]
        modality_keys.sort()
        return "".join(modality_keys)

    def __init__(
        self,
        data_fp: Path | PathLike,
        split: str,
        target_modality: Modality | str = Modality.MULTIMODAL,
        *,
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
        selected_patterns: Optional[List[str]] = None,
        audio_column: str = "audio",
        image_column: str = "image",
        labels_column: str = "label",
        split_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the AVMNIST dataset.

        Args:
            data_fp (PathLike): Path to the data CSV file.
            split (str): Dataset split ("train", "valid", or "test").
            target_modality (Modality | str): Target modality for the dataset.
            missing_patterns (Optional[Dict[str, Dict[str, float]]]): Dict of pattern configurations.
            selected_patterns (Optional[List[str]]): List of selected patterns for evaluation.
            audio_column (str): Name of the audio column in the CSV.
            image_column (str): Name of the image column in the CSV.
            labels_column (str): Name of the labels column in the CSV.
            split_indices (Optional[List[int]]): Optional indices for dataset splitting.
        """
        m_patterns = missing_patterns or {
            "ai": {"audio": 1.0, "image": 1.0},  # Both modalities present
            "a": {"audio": 1.0, "image": 0.0},  # Audio only
            "i": {"audio": 0.0, "image": 1.0},  # Image only
        }
        super().__init__(split=split, selected_patterns=selected_patterns, missing_patterns=m_patterns)

        assert split in AVMNIST.VALID_SPLITS, f"Invalid split provided, must be one of {AVMNIST.VALID_SPLITS}"

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

        # Load and process data
        self._load_data(split_indices)

        self.num_samples = len(self.data)
        self.set_pattern_indices(self.num_samples)

        if isinstance(target_modality, str):
            target_modality = Modality.from_str(target_modality)
        assert isinstance(
            target_modality, Modality
        ), f"Invalid modality provided, must be a Modality Enum, not {type(target_modality)}"
        assert target_modality in [
            Modality.AUDIO,
            Modality.IMAGE,
            Modality.MULTIMODAL,
        ], "Invalid modality provided, must be one of [audio, image, multimodal]"
        self.target_modality = target_modality

        logger.info(
            f"Initialized AVMNIST dataset:"
            f"\n  Split: {split}"
            f"\n  Target Modality: {target_modality}"
            f"\n  Samples: {self.num_samples}"
            f"\n  Patterns: {', '.join(self.selected_patterns)}"
        )

    def _load_data(self, split_indices: Optional[List[int]] = None) -> None:
        """
        Load and validate dataset from a CSV file.

        Args:
            split_indices (Optional[List[int]]): Optional indices for filtering rows.
        """
        self.data = pd.read_csv(self.data_fp)
        if split_indices is not None:
            self.data = self.data.iloc[split_indices].reset_index(drop=True)

        # Validate required columns
        required_columns = [self.audio_column, self.image_column, self.labels_column]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def __len__(self) -> int:
        """
        Return the total length of the dataset.

        Returns:
            int: Total number of samples.
        """
        if self.split == "train":
            return self.num_samples
        else:
            return self.num_samples * len(self.selected_patterns)

    @lru_cache(maxsize=1000)
    def _load_audio(self, path: str) -> torch.Tensor:
        """
        Load audio data from a file with caching.

        Args:
            path (str): Path to the audio file.

        Returns:
            torch.Tensor: Loaded audio data as a tensor.
        """
        return torch.load(path, weights_only=True)

    @lru_cache(maxsize=1000)
    def _load_image(self, path: str) -> torch.Tensor:
        """
        Load and process image data from a file with caching.

        Args:
            path (str): Path to the image file.

        Returns:
            torch.Tensor: Processed image data as a tensor.
        """
        img_data = np.array(torch.load(path, weights_only=False))
        img = Image.fromarray(np.uint8(cm.gist_earth(img_data) * 255)).convert("L")
        img_tensor = self.transforms["pil_to_tensor"](img)
        return self.transforms["scale"](img_tensor)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a dataset sample by index with missing pattern applied.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, Any]: A dictionary containing the sample data and metadata.
        """
        pattern_name, sample_idx = self._get_pattern_and_sample_idx(idx)
        pattern = self.missing_patterns[pattern_name]
        row = self.data.iloc[sample_idx]

        sample = {
            "label": torch.tensor(row[self.labels_column]),
            "pattern_name": pattern_name,
            "missing_mask": {},
            "sample_idx": sample_idx,
        }

        # Load and apply masking for each modality
        modality_loaders = {
            "audio": (lambda x: self._load_audio(self.data.iloc[x][self.audio_column]), Modality.AUDIO),
            "image": (lambda x: self._load_image(self.data.iloc[x][self.image_column]), Modality.IMAGE),
        }
        sample = self.get_sample_and_apply_mask(pattern, sample, modality_loaders, sample_idx)
        return sample

    def get_pattern_batches(self, batch_size: int, **dataloader_kwargs) -> Dict[str, DataLoader]:
        """
        Get separate DataLoaders for each pattern.

        Args:
            batch_size (int): Batch size for the DataLoader.
            **dataloader_kwargs: Additional DataLoader keyword arguments.

        Returns:
            Dict[str, DataLoader]: A dictionary of DataLoaders for each pattern.
        """
        if self.split == "train":
            raise ValueError("Pattern-specific batches only available for validation/test")

        pattern_loaders = {}
        for pattern in self.selected_patterns:
            pattern_dataset = PatternSpecificDataset(self, pattern)
            pattern_loaders[pattern] = DataLoader(
                pattern_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn, **dataloader_kwargs
            )
        return pattern_loaders

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collation function for batching.

        Args:
            batch (List[Dict[str, Any]]): List of samples.

        Returns:
            Dict[str, Any]: Collated batch of samples.
        """
        device = batch[0]["label"].device

        collated = {
            "label": torch.stack([b["label"] for b in batch]),
            "pattern_names": [b["pattern_name"] for b in batch],
            "missing_masks": {
                mod: torch.tensor([b["missing_mask"][mod] for b in batch], device=device)
                for mod in [Modality.AUDIO, Modality.IMAGE]
                if mod in batch[0]["missing_mask"]
            },
        }

        if self.target_modality == Modality.MULTIMODAL:
            for mod in [Modality.AUDIO, Modality.IMAGE]:
                if mod in batch[0]:
                    collated[mod] = torch.stack([b[mod] for b in batch])
        else:
            collated[self.target_modality] = torch.stack([b[self.target_modality] for b in batch])

        return collated
