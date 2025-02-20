import random
from itertools import combinations
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from experiment_utils.printing import get_console
from experiment_utils.logging import get_logger
from experiment_utils.utils import AccessError, NestedDictAccess
from modalities import Modality, create_missing_mask
from torch.utils.data import Dataset

console = get_console()
logger = get_logger()


class MultimodalBaseDataset(Dataset):
    """Base class for multimodal datasets with mi ssing modality support."""

    _ndict_accessor = NestedDictAccess(max_depth=5, logger=logger)

    def __init__(
        self,
        split: Literal["train", "valid", "test"],
        selected_patterns: Optional[List[str]] = None,
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
        batch_size: int = 1,
        _id: int = 1,
    ) -> None:
        self.split = split.lower()
        assert split in self.VALID_SPLITS, f"Invalid split provided, must be one of {self.VALID_SPLITS}"

        self.missing_patterns = missing_patterns

        # Handle pattern selection
        if selected_patterns is not None:
            self.selected_patterns = self.validate_patterns(selected_patterns)
        else:
            self.selected_patterns = self.get_all_possible_patterns()
        self.pattern_indices = None
        self._batch_size = batch_size
        self.current_pattern = None

        assert isinstance(_id, int), "ID must be an integer."
        self._id = _id

    def _initialise_missing_masks(
        self, missing_patterns: Optional[Dict[str, Dict[Modality, float]]], batch_size: int = 1
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Initialise missing masks for each pattern."""
        masks = {}
        if missing_patterns is not None:
            for pattern, modality_patterns in missing_patterns.items():
                _mask = create_missing_mask(
                    len(self.AVAILABLE_MODALITIES), batch_size, [1 - pct for _, pct in modality_patterns.items()]
                ).squeeze()

                masks[pattern] = {modality: col for modality, col in zip(modality_patterns.keys(), _mask.T)}

        return masks

    def get_samples(
        self,
        sample: Dict[str, Any],
        modality_loaders: Dict[str, Tuple[Callable, Modality]],
    ) -> Dict[str, Any]:
        """Load data for each modality."""
        for _mod_name, (loader_fn, mod_enum) in modality_loaders.items():
            if self.target_modality == Modality.MULTIMODAL or self.target_modality == mod_enum:
                sample[f"{mod_enum}_original"] = loader_fn()
                mask = sample[f"{mod_enum}_missing_index"]
                sample[mod_enum] = sample[f"{mod_enum}_original"] * mask
                sample[f"{mod_enum}_reverse"] = sample[f"{mod_enum}_original"] * -1 * (mask - 1)

        return sample

    def _get_pattern_and_sample_idx(self, idx: int) -> Tuple[str, int]:
        """
        Get the pattern and corresponding sample index for a given dataset index.

        Args:
            idx (int): Dataset index.

        Returns:
            Tuple[str, int]: Tuple containing the pattern name and sample index.
        """
        if self.split == "train" or self.split == "trn":
            mp = random.choice(self.selected_patterns)
            return mp, idx
        else:
            pattern_idx = idx // self.num_samples
            sample_idx = idx % self.num_samples
            return self.selected_patterns[pattern_idx], sample_idx

    def set_pattern_indices(self, n_samples: int) -> None:
        # For validation/test, organize samples by pattern
        if self.split != "train":
            self.pattern_indices = {pattern: list(range(n_samples)) for pattern in self.selected_patterns}

    def get_split(self) -> str:
        return self.split

    def get_selected_patterns(self) -> List[str]:
        return self.selected_patterns

    def get_missing_patterns(self) -> Dict[str, Dict[str, float]]:
        return self.missing_patterns

    @staticmethod
    def get_full_modality() -> str:
        """Return the name of the full modality."""
        raise NotImplementedError("Method get_full_modality must be implemented in the derived class.")

    @classmethod
    def get_all_possible_patterns(cls) -> List[str]:
        """Generate all possible modality combinations excluding empty set."""
        modalities = list(cls.AVAILABLE_MODALITIES.keys())
        patterns = []
        for r in range(1, len(modalities) + 1):
            for combo in combinations(modalities, r):
                pattern_name = "".join(m[0] for m in sorted(combo))
                patterns.append(pattern_name)
        return sorted(patterns)

    def validate_patterns(self, patterns: List[str]) -> List[str]:
        """Validate and normalize pattern names."""
        all_patterns = self.get_all_possible_patterns()
        invalid_patterns = set(patterns) - set(all_patterns)
        if invalid_patterns:
            raise ValueError(f"Invalid patterns: {invalid_patterns}\n" f"Valid patterns are: {all_patterns}")
        return patterns

    def set_selected_pattern(self, pattern: str) -> None:
        assert hasattr(self, "selected_pattern"), "Dataset must have attribute selected_pattern"
        assert pattern in self.get_all_possible_patterns(), "Invalid pattern"
        self.selected_pattern = pattern

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pattern, sample_idx = self._get_pattern_and_sample_idx(idx)

        data = {
            "pattern": pattern,
            "sample_idx": sample_idx,
        }

        for modality in self.AVAILABLE_MODALITIES.values():
            try:
                mask = MultimodalBaseDataset._ndict_accessor.get(self.masks, [pattern, modality, sample_idx])
            except Exception as e:
                console.error(f"Error accessing missing mask: {e}")
                exit(1)

            data[f"{str(modality)}_missing_index"] = mask

        return data
