import random
from itertools import combinations
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch

from experiment_utils.printing import get_console
from modalities import Modality, create_missing_mask
from torch.utils.data import Dataset

console = get_console()


class MultimodalBaseDataset(Dataset):
    """Base class for multimodal datasets with mi ssing modality support."""

    def __init__(
        self,
        split: Literal["train", "valid", "test"],
        selected_patterns: Optional[List[str]] = None,
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.split = split.lower()
        assert split in self.VALID_SPLITS, f"Invalid split provided, must be one of {self.VALID_SPLITS}"

        self.missing_patterns = missing_patterns
        # Handle pattern selection
        if selected_patterns is not None:
            self.selected_patterns = self.validate_patterns(selected_patterns)
        else:
            self.selected_patterns = self.get_all_possible_patterns()
        # if "m" in self.missing_patterns:
        #     full_condition = "".join([k for k in self.AVAILABLE_MODALITIES.keys()])
        #     self.missing_patterns[full_condition] = self.missing_patterns["m"]
        #     del self.missing_patterns["m"]
        self.pattern_indices = None

    def get_sample_and_apply_mask(
        self, patterns: Dict[str, float], sample, modality_loaders: Dict[str, Tuple[Callable, Modality]]
    ) -> Dict[str, Any]:
        """Load data for each modality and apply masking."""
        for mod_name, (loader_fn, mod_enum) in modality_loaders.items():
            if self.target_modality == Modality.MULTIMODAL or self.target_modality == mod_enum:
                # Load data
                data = loader_fn()

                # Apply masking
                if mod_name in patterns:
                    prob = patterns[mod_name]
                    mask = prob  ## TODO: change this to create_missing_mask(1, data.shape[0], pct_missing) and utilise the ModalityConfig missing rates
                    # mask = create_missing_mask(1, data.shape[0], prob)
                else:
                    mask = torch.ones_like(data)

                sample[f"{str(mod_enum)}_original"] = data * mask
                sample[mod_enum] = data * mask
                sample[f"{str(mod_enum)}_reverse"] = data * -1 * (mask - 1)
                sample["missing_mask"][mod_enum] = mask

        return sample

    def _get_pattern_and_sample_idx(self, idx: int) -> Tuple[str, int]:
        """
        Get the pattern and corresponding sample index for a given dataset index.

        Args:
            idx (int): Dataset index.

        Returns:
            Tuple[str, int]: Tuple containing the pattern name and sample index.
        """
        if self.split == "train":
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
