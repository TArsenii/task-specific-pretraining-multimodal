from torch.utils.data import Dataset
from itertools import combinations
from typing import List


class MultimodalBaseDataset(Dataset):
    """Base class for multimodal datasets with missing modality support."""

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
