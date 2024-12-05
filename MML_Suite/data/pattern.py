from typing import Any, Dict

from data.base_dataset import MultimodalBaseDataset


class PatternSpecificDataset(MultimodalBaseDataset):
    """View of the main dataset that only shows samples for a specific pattern."""

    def __init__(self, parent_dataset: MultimodalBaseDataset, pattern: str):
        self.parent = parent_dataset
        self.pattern = pattern
        self.sample_indices = self.parent.pattern_indices[pattern]

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = idx + (self.parent.selected_patterns.index(self.pattern) * self.parent.num_samples)
        return self.parent[real_idx]
