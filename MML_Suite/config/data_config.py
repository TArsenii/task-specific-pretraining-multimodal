from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import chain, combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config import BaseConfig
from experiment_utils import get_console, get_logger
from experiment_utils.utils import format_path_with_env
from torch.utils.data import DataLoader, Dataset

from .resolvers import resolve_dataset_name

logger = get_logger()
console = get_console()


@dataclass
class ModalityConfig:
    """Configuration for a single modality's missing rate"""

    missing_rate: float = 0.0

    def __post_init__(self):
        if not 0 <= self.missing_rate <= 1:
            raise ValueError(f"Missing rate must be between 0 and 1, got {self.missing_rate}")


@dataclass
class MissingPatternConfig:
    """Configuration for missing patterns"""

    modalities: Dict[str, ModalityConfig] = field(default_factory=OrderedDict)
    force_binary: bool = False
    selected_patterns: Optional[List[str]] = None

    def __post_init__(self):
        if self.selected_patterns:
            self.selected_patterns = ["".join(sorted(p)) for p in self.selected_patterns]

    @property
    def available_modalities(self) -> Set[str]:
        """Get set of available modalities"""
        return set(self.modalities.keys()) | {"multimodal"}

    def generate_patterns(self) -> Dict[str, Dict[str, float]]:
        """Generate all possible missing patterns"""
        base_mods = set(self.modalities.keys())
        all_mods = base_mods | {"multimodal"}

        # Generate powerset excluding empty set
        mod_combinations = chain.from_iterable(combinations(base_mods, r) for r in range(1, len(base_mods) + 1))

        patterns = {}
        for combo in mod_combinations:
            combo = sorted(combo)
            pattern_name = "".join(m[0] for m in combo)

            probs = {}
            for modality in all_mods:
                if modality in combo:
                    # Present modality - use configured missing rate
                    rate = 0.0 if modality == "multimodal" else self.modalities[modality].missing_rate
                    probs[modality] = 1.0 if self.force_binary else (1 - rate)
                else:
                    # Absent modality - always mask
                    probs[modality] = 0.0

            patterns[pattern_name] = probs


        # Filter patterns if selected_patterns is specified
        if self.selected_patterns:
            patterns = {k: v for k, v in patterns.items() if k in self.selected_patterns}

        return patterns


@dataclass
class DatasetConfig(BaseConfig):
    """Enhanced dataset configuration with missing pattern support"""

    # Existing fields
    dataset: str
    data_fp: str
    target_modality: str
    split: str
    batch_size: int = 32
    shuffle: bool = False
    pin_memory: bool = False
    drop_last: bool = False
    num_workers: int = 0
    selected_missing_types: Optional[List[str]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # New missing pattern configuration
    missing_patterns: Optional[MissingPatternConfig] = None

    def __post_init__(self):
        """Validate configuration on initialization."""
        self.data_fp = format_path_with_env(self.data_fp)
        self._validate_config()

        # Initialize default missing pattern config if none provided
        if self.missing_patterns is None:
            self.missing_patterns = MissingPatternConfig()

    def get_dataset_args(self) -> Dict[str, Any]:
        """Extract dataset specific arguments."""
        args = {
            "data_fp": self.data_fp,
            "split": self.split,
            "target_modality": self.target_modality,
        }

        # Add missing pattern configuration if available
        if self.missing_patterns:
            args.update(
                {
                    "missing_patterns": self.missing_patterns.generate_patterns(),
                    "force_binary": self.missing_patterns.force_binary,
                    "selected_patterns": self.missing_patterns.selected_patterns,
                }
            )

        # Add any additional kwargs
        args.update(self.kwargs)

        logger.debug(f"Dataset arguments: {args}")
        return args

    def __str__(self) -> str:
        """Return a string representation of the configuration."""

        data_loader_str = "\n".join(f"{key}={getattr(self, key)}" for key in self.get_dataloader_args())

        # dataset_str = "\n".join(f"{key}={getattr(self, key)}" for key, value in self.get_dataset_args().items())

        dataset_str = f"dataset={self.dataset}\ndata_fp={self.data_fp}\nsplit={self.split}\ntarget_modality={self.target_modality}"
        missing_str = f"missing_patterns={self.missing_patterns}" if self.missing_patterns else ""

        return f"{dataset_str}\n{data_loader_str}\n{missing_str}"

    def _validate_config(self) -> None:
        """Validate the dataset configuration."""
        try:
            # Validate data path
            data_path = Path(self.data_fp)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_fp}")

            # Validate dataset class

            self._dataset_cls = resolve_dataset_name(self.dataset)

            logger.info(f"Validated dataset class: {self.dataset}")
            console.print(f"[green]✓[/] Dataset class verified: {self._dataset_cls.__name__}")

        except Exception as e:
            error_msg = f"Configuration validation failed: {str(e)}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            raise

    def get_dataloader_args(self) -> Dict[str, Any]:
        """Extract DataLoader specific arguments."""
        args = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }
        logger.debug(f"DataLoader arguments: {args}")
        return args

    def build_dataset(self) -> Dataset:
        """Build the dataset instance."""
        try:
            dataset_args = self.get_dataset_args()
            dataset = self._dataset_cls(**dataset_args)
            logger.info(f"Created {self._dataset_cls.__name__} dataset for {self.split} split")
            console.print(f"[green]✓[/] Created dataset: {self._dataset_cls.__name__} ({len(dataset)} samples)")
            return dataset
        except Exception as e:
            error_msg = f"Failed to create dataset: {str(e)}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            raise e


@dataclass
class DataConfig(BaseConfig):
    """Enhanced data configuration with advanced builder patterns."""

    datasets: Dict[str, DatasetConfig]
    default_batch_size: int = 32

    def __post_init__(self):
        """Initialize and validate the configuration."""
        self._validate_configs()

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        return "\n".join(str(config) for config in self.datasets.values())

    def _validate_configs(self) -> None:
        """Validate all dataset configurations."""
        if not self.datasets:
            raise ValueError("No datasets configured")

        for name, config in self.datasets.items():
            try:
                if not isinstance(config, DatasetConfig):
                    self.datasets[name] = DatasetConfig.from_dict(config)
                logger.info(f"Validated configuration for {name} dataset")
            except Exception as e:
                error_msg = f"Invalid configuration for {name} dataset: {str(e)}"
                logger.error(error_msg)
                console.print(f"[red]✗[/] {error_msg}")
                raise

    def build_dataloader(
        self,
        target_split: str,
    ) -> DataLoader:
        """
        Build a DataLoader for the specified split with enhanced error handling
        and logging.

        Args:
            target_split: The split to build the DataLoader for
            batch_size: Optional batch size override
            print_fn: Function to use for printing status messages

        Returns:
            DataLoader instance for the specified split
        """
        if target_split not in self.datasets:
            raise KeyError(f"Split '{target_split}' not found in configuration")

        try:
            dataset_config = self.datasets[target_split]

            # Build the dataset
            dataset = dataset_config.build_dataset()

            # Get DataLoader arguments
            dataloader_args = dataset_config.get_dataloader_args()

            # Add collate function if available
            if hasattr(dataset, "collate"):
                dataloader_args["collate_fn"] = dataset.collate
                logger.debug("Using custom collate function from dataset")

            # Create the DataLoader
            dataloader = DataLoader(dataset, **dataloader_args)

            # Log success
            logger.info(f"Created DataLoader for {target_split} split " f"(batch_size={dataloader_args['batch_size']})")
            console.print(f"[green]✓[/] Created DataLoader for {target_split} split")

            return dataloader

        except Exception as e:
            error_msg = f"Failed to build DataLoader for {target_split}: {str(e)}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            raise e

    def build_all_dataloaders(self) -> Dict[str, DataLoader]:
        """Build DataLoaders for all configured splits."""
        dataloaders = {}
        for split in self.datasets:
            try:
                dataloaders[split] = self.build_dataloader(split)
            except Exception as e:
                logger.error(f"Failed to build DataLoader for {split}: {str(e)}")
        return dataloaders
