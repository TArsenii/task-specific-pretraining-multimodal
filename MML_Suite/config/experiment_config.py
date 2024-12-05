import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from experiment_utils import get_console, get_logger

from .base_config import BaseConfig

logger = get_logger()
console = get_console()


@dataclass
class ExperimentConfig(BaseConfig):
    """
    Configuration for experiment execution with automatic seed management
    and device selection.
    """

    name: str
    seed: Optional[int] = None
    device: str = "cuda"
    debug: bool = False
    run_id: int = field(default_factory=lambda: int(time.time()))
    is_test: bool = True
    is_train: bool = True
    train_print_interval_epochs: int = 1
    validation_print_interval_epochs: int = 1
    dry_run: bool = False

    def __post_init__(self):
        """Initialize experiment configuration and set up environment."""
        console.rule(
            f"[heading]{self.name}[/]",
        )
        assert self.train_print_interval_epochs > 0, "Print interval must be positive"
        assert self.validation_print_interval_epochs > 0, "Print interval must be positive"
        self._setup_seed()
        self._setup_device()
        self._display_config()

    def _setup_seed(self) -> None:
        """Set up and validate random seeds for reproducibility."""
        if self.seed is None:
            self.seed = int(time.time())
            logger.info(f"Generated random seed: {self.seed}")

        try:
            # Set seeds for all relevant components
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

            # Configure CUDA backend
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True

            logger.info("Successfully set random seeds and CUDA configuration")
            console.print(f"[bold green]✓[/] Set random seed: {self.seed}")
        except Exception as e:
            error_msg = f"Error setting random seeds: {str(e)}"
            logger.error(error_msg)
            console.print(f"[boldred]✗[/] {error_msg}")
            raise

    def _setup_device(self) -> None:
        """Set up and validate compute device."""
        try:
            if self.device.lower() == "cuda" and not torch.cuda.is_available():
                # logger.warning("CUDA requested but not available, falling back to CPU")
                console.print("[bold yellow]![/] CUDA not available, using CPU")
                self.device = "cpu"

            self.device = torch.device(self.device)

            if self.device.type == "cuda":
                device_name = torch.cuda.get_device_name(self.device)
                logger.info(f"Using CUDA device: {device_name}")
                console.print(f"[bold green]✓[/] Using CUDA device: {device_name}")
            else:
                logger.info("Using CPU device")
                console.print("[bold green]✓[/] Using CPU device")

        except Exception as e:
            error_msg = f"Error setting up device: {str(e)}"
            logger.error(error_msg)
            console.print(f"[bold red]✗[/] {error_msg}")
            raise

    def _display_config(self) -> None:
        """Display the experiment configuration."""
        console.print("\n[bold cyan]Experiment Configuration:[/]")
        console.print(f"  Name: {self.name}")
        console.print(f"  Run ID: {self.run_id}")
        console.print(f"  Seed: {self.seed}")
        console.print(f"  Device: {self.device}")
        console.print(f"  Debug Mode: {'Enabled' if self.debug else 'Disabled'}")
        console.print(f"  Test Mode: {'Enabled' if self.is_test else 'Disabled'}")
        console.print(f"  Train Mode: {'Enabled' if self.is_train else 'Disabled'}")
        console.print(f"  Train Print Interval: {self.train_print_interval_epochs} epochs")
        console.print(f"  Validation Print Interval: {self.validation_print_interval_epochs} epochs")

        if self.debug:
            console.print("\n[yellow]Warning: Debug mode is enabled[/]")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            "name": self.name,
            "seed": self.seed,
            "device": str(self.device),
            "debug": self.debug,
            "run_id": self.run_id,
            "is_test": self.is_test,
            "is_train": self.is_train,
            "train_print_interval_epochs": self.train_print_interval_epochs,
            "validation_print_interval_epochs": self.validation_print_interval_epochs,
        }

    def __str__(self) -> str:
        """Return string representation of the configuration."""
        return str("\n".join([f"{k}: {v}" for k, v in self.to_dict().items()]))
