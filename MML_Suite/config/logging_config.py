import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_utils import SafeDict, get_console, get_logger
from experiment_utils.utils import format_path_with_env
from rich.table import Table

from .base_config import BaseConfig

# Get logger and console
logger = get_logger()
console = get_console()


@dataclass
class LoggingConfig(BaseConfig):
    """
    Configuration for logging with automatic path formatting, creation,
    and enhanced logging capabilities using Rich console.
    """

    log_path: str
    run_id: str
    experiment_name: str
    save_metric: Optional[str] = None
    model_output_path: Optional[str] = None
    metrics_path: Optional[str] = None
    iid_metrics_path: Optional[str] = None
    monitor_path: Optional[str] = None
    tensorboard_path: Optional[str] = None
    tb_record_only: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], run_id: int | str, experiment_name: str) -> "LoggingConfig":
        """Create LoggingConfig from a dictionary with logging."""

        if isinstance(run_id, int):
            run_id = str(run_id)
        logger.info(f"Creating LoggingConfig for experiment: {experiment_name}, run: {run_id}")

        data["log_path"] = format_path_with_env(data["log_path"])
        data["metrics_path"] = format_path_with_env(data["metrics_path"])
        data["model_output_path"] = format_path_with_env(data["model_output_path"])
        if "iid_metrics_path" in data:
            data["iid_metrics_path"] = format_path_with_env(data["iid_metrics_path"])
        if "monitor_path" in data:
            data["monitor_path"] = format_path_with_env(data["monitor_path"])

        if "tensorboard_path" in data:
            data["tensorboard_path"] = format_path_with_env(data["tensorboard_path"])

        config_data = {"run_id": run_id, "experiment_name": experiment_name, **data}

        log_config = cls(**config_data)

        # log_config.__post_init__()
        return log_config

    def __post_init__(self):
        """Format paths, create directories, and log initial setup."""
        # console.print(Panel("[bold blue]Initializing Logging Configuration[/]"))

        # Sanitize experiment name first as it's used in other paths
        self.experiment_name = self._sanitize_string(self.experiment_name)
        self.run_id = self._sanitize_string(self.run_id)

        # Format and create all paths
        self._process_paths()
        self._create_directories()
        self._log_configuration()

    @staticmethod
    def _sanitize_string(s: str) -> str:
        """Convert spaces, hyphens, and other special characters to underscores."""
        return re.sub(r"[^\w\s-]|[\s-]+", "_", s).strip("_")

    def format_path(self, path: str) -> Path:
        """Format a path string with config values and return a Path object."""
        if not path:
            return None

        format_vars = SafeDict(
            {
                "experiment_name": self.experiment_name,
                "run_id": self.run_id,
                "save_metric": f"_{self.save_metric}" if self.save_metric else "",
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
        )
        formatted_path = path.format_map(format_vars)
        # formatted_path = self._sanitize_string()
        return Path(formatted_path)

    def _process_paths(self) -> None:
        """Process and format all paths in the config."""
        path_attrs = {
            "log_path": "Logging",
            "model_output_path": "Model Output",
            "metrics_path": "Metrics",
            "monitor_path": "Monitoring",
            "iid_metrics_path": "IID Metrics",
            "tensorboard_path": "Tensorboard",
        }

        console.rule("[heading]Initialising Logging[/]")

        # Create a table for path information
        table = Table(title="Path Configuration", title_style="bold white", expand=True)
        table.add_column("Path Type", style="cyan")
        table.add_column("Location", style="green")
        table.add_column("Status", style="yellow")

        for attr, desc in path_attrs.items():
            if hasattr(self, attr) and getattr(self, attr):
                try:
                    path = self.format_path(getattr(self, attr))
                    setattr(self, attr, path)
                    table.add_row(desc, str(path), "✓ Formatted")
                    logger.debug(f"Processed {attr}: {path}")
                except Exception as e:
                    error_msg = f"Error processing {attr}: {str(e)}"
                    table.add_row(desc, str(getattr(self, attr)), "✗ Error")
                    logger.error(error_msg)
                    console.print(f"[red]{error_msg}[/]")

        console.print(table)

    def _create_directories(self) -> None:
        """Create directories for all paths if they don't exist."""
        console.print("\n[bold yellow]Creating Directories...[/]")

        for attr, value in vars(self).items():
            if isinstance(value, Path):
                try:
                    # Create parent directory
                    value.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {value.parent}")
                    console.print(f"[bold green]✓[/] Created: {value.parent}")
                except Exception as e:
                    error_msg = f"Failed to create directory {value.parent}: {str(e)}"
                    logger.error(error_msg)
                    console.print(f"[bold red]✗[/] {error_msg}")

    def _log_configuration(self) -> None:
        """Log the complete configuration."""
        # Log to file
        logger.info("Logging Configuration Initialized:")
        for key, value in self.to_dict().items():
            logger.info(f"{key}: {value}")

        # Create a rich table for console display
        table = Table(title="Logging Configuration Summary", title_style="bold white", expand=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        for key, value in self.to_dict().items():
            table.add_row(key, str(value))

        console.print(table)

    def validate_paths(self) -> bool:
        """Validate all paths are properly set up."""
        console.print("\n[bold yellow]Validating Paths...[/]")
        valid = True

        for attr, value in vars(self).items():
            if isinstance(value, Path):
                try:
                    # Check if parent directory exists and is writable
                    if not value.parent.exists():
                        raise ValueError(f"Directory does not exist: {value.parent}")
                    if not os.access(value.parent, os.W_OK):
                        raise ValueError(f"Directory not writable: {value.parent}")

                    console.print(f"[bold green]✓[/] {attr}: Valid")
                    logger.info(f"Validated path for {attr}: {value}")
                except Exception as e:
                    valid = False
                    error_msg = f"Path validation failed for {attr}: {str(e)}"
                    console.print(f"[bold red]✗[/] {error_msg}")
                    logger.error(error_msg)

        return valid

    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.to_dict().items()])
