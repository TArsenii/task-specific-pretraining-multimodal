from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .logging import get_logger
from .printing import get_console

logger = get_logger()
console = get_console()


class CheckpointManager:
    """Manages model checkpointing and loading."""

    def __init__(
        self,
        model_dir: Path,
        save_metric: str = "loss",
        mode: str = "minimize",
        device: str = "cuda",
    ):
        self.model_dir = Path(model_dir)
        self.save_metric = save_metric
        self.mode = mode
        self.device = device
        self.best_metric = float("inf") if mode == "minimize" else float("-inf")
        self.best_epoch = -1

        # Create directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def is_better(self, current: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "minimize":
            return current < self.best_metric
        return current > self.best_metric

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint."""
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.model_dir / f"epoch_{epoch}.pth"
        torch.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint for epoch {epoch}")

        # Save best checkpoint if applicable
        if is_best:
            best_path = self.model_dir / "best.pth"
            torch.save(state, best_path)
            logger.info(f"Saved best checkpoint (epoch {epoch})")
            console.print(f"[green]✓[/] New best model saved (epoch {epoch})")

        # Update best metric if needed
        try:
            metric_value = metrics[self.save_metric]
        except KeyError as ke:
            error_msg = f"Error saving checkpoint: metric '{self.save_metric}' not found in metrics. Available metrics: {metrics.keys()}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            raise ke
        if self.is_better(metric_value):
            self.best_metric = metric_value
            self.best_epoch = epoch

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        try:
            if load_best:
                checkpoint_path = self.model_dir / "best.pth"
                console.print(f" [cyan]Loading best checkpoint (Epoch {self.best_epoch}): ({checkpoint_path}) ...[/]")
            elif epoch is not None:
                checkpoint_path = self.model_dir / f"epoch_{epoch}.pth"
                console.print(f"[cyan]Loading checkpoint from epoch {epoch}...[/]")
            else:
                checkpoint_path = self.model_dir / "last.pth"
                console.print("[cyan]Loading last checkpoint...[/]")

            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

            # Load model state
            model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer state if provided
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load scheduler state if provided
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            console.print("[green]✓[/] Successfully loaded checkpoint")

            return checkpoint

        except Exception as e:
            error_msg = f"Error loading checkpoint: {str(e)}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            raise

    def __str__(self) -> str:
        """Return string representation of the CheckpointManager."""
        return (
            f"CheckpointManager("
            f"model_dir='{self.model_dir}', "
            f"save_metric='{self.save_metric}', "
            f"mode='{self.mode}', "
            f"device='{self.device}', "
            f"best_metric={self.best_metric:.4f}, "
            f"best_epoch={self.best_epoch})"
        )
