import os
import time
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from config import StandardMultimodalConfig
from config.resolvers import resolve_model_name
from experiment_utils import (
    CheckpointManager,
    EmbeddingVisualizationReport,
    EnhancedConsole,
    ExperimentMonitor,
    ExperimentReportGenerator,
    LoggerSingleton,
    LossFunctionGroup,
    MetricRecorder,
    MetricsReport,
    ModelReport,
    TimingReport,
    clean_checkpoints,
    configure_logger,
    get_console,
    get_logger,
)
from modalities import add_modality
from rich import box
from rich.panel import Panel
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "error",
    message="Degrees of freedom <= 0 for slice",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "error",
    message="divide by zero encountered in divide",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "error",
    message="invalid value encountered in multiply",
    category=RuntimeWarning,
)

# Set other warnings to default or ignore behavior as needed
# warnings.filterwarnings("default")  # Treat all other warnings as default

add_modality("video")

console = get_console()


def setup_experiment(
    config_path: str, run_id: int
) -> tuple[StandardMultimodalConfig, EnhancedConsole, LoggerSingleton]:
    """Setup experiment configuration and logging."""
    config = StandardMultimodalConfig.load(config_path, run_id)

    # Configure logging
    configure_logger(log_path=config.logging.log_path, suffix="")

    logger = get_logger()
    # Log initial information
    logger.info(f"Starting experiment with run ID {run_id}")
    logger.info(f"Configuration:\n{config}")
    console.print(f"Starting experiment with run ID {run_id}")

    return config, console, logger


def setup_dataloaders(
    config: StandardMultimodalConfig, console: EnhancedConsole, logger: LoggerSingleton
) -> Dict[str, DataLoader]:
    """Setup data loaders for training and evaluation."""
    logger.debug("Building dataloaders...")

    dataloaders = config.data.build_all_dataloaders()
    console.print(f"Finished building dataloaders. Created: {list(dataloaders.keys())}")

    # Log dataset sizes
    for split, loader in dataloaders.items():
        dataset_size = len(loader.dataset)
        logger.debug(f"{split} dataset size: {dataset_size}")
        if split in ["train", "validation"]:
            total_iterations = config.training.epochs * (dataset_size // loader.batch_size)
            logger.debug(f"Total {split} iterations: {total_iterations}")

    return dataloaders


def setup_model_components(
    config: StandardMultimodalConfig,
    console: EnhancedConsole,
    logger: LoggerSingleton,
    dataloaders: DataLoader | Dict[str, DataLoader] = None,
) -> tuple[Module, Optimizer, LossFunctionGroup, LRScheduler, torch.device, MetricRecorder]:
    """Setup model and training components."""
    logger.debug("Building model...")

    # Initialize model
    model_cls: Module = resolve_model_name(config.model.name)
    model = model_cls(
        **config.model.kwargs,
    )

    if (
        hasattr(model, "post_init_with_dataloaders")
        and callable(getattr(model, "post_init_with_dataloaders"))
        and dataloaders is not None
    ):
        console.print("Initializing model with dataloaders")
        model.post_init_with_dataloaders(dataloaders)

    console.print("[green]✓[/] Model created successfully")

    model_panel = Panel(str(model), box=box.SQUARE, highlight=True, expand=True, title="[heading]Model Architecture[/]")
    console.print(model_panel)

    logger.info(f"Model: {model}")

    # Move model to device
    device = config.experiment.device
    model.to(device)

    # Setup optimizer and criterion
    optimizer = config.get_optimizer(model)
    criterion: LossFunctionGroup = config.get_criterion(
        criterion_info=config.training.criterion,
        criterion_kwargs=config.training.criterion_kwargs,
    )

    console.print("[green]✓[/] Optimizer and criterion created")
    logger.info(f"Optimizer and criterion created\n{optimizer}\n{criterion}")

    # Setup scheduler if specified
    scheduler = None
    if config.training.scheduler is not None:
        scheduler = config.get_scheduler(optimizer=optimizer)
        console.print("[green]✓[/] Scheduler created")
        logger.info(f"Scheduler created\n{scheduler}")
    else:
        console.print("[bold yellow]![/] No scheduler")

    metric_recorder = MetricRecorder(
        config.metrics, tensorboard_path=config.logging.tensorboard_path, tb_record_only=config.logging.tb_record_only
    )

    return model, optimizer, criterion, scheduler, device, metric_recorder


def train_epoch(
    model: Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: LossFunctionGroup,
    device: torch.device,
    console: EnhancedConsole,
    epoch: int,
    metric_recorder: MetricRecorder,
    monitor: ExperimentMonitor = None,
) -> tuple[float, float]:
    """Run one epoch of training."""
    model.train()
    start_time = time.time()

    console.start_task("Training", total=len(train_loader), style="light slate_blue")
    losses = []
    for batch in train_loader:
        train_loss = model.train_step(
            batch, criterion=criterion, optimizer=optimizer, device=device, epoch=epoch, metric_recorder=metric_recorder
        )
        losses.append(train_loss)
        if monitor:
            monitor.step()

        console.update_task("Training", advance=1)

    console.complete_task("Training")

    return np.mean([l["loss"] for l in losses]), (time.time() - start_time) / len(train_loader)


def validate_epoch(
    model: Module,
    val_loader: DataLoader,
    criterion: LossFunctionGroup,
    device: torch.device,
    console: EnhancedConsole,
    metric_recorder: MetricRecorder,
    monitor: ExperimentMonitor = None,
    task_name: str = "Validation",
) -> tuple[float, float]:
    """Run one epoch of validation."""
    model.eval()
    start_time = time.time()

    console.start_task(task_name, total=len(val_loader), style="bright yellow")
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            validation_loss = model.validation_step(
                batch, criterion=criterion, device=device, metric_recorder=metric_recorder
            )
            # epoch_metrics.update_from_dict(validation_results)
            losses.append(validation_loss)
            if monitor:
                monitor.step()
            console.update_task(task_name, advance=1)

    console.complete_task(task_name)
    return np.mean([l["loss"] for l in losses]), (time.time() - start_time) / len(val_loader)


def check_early_stopping(
    val_metrics: Dict[str, Any],
    best_metrics: Dict[str, Any],
    patience: int,
    min_delta: float,
    wait: int = 0,
    mode="minimize",
) -> tuple[bool, int]:
    """Check early stopping conditions."""
    metric_value = val_metrics["loss"] if mode == "minimize" else val_metrics[mode]
    best_value = best_metrics["loss"] if mode == "minimize" else best_metrics[mode]

    if mode == "minimize":
        improved = metric_value < (best_value - min_delta)
    else:
        improved = metric_value > (best_value + min_delta)

    if improved:
        wait = 0
        return True, wait

    wait += 1
    if wait >= patience:
        return False, wait

    return True, wait


def setup_tracking(
    config: StandardMultimodalConfig, output_dir: Path, model: Module
) -> tuple[CheckpointManager, dict[str, Any], ExperimentReportGenerator, ExperimentMonitor]:
    """Setup experiment tracking components."""
    checkpoint_manager = CheckpointManager(
        model_dir=config.logging.model_output_path,
        save_metric=config.logging.save_metric,
        mode="minimize" if config.logging.save_metric == "loss" else "maximize",
        device=config.experiment.device,
    )

    if config.model.pretrained_path is not None:
        checkpoint_manager.model_dir = Path(config.model.pretrained_path).parent
        console.print(f"Using pretrained model from: {config.model.pretrained_path}")

    # Initialize experiment data collector
    experiment_data = {
        "metrics_history": {"train": [], "validation": [], "test": []},
        "timing_history": {"train": [], "validation": []},
        "embeddings": None,  # Will store embeddings by modality
        "model_info": {},
    }

    # Initialize subreports
    subreports = {
        "metrics": MetricsReport(
            output_dir=config.logging.metrics_path,
            metric_keys=list(config.metrics.metrics.keys()),
        ),
        "embeddings": EmbeddingVisualizationReport(
            output_dir=config.logging.metrics_path / "embeddings",
            visualization_fn=model.visualize_embeddings
            if hasattr(model, "visualize_embeddings")
            else lambda x, y: (x, y),
        ),
        "model": ModelReport(output_dir=config.logging.metrics_path),
        "timing": TimingReport(output_dir=config.logging.metrics_path),
    }

    report_generator = ExperimentReportGenerator(output_dir=output_dir, config=config, subreports=subreports)
    console.print(f"Checkpoints Manager: {checkpoint_manager}")
    console.print(f"Report Generator: {report_generator}")
    if config.monitoring.enabled:
        monitor = ExperimentMonitor(config.monitoring, model=model, log_dir=config.logging.monitor_path)
        model.attach_monitor(monitor)
        console.print(f"Monitor: {monitor}")
    else:
        monitor = None

    return checkpoint_manager, experiment_data, report_generator, monitor


def setup_managers():
    pass


def main(
    config: StandardMultimodalConfig, console: EnhancedConsole, logger: LoggerSingleton
) -> tuple[Module, dict[str, Any], Path]:
    """Main training loop with tracking and reporting."""
    # Setup output directory
    output_dir = Path(config.logging.log_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean old checkpoints
    logger.debug("Cleaning up old checkpoints...")
    clean_checkpoints(os.path.join(os.path.dirname(config.logging.model_output_path), str(config.experiment.run_id)))
    dataloaders = setup_dataloaders(config, console, logger)

    model, optimizer, criterion, scheduler, device, metric_recorder = setup_model_components(
        config, console, logger, dataloaders
    )

    # Setup tracking components
    checkpoint_manager, experiment_data, report_generator, monitor = setup_tracking(config, output_dir, model)
    # Initialize early stopping variables
    wait = 0
    early_stopping_triggered = False

    if config.experiment.dry_run:
        console.print("Dry run, exitting")
        exit(0)

    try:
        # Training loop
        if config.experiment.is_train:
            console.start_task("Epoch", total=config.training.epochs)

            for epoch in range(1, config.training.epochs + 1):
                # Reset metrics
                metric_recorder.reset()
                epoch_metrics = metric_recorder

                if monitor:
                    monitor.start_epoch(epoch)

                # Training phase
                train_loss, train_time = train_epoch(
                    model=model,
                    train_loader=dataloaders["train"],
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    console=console,
                    metric_recorder=epoch_metrics,
                    epoch=epoch,
                    monitor=monitor,
                )

                # Record training data
                console.print("Calculating training metrics")

                train_metrics = epoch_metrics.calculate_metrics(metric_group="Train", epoch=epoch, loss=train_loss)
                train_metrics["loss"] = train_loss
                experiment_data["metrics_history"]["train"].append(train_metrics.copy())
                experiment_data["timing_history"]["train"].append(train_time)

                ## save the train_metrics

                if epoch % config.experiment.train_print_interval_epochs == 0:
                    console.display_validation_metrics(train_metrics)

                # Validation phase
                metric_recorder.reset()
                epoch_metrics = metric_recorder

                val_loss, val_time = validate_epoch(
                    model=model,
                    val_loader=dataloaders["validation"],
                    criterion=criterion,
                    device=device,
                    console=console,
                    metric_recorder=epoch_metrics,
                    monitor=monitor,
                )

                if monitor:
                    monitor.end_epoch()

                # Record validation data
                console.print("Calculating validation metrics")
                val_metrics = epoch_metrics.calculate_metrics(metric_group="Validation", epoch=epoch, loss=val_loss)
                val_metrics["loss"] = val_loss
                experiment_data["metrics_history"]["validation"].append(val_metrics.copy())
                experiment_data["timing_history"]["validation"].append(val_time)

                if epoch % config.experiment.validation_print_interval_epochs == 0:
                    console.display_validation_metrics(val_metrics)

                console.print(f"Checking early stopping at epoch {epoch}")
                is_best = checkpoint_manager.is_better(val_metrics[config.logging.save_metric])

                # Save checkpoint and check if best
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=is_best,
                )

                # Reset wait counter if we found a new best model
                if is_best:
                    wait = 0
                    console.print(f"[green]>> New best model saved at epoch {epoch}[/]")
                else:
                    wait += 1

                # Early stopping check
                if config.training.early_stopping and wait >= config.training.early_stopping_patience:
                    console.print(
                        f"[yellow]Early stopping triggered at epoch {epoch}. " f"No improvement for {wait} epochs.[/]"
                    )
                    early_stopping_triggered = True
                    break

                # Update learning rate
                if scheduler is not None:
                    console.print("Updating learning rate")
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_metrics["loss"])
                    else:
                        scheduler.step()
                console.print(f"Epoch {epoch} completed")
                console.update_task("Epoch", advance=1)

            console.complete_task("Epoch")
            # Log early stopping status
            if early_stopping_triggered:
                logger.info(
                    f"Training stopped early at epoch {epoch}. " f"Best epoch was {checkpoint_manager.best_epoch}"
                )
        # Testing phase
        if config.experiment.is_test:
            # Load best model for testing
            checkpoint_manager.load_checkpoint(model=model, load_best=True)

            for _test_dataloader in [d for d in dataloaders if d not in ["train", "validation", "embeddings"]]:
                experiment_data["best_epoch"] = checkpoint_manager.best_epoch
                experiment_data["timing_history"][_test_dataloader] = []

                metric_recorder.reset()
                test_metrics = metric_recorder

                if monitor:
                    monitor.start_epoch(_test_dataloader)

                console.print(f"\n[bold cyan]Starting Testing Phase for {_test_dataloader}[/]")
                with torch.no_grad():
                    test_loss, test_time = validate_epoch(
                        model=model,
                        val_loader=dataloaders[_test_dataloader],
                        criterion=criterion,
                        device=device,
                        console=console,
                        metric_recorder=test_metrics,
                        task_name=f"Testing {_test_dataloader}",
                    )

                final_test_metrics = test_metrics.calculate_metrics(
                    metric_group="Test", epoch=checkpoint_manager.best_epoch
                )
                final_test_metrics["loss"] = test_loss
                experiment_data["metrics_history"][_test_dataloader] = final_test_metrics
                experiment_data["timing_history"][_test_dataloader].append(test_time)

                console.display_validation_metrics(final_test_metrics)

                if monitor:
                    monitor.end_epoch()
    finally:
        if monitor:
            monitor.close()
            model.detach_monitor()
    has_embeddings_dataset = "embeddings" in dataloaders
    if has_embeddings_dataset:
        if hasattr(model, "get_embeddings"):
            test_embeddings = model.get_embeddings(dataloaders["embeddings"], device=device)

            for modality, embeddings in test_embeddings.items():
                if experiment_data["embeddings"] is None:
                    experiment_data["embeddings"] = {}
                embeddings = np.concat(embeddings)
                np.save(
                    f"{os.path.join(config.logging.metrics_path, 'embeddings', str(modality) +'.npy')}",
                    embeddings,
                )
                experiment_data["embeddings"][modality] = embeddings

    # Add model information
    experiment_data["model_info"] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "size": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),  # MB
        "architecture": str(model),
    }

    # Generate final report
    report_path = report_generator.generate_report(experiment_data)
    console.print(f"\n[green]Report generated at: {report_path}[/]")

    return model, experiment_data, output_dir


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a multimodal model and evaluate using missing data imputation.")

    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--run_id", type=int, default=-1, help="The run ID for this experiment.")

    optional_args = parser.add_argument_group("Optional arguments")
    optional_args.add_argument("--dry-run", action="store_true", help="Run a dry run of the experiment.")
    optional_args.add_argument("--skip-train", action="store_true", default=None, help="Skip training phase.")
    optional_args.add_argument("--skip-test", action="store_true", default=None, help="Skip testing phase.")

    optional_args.add_argument(
        "--disable_monitoring", action="store_false", help="Enable monitoring of model weights and gradients."
    )

    args = parser.parse_args()

    # Setup experiment
    config, console, logger = setup_experiment(args.config, args.run_id)

    config.experiment.dry_run = args.dry_run
    config.experiment.is_train = args.skip_train if args.skip_train is not None else config.experiment.is_train
    config.experiment.is_test = not args.skip_test if args.skip_test is not None else config.experiment.is_test

    if not args.disable_monitoring:
        config.monitoring.enabled = False

    # Run experiment
    model, metrics, output_dir = main(config, console, logger)
    clean_checkpoints(os.path.join(os.path.dirname(config.logging.model_output_path), str(config.experiment.run_id)))
    print(os.path.dirname(os.path.dirname(config.logging.metrics_path)))
