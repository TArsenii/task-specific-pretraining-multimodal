import atexit
from collections import defaultdict
import os
import subprocess
import time
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from config.multimodal_training_config import StandardMultimodalConfig
from config.resolvers import resolve_init_fn, resolve_model_name
from experiment_utils.checkpoints import CheckpointManager
from experiment_utils.experiment_report import (
    ExperimentReportGenerator,
    EmbeddingVisualizationReport,
    MetricsReport,
    ModelReport,
    TimingReport,
)
from experiment_utils.logging import LoggerSingleton, configure_logger, get_logger
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.monitoring import ExperimentMonitor
from experiment_utils.printing import EnhancedConsole, get_console
from experiment_utils.utils import PARAMETER_SIZE_BYTES, clean_checkpoints, gpu_memory
from modalities import add_modality
from models.protocols import MultimodalModelProtocol
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


def shutdown_cursor_reset_hook() -> None:
    subprocess.run(["tput", "cnorm"])


atexit.register(shutdown_cursor_reset_hook)

# Add modalities and initialize utilities
add_modality("video")
console: EnhancedConsole = get_console()
logger: Optional[LoggerSingleton] = None


def setup_experiment(config_path: str, run_id: int) -> StandardMultimodalConfig:
    """
    Set up the experiment configuration and logging.

    Args:
        config_path (str): Path to the experiment configuration file.
        run_id (int): Unique identifier for this experiment run.

    Returns:
        StandardMultimodalConfig: Loaded experiment configuration.
    """
    global logger
    config = StandardMultimodalConfig.load(config_path, run_id)

    # Configure logging
    configure_logger(log_path=config.logging.log_path, suffix="")
    logger = get_logger()

    logger.info(f"Starting experiment with run ID {run_id}")
    logger.info(f"Configuration:\n{config}")
    console.rule(f"Starting experiment with run ID {run_id}")

    return config


def setup_dataloaders(config: StandardMultimodalConfig) -> Dict[str, DataLoader]:
    """
    Set up the data loaders for training and evaluation.

    Args:
        config (StandardMultimodalConfig): Experiment configuration.

    Returns:
        Dict[str, DataLoader]: Dictionary of data loaders for different splits.
    """
    logger.debug("Building dataloaders...")

    dataloaders = config.data.build_all_dataloaders(
        is_train=config.experiment.is_train, is_test=config.experiment.is_test
    )
    console.print(f"Finished building dataloaders. Created: {list(dataloaders.keys())}")

    for split, loader in dataloaders.items():
        dataset_size = len(loader.dataset)
        logger.debug(f"{split} dataset size: {dataset_size}")
        if split in ["train", "validation"]:
            total_iterations = config.training.epochs * (dataset_size // loader.batch_size)
            logger.debug(f"Total {split} iterations: {total_iterations}")

    return dataloaders


def setup_model_components(
    config: StandardMultimodalConfig,
    dataloaders: Optional[DataLoader | Dict[str, DataLoader]] = None,
) -> Tuple[MultimodalModelProtocol, Optimizer, LossFunctionGroup, Optional[LRScheduler], torch.device, MetricRecorder]:
    """
    Set up the model and its training components.

    Args:
        config (StandardMultimodalConfig): Experiment configuration.
        dataloaders (Optional[Union[DataLoader, Dict[str, DataLoader]]]): Optional dataloaders for initialization.

    Returns:
        Tuple[Module, Optimizer, LossFunctionGroup, Optional[LRScheduler], torch.device, MetricRecorder]:
            Model, optimizer, criterion, scheduler, device, and metric recorder.
    """
    logger.debug("Building model...")
    model_cls: MultimodalModelProtocol = resolve_model_name(config.model.name)
    model: MultimodalModelProtocol = model_cls(**config.model.kwargs)

    if hasattr(model, "post_init_with_dataloaders") and callable(model.post_init_with_dataloaders) and dataloaders:
        console.print("Initializing model with dataloaders")
        model.post_init_with_dataloaders(dataloaders)

    if config.model.init_fn is not None:
        init_fn = resolve_init_fn(config.model.init_fn)
        init_fn(model)
        console.print(f"[green]✓[/] Initialized model with {config.model.init_fn}")

    console.print("[green]✓[/] Model created successfully")
    console.print(
        Panel(str(model), box=box.SQUARE, highlight=True, expand=True, title="[heading]Model Architecture[/]")
    )
    logger.info(f"Model: {model}")

    device = config.experiment.device
    model.to(device)

    optimizer = config.get_optimizer(model)
    criterion = config.training.loss_functions
    console.print("[green]✓[/] Optimizer and criterion created")
    logger.info(f"Optimizer and criterion created\n{optimizer}\n{criterion}")

    scheduler = None
    if config.training.scheduler:
        scheduler = config.get_scheduler(optimizer=optimizer)
        console.print("[green]✓[/] Scheduler created")
        logger.info(f"Scheduler created\n{scheduler}")
    else:
        console.print("[bold yellow]![/] No scheduler")

    metric_recorder = MetricRecorder(
        config.metrics, tensorboard_path=config.logging.tensorboard_path, tb_record_only=config.logging.tb_record_only
    )

    return model, optimizer, criterion, scheduler, device, metric_recorder


def check_early_stopping(
    val_metrics: Dict[str, Any],
    best_metrics: Optional[Dict[str, Any]],
    patience: int,
    min_delta: float,
    wait: int,
    mode: Literal["minimize", "maximize"] = "minimize",
    target_metric: str = "loss",
) -> Tuple[bool, bool, int]:
    """
    Check early stopping conditions based on validation metrics.

    Args:
        val_metrics (Dict[str, Any]): Current validation metrics.
        best_metrics (Optional[Dict[str, Any]]): Best metrics recorded so far.
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum improvement threshold to reset patience.
        wait (int): Current wait count since the last improvement.
        mode (str): "minimize" to minimize the metric, or "maximize" to maximize.

    Returns:
        Tuple[bool, bool, int]:
            is_best (bool): Whether the current metrics are the best so far.
            should_continue (bool): Whether training should continue.
            wait (int): Updated wait count.
    """
    if best_metrics is None:
        # No best metrics yet; current metrics are the best by default
        return True, True, 0

    metric_value = val_metrics.get(target_metric, None)
    best_value = best_metrics.get(target_metric, None)

    if metric_value is None or best_value is None:
        raise ValueError(f"Metric '{target_metric}' not found in val_metrics or best_metrics.")

    # Check for improvement
    if (mode == "minimize" and metric_value < best_value - min_delta) or (
        mode == "maximize" and metric_value > best_value + min_delta
    ):
        console.print(f"[bold green]>>[/] Improvement detected: {best_value:.4f} -> {metric_value:.4f}")
        return True, True, 0  # Improvement detected, reset wait

    # No improvement
    wait += 1
    should_continue = wait < patience
    return False, should_continue, wait


def setup_tracking(
    config: StandardMultimodalConfig, output_dir: Path, model: Module
) -> Tuple[CheckpointManager, Dict[str, Any], ExperimentReportGenerator, Optional[ExperimentMonitor]]:
    """
    Set up tracking components for the experiment.

    Args:
        config (StandardMultimodalConfig): Experiment configuration.
        output_dir (Path): Directory to store outputs.
        model (Module): Model being trained.

    Returns:
        Tuple[CheckpointManager, Dict[str, Any], ExperimentReportGenerator, Optional[ExperimentMonitor]]:
            Checkpoint manager, experiment data dictionary, report generator, and optional monitor.
    """
    checkpoint_manager = CheckpointManager(
        model_dir=config.logging.model_output_path,
        save_metric=config.logging.save_metric,
        mode="minimize" if config.logging.save_metric == "loss" else "maximize",
        device=config.experiment.device,
    )

    if config.model.pretrained_path:
        checkpoint_manager.model_dir = Path(config.model.pretrained_path).parent
        console.print(f"Using pretrained model from: {config.model.pretrained_path}")

    experiment_data = {
        "metrics_history": {"train": [], "validation": [], "test": []},
        "timing_history": {"train": [], "validation": []},
        "embeddings": None,
        "model_info": {},
    }

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
    monitor = None
    if config.monitoring.enabled:
        monitor = ExperimentMonitor(config.monitoring, model=model, log_dir=config.logging.monitor_path)
        model.attach_monitor(monitor)
        console.print(f"Monitor: {monitor}")

    console.print(f"Checkpoints Manager: {checkpoint_manager}")
    console.print(f"Report Generator: {report_generator}")
    return checkpoint_manager, experiment_data, report_generator, monitor


def train_epoch(
    model: MultimodalModelProtocol,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_functions: LossFunctionGroup,
    device: torch.device,
    epoch: int,
    metric_recorder: MetricRecorder,
    monitor: Optional[ExperimentMonitor] = None,
) -> Tuple[float, float, Dict[str, List[float]]]:
    """
    Run one training epoch.

    Args:
        model (MultimodalModelProtocol): Model to train.
        train_loader (DataLoader): Data loader for training data.
        optimizer (Optimizer): Optimizer for model parameters.
        criterion (LossFunctionGroup): Loss function group.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.
        metric_recorder (MetricRecorder): Recorder for metrics.
        monitor (Optional[ExperimentMonitor]): Optional monitor for experiment progress.

    Returns:
        Tuple[float, float]: Average loss and time per batch for this epoch.
    """
    model.train()
    losses = defaultdict(list)

    console.start_task("Training", total=len(train_loader), style="light slate_blue")
    start_time = time.time()
    for batch in train_loader:
        train_output: Dict[str, Any] = model.train_step(
            batch,
            loss_functions=loss_functions,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            metric_recorder=metric_recorder,
        )

        loss: float = train_output["loss"]
        other_losses = train_output.get("losses", None)

        losses["loss"].append(loss)
        if other_losses:
            for key, value in other_losses.items():
                losses[key].append(value)
        if monitor:
            monitor.step()
        console.update_task("Training", advance=1)
    time_taken = time.time() - start_time

    console.complete_task("Training")

    losses = {key: np.mean(value) for key, value in losses.items()}
    return losses["loss"], time_taken, losses


def validate_epoch(
    model: MultimodalModelProtocol,
    val_loader: DataLoader,
    loss_functions: LossFunctionGroup,
    device: torch.device,
    console: EnhancedConsole,
    metric_recorder: MetricRecorder,
    monitor: Optional[ExperimentMonitor] = None,
    task_name: str = "Validation",
) -> Tuple[float, float, Dict[str, List[float]]]:
    """
    Run one validation epoch.

    Args:
        model (MultimodalModelProtocol): Model to validate.
        val_loader (DataLoader): Data loader for validation data.
        loss_functions (LossFunctionGroup): Loss function group.
        device (torch.device): Device to run validation on.
        console (EnhancedConsole): Console for displaying progress.
        metric_recorder (MetricRecorder): Recorder for metrics.
        monitor (Optional[ExperimentMonitor]): Optional monitor for experiment progress.
        task_name (str): Task name for console display.

    Returns:
        Tuple[float, float]: Average loss and time per batch for this epoch.
    """
    model.eval()
    start_time = time.time()
    losses = defaultdict(list)

    console.start_task(task_name, total=len(val_loader), style="bright yellow")

    with torch.no_grad():
        for batch in val_loader:
            validation_output = model.validation_step(
                batch, loss_functions=loss_functions, device=device, metric_recorder=metric_recorder
            )

            loss = validation_output["loss"]
            other_losses = validation_output.get("losses", None)

            losses["loss"].append(loss)

            if other_losses:
                for key, value in other_losses.items():
                    losses[key].append(value)

            if monitor:
                monitor.step()
            console.update_task(task_name, advance=1)
    console.complete_task(task_name)

    losses = {key: np.mean(value) for key, value in losses.items()}
    time_taken = time.time() - start_time
    return losses["loss"], time_taken, losses


def _train_loop(
    config: StandardMultimodalConfig,
    model: MultimodalModelProtocol,
    dataloaders: Dict[str, DataLoader],
    optimizer: Optimizer,
    loss_functions: LossFunctionGroup,
    device: torch.device,
    metric_recorder: MetricRecorder,
    checkpoint_manager: CheckpointManager,
    scheduler: Optional[LRScheduler] = None,
    experiment_data: Optional[Dict[str, Any]] = None,
    monitor: Optional[ExperimentMonitor] = None,
    checkpoint_mode: Literal["minimize", "maximize"] = "minimize",
) -> Dict[str, Any]:
    """
    Perform the training loop over all epochs.

    Args:
        config (StandardMultimodalConfig): Experiment configuration.
        model (MultimodalModelProtocol): Model to train.
        dataloaders (Dict[str, DataLoader]): Data loaders for train/validation splits.
        optimizer (Optimizer): Optimizer for model parameters.
        loss_functions (LossFunctionGroup): Loss function group.
        device (torch.device): Device to train on.
        metric_recorder (MetricRecorder): Recorder for metrics.
        checkpoint_manager (CheckpointManager): Manager for saving/loading checkpoints.
        scheduler (Optional[LRScheduler]): Learning rate scheduler.
        experiment_data (Optional[Dict[str, Any]]): Dictionary to store experiment data.
        monitor (Optional[ExperimentMonitor]): Optional experiment monitor.

    Returns:
        Dict[str, Any]: Dictionary containing the best metrics achieved during training.
    """
    best_metrics = None
    wait = 0
    console.start_task("Epoch", total=config.training.epochs)

    for epoch in range(1, config.training.epochs + 1):
        if monitor:
            monitor.start_epoch(epoch)

        metric_recorder.reset()
        train_loss, train_time, train_loss_info = train_epoch(
            model=model,
            train_loader=dataloaders["train"],
            optimizer=optimizer,
            loss_functions=loss_functions,
            device=device,
            epoch=epoch,
            metric_recorder=metric_recorder,
            monitor=monitor,
        )
        train_metrics = metric_recorder.calculate_metrics(metric_group="Train", epoch=epoch, loss=train_loss)
        train_metrics["loss"] = train_loss
        experiment_data["metrics_history"]["train"].append(train_metrics.copy())
        experiment_data["timing_history"]["train"].append(train_time)
        console.display_validation_metrics(train_metrics)

        metric_recorder.reset()
        val_loss, val_time, val_loss_info = validate_epoch(
            model=model,
            val_loader=dataloaders["validation"],
            loss_functions=loss_functions,
            device=device,
            console=console,
            metric_recorder=metric_recorder,
            monitor=monitor,
            task_name="Validation",
        )
        val_metrics = metric_recorder.calculate_metrics(metric_group="Validation", epoch=epoch, loss=val_loss)
        val_metrics["loss"] = val_loss
        experiment_data["metrics_history"]["validation"].append(val_metrics.copy())
        experiment_data["timing_history"]["validation"].append(val_time)
        console.display_validation_metrics(val_metrics)

        if metric_recorder.writer is not None:
            for loss_name in train_loss_info:
                logger.debug(f"Logging {loss_name} loss")
                train_value = train_loss_info[loss_name]
                val_value = val_loss_info[loss_name]
                metric_recorder.writer.add_scalars(
                    f"{loss_name} Loss", {"Train": train_value, "Validation": val_value}, epoch
                )

        is_best, should_continue, wait = check_early_stopping(
            val_metrics=val_metrics,
            best_metrics=best_metrics,
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            wait=wait,
            mode=checkpoint_mode,
            target_metric=config.logging.save_metric,
        )

        if is_best:
            best_metrics = val_metrics.copy()
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                is_best=True,
            )
            console.print(f"[green]>> New best model saved at epoch {epoch}[/]")

        if not should_continue:
            console.print("[bold red]Early stopping triggered. Stopping training.[/]")
            break

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()
            console.print(f"[grey] - Learning rate: {optimizer.param_groups[0]['lr']:.2e}][/]")

        console.update_task("Epoch", advance=1)
        if monitor:
            monitor.end_epoch()

    console.complete_task("Epoch")
    return best_metrics


def _test(
    model: MultimodalModelProtocol,
    dataloaders: Dict[str, DataLoader],
    loss_functions: LossFunctionGroup,
    device: torch.device,
    metric_recorder: MetricRecorder,
    checkpoint_manager: CheckpointManager,
    experiment_data: Optional[Dict[str, Any]] = None,
    monitor: Optional[ExperimentMonitor] = None,
) -> Dict[str, Any]:
    """
    Perform testing on the model using the specified data loaders.

    Args:
        config (StandardMultimodalConfig): Experiment configuration.
        model (Module): Model to test.
        dataloaders (Dict[str, DataLoader]): Data loaders for testing splits.
        loss_functions (LossFunctionGroup): Loss function group.
        device (torch.device): Device for computation.
        metric_recorder (MetricRecorder): Recorder for metrics.
        checkpoint_manager (CheckpointManager): Manager for loading the best checkpoint.
        experiment_data (Optional[Dict[str, Any]]): Experiment data storage dictionary.
        monitor (Optional[ExperimentMonitor]): Optional experiment monitor.

    Returns:
        Dict[str, Any]: Metrics recorded during testing.
    """
    checkpoint_manager.load_checkpoint(model=model, load_best=True)

    for split_name, loader in dataloaders.items():
        if split_name in ["train", "validation", "embeddings"]:
            continue

        metric_recorder.reset()
        console.print(f"\n[bold cyan]Testing on {split_name} split[/]")

        with torch.no_grad():
            test_loss, test_time, test_loss_info = validate_epoch(
                model=model,
                val_loader=loader,
                loss_functions=loss_functions,
                device=device,
                console=console,
                metric_recorder=metric_recorder,
                monitor=monitor,
                task_name=f"Testing {split_name}",
            )

        metrics = metric_recorder.calculate_metrics(loss=test_loss, skip_tensorboard=True)
        metrics.update({k: np.mean(v) for k, v in test_loss_info.items()})
        experiment_data["metrics_history"][split_name] = metrics
        experiment_data["timing_history"][split_name] = [test_time]
        console.display_validation_metrics(metrics)

    return experiment_data["metrics_history"]


def main_cross_validation(config: StandardMultimodalConfig) -> Tuple[Module, Dict[str, Any], Path]:
    """
    Perform cross-validation training and evaluation.

    Args:
        config (StandardMultimodalConfig): Experiment configuration.

    Returns:
        Tuple[Module, Dict[str, Any], Path]: Trained model, experiment data, and output directory.
    """
    n_folds = config.experiment.cross_validation
    fold_metrics = {}

    console.start_task("Cross-Validation", total=n_folds)

    for fold_index in range(n_folds):
        console.print(f"\n[bold cyan]Starting Fold {fold_index + 1}/{n_folds}[/]")
        logger.info(f"Starting Fold {fold_index + 1}/{n_folds}")

        # Update paths for the fold
        fold_output_dir = Path(config.logging.model_output_path) / f"fold_{fold_index}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        config.logging.model_output_path = str(fold_output_dir)

        # Prepare fold-specific components
        dataloaders = setup_dataloaders(config)
        model, optimizer, loss_functions, scheduler, device, metric_recorder = setup_model_components(
            config=config, dataloaders=dataloaders
        )
        checkpoint_manager, experiment_data, report_generator, monitor = setup_tracking(
            config=config, output_dir=fold_output_dir, model=model
        )

        try:
            # Train and validate
            _ = _train_loop(
                config=config,
                model=model,
                dataloaders=dataloaders,
                optimizer=optimizer,
                loss_functions=loss_functions,
                device=device,
                metric_recorder=metric_recorder,
                checkpoint_manager=checkpoint_manager,
                scheduler=scheduler,
                experiment_data=experiment_data,
                monitor=monitor,
            )

            # Test
            test_metrics = _test(
                model=model,
                dataloaders=dataloaders,
                loss_functions=loss_functions,
                device=device,
                metric_recorder=metric_recorder,
                checkpoint_manager=checkpoint_manager,
                experiment_data=experiment_data,
                monitor=monitor,
            )
            fold_metrics[fold_index] = test_metrics

        finally:
            if monitor:
                monitor.close()
                model.detach_monitor()

        console.update_task("Cross-Validation", advance=1)

    console.complete_task("Cross-Validation")

    # Aggregate and log cross-validation results
    aggregated_metrics = {
        metric: {
            "mean": np.mean([fold_metrics[fold][metric] for fold in fold_metrics]),
            "std": np.std([fold_metrics[fold][metric] for fold in fold_metrics]),
        }
        for metric in fold_metrics[0]
    }

    console.print("\n[bold green]Cross-validation results[/]")
    for metric, values in aggregated_metrics.items():
        console.print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        logger.info(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")

    return model, experiment_data, fold_output_dir


def main(
    config: StandardMultimodalConfig,
) -> Tuple[Module, Dict[str, Any], Path]:
    """
    Main function to perform training, validation, and testing.

    Args:
        config (StandardMultimodalConfig): Experiment configuration.

    Returns:
        Tuple[Module, Dict[str, Any], Path]: Final trained model, experiment data, and output directory.
    """
    # Setup output directory
    output_dir = Path(config.logging.log_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean old checkpoints
    logger.debug("Cleaning up old checkpoints...")
    clean_checkpoints(os.path.join(os.path.dirname(config.logging.model_output_path), str(config.experiment.run_id)))

    # Setup components
    dataloaders = setup_dataloaders(config)
    model, optimizer, loss_functions, scheduler, device, metric_recorder = setup_model_components(
        config=config, dataloaders=dataloaders
    )
    checkpoint_manager, experiment_data, report_generator, monitor = setup_tracking(
        config=config, output_dir=output_dir, model=model
    )

    experiment_data["model_info"]["architecture"] = str(model)
    experiment_data["model_info"]["parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    experiment_data["model_info"]["size"] = (
        sum(p.numel() for p in model.parameters()) * PARAMETER_SIZE_BYTES / 1024 / 1024
    )  # in MB
    if config.experiment.dry_run:
        console.print("[yellow]Dry run completed. Exiting.[/]")
        return model, experiment_data, output_dir
    logger.info(gpu_memory)
    try:
        # Training
        if config.experiment.is_train:
            _train_loop(
                config=config,
                model=model,
                dataloaders=dataloaders,
                optimizer=optimizer,
                loss_functions=loss_functions,
                device=device,
                metric_recorder=metric_recorder,
                checkpoint_manager=checkpoint_manager,
                scheduler=scheduler,
                experiment_data=experiment_data,
                monitor=monitor,
                checkpoint_mode="minimize" if config.logging.save_metric == "loss" else "maximize",
            )

        # Testing
        if config.experiment.is_test:
            _test(
                model=model,
                dataloaders=dataloaders,
                loss_functions=loss_functions,
                device=device,
                metric_recorder=metric_recorder,
                checkpoint_manager=checkpoint_manager,
                experiment_data=experiment_data,
                monitor=monitor,
            )

            if hasattr(model, "get_embeddings"):
                console.print("[bold cyan]Generating embeddings for visualization...[/]")
                embeddings = model.get_embeddings(dataloaders["embeddings"], device=device)

                if embeddings is not None and isinstance(embeddings, dict):
                    for modality, embds in embeddings.items():
                        save_fp = config.logging.metrics_path / "embeddings" / f"{modality}_embeddings.npy"
                        os.makedirs(save_fp.parent, exist_ok=True)
                        np.save(save_fp, embds)
                        console.print(f"[green]✓[/] Saved {modality} embeddings to: {save_fp}")
                elif embeddings is not None and isinstance(embeddings, tuple):
                    embeddings, reconstructions = embeddings
                    for modality, embeddings in embeddings.items():
                        embeddings_save_fp = config.logging.metrics_path / "embeddings" / f"{modality}_embeddings.npy"
                        os.makedirs(embeddings_save_fp.parent, exist_ok=True)
                        np.save(embeddings_save_fp, embeddings)
                        console.print(f"[green]✓[/] Saved embeddings to: {embeddings_save_fp}")

                    for modality, reconstructions in reconstructions.items():
                        reconstructions_save_fp = (
                            config.logging.metrics_path / "embeddings" / f"{modality}_reconstructions.npy"
                        )
                        os.makedirs(reconstructions_save_fp.parent, exist_ok=True)
                        np.save(reconstructions_save_fp, reconstructions)
                        console.print(f"[green]✓[/] Saved reconstructions to: {reconstructions_save_fp}")
            else:
                console.print("[bold yellow]![/] Model does not support gathering embeddings")

    finally:
        if monitor:
            monitor.close()
            model.detach_monitor()

    # Generate final report
    report_path = report_generator.generate_report(experiment_data)
    console.print(f"[green]Experiment completed. Report saved at: {report_path}[/]")

    return model, experiment_data, output_dir


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a multimodal model and evaluate using missing data imputation.")

    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--run_id", type=int, default=-1, help="The run ID for this experiment.")

    optional_args = parser.add_argument_group("Optional arguments")
    optional_args.add_argument("--dry-run", action="store_true", help="Run a dry run of the experiment.")
    optional_args.add_argument("--skip-train", action="store_true", help="Skip training phase.")
    optional_args.add_argument("--skip-test", action="store_true", help="Skip testing phase.")
    optional_args.add_argument(
        "--disable_monitoring", action="store_true", help="Disable monitoring of model weights and gradients."
    )

    args = parser.parse_args()

    # Setup experiment
    config = setup_experiment(args.config, args.run_id)
    config.experiment.dry_run = args.dry_run
    config.experiment.is_train = not args.skip_train
    config.experiment.is_test = not args.skip_test

    if args.disable_monitoring:
        config.monitoring.enabled = False

    if config.experiment.cross_validation:
        main_cross_validation(config)
    else:
        main(config)

    shutdown_cursor_reset_hook()
