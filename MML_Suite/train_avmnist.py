import atexit
import json
import os
import subprocess
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from config.multimodal_training_config import StandardMultimodalConfig
from config.resolvers import resolve_init_fn, resolve_model_name
from experiment_utils.checkpoints import CheckpointManager
from experiment_utils.experiment_report import (
    EmbeddingVisualizationReport,
    ExperimentReportGenerator,
    MetricsReport,
    ModelReport,
    TimingReport,
)
from experiment_utils.logging import LoggerSingleton, configure_logger, get_logger
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.monitoring import ExperimentMonitor
from experiment_utils.printing import EnhancedConsole, get_console
from experiment_utils.utils import (
    PARAMETER_SIZE_BYTES,
    clean_checkpoints,
    flatten_dict,
    gpu_memory,
    prepare_metrics_for_json,
    format_path_with_env,
)
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

    # Загрузка предобученных весов энкодеров, если они указаны в конфигурации
    if hasattr(config.model, "pretrained_encoders") and config.model.pretrained_encoders:
        console.print("[yellow]Loading pretrained encoders[/]")
        for modality, encoder_path in config.model.pretrained_encoders.items():
            # Обрабатываем переменные окружения в пути
            encoder_path_str = format_path_with_env(encoder_path)
            encoder_path = Path(encoder_path_str)
            if encoder_path.exists():
                try:
                    # Получаем энкодер из модели в зависимости от модальности
                    if modality.lower() == "image":
                        encoder = model.netI if hasattr(model, "netI") else (
                            model.image_model if hasattr(model, "image_model") else model.image_encoder
                        )
                    elif modality.lower() == "text":
                        encoder = model.netT if hasattr(model, "netT") else (
                            model.text_model if hasattr(model, "text_model") else model.text_encoder
                        )
                    elif modality.lower() == "audio":
                        encoder = model.netA if hasattr(model, "netA") else (
                            model.audio_model if hasattr(model, "audio_model") else model.audio_encoder
                        )
                    elif modality.lower() == "video":
                        encoder = model.netV if hasattr(model, "netV") else (
                            model.video_model if hasattr(model, "video_model") else model.video_encoder
                        )
                    else:
                        console.print(f"[red]Unknown modality: {modality}[/]")
                        continue
                    
                    # Загружаем веса
                    state_dict = torch.load(encoder_path, map_location=config.experiment.device)
                    encoder.load_state_dict(state_dict)
                    console.print(f"[green]✓[/] Loaded pretrained {modality} encoder from {encoder_path}")
                except Exception as e:
                    console.print(f"[red]Error loading encoder {modality}: {str(e)}[/]")
            else:
                console.print(f"[red]Encoder path not found: {encoder_path}")
                # Попробуем наиболее распространенный путь
                try:
                    actual_path = Path(f"experiments_output/MMIMDb_{modality.title()}_Encoder_Pretrain/models/1/encoder_{modality.lower()}_best.pth")
                    if actual_path.exists():
                        encoder = getattr(model, f"{modality.lower()}_model")
                        state_dict = torch.load(actual_path, map_location=config.experiment.device)
                        encoder.load_state_dict(state_dict)
                        console.print(f"[green]✓[/] Loaded pretrained {modality} encoder from {actual_path}")
                    else:
                        console.print(f"[red]Alternative encoder path not found: {actual_path}")
                except Exception as e:
                    console.print(f"[red]Error loading alternative encoder path: {str(e)}")

    console.print("[green]✓[/] Model created successfully")
    console.print(
        Panel(str(model), box=box.SQUARE, highlight=True, expand=True, title="[heading]Model Architecture[/]")
    )
    logger.info(f"Model: {model}")

    device = config.experiment.device
    model.to(device)

    # Создаем оптимизатор с различными скоростями обучения для энкодеров и остальных частей модели
    if hasattr(config.training, "encoder_optimizer") and config.model.pretrained_encoders:
        # Параметры энкодеров с меньшей скоростью обучения
        encoder_params = []
        if hasattr(model, "image_model"):
            encoder_params.extend(model.image_model.parameters())
        if hasattr(model, "text_model"):
            encoder_params.extend(model.text_model.parameters())
        if hasattr(model, "audio_model"):
            encoder_params.extend(model.audio_model.parameters())
        if hasattr(model, "video_model"):
            encoder_params.extend(model.video_model.parameters())
        
        # Параметры остальных частей модели
        other_params = [p for p in model.parameters() if not any(p is ep for ep in encoder_params)]
        
        # Create a copy of parameter settings to avoid changing the original configuration
        encoder_kwargs = dict(config.training.encoder_optimizer.default_kwargs)
        base_kwargs = dict(config.training.optimizer.default_kwargs)
        
        # Type checking and conversion if needed
        if 'lr' in encoder_kwargs and not isinstance(encoder_kwargs['lr'], float):
            console.print(f"[yellow]Warning: encoder lr is not float, converting from {type(encoder_kwargs['lr'])} to float[/]")
            encoder_kwargs['lr'] = float(encoder_kwargs['lr'])
        
        if 'lr' in base_kwargs and not isinstance(base_kwargs['lr'], float):
            console.print(f"[yellow]Warning: base lr is not float, converting from {type(base_kwargs['lr'])} to float[/]")
            base_kwargs['lr'] = float(base_kwargs['lr'])
        
        # Prepare specific optimizers for individual encoders
        audio_params = []
        video_params = []
        text_params = []
        
        # Separate parameters for each encoder if they exist in the model
        if hasattr(model, "netA") or hasattr(model, "audio_model"):
            audio_params = list(model.netA.parameters() if hasattr(model, "netA") else model.audio_model.parameters())
            
        if hasattr(model, "netV") or hasattr(model, "video_model"):
            video_params = list(model.netV.parameters() if hasattr(model, "netV") else model.video_model.parameters())
            
        if hasattr(model, "netT") or hasattr(model, "text_model"):
            text_params = list(model.netT.parameters() if hasattr(model, "netT") else model.text_model.parameters())
        
        # Form parameter groups based on the availability of specific optimizers
        param_groups = []
        
        # Specific optimizer for audio encoder
        if hasattr(config.training, "audio_optimizer") and audio_params:
            audio_kwargs = dict(config.training.audio_optimizer.default_kwargs)
            if 'lr' in audio_kwargs and not isinstance(audio_kwargs['lr'], float):
                audio_kwargs['lr'] = float(audio_kwargs['lr'])
                
            console.print(f"[green]Using specific audio optimizer with lr={audio_kwargs['lr']}[/]")
            param_groups.append({"params": audio_params, **audio_kwargs})
            # Remove audio parameters from general encoder parameters
            encoder_params = [p for p in encoder_params if not any(p is ap for ap in audio_params)]
            
        # Specific optimizer for video encoder
        if hasattr(config.training, "video_optimizer") and video_params:
            video_kwargs = dict(config.training.video_optimizer.default_kwargs)
            if 'lr' in video_kwargs and not isinstance(video_kwargs['lr'], float):
                video_kwargs['lr'] = float(video_kwargs['lr'])
                
            console.print(f"[green]Using specific video optimizer with lr={video_kwargs['lr']}[/]")
            param_groups.append({"params": video_params, **video_kwargs})
            # Remove video parameters from general encoder parameters
            encoder_params = [p for p in encoder_params if not any(p is vp for vp in video_params)]
            
        # Specific optimizer for text encoder
        if hasattr(config.training, "text_optimizer") and text_params:
            text_kwargs = dict(config.training.text_optimizer.default_kwargs)
            if 'lr' in text_kwargs and not isinstance(text_kwargs['lr'], float):
                text_kwargs['lr'] = float(text_kwargs['lr'])
                
            console.print(f"[green]Using specific text optimizer with lr={text_kwargs['lr']}[/]")
            param_groups.append({"params": text_params, **text_kwargs})
            # Remove text parameters from general encoder parameters
            encoder_params = [p for p in encoder_params if not any(p is tp for tp in text_params)]
        
        # Add remaining encoder parameters (if any)
        if encoder_params:
            param_groups.append({"params": encoder_params, **encoder_kwargs})
            
        # Add all other model parameters
        param_groups.append({"params": other_params, **base_kwargs})
        
        # Create optimizer with parameter groups
        OptClass = getattr(torch.optim, config.training.optimizer.name)
        optimizer = OptClass(param_groups)
        console.print("[green]✓[/] Created optimizer with different learning rates for components")
    else:
        # Стандартный оптимизатор для всей модели
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
    for i, batch in enumerate(train_loader):
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
    
    # Инициализируем список для хранения метрик по эпохам
    epoch_metrics = []
    metrics_file = Path(config.logging.metrics_path) / "epoch_metrics.json"
    
    def _save_metrics_json():
        """Сохраняет текущие метрики по эпохам в JSON-файл"""
        with open(metrics_file, 'w') as f:
            json.dump(epoch_metrics, f, indent=4)
        console.print(f"[green]✓[/] Сохранены метрики в: {metrics_file}")

    for epoch in range(1, config.training.epochs + 1):
        if monitor:
            monitor.start_epoch(epoch)

        metric_recorder.reset()
        train_loss, train_timing, train_loss_info = train_epoch(
            model=model,
            train_loader=dataloaders["train"],
            optimizer=optimizer,
            loss_functions=loss_functions,
            device=device,
            epoch=epoch,
            metric_recorder=metric_recorder,
            monitor=monitor,
        )
        train_metrics = metric_recorder.calculate_all_groups(epoch=epoch, loss=train_loss)
        train_metrics = flatten_dict(train_metrics)
        train_metrics["loss"] = train_loss
        experiment_data["metrics_history"]["train"].append(train_metrics.copy())
        experiment_data["timing_history"]["train"].append(train_timing)
        console.display_validation_metrics(train_metrics)

        metric_recorder.reset()
        val_loss, val_timing, val_loss_info = validate_epoch(
            model=model,
            val_loader=dataloaders["validation"],
            loss_functions=loss_functions,
            device=device,
            console=console,
            metric_recorder=metric_recorder,
            monitor=monitor,
            task_name="Validation",
        )
        val_metrics = metric_recorder.calculate_all_groups(epoch=epoch, loss=val_loss)
        val_metrics = flatten_dict(val_metrics)
        val_metrics["loss"] = val_loss
        experiment_data["metrics_history"]["validation"].append(val_metrics.copy())
        experiment_data["timing_history"]["validation"].append(val_timing)
        console.display_validation_metrics(val_metrics)

        # Сохраняем метрики в формате epoch_metrics.json
        epoch_data = {
            "epoch": epoch,
            "train": {
                "loss": train_loss,
                "timing": {
                    "total_time": train_timing,
                    "avg_batch_time": train_timing / len(dataloaders["train"])
                }
            },
            "validation": {
                "loss": val_loss,
                "timing": {
                    "total_time": val_timing,
                    "avg_batch_time": val_timing / len(dataloaders["validation"])
                }
            }
        }
        
        # Метрики для AVMNIST (A - аудио, I - изображение, AI - обе модальности)
        modalities = ["AI", "A", "I"]
        metric_prefixes = [
            "accuracy", "balanced_accuracy", 
            "f1_macro", "f1_micro", "f1_weighted", 
            "precision_macro", "precision_micro", "precision_weighted",
            "recall_macro", "recall_micro", "recall_weighted"
        ]
        
        # Добавляем все метрики для train
        for key, value in train_metrics.items():
            # Пропускаем специальные ключи, которые уже добавлены
            if key == "loss" or not isinstance(value, (int, float)):
                continue
                
            # Проверяем метрики AVMNIST
            for modality in modalities:
                if key.endswith(f"_{modality}"):
                    # Получаем базовое имя метрики без модальности
                    base_metric = key.replace(f"_{modality}", "")
                    
                    # Создаем секцию для модальности, если её ещё нет
                    if modality not in epoch_data["train"]:
                        epoch_data["train"][modality] = {}
                        
                    # Добавляем метрику
                    epoch_data["train"][modality][base_metric] = value
                    break
            else:
                # Для всех остальных метрик
                if "metrics" not in epoch_data["train"]:
                    epoch_data["train"]["metrics"] = {}
                epoch_data["train"]["metrics"][key] = value
                    
        # Добавляем все метрики для validation
        for key, value in val_metrics.items():
            # Пропускаем специальные ключи, которые уже добавлены
            if key == "loss" or not isinstance(value, (int, float)):
                continue
                
            # Проверяем метрики AVMNIST
            for modality in modalities:
                if key.endswith(f"_{modality}"):
                    # Получаем базовое имя метрики без модальности
                    base_metric = key.replace(f"_{modality}", "")
                    
                    # Создаем секцию для модальности, если её ещё нет
                    if modality not in epoch_data["validation"]:
                        epoch_data["validation"][modality] = {}
                        
                    # Добавляем метрику
                    epoch_data["validation"][modality][base_metric] = value
                    break
            else:
                # Для всех остальных метрик
                if "metrics" not in epoch_data["validation"]:
                    epoch_data["validation"]["metrics"] = {}
                epoch_data["validation"]["metrics"][key] = value
        
        epoch_metrics.append(epoch_data)
        _save_metrics_json()
        
        # Сохраняем метрики в файл total.txt (оставляем существующую функцию)
        # save_epoch_metrics(
        #     epoch=epoch,
        #     train_metrics=train_metrics,
        #     val_metrics=val_metrics,
        #     train_timing=train_timing,
        #     val_timing=val_timing,
        #     output_dir=Path(config.logging.log_path),
        # )

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

        config_do_early_stopping: bool = config.training.early_stopping

        ## Only stop early if the config says to do so AND the check_early_stopping function says to do so.
        if config_do_early_stopping and not should_continue:
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

    # Добавляем тестовые метрики, если есть
    if "test" in dataloaders:
        metric_recorder.reset()
        console.print(f"\n[bold cyan]Testing on test split[/]")
        
        with torch.no_grad():
            test_loss, test_timing, test_loss_info = validate_epoch(
                model=model,
                val_loader=dataloaders["test"],
                loss_functions=loss_functions,
                device=device,
                console=console,
                metric_recorder=metric_recorder,
                monitor=monitor,
                task_name="Testing test",
            )
            
        test_metrics = metric_recorder.calculate_all_groups(loss=test_loss, skip_tensorboard=True)
        test_metrics = flatten_dict(test_metrics)
        test_metrics.update({k: np.mean(v) for k, v in test_loss_info.items()})
        experiment_data["metrics_history"]["test"] = test_metrics
        experiment_data["timing_history"]["test"] = [test_timing]
        console.display_validation_metrics(test_metrics)
        
        # Сохраняем тестовые метрики в формате epoch_metrics.json
        test_epoch_data = {
            "test": {
                "loss": test_loss,
                "timing": {
                    "total_time": test_timing,
                    "avg_batch_time": test_timing / len(dataloaders["test"])
                }
            }
        }
        
        # Метрики для AVMNIST (A - аудио, I - изображение, AI - обе модальности)
        modalities = ["AI", "A", "I"]
        
        # Добавляем метрики тестирования
        for key, value in test_metrics.items():
            # Пропускаем специальные ключи, которые уже добавлены
            if key == "loss" or not isinstance(value, (int, float)):
                continue
                
            # Проверяем метрики AVMNIST
            for modality in modalities:
                if key.endswith(f"_{modality}"):
                    # Получаем базовое имя метрики без модальности
                    base_metric = key.replace(f"_{modality}", "")
                    
                    # Создаем секцию для модальности, если её ещё нет
                    if modality not in test_epoch_data["test"]:
                        test_epoch_data["test"][modality] = {}
                        
                    # Добавляем метрику
                    test_epoch_data["test"][modality][base_metric] = value
                    break
            else:
                # Для всех остальных метрик
                if "metrics" not in test_epoch_data["test"]:
                    test_epoch_data["test"]["metrics"] = {}
                test_epoch_data["test"]["metrics"][key] = value
        
        # Сохраняем тестовые метрики в файл epoch_metrics.json
        metrics_path = Path(config.logging.metrics_path) / str(config.experiment.run_id) / "epoch_metrics.json"
        
        # Если файл существует, загружаем его и добавляем тестовые метрики
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                epoch_metrics_data = json.load(f)
                epoch_metrics_data.append(test_epoch_data)
        else:
            epoch_metrics_data = [test_epoch_data]
            
        # Сохраняем обновленные метрики
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(epoch_metrics_data, f, indent=4)
        
    return best_metrics


def test(
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

        metrics = metric_recorder.calculate_all_groups(loss=test_loss, skip_tensorboard=True)
        metrics = flatten_dict(metrics)
        metrics.update({k: np.mean(v) for k, v in test_loss_info.items()})
        experiment_data["metrics_history"][split_name] = metrics
        experiment_data["timing_history"][split_name] = [test_time]
        console.display_validation_metrics(metrics)
        
        # Сохраняем тестовые метрики в формате epoch_metrics.json
        test_epoch_data = {
            "test": {
                "loss": test_loss,
                "timing": {
                    "total_time": test_time,
                    "avg_batch_time": test_time / len(loader)
                }
            }
        }
        
        # Метрики для AVMNIST (A - аудио, I - изображение, AI - обе модальности)
        modalities = ["AI", "A", "I"]
        
        # Добавляем метрики тестирования
        for key, value in metrics.items():
            # Пропускаем специальные ключи, которые уже добавлены
            if key == "loss" or not isinstance(value, (int, float)):
                continue
                
            # Проверяем метрики AVMNIST
            for modality in modalities:
                if key.endswith(f"_{modality}"):
                    # Получаем базовое имя метрики без модальности
                    base_metric = key.replace(f"_{modality}", "")
                    
                    # Создаем секцию для модальности, если её ещё нет
                    if modality not in test_epoch_data["test"]:
                        test_epoch_data["test"][modality] = {}
                        
                    # Добавляем метрику
                    test_epoch_data["test"][modality][base_metric] = value
                    break
            else:
                # Для всех остальных метрик
                if "metrics" not in test_epoch_data["test"]:
                    test_epoch_data["test"]["metrics"] = {}
                test_epoch_data["test"]["metrics"][key] = value
        
        # Сохраняем тестовые метрики в файл epoch_metrics.json
        metrics_path = Path(config.logging.metrics_path) / str(config.experiment.run_id) / "epoch_metrics.json"
        
        # Если файл существует, загружаем его и добавляем тестовые метрики
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                epoch_metrics_data = json.load(f)
                epoch_metrics_data.append(test_epoch_data)
        else:
            epoch_metrics_data = [test_epoch_data]
            
        # Сохраняем обновленные метрики
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(epoch_metrics_data, f, indent=4)

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

    models_output_path = Path(config.logging.model_output_path)

    for fold_index in range(1, n_folds + 1):
        console.print(f"\n[bold cyan]Starting Fold {fold_index}/{n_folds}[/]")
        logger.info(f"Starting Fold {fold_index}/{n_folds}")

        # Update paths for the fold
        fold_output_dir = models_output_path / f"fold_{fold_index}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        config.logging.model_output_path = str(fold_output_dir)

        for dataset_config in config.data.datasets.values():
            dataset_config.kwargs["cv_no"] = fold_index

        # Prepare fold-specific components
        dataloaders = setup_dataloaders(config)
        console.print(f"Finished building dataloaders for split {fold_index}.")
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
            test_metrics = test(
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

    _train_metrics = []
    _val_metrics = []
    _test_metrics = []

    for fold in fold_metrics:
        ## Basically, the train and validation is just a list of metric dicts for each epoch of training.
        train_metrics: List[Dict[str, Any]] = fold_metrics[fold]["train"]
        val_metrics: List[Dict[str, Any]] = fold_metrics[fold]["validation"]

        ## The test metrics is a single dict with the metrics for the test split.
        test_metrics: Dict[str, Any] = fold_metrics[fold]["test"]

        ## Need to both save the metrics per fold and also aggregate them for the final report.

        train_output_path = config.logging.metrics_path / f"fold_{fold}" / "train_metrics.json"
        val_output_path = config.logging.metrics_path / f"fold_{fold}" / "validation_metrics.json"
        test_output_path = config.logging.metrics_path / f"fold_{fold}" / "test_metrics.json"

        ## TODO Move this functionality elsewhere. I don't think it belongs in the main code.

        os.makedirs(train_output_path.parent, exist_ok=True)
        os.makedirs(val_output_path.parent, exist_ok=True)
        os.makedirs(test_output_path.parent, exist_ok=True)

        with open(train_output_path, "w") as f:
            json.dump(prepare_metrics_for_json(train_metrics), f, indent=4)

        with open(val_output_path, "w") as f:
            json.dump(prepare_metrics_for_json(val_metrics), f, indent=4)

        with open(test_output_path, "w") as f:
            json.dump(prepare_metrics_for_json([test_metrics]), f, indent=4)

        _train_metrics.append(train_metrics)
        _val_metrics.append(val_metrics)
        _test_metrics.append(test_metrics)

    ## Aggregate metrics for final report
    ## TODO Move this functionality elsewhere. I don't think it belongs in the main code.

    def aggregate_cv_metrics(fold_metrics: List[List[Dict[str, Any]]] | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate metrics across CV folds by averaging per epoch.

        Args:
            fold_metrics: List of metrics per fold, where each fold contains a list of metric dicts per epoch

        Returns:
            List of dicts containing averaged metrics per epoch
        """
        # Handle single-epoch test metrics
        if isinstance(fold_metrics[0], dict):
            fold_metrics = [[m] for m in fold_metrics]

        # Validate all folds have same number of epochs
        n_epochs = len(fold_metrics[0])
        if not all(len(fold) == n_epochs for fold in fold_metrics):
            raise ValueError("All folds must have the same number of epochs")

        # Validate all epochs have the same metric keys across folds
        metric_keys = set(fold_metrics[0][0].keys())
        for fold in fold_metrics:
            for epoch_metrics in fold:
                if set(epoch_metrics.keys()) != metric_keys:
                    raise ValueError("All epochs must have the same metric keys")

        # Initialize storage for aggregated metrics
        aggregated_metrics = []

        # For each epoch
        for epoch in range(n_epochs):
            epoch_metrics = defaultdict(list)

            # Collect metrics from each fold for this epoch
            for fold in fold_metrics:
                for metric_name, value in fold[epoch].items():
                    # Skip non-numeric values
                    if isinstance(value, (int, float)):
                        epoch_metrics[metric_name].append(value)

            # Average metrics across folds
            averaged_metrics = {metric_name: float(np.mean(values)) for metric_name, values in epoch_metrics.items()}

            aggregated_metrics.append(averaged_metrics)

        return aggregated_metrics

    train_metrics = aggregate_cv_metrics(_train_metrics)
    val_metrics = aggregate_cv_metrics(_val_metrics)
    test_metrics = aggregate_cv_metrics(_test_metrics)

    train_output_path = config.logging.metrics_path / "train_metrics_agg.json"
    val_output_path = config.logging.metrics_path / "validation_metrics_agg.json"
    test_output_path = config.logging.metrics_path / "test_metrics_agg.json"

    with open(train_output_path, "w") as f:
        json.dump(prepare_metrics_for_json(train_metrics), f, indent=4)
    console.print(f"[green]✓[/] Saved aggregated train metrics to: {train_output_path}")

    with open(val_output_path, "w") as f:
        json.dump(prepare_metrics_for_json(val_metrics), f, indent=4)
    console.print(f"[green]✓[/] Saved aggregated validation metrics to: {val_output_path}")

    with open(test_output_path, "w") as f:
        json.dump(prepare_metrics_for_json(test_metrics), f, indent=4)
    console.print(f"[green]✓[/] Saved aggregated test metrics to: {test_output_path}")

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
            test(
                model=model,
                dataloaders=dataloaders,
                loss_functions=loss_functions,
                device=device,
                metric_recorder=metric_recorder,
                checkpoint_manager=checkpoint_manager,
                experiment_data=experiment_data,
                monitor=monitor,
            )

            if hasattr(model, "get_embeddings") and "embeddings" in dataloaders:
                console.print("[bold cyan]Generating embeddings for visualization...[/]")
                embeddings = model.get_embeddings(dataloaders["embeddings"], device=device)

                if embeddings is not None and isinstance(embeddings, dict):
                    for modality, embds in embeddings.items():
                        if not isinstance(modality, str) and isinstance(embds, list):
                            embds = np.concatenate(embds, axis=0)
                            console.print(f"Embeddings shape: {embds.shape}")

                        if isinstance(modality, str):
                            ## We're dealing with labels
                            save_fp = config.logging.metrics_path / "embeddings" / "labels.npy"
                        else:
                            save_fp = config.logging.metrics_path / "embeddings" / f"{modality}_embeddings.npy"
                        os.makedirs(save_fp.parent, exist_ok=True)
                        try:
                            np.save(save_fp, embds)
                        except Exception as e:
                            console.print(f"[bold red]Error saving embeddings: {e}[/]")
                            console.print(f"Embedings shape: {len(embds)}")
                            raise e
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
                console.print("[bold yellow]![/] Model / Data configuration does not support gathering embeddings")

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
