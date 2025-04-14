import atexit
import json
import os
import subprocess
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import cm

from config.multimodal_training_config import StandardMultimodalConfig
from experiment_utils.checkpoints import CheckpointManager
from experiment_utils.experiment_report import ExperimentReportGenerator, MetricsReport, ModelReport, TimingReport
from experiment_utils.logging import LoggerSingleton, configure_logger, get_logger
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.printing import EnhancedConsole, get_console
from experiment_utils.utils import clean_checkpoints, gpu_memory, prepare_metrics_for_json, flatten_dict

# Set up logger and console
logger = get_logger()
console = get_console()

warnings.filterwarnings(
    "error",
    message="Degrees of freedom <= 0 for slice",
    category=RuntimeWarning,
)

# Register cursor reset hook to fix terminal
def shutdown_cursor_reset_hook():
    try:
        if os.name == "posix":
            subprocess.run(["tput", "cnorm"])
    except Exception as e:
        logger.warning(f"Failed to reset cursor: {e}")

atexit.register(shutdown_cursor_reset_hook)


# Добавляем функции для загрузки аудио и изображений из AVMNIST
def load_avmnist_audio(path: str) -> torch.Tensor:
    """Загружает аудио-данные из файла"""
    return torch.load(path, weights_only=True)

def load_avmnist_image(path: str) -> torch.Tensor:
    """Загружает и обрабатывает изображение из файла"""
    img_data = np.array(torch.load(path, weights_only=False))
    img = Image.fromarray(np.uint8(cm.gist_earth(img_data) * 255)).convert("L")
    pil_to_tensor = getattr(torch, 'from_numpy', lambda x: torch.tensor(x))
    scale = lambda x: x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
    return scale(pil_to_tensor(np.array(img)))


class MonomodalEncoder(Module):
    """
    Wrapper for single-modality encoders to be trained independently.
    """
    def __init__(self, encoder: Module, output_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(output_dim, num_classes)
        
    def forward(self, x):
        """Forward pass through the encoder and classifier."""
        # Обрабатываем возможные различные форматы входных данных
        if isinstance(x, list):
            # Если входные данные представляют собой список тензоров
            encoded = self.encoder(torch.stack(x))
        else:
            encoded = self.encoder(x)
            
        # Исправляем размерность для классификатора
        if len(encoded.shape) > 2:
            # Reshape tensor to (batch_size, -1)
            batch_size = encoded.shape[0]
            encoded = encoded.reshape(batch_size, -1)
        
        # Print shape for debugging
        # print(f"[Debug] Encoder output shape: {encoded.shape}, classifier expects input: {self.classifier.in_features}")
            
        return self.classifier(encoded)
    
    def get_encoder(self):
        """Return the encoder module for saving."""
        return self.encoder
    
    def train_step(self, batch, optimizer, loss_functions, device, metric_recorder, config, **kwargs):
        """Training step for a single modality."""
        optimizer.zero_grad()
        
        # Получаем данные модальности и метки
        # Проверяем ключи как строки и как объекты класса Modality
        modality_key = None
        
        # Ищем подходящий ключ модальности
        for key in batch.keys():
            key_str = str(key)
            
            # Пропускаем неподходящие ключи
            if key_str in ["labels", "label", "genres", "imdb_ids", "pattern_name", "missing_masks", "sample_idx"] or \
               key_str.endswith("_missing_index") or key_str.endswith("_reverse"):
                continue
            
            # Проверяем, соответствует ли ключ нужной модальности
            # Для обучения IMAGE энкодера выбираем IMAGE, для AUDIO энкодера - AUDIO
            if "AVMNIST_Image_Encoder" in config.experiment.name and "IMAGE" in key_str:
                modality_key = key
                break
            elif "AVMNIST_Audio_Encoder" in config.experiment.name and "AUDIO" in key_str:
                modality_key = key
                break
            # Если не нашли конкретную модальность, берем первую доступную
            else:
                modality_key = key
                # Не делаем break здесь, чтобы продолжить поиск нужной модальности
        
        if not modality_key:
            raise ValueError(f"No modality data found in batch. Available keys: {list(batch.keys())}")
        
        # Получаем данные и метки
        if f"{modality_key}_original" in batch:
            modal_data_raw = batch[f"{modality_key}_original"]
        else:
            modal_data_raw = batch[modality_key]
        
        # Проверяем содержимое данных и обрабатываем соответственно
        if isinstance(modal_data_raw, list):
            # Если это список строк или путей к файлам, нужно загрузить данные
            if len(modal_data_raw) > 0 and isinstance(modal_data_raw[0], str):
                # Загружаем данные из путей для AVMNIST
                try:
                    console.print(f"[yellow]Info:[/] Loading data from paths for modality {modality_key}")
                    
                    # Определяем функцию загрузки на основе типа модальности
                    mod_str = str(modality_key).lower()
                    if "audio" in mod_str:
                        loader_fn = load_avmnist_audio
                    elif "image" in mod_str:
                        loader_fn = load_avmnist_image
                    else:
                        raise ValueError(f"Unknown modality type: {modality_key}")
                        
                    # Загружаем данные из файлов
                    converted_items = []
                    for path in modal_data_raw:
                        tensor = loader_fn(path)
                        converted_items.append(tensor)
                    
                    # Объединяем в пакет
                    modal_data = torch.stack(converted_items)
                    
                except Exception as e:
                    # В случае ошибки выводим подробности для отладки
                    sample_paths = modal_data_raw[:5] if len(modal_data_raw) > 5 else modal_data_raw
                    raise RuntimeError(
                        f"Failed to load data from paths for modality {modality_key}: {e}. "
                        f"Sample paths: {sample_paths}"
                    )
            else:
                # Стандартная обработка для числовых данных в списке
                try:
                    # Преобразуем в тензор, обрабатывая разные типы данных
                    converted_items = []
                    for item in modal_data_raw:
                        if torch.is_tensor(item):
                            converted_items.append(item)
                        elif isinstance(item, (int, float)):
                            converted_items.append(torch.tensor(item))
                        elif isinstance(item, np.ndarray):
                            converted_items.append(torch.from_numpy(item))
                        else:
                            # Пропускаем неподдерживаемые типы
                            raise TypeError(f"Unsupported data type: {type(item)}")
                    
                    modal_data = torch.stack(converted_items)
                except (TypeError, ValueError) as e:
                    # Если преобразование не удалось, выводим информацию и поднимаем ошибку
                    raise TypeError(f"Failed to convert data to tensor: {e}. Data type: {type(modal_data_raw)}, First element type: {type(modal_data_raw[0]) if len(modal_data_raw) > 0 else 'empty list'}")
        else:
            # Если уже тензор, просто используем его
            modal_data = modal_data_raw
            
        # Переводим данные на устройство
        modal_data = modal_data.to(device)
        
        # Обрабатываем метки - они могут быть под ключом "label", "labels" или "genres"
        if "label" in batch:
            labels_raw = batch["label"]
        elif "labels" in batch:
            labels_raw = batch["labels"]
        elif "genres" in batch:
            labels_raw = batch["genres"]
        else:
            raise ValueError(f"No labels found in batch. Available keys: {list(batch.keys())}")
        
        # Преобразуем метки, если нужно
        if isinstance(labels_raw, list):
            try:
                # Проверяем тип данных в метках
                if len(labels_raw) > 0 and isinstance(labels_raw[0], str):
                    # Для категориальных меток - строк, мы должны конвертировать их в числа
                    # Но для этого нужен словарь меток, которого у нас нет
                    raise TypeError("Cannot convert string labels to tensor without label mapping")
                
                labels = torch.tensor(labels_raw).to(device)
            except (TypeError, ValueError) as e:
                raise TypeError(f"Failed to convert labels to tensor: {e}. Labels type: {type(labels_raw)}, First label type: {type(labels_raw[0]) if len(labels_raw) > 0 else 'empty list'}")
        else:
            labels = labels_raw.to(device)
        
        # Forward pass
        logits = self.forward(modal_data)
        
        # Calculate loss
        loss = loss_functions(logits, labels)["total_loss"]
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            # Для правильной работы метрик, преобразуем предсказания в вероятности или метки
            # В случае многоклассовой классификации с одной меткой
            if labels.dim() == 1:
                predictions = torch.argmax(logits, dim=1)
            # В случае многометочной классификации
            else:
                predictions = torch.sigmoid(logits) > 0.5
            
            # Используем update_group для каждой группы метрик
            for group_name in metric_recorder.config.groups:
                metric_recorder.update_group(
                    group_name=group_name,
                    predictions=predictions,
                    targets=labels,
                    modality=str(modality_key)
                )
            
            metrics = {
                "loss": loss.item()
            }
            if labels.dim() == 1:
                acc = (predictions == labels).float().mean().item()
                metrics["accuracy"] = acc
        
        return {
            "loss": loss.item(),
            "metrics": metrics
        }
    
    def validation_step(self, batch, loss_functions, device, metric_recorder, config, **kwargs):
        """Validation step for a single modality."""
        with torch.no_grad():
            # Получаем данные модальности и метки
            # Проверяем ключи как строки и как объекты класса Modality
            modality_key = None
            
            # Ищем подходящий ключ модальности
            for key in batch.keys():
                key_str = str(key)
                
                # Пропускаем неподходящие ключи
                if key_str in ["labels", "label", "genres", "imdb_ids", "pattern_name", "missing_masks", "sample_idx"] or \
                   key_str.endswith("_missing_index") or key_str.endswith("_reverse"):
                    continue
                
                # Проверяем, соответствует ли ключ нужной модальности
                # Для обучения IMAGE энкодера выбираем IMAGE, для AUDIO энкодера - AUDIO
                if "AVMNIST_Image_Encoder" in config.experiment.name and "IMAGE" in key_str:
                    modality_key = key
                    break
                elif "AVMNIST_Audio_Encoder" in config.experiment.name and "AUDIO" in key_str:
                    modality_key = key
                    break
                # Если не нашли конкретную модальность, берем первую доступную
                else:
                    modality_key = key
                    # Не делаем break здесь, чтобы продолжить поиск нужной модальности
            
            if not modality_key:
                raise ValueError(f"No modality data found in batch. Available keys: {list(batch.keys())}")
            
            # Получаем данные и метки
            if f"{modality_key}_original" in batch:
                modal_data_raw = batch[f"{modality_key}_original"]
            else:
                modal_data_raw = batch[modality_key]
            
            # Проверяем содержимое данных и обрабатываем соответственно
            if isinstance(modal_data_raw, list):
                # Если это список строк или путей к файлам, нужно загрузить данные
                if len(modal_data_raw) > 0 and isinstance(modal_data_raw[0], str):
                    # Загружаем данные из путей для AVMNIST
                    try:
                        console.print(f"[yellow]Info:[/] Loading data from paths for modality {modality_key}")
                        
                        # Определяем функцию загрузки на основе типа модальности
                        mod_str = str(modality_key).lower()
                        if "audio" in mod_str:
                            loader_fn = load_avmnist_audio
                        elif "image" in mod_str:
                            loader_fn = load_avmnist_image
                        else:
                            raise ValueError(f"Unknown modality type: {modality_key}")
                            
                        # Загружаем данные из файлов
                        converted_items = []
                        for path in modal_data_raw:
                            tensor = loader_fn(path)
                            converted_items.append(tensor)
                        
                        # Объединяем в пакет
                        modal_data = torch.stack(converted_items)
                        
                    except Exception as e:
                        # В случае ошибки выводим подробности для отладки
                        sample_paths = modal_data_raw[:5] if len(modal_data_raw) > 5 else modal_data_raw
                        raise RuntimeError(
                            f"Failed to load data from paths for modality {modality_key}: {e}. "
                            f"Sample paths: {sample_paths}"
                        )
                else:
                    # Стандартная обработка для числовых данных в списке
                    try:
                        # Преобразуем в тензор, обрабатывая разные типы данных
                        converted_items = []
                        for item in modal_data_raw:
                            if torch.is_tensor(item):
                                converted_items.append(item)
                            elif isinstance(item, (int, float)):
                                converted_items.append(torch.tensor(item))
                            elif isinstance(item, np.ndarray):
                                converted_items.append(torch.from_numpy(item))
                            else:
                                # Пропускаем неподдерживаемые типы
                                raise TypeError(f"Unsupported data type: {type(item)}")
                        
                        modal_data = torch.stack(converted_items)
                    except (TypeError, ValueError) as e:
                        # Если преобразование не удалось, выводим информацию и поднимаем ошибку
                        raise TypeError(f"Failed to convert data to tensor: {e}. Data type: {type(modal_data_raw)}, First element type: {type(modal_data_raw[0]) if len(modal_data_raw) > 0 else 'empty list'}")
            else:
                # Если уже тензор, просто используем его
                modal_data = modal_data_raw
                
            # Переводим данные на устройство
            modal_data = modal_data.to(device)
            
            # Обрабатываем метки - они могут быть под ключом "label", "labels" или "genres"
            if "label" in batch:
                labels_raw = batch["label"]
            elif "labels" in batch:
                labels_raw = batch["labels"]
            elif "genres" in batch:
                labels_raw = batch["genres"]
            else:
                raise ValueError(f"No labels found in batch. Available keys: {list(batch.keys())}")
            
            # Преобразуем метки, если нужно
            if isinstance(labels_raw, list):
                try:
                    # Проверяем тип данных в метках
                    if len(labels_raw) > 0 and isinstance(labels_raw[0], str):
                        # Для категориальных меток - строк, мы должны конвертировать их в числа
                        # Но для этого нужен словарь меток, которого у нас нет
                        raise TypeError("Cannot convert string labels to tensor without label mapping")
                    
                    labels = torch.tensor(labels_raw).to(device)
                except (TypeError, ValueError) as e:
                    raise TypeError(f"Failed to convert labels to tensor: {e}. Labels type: {type(labels_raw)}, First label type: {type(labels_raw[0]) if len(labels_raw) > 0 else 'empty list'}")
            else:
                labels = labels_raw.to(device)
            
            # Forward pass
            logits = self.forward(modal_data)
            
            # Calculate loss
            loss = loss_functions(logits, labels)["total_loss"]
            
            # Для правильной работы метрик, преобразуем предсказания в вероятности или метки
            # В случае многоклассовой классификации с одной меткой
            if labels.dim() == 1:
                predictions = torch.argmax(logits, dim=1)
            # В случае многометочной классификации
            else:
                predictions = torch.sigmoid(logits) > 0.5
            
            # Используем update_group для каждой группы метрик
            for group_name in metric_recorder.config.groups:
                metric_recorder.update_group(
                    group_name=group_name,
                    predictions=predictions,
                    targets=labels,
                    modality=str(modality_key)
                )
            
            metrics = {
                "loss": loss.item()
            }
            if labels.dim() == 1:
                acc = (predictions == labels).float().mean().item()
                metrics["accuracy"] = acc
            
            return {
                "loss": loss.item(),
                "metrics": metrics
            }


def setup_experiment(config_path: str, run_id: int) -> Tuple[MonomodalEncoder, StandardMultimodalConfig]:
    """
    Set up the monomodal training experiment.
    
    Args:
        config_path: Path to the monomodal encoder configuration file
        run_id: Run identifier
    
    Returns:
        Tuple containing the monomodal model and configuration
    """
    # Load the configuration
    config = StandardMultimodalConfig.load(config_path, run_id)
    
    # Identify the encoder from the configuration
    encoder = None
    encoder_output_dim = None
    num_classes = None
    
    # Detect the encoder type
    if hasattr(config.model, 'image_encoder'):
        console.print("Found image encoder in configuration")
        encoder = config.model.image_encoder
        if hasattr(encoder, 'output_dim'):
            encoder_output_dim = encoder.output_dim
        
    elif hasattr(config.model, 'text_encoder'):
        console.print("Found text encoder in configuration")
        encoder = config.model.text_encoder
        if hasattr(encoder, 'output_dim'):
            encoder_output_dim = encoder.output_dim
        
    elif hasattr(config.model, 'audio_encoder'):
        console.print("Found audio encoder in configuration")
        encoder = config.model.audio_encoder
        if hasattr(encoder, 'hidden_dim'):
            encoder_output_dim = encoder.hidden_dim
        elif hasattr(encoder, 'input_dim') and hasattr(encoder, 'layers'):
            # Проверяем FcEncoder и берем последний слой как выход
            encoder_output_dim = encoder.layers[-1]
            console.print(f"Determined output dimension from FC layers: {encoder_output_dim}")
        
    elif hasattr(config.model, 'video_encoder'):
        console.print("Found video encoder in configuration")
        encoder = config.model.video_encoder
        if hasattr(encoder, 'hidden_dim'):
            encoder_output_dim = encoder.hidden_dim
        
    elif hasattr(config.model, 'kwargs'):
        # Try to find in kwargs
        for key, value in config.model.kwargs.items():
            if key.endswith('_encoder') or key.startswith('net'):
                encoder = value
                console.print(f"Found encoder in kwargs: {key}")
                if hasattr(encoder, 'output_dim'):
                    encoder_output_dim = encoder.output_dim
                elif hasattr(encoder, 'hidden_dim'):
                    encoder_output_dim = encoder.hidden_dim
                break
    
    if encoder is None:
        raise ValueError("No encoder found in configuration")
    
    # If encoder dim is still not found, try to look for it in model level
    if encoder_output_dim is None:
        if hasattr(config.model, 'output_dim'):
            encoder_output_dim = config.model.output_dim
            console.print(f"Using output_dim from model config: {encoder_output_dim}")
        elif hasattr(config.model, 'hidden_dim'):
            encoder_output_dim = config.model.hidden_dim
        else:
            # Try some standard dimensions based on model type
            model_type = config.model.model_type.lower()
            if "mmimdb" in model_type:
                encoder_output_dim = 512
            elif "avmnist" in model_type:
                encoder_output_dim = 128
            elif "utt" in model_type or "mosi" in model_type:
                encoder_output_dim = 64
            else:
                encoder_output_dim = 512  # Default value
                console.print(f"[yellow]Warning:[/] Could not determine encoder output dimension. Using default: {encoder_output_dim}")

    # Find number of classes from the configuration or dataset
    # First try common places in model config
    if hasattr(config.model, 'num_classes'):
        num_classes = config.model.num_classes
    elif hasattr(config.model, 'classifier') and hasattr(config.model.classifier, 'output_size'):
        num_classes = config.model.classifier.output_size
    else:
        # Try to determine from the model type
        model_type = config.model.model_type.lower()
        if "mmimdb" in model_type:
            num_classes = 23
        elif "avmnist" in model_type:
            num_classes = 10
        elif "utt" in model_type or "mosi" in model_type:
            num_classes = 3
        else:
            # Default to binary classification if we can't determine
            num_classes = 2
            console.print(f"[yellow]Warning:[/] Could not determine number of classes. Using default: {num_classes}")

    # Create the monomodal model
    model = MonomodalEncoder(
        encoder=encoder,
        output_dim=encoder_output_dim,
        num_classes=num_classes
    )
    
    console.print(f"Created monomodal model with encoder output dim: {encoder_output_dim}, num classes: {num_classes}")
    
    return model, config


def train_monomodal(model: MonomodalEncoder, config: StandardMultimodalConfig):
    """
    Train a monomodal encoder.
    
    Args:
        model: The monomodal model to train
        config: Configuration for the experiment
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and config.experiment.device == "cuda" else "cpu")
    if device.type == "cuda":
        console.print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        console.print("! CUDA not available, using CPU")
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer = config.get_optimizer(model)
    
    # Create loss function
    loss_functions = config.training.loss_functions
    
    # Setup logging paths
    exp_name = config.experiment.name
    run_id = config.experiment.run_id
    
    # Получаем пути для логов и моделей. Обрабатываем строки и объекты Path
    if isinstance(config.logging.log_path, Path):
        log_path = config.logging.log_path
    else:
        log_path = Path(str(config.logging.log_path).format(experiment_name=exp_name, run_id=run_id))
        
    if isinstance(config.logging.model_output_path, Path):
        model_output_path = config.logging.model_output_path
    else:
        model_output_path = Path(str(config.logging.model_output_path).format(experiment_name=exp_name, run_id=run_id))
        
    if isinstance(config.logging.metrics_path, Path):
        metrics_path = config.logging.metrics_path
    else:
        metrics_path = Path(str(config.logging.metrics_path).format(experiment_name=exp_name, run_id=run_id))
    
    # Пути для tensorboard и мониторинга, если они есть в конфигурации
    if hasattr(config.logging, 'tensorboard_path'):
        if isinstance(config.logging.tensorboard_path, Path):
            tensorboard_path = config.logging.tensorboard_path
        else:
            tensorboard_path = Path(str(config.logging.tensorboard_path).format(experiment_name=exp_name, run_id=run_id))
        os.makedirs(tensorboard_path, exist_ok=True)
    else:
        tensorboard_path = None
        
    if hasattr(config.logging, 'monitor_path'):
        if isinstance(config.logging.monitor_path, Path):
            monitor_path = config.logging.monitor_path
        else:
            monitor_path = Path(str(config.logging.monitor_path).format(experiment_name=exp_name, run_id=run_id))
        os.makedirs(monitor_path, exist_ok=True)
    else:
        monitor_path = None
    
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    
    # Build dataloaders
    dataloaders = config.data.build_all_dataloaders(
        is_train=config.experiment.is_train, 
        is_test=config.experiment.is_test
    )
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        model_dir=model_output_path,
        save_metric=config.logging.save_metric,
        mode="minimize" if config.logging.save_metric == "loss" else "maximize",
        device=device,
    )
    
    # Setup metric recorder
    metric_recorder = MetricRecorder(
        config.metrics,
        tensorboard_path=tensorboard_path,
        tb_record_only=getattr(config.logging, 'tb_record_only', False) if hasattr(config.logging, 'tb_record_only') else False
    )
    
    # Setup report generator
    report_generator = ExperimentReportGenerator(
        output_dir=log_path,
        config=config,
        subreports={
            "model": ModelReport(output_dir=metrics_path), 
            "metrics": MetricsReport(output_dir=metrics_path), 
            "timing": TimingReport(output_dir=metrics_path)
        }
    )
    
    # Train for the specified number of epochs
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0
    
    max_patience = config.training.early_stopping_patience if hasattr(config.training, "early_stopping_patience") else 10
    use_early_stopping = config.training.early_stopping if hasattr(config.training, "early_stopping") else True
    
    total_epochs = config.training.epochs
    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("validation")
    test_loader = dataloaders.get("test")
    
    if not train_loader or not val_loader:
        raise ValueError("Training and validation dataloaders are required")
    
    experiment_data = {
        "metrics_history": {"train": [], "validation": [], "test": []},
        "timing_history": {"train": [], "validation": []},
    }
    
    console.rule(f"Starting training for {total_epochs} epochs")
    
    for epoch in range(total_epochs):
        # Training phase
        model.train()
        train_metrics = defaultdict(list)
        
        console.start_task("Training", total=len(train_loader), style="light slate_blue")
        start_time = time.time()
        
        for batch in train_loader:
            train_output = model.train_step(
                batch=batch,
                optimizer=optimizer,
                loss_functions=loss_functions,
                device=device,
                metric_recorder=metric_recorder,
                config=config,
            )
            
            # Record metrics
            for k, v in train_output["metrics"].items():
                train_metrics[k].append(v)
            
            console.update_task("Training", advance=1)
            
        console.complete_task("Training")
        train_time = time.time() - start_time
        
        # Вычислить метрики для обучающей выборки
        calculated_train_metrics = metric_recorder.calculate_all_groups(
            epoch=epoch,
            loss=np.mean(train_metrics["loss"]) if train_metrics.get("loss") else None,
            skip_tensorboard=True
        )
        calculated_train_metrics = flatten_dict(calculated_train_metrics)
        
        # Объединяем рассчитанные метрики с метриками, собранными во время обучения
        avg_train_metrics = {}
        # Сначала добавляем средние значения метрик из train_metrics
        for k, v in train_metrics.items():
            avg_train_metrics[k] = np.mean(v)
        # Затем добавляем метрики, вычисленные через metric_recorder
        avg_train_metrics.update(calculated_train_metrics)
        
        experiment_data["metrics_history"]["train"].append(avg_train_metrics)
        experiment_data["timing_history"]["train"].append(train_time)
        
        # Сбрасываем состояние для следующей эпохи
        metric_recorder.reset()
        
        # Validation phase
        model.eval()
        val_metrics = defaultdict(list)
        
        console.start_task("Validation", total=len(val_loader), style="bright yellow")
        start_time = time.time()
        
        with torch.no_grad():
            for batch in val_loader:
                val_output = model.validation_step(
                    batch=batch,
                    loss_functions=loss_functions,
                    device=device,
                    metric_recorder=metric_recorder,
                    config=config,
                )
                
                # Record metrics
                for k, v in val_output["metrics"].items():
                    val_metrics[k].append(v)
                
                console.update_task("Validation", advance=1)
                
        console.complete_task("Validation")
        val_time = time.time() - start_time
        
        # Вычислить метрики для валидационной выборки
        calculated_val_metrics = metric_recorder.calculate_all_groups(
            epoch=epoch,
            loss=np.mean(val_metrics["loss"]) if val_metrics.get("loss") else None,
            skip_tensorboard=True
        )
        calculated_val_metrics = flatten_dict(calculated_val_metrics)
        
        # Объединяем рассчитанные метрики с метриками, собранными во время валидации
        avg_val_metrics = {}
        # Сначала добавляем средние значения метрик из val_metrics
        for k, v in val_metrics.items():
            avg_val_metrics[k] = np.mean(v)
        # Затем добавляем метрики, вычисленные через metric_recorder
        avg_val_metrics.update(calculated_val_metrics)
        
        experiment_data["metrics_history"]["validation"].append(avg_val_metrics)
        experiment_data["timing_history"]["validation"].append(val_time)
        
        # Сбрасываем состояние для следующей эпохи
        metric_recorder.reset()
        
        # Print training and validation metrics
        console.print(f"Epoch {epoch+1}/{total_epochs}")
        console.print(f"Train loss: {avg_train_metrics['loss']:.4f}" + 
                     (f", Train accuracy: {avg_train_metrics.get('accuracy', 0):.4f}" if 'accuracy' in avg_train_metrics else ""))
        console.print(f"Val loss: {avg_val_metrics['loss']:.4f}" + 
                     (f", Val accuracy: {avg_val_metrics.get('accuracy', 0):.4f}" if 'accuracy' in avg_val_metrics else ""))
        
        # Check for improvement and save checkpoint
        current_val_loss = avg_val_metrics["loss"]
        current_val_accuracy = avg_val_metrics.get("accuracy", 0)
        
        is_best = False
        if config.logging.save_metric == "loss" and current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            is_best = True
            patience_counter = 0
        elif config.logging.save_metric != "loss" and current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            is_best = True
            patience_counter = 0
        else:
            patience_counter += 1
        
        if is_best:
            # Save the model checkpoint
            checkpoint_manager.save_checkpoint(
                model=model, 
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch, 
                metrics=avg_val_metrics,
                is_best=True
            )
            
            # Also save just the encoder state dict for later use in multimodal training
            encoder_state_dict = model.get_encoder().state_dict()
            
            # Extract modality name from experiment name (assumes format like "MMIMDb_Image_Encoder_Pretrain")
            exp_parts = exp_name.lower().split('_')
            modality = "unknown"
            for part in exp_parts:
                if part in ["image", "text", "audio", "video"]:
                    modality = part
                    break
            
            encoder_path = model_output_path / f"encoder_{modality}_best.pth"
            torch.save(encoder_state_dict, encoder_path)
            console.print(f"Saved best encoder to {encoder_path}")
        
        # Early stopping check
        if use_early_stopping and patience_counter >= max_patience:
            console.print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Update learning rate if scheduler is configured
        if hasattr(config.training, "scheduler") and config.training.scheduler:
            scheduler = config.get_scheduler(optimizer=optimizer)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_val_loss)
            else:
                scheduler.step()
    
    # Test phase if test loader is available
    if test_loader and config.experiment.is_test:
        console.rule("Testing best model")
        
        # Load best model
        checkpoint_manager.load_checkpoint(model=model, load_best=True)
        
        # Test
        model.eval()
        test_metrics = defaultdict(list)
        
        console.start_task("Testing", total=len(test_loader), style="bright cyan")
        start_time = time.time()
        
        with torch.no_grad():
            for batch in test_loader:
                test_output = model.validation_step(
                    batch=batch,
                    loss_functions=loss_functions,
                    device=device,
                    metric_recorder=metric_recorder,
                    config=config,
                )
                
                # Record metrics
                for k, v in test_output["metrics"].items():
                    test_metrics[k].append(v)
                
                console.update_task("Testing", advance=1)
                
        console.complete_task("Testing")
        test_time = time.time() - start_time
        
        # Вычислить метрики для тестовой выборки
        calculated_test_metrics = metric_recorder.calculate_all_groups(
            loss=np.mean(test_metrics["loss"]) if test_metrics.get("loss") else None,
            skip_tensorboard=True
        )
        calculated_test_metrics = flatten_dict(calculated_test_metrics)
        
        # Объединяем рассчитанные метрики с метриками, собранными во время тестирования
        avg_test_metrics = {}
        # Сначала добавляем средние значения метрик из test_metrics
        for k, v in test_metrics.items():
            avg_test_metrics[k] = np.mean(v)
        # Затем добавляем метрики, вычисленные через metric_recorder
        avg_test_metrics.update(calculated_test_metrics)
        
        experiment_data["metrics_history"]["test"] = avg_test_metrics
        experiment_data["timing_history"]["test"] = [test_time]
        
        # Сбрасываем состояние для следующего использования
        metric_recorder.reset()
        
        # Print test metrics
        console.print(f"Test loss: {avg_test_metrics['loss']:.4f}" + 
                     (f", Test accuracy: {avg_test_metrics.get('accuracy', 0):.4f}" if 'accuracy' in avg_test_metrics else ""))
    
    # Generate final report
    report_generator.generate_report(experiment_data)
    
    # Final message
    console.rule("Training completed")
    console.print(f"Best validation loss: {best_val_loss:.4f}" +
                 (f", Best validation accuracy: {best_val_accuracy:.4f}" if best_val_accuracy > 0 else ""))
    console.print(f"Encoder weights saved to: {encoder_path}")
    
    return encoder_path


def main():
    parser = ArgumentParser(description="Train a monomodal encoder for a multimodal model")
    parser.add_argument("--config", type=str, required=True, help="Path to the monomodal encoder config file")
    parser.add_argument("--run_id", type=int, default=1, help="Run identifier")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Setup the experiment
    model, config = setup_experiment(args.config, args.run_id)
    
    # Start training
    encoder_path = train_monomodal(model, config)
    
    console.print(f"Обучение завершено. Лучший энкодер сохранен в: {encoder_path}")


if __name__ == "__main__":
    main()
