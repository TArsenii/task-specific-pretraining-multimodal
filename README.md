# MML_Suite

MML_Suite is a modular tool for multimodal machine learning research. It supports configurable experiments, federated learning, and the novel C-MAMs approach for handling missing modalities.

> **Note:** This repository is a fork of the original [MML_Suite](https://github.com/jmg049/MML_Suite) project by Jack Geraghty (jmg049). This fork includes implementations and improvements for MMIMDB, AVMNIST, and MOSI datasets, as well as additional training scripts.

## Getting Starting
The project manages dependencies through [Poetry](https://python-poetry.org/). To install all the dependencies, from the root directory, run ``poetry install`` or ``pip install .``.

## Running an Experimennt
This fork provides two main training scripts:

1. For pretraining individual modality encoders:
```bash
python train_monomodal.py --config ./path/to/config/file.yaml --run_id 1
```

2. For training multimodal models:
```bash
python train_multimodal.py --config ./path/to/config/file.yaml --run_id 1
```

Additional CLI args:
- ``--skip-train`` - Skips the training phase.
- ``--skip-test`` - Skips the testing phase.
- ``--dry-run`` - Performs a dry run and stops just before training.
- ``--disable-monitoring`` - Force monitoring (of gradients etc.) off. Overrides the config file.


# Configuration

The project uses a hierarchical YAML-based configuration system with several specialized configuration classes to manage different aspects of the experiment pipeline. The configuration system is designed to be modular, extensible, and relatively type-safe.


## Configuration Structure

A complete experiment configuration consists of the following main components:

- `experiment`: General experiment settings
- `data`: Dataset and dataloader configuration
- `model`: Model architecture and parameters
- `logging`: Logging paths and settings
- `metrics`: Evaluation metrics
- `training`: Training parameters
- `monitoring`: Experiment monitoring settings

## Creating a Configuration

Create a YAML file with the following structure:

```yaml
experiment:
  name: "experiment_name"
  device: "cuda"  # or "cpu"
  seed: 42  # optional
  debug: false
  do_test: true
  do_train: true
  train_print_interval_epochs: 1
  validation_print_interval_epochs: 1

data:
  datasets:
    train:
      dataset: "dataset_name"
      data_fp: "${DATA_DIR}/path/to/data"
      target_modality: "target_mod"
      split: "train"
      batch_size: 32
      shuffle: true
      num_workers: 4
      missing_patterns:  # optional
        modalities:
          modality1:
            missing_rate: 0.2
          modality2:
            missing_rate: 0.3
        selected_patterns: ["m1", "m2"]

model:
  name: "model_name"
  model_type: "model.path.ModelClass"
  pretrained_path: null  # optional
  # model-specific parameters

logging:
  log_path: "${LOG_DIR}/experiments/{experiment_name}/{run_id}"
  metrics_path: "${LOG_DIR}/metrics/{experiment_name}/{run_id}"
  model_output_path: "${MODEL_DIR}/{experiment_name}/{run_id}"
  monitor_path: "${LOG_DIR}/monitoring/{experiment_name}/{run_id}"

metrics:
  metrics:
    accuracy:
      function: "metrics.classification.accuracy"
      level: "batch"  # or "epoch" or "both"
    loss:
      function: "metrics.losses.cross_entropy"
      kwargs:
        reduction: "mean"
  groups:
    classification:
      - "accuracy"
      - "loss"

training:
  epochs: 100
  num_modalities: 3
  optimizer:
    name: "adam"
    param_groups:
      - name: "encoder"
        pattern: ["encoder.*"]
        lr: 0.001
        weight_decay: 0.01
    default_kwargs:
      lr: 0.001
      weight_decay: 0.0
  scheduler:  # optional
    name: "cosine"
    args:
      T_max: 100
  criterion: "cross_entropy"
  validation_interval: 1
  early_stopping: true
  early_stopping_patience: 10
  early_stopping_min_delta: 0.001

monitoring:
  enabled: true
  gradient_interval: 100
  activation_interval: 100
  weight_interval: 200
  buffer_size: 1000
  compression: "gzip"
```

## Configuration Components

### Experiment Config
Controls high-level experiment settings:
- `name`: Experiment identifier
- `device`: Computing device (cuda/cpu)
- `seed`: Random seed for reproducibility
- `debug`: Enable debug mode
- `do_test/do_train`: Control training and testing phases

### Data Config
Manages dataset and dataloader configuration:
- Supports multiple datasets (train/val/test)
- Configurable batch size, shuffling, and workers
- Missing pattern support for multimodal scenarios
- Environment variable expansion in paths (e.g., `${DATA_DIR}`)

### Model Config
Defines model architecture and parameters:
- Model type specification
- Optional pretrained model loading
- Flexible kwargs for model-specific parameters

### Logging Config
Controls experiment logging:
- Configurable paths for logs, metrics, and model outputs
- Support for environment variables
- Automatic directory creation and validation

### Metrics Config
Defines evaluation metrics:
- Supports multiple metric functions
- Configurable metric groups
- Batch-level and epoch-level metrics
- Custom metric parameters

### Training Config
Manages training parameters:
- Optimizer configuration with parameter groups
- Learning rate scheduling
- Loss criterion selection
- Early stopping configuration
- Validation intervals

### Monitoring Config
Controls experiment monitoring:
- Gradient tracking
- Activation monitoring
- Weight tracking
- Buffer settings for efficient storage

## Usage

Load a configuration using the `StandardMultimodalConfig` class:

```python
from config import StandardMultimodalConfig

config = StandardMultimodalConfig.load("path/to/config.yaml", run_id=123)
```

The configuration system will:
1. Validate all components
2. Create necessary directories
3. Set up logging
4. Initialize monitoring
5. Prepare metrics tracking

## Environment Variables

The configuration system supports environment variable expansion in paths, for example:
- `${DATA_DIR}`: Base directory for datasets
- `${LOG_DIR}`: Directory for logs and metrics
- `${MODEL_DIR}`: Directory for model checkpoints

Ensure these environment variables are set before running experiments.

## The General Procedure for Adding Custom Models/Datasets
To add in a new model and/or dataset there are some general steps to follow. For models, the model should be (best practice) added to somewhere in the ``model`` directory. Note that to support C-MAMs there are a few more steps, see [C-MAMs](#cross-modal-association-models) and [Federated Learning](#federated-learning). 

### Models
Each model should implement the following functions:

- train_step(self, batch, optimizer, loss_functions: [LossFunctionGroup](./MML_Suite/experiment_utils/loss.py), device, metric_recorder: [MetricRecorder](./MML_Suite/experiment_utils/metric_recorder.py), **kwargs)

- validation_step(self, batch, loss_functions: [LossFunctionGroup](./MML_Suite/experiment_utils/loss.py), device, metric_recorder: [MetricRecorder](./MML_Suite/experiment_utils/metric_recorder.py), **kwargs)

After this, go to the [resolvers](./MML_Suite/config/resolvers.py) file and the ``resolve_model_name`` function. In the match statement add in the name of your model and return the class from the function. *Note this is how you add an unsupported loss function, optimizer, scheduler, etc.*

### Datasets
To add a PyTorch dataset implement it as normal for PyTorch and then, like for adding a model, go to [resolvers](./MML_Suite/config/resolvers.py) and add your dataset to the match statement in ``resolve_dataset_name`` function.

## Cross-Modal Association Models


## Federated Learning
