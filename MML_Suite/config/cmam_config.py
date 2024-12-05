from dataclasses import dataclass
from config.model_config import ModelConfig
from config.multimodal_training_config import BaseExperimentConfig
from config.base_config import BaseConfig
from modalities import Modality


@dataclass(kw_only=True)
class AssociationNetworkConfig(BaseConfig):
    input_size: int
    hidden_size: int
    output_size: int
    batch_norm: bool = False
    dropout: float = 0.0


@dataclass(kw_only=True)
class CMAMConfig(BaseExperimentConfig):
    # experiment: ExperimentConfig
    #     data: DataConfig
    #     model: ModelConfig
    #     logging: LoggingConfig
    #     metrics: MetricConfig
    #     training: TrainingConfig
    #     monitoring: MonitorConfig
    #     _config_path: Optional[str] = None

    cmam: ModelConfig
    target_modality: Modality

    def __post_init__(self):
        super().__post_init__()
