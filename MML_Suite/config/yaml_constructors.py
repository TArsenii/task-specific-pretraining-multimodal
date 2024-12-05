import yaml
from config import (
    DataConfig,
    DatasetConfig,
    ModalityConfig,
    MissingPatternConfig,
    ExperimentConfig,
    LoggingConfig,
    MetricConfig,
    ModelConfig,
    StandardMultimodalConfig,
    ParameterGroupConfig,
    OptimizerConfig,
)
from experiment_utils import get_logger
from modalities import add_modality
from experiment_utils.managers import CenterManager, FeatureManager, LabelManager
from models import ConvBlockArgs, MNISTAudio, MNISTImage, Self_MM, AuViSubNet, BertTextEncoder

logger = get_logger()


# YAML constructors
def register_constructor(tag, cls, from_dict=False, deep=False):
    def constructor(loader, node):
        data = loader.construct_mapping(node, deep=deep)
        return cls.from_dict(data) if from_dict else cls(**data)

    yaml.SafeLoader.add_constructor(tag, constructor)
    logger.debug(f"Added {cls.__name__} constructor to YAML Loader.")


# Specific YAML constructor functions
def register_scalar_constructor(tag, func):
    def constructor(loader, node):
        value = loader.construct_scalar(node) if func == add_modality else loader.construct_mapping(node)
        return func(value)

    yaml.SafeLoader.add_constructor(tag, constructor)
    logger.debug(f"Added {func.__name__} constructor to YAML Loader.")


# Registering YAML constructors for various configurations
register_constructor("!DatasetConfig", DatasetConfig, from_dict=True, deep=True)
register_constructor("!DataConfig", DataConfig, from_dict=True, deep=True)
register_constructor("!MetricConfig", MetricConfig, from_dict=True)
register_constructor("!LoggingConfig", LoggingConfig, from_dict=True)
register_constructor("!ModelConfig", ModelConfig, from_dict=True)
register_constructor("!ExperimentConfig", ExperimentConfig)
register_constructor("!StandardConfig", StandardMultimodalConfig)
register_constructor("!ParameterGroupConfig", ParameterGroupConfig)
register_constructor("!Optimizer", OptimizerConfig)

# Registering constructors for models and other components
register_scalar_constructor("!Modality", add_modality)
register_constructor("!MNISTAudio", MNISTAudio)
register_constructor("!MNISTImage", MNISTImage)
register_constructor("!ConvBlock", ConvBlockArgs)
register_constructor("!Self_MM", Self_MM)
register_constructor("!AuViSubNet", AuViSubNet)
register_constructor("!FeatureManager", FeatureManager, deep=True, )
register_constructor("!CenterManager", CenterManager, deep=True)
register_constructor("!LabelManager", LabelManager, deep=True)
register_constructor("!BertTextEncoder", BertTextEncoder)
register_constructor("!MissingPatternConfig", MissingPatternConfig)
register_constructor("!ModalityConfig", ModalityConfig)

logger.debug("All YAML constructors registered successfully.")
