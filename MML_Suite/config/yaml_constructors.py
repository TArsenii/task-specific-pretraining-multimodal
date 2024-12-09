import yaml
from config import (
    AssociationNetworkConfig,
    CMAMConfig,
    DataConfig,
    DatasetConfig,
    ExperimentConfig,
    LoggingConfig,
    MetricConfig,
    MissingPatternConfig,
    ModalityConfig,
    ModelConfig,
    OptimizerConfig,
    ParameterGroupConfig,
    StandardMultimodalConfig,
)
from experiment_utils import get_logger
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.managers import CenterManager, FeatureManager, LabelManager
from modalities import add_modality
from models import (
    CMAM,
    AuViSubNet,
    BertTextEncoder,
    ConvBlockArgs,
    FcClassifier,
    InputEncoders,
    LSTMEncoder,
    MNISTAudio,
    MNISTImage,
    ResidualAE,
    Self_MM,
    TextCNN,
    UttFusionModel,
    MMIMDbModalityEncoder, 
    GatedBiModalNetwork, GMUModel, MLPGenreClassifier, MaxOut
)

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
register_constructor("!LoggingConfig", LoggingConfig)
register_constructor("!ModelConfig", ModelConfig, from_dict=True)
register_constructor("!ExperimentConfig", ExperimentConfig)
register_constructor("!StandardConfig", StandardMultimodalConfig)
register_constructor("!ParameterGroupConfig", ParameterGroupConfig)
register_constructor("!Optimizer", OptimizerConfig, from_dict=True, deep=True)

# Registering constructors for models and other components
register_scalar_constructor("!Modality", add_modality)
register_constructor(
    "!MNISTAudio",
    MNISTAudio,
    deep=True,
)
register_constructor(
    "!MNISTImage",
    MNISTImage,
    deep=True,
)
register_constructor(
    "!ConvBlock",
    ConvBlockArgs,
    deep=True,
)
register_constructor(
    "!Self_MM",
    Self_MM,
    deep=True,
)
register_constructor(
    "!AuViSubNet",
    AuViSubNet,
    deep=True,
)
register_constructor(
    "!LSTMEncoder",
    LSTMEncoder,
    deep=True,
)
register_constructor(
    "!TextCNN",
    TextCNN,
    deep=True,
)
register_constructor(
    "!FcClassifier",
    FcClassifier,
    deep=True,
)
register_constructor(
    "!ResidualAE",
    ResidualAE,
    deep=True,
)
register_constructor("!LossFunctionGroup", LossFunctionGroup)
register_constructor(
    "!UttFusionModel",
    UttFusionModel,
    deep=True,
)
register_constructor(
    "!MMIMDbModalityEncoder",
    MMIMDbModalityEncoder,
    deep=True,
)
register_constructor(
    "!MaxOut",
    MaxOut,
    deep=True,
)
register_constructor(
    "!GatedBiModalNetwork",
    GatedBiModalNetwork,
    deep=True,
)
register_constructor(
    "!GMUModel",
    GMUModel,
    deep=True,
)
register_constructor(
    "!MLPGenreClassifier",
    MLPGenreClassifier,
    deep=True,
)
register_constructor(
    "!FeatureManager",
    FeatureManager,
    deep=True,
)
register_constructor("!CenterManager", CenterManager, deep=True)
register_constructor("!LabelManager", LabelManager, deep=True)
register_constructor("!BertTextEncoder", BertTextEncoder)
register_constructor("!MissingPatternConfig", MissingPatternConfig)
register_constructor("!ModalityConfig", ModalityConfig)

register_constructor("!AssociationNetworkConfig", AssociationNetworkConfig)
register_constructor("!InputEncoders", InputEncoders)
register_constructor("!CMAM", CMAM)
register_constructor("!CMAMConfig", CMAMConfig)

logger.debug("All YAML constructors registered successfully.")
