from typing import Type
from cmam_loss import CMAMLoss
from experiment_utils import get_console, get_logger
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from data import AVMNIST, MOSI, MOSEI, IEMOCAP, MSP_IMPROV, MM_IMDb

logger = get_logger()
console = get_console()


def resolve_model_name(_type: str):
    match _type.lower():
        case "avmnist":
            from models.avmnist import AVMNIST

            return AVMNIST
        case "self-mm":
            from models.msa.self_mm import Self_MM

            return Self_MM
        case "utt-fusion":
            from models.msa.utt_fusion import UttFusionModel

            return UttFusionModel
        case "mmin":
            from models.msa.mmin import MMIN

            return MMIN
        case "cmam":
            from models.cmams import CMAM

            return CMAM

        case _:
            raise ValueError(f"Unknown model type: {_type}")


def resolve_encoder(_type: str):
    from models.msa.networks import LSTMEncoder, TextCNN

    match _type.lower():
        case "lstmencoder":
            return LSTMEncoder
        case "textcnn":
            return TextCNN
        # case "mmimdbmodalityencoder":
        #     return MMIMDbModalityEncoder
        case _:
            raise ValueError(f"Unknown encoder type: {_type}")


def resolve_optimizer(optimizer_name: str) -> Type[optim.Optimizer]:
    """
    Resolve optimizer class from string name.

    Args:
        optimizer_name: Name of the optimizer (case-insensitive)

    Returns:
        Optimizer class
    """
    optimizer_map = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
        "adadelta": optim.Adadelta,
        "adamax": optim.Adamax,
        "asgd": optim.ASGD,
        "lbfgs": optim.LBFGS,
        "sparse_adam": optim.SparseAdam,
    }

    optimizer_name = optimizer_name.lower()

    if optimizer_name not in optimizer_map:
        error_msg = f"Unknown optimizer: {optimizer_name}. Available optimizers: {list(optimizer_map.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Resolved optimizer: {optimizer_name}")
    return optimizer_map[optimizer_name]


def resolve_scheduler(scheduler_name: str) -> Type[lr_scheduler._LRScheduler]:
    """
    Resolve scheduler class from string name.

    Args:
        scheduler_name: Name of the scheduler (case-insensitive)

    Returns:
        Scheduler class
    """
    scheduler_map = {
        "step": lr_scheduler.StepLR,
        "multistep": lr_scheduler.MultiStepLR,
        "exponential": lr_scheduler.ExponentialLR,
        "cosine": lr_scheduler.CosineAnnealingLR,
        "plateau": lr_scheduler.ReduceLROnPlateau,
        "cyclic": lr_scheduler.CyclicLR,
        "onecycle": lr_scheduler.OneCycleLR,
        "cosine_warmup": lr_scheduler.CosineAnnealingWarmRestarts,
        "lambda": lr_scheduler.LambdaLR,
    }

    scheduler_name = scheduler_name.lower()

    if scheduler_name not in scheduler_map:
        error_msg = f"Unknown scheduler: {scheduler_name}. Available schedulers: {list(scheduler_map.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Resolved scheduler: {scheduler_name}")
    return scheduler_map[scheduler_name]


def resolve_criterion(criterion_name: str) -> Type[nn.Module]:
    """
    Resolve loss criterion class from string name.

    Args:
        criterion_name: Name of the criterion (case-insensitive)

    Returns:
        Criterion class
    """
    criterion_map = {
        "cross_entropy": nn.CrossEntropyLoss,
        "nll": nn.NLLLoss,
        "mse": nn.MSELoss,
        "bce": nn.BCELoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "l1": nn.L1Loss,
        "smooth_l1": nn.SmoothL1Loss,
        "kl_div": nn.KLDivLoss,
        "huber": nn.HuberLoss,
        "triplet": nn.TripletMarginLoss,
        "cosine": nn.CosineEmbeddingLoss,
        "margin_ranking": nn.MarginRankingLoss,
        "multi_margin": nn.MultiMarginLoss,
        "soft_margin": nn.SoftMarginLoss,
        "multi_label_margin": nn.MultiLabelMarginLoss,
        "hinge_embedding": nn.HingeEmbeddingLoss,
        "poisson_nll": nn.PoissonNLLLoss,
        "gaussian_nll": nn.GaussianNLLLoss,
        "ctc": nn.CTCLoss,
        "cmam": CMAMLoss,
        "na": lambda x: x,
        "cycle": nn.MSELoss,
    }

    criterion_name = criterion_name.lower()

    if criterion_name not in criterion_map:
        error_msg = f"Unknown criterion: {criterion_name}. Available criteria: {list(criterion_map.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Resolved criterion: {criterion_name}")
    return criterion_map[criterion_name]


def resolve_dataset_name(dataset_name: str) -> Type[Dataset]:
    """
    Resolve dataset class from string

    Args:
        dataset_name: Name of the dataset (case-insensitive)

    Returns:
        Dataset class
    """

    dataset_map = {
        "avmnist": AVMNIST,
        "mosi": MOSI,
        "mosei": MOSEI,
        "iemocap": IEMOCAP,
        "msp_improv": MSP_IMPROV,
        "mm_imdb": MM_IMDb,
    }

    dataset_name = dataset_name.lower()

    if dataset_name not in dataset_map:
        error_msg = f"Unknown dataset: {dataset_name}. Available datasets: {list(dataset_map.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Resolved dataset: {dataset_name}")
    return dataset_map[dataset_name]
