__all__ = [
    "BasicCMAM",
    "DualCMAM",
    "MNISTAudio",
    "MNISTImage",
    "ConvBlockArgs",
    "ConvBlock",
    "AVMNIST",
    "LSTMEncoder",
    "TextCNN",
    "kaiming_init",
    "resolve_encoder",
    "MultimodalModelProtocol",
    "Self_MM",
    "UttFusionModel",
    "msa_binarize",
    "AuViSubNet",
    "BertTextEncoder",
    "InputEncoders",
    "CMAM",
    "DualCMAM",
    "FcClassifier",
    "ResidualAE",
]
from torch.nn import BatchNorm2d, Conv2d, Linear, init

from .msa import (
    Self_MM,
    UttFusionModel,
    msa_binarize,
    AuViSubNet,
    TextCNN,
    LSTMEncoder,
    FcClassifier,
    ResidualAE,
    BertTextEncoder,
)
from .avmnist import AVMNIST, MNISTAudio, MNISTImage
from .cmams import InputEncoders, CMAM, DualCMAM
from .conv import ConvBlock, ConvBlockArgs

from .protocols import MultimodalModelProtocol


def kaiming_init(module):
    if isinstance(module, (Conv2d, Linear)):
        init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
