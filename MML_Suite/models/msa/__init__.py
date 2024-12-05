__all__ = ["UttFusionModel", "msa_binarize", "Self_MM", "AuViSubNet", "LSTMEncoder", "TextCNN", "ResidualAE", "FcClassifier"]

from .utt_fusion import UttFusionModel
from .self_mm import Self_MM, AuViSubNet
from .networks.lstm import LSTMEncoder
from .networks.textcnn import TextCNN
from .networks.autoencoder import ResidualAE
from .networks.classifier import FcClassifier

def msa_binarize(preds, labels):
    test_preds = preds - 1
    test_truth = labels - 1
    non_zeros_mask = test_truth != 0

    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0

    return (
        binary_preds,
        binary_truth,
        non_zeros_mask,
    )
