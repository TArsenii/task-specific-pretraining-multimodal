from .corr import pearson
from .msa import msa_binary_classification, confusion_matrix_from_logits
import sklearn

__all__ = ["pearson", "msa_binary_classification", "confusion_matrix_from_logits"]


def cosine_similarity(a, b, dense_output=True):
    return sklearn.metrics.pairwise.cosine_similarity(a, b, dense_output).mean()
