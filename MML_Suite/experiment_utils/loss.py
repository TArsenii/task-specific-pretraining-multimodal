from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from torch import Tensor, Type
from torch.nn import (
    Module,
    CrossEntropyLoss,
    NLLLoss,
    MSELoss,
    BCELoss,
    BCEWithLogitsLoss,
    L1Loss,
    SmoothL1Loss,
    KLDivLoss,
    HuberLoss,
    TripletMarginLoss,
    CosineEmbeddingLoss,
    MarginRankingLoss,
    MultiMarginLoss,
    SoftMarginLoss,
    MultiLabelMarginLoss,
    HingeEmbeddingLoss,
    PoissonNLLLoss,
    GaussianNLLLoss,
    CTCLoss,
)
from experiment_utils.logging import get_logger
from experiment_utils.printing import get_console
from cmam_loss import CMAMLoss

logger = get_logger()
console = get_console()


def resolve_criterion(criterion_name: str) -> Type[Module]:
    """
    Resolve loss criterion class from string name.

    Args:
        criterion_name: Name of the criterion (case-insensitive)

    Returns:
        Criterion class
    """
    criterion_map = {
        "cross_entropy": CrossEntropyLoss,
        "nll": NLLLoss,
        "mse": MSELoss,
        "bce": BCELoss,
        "bce_with_logits": BCEWithLogitsLoss,
        "l1": L1Loss,
        "smooth_l1": SmoothL1Loss,
        "kl_div": KLDivLoss,
        "huber": HuberLoss,
        "triplet": TripletMarginLoss,
        "cosine": CosineEmbeddingLoss,
        "margin_ranking": MarginRankingLoss,
        "multi_margin": MultiMarginLoss,
        "soft_margin": SoftMarginLoss,
        "multi_label_margin": MultiLabelMarginLoss,
        "hinge_embedding": HingeEmbeddingLoss,
        "poisson_nll": PoissonNLLLoss,
        "gaussian_nll": GaussianNLLLoss,
        "ctc": CTCLoss,
        "cmam": CMAMLoss,
        "na": lambda x: x,
        "cycle": MSELoss,
    }

    criterion_name = criterion_name.lower()

    if criterion_name not in criterion_map:
        error_msg = f"Unknown criterion: {criterion_name}. Available criteria: {list(criterion_map.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Resolved criterion: {criterion_name}")
    return criterion_map[criterion_name]


@dataclass
class WeightedLossTerm:
    loss_fn: Module
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WeightedLossTerm:
        loss_name = data["loss_name"]
        loss_kwargs = data.get("loss_kwargs", {})
        weight = data.get("weight", 1.0)

        loss_fn_type = resolve_criterion(loss_name)

        return cls(loss_fn=loss_fn_type(**loss_kwargs), weight=weight)


class LossFunctionGroup(Dict[str, WeightedLossTerm]):
    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> LossFunctionGroup:
        l_group = cls({key: WeightedLossTerm.from_dict(value) for key, value in data.items()})
        console.print(f"Created LossFunctionGroup with keys: {list(l_group.keys())}")
        return l_group

    def __call__(
        self, key: Optional[str] = None, override_weight_with: Optional[float] = None, *args, **kwargs
    ) -> Tensor:
        if not key:
            ## assume a single loss function
            key = list(self.keys())[0]
        return self[key].loss_fn(*args, **kwargs) * (
            self[key].weight if override_weight_with is None else override_weight_with
        )
