from typing import Any, Dict
from torch.nn import Module


class LossFunctionGroup(Dict[str, Module]):
    def __init__(self, weights: list[float] = None, **kwargs):
        super(LossFunctionGroup, self).__init__(**kwargs)
        self.update(kwargs)
        self.weights = weights or [1.0] * len(self)

    def __call__(self, *args, **kwargs) -> Any | Dict[str, Any]:
        loss_terms = {k: v(*args, **kwargs) * self.weights[i] for i, (k, v) in enumerate(self.items())}
        if len(loss_terms) == 1:
            ## If there is only one loss term, return it directly
            return list(loss_terms.values())[0]
        return loss_terms
