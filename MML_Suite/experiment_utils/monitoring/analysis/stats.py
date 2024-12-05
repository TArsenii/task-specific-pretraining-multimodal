# monitoring/analysis/stats.py
from typing import Any, Dict

import numpy as np
from scipy import stats


class StatisticalMeasures:
    """Statistical measurement utilities for monitoring data."""
    
    @staticmethod
    def distribution_stats(data: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive distribution statistics."""
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "median": float(np.median(data)),
            "skewness": float(stats.skew(data.flatten())),
            "kurtosis": float(stats.kurtosis(data.flatten())),
            "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
            "range": float(np.ptp(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "percentile_5": float(np.percentile(data, 5)),
            "percentile_95": float(np.percentile(data, 95))
        }
    
    @staticmethod
    def compute_gradient_stats(grad_data: np.ndarray) -> Dict[str, Any]:
        """Compute gradient-specific statistics."""
        basic_stats = StatisticalMeasures.distribution_stats(grad_data)
        
        return {
            **basic_stats,
            "l2_norm": float(np.linalg.norm(grad_data)),
            "l1_norm": float(np.sum(np.abs(grad_data))),
            "zero_fraction": float(np.mean(np.abs(grad_data) < 1e-7)),
            "sign_changes": float(np.mean(np.diff(np.signbit(grad_data)) != 0)),
            "positive_fraction": float(np.mean(grad_data > 0)),
            "negative_fraction": float(np.mean(grad_data < 0))
        }
    
    @staticmethod
    def compute_activation_stats(activation_data: np.ndarray) -> Dict[str, Any]:
        """Compute activation-specific statistics."""
        basic_stats = StatisticalMeasures.distribution_stats(activation_data)
        
        return {
            **basic_stats,
            "dead_fraction": float(np.mean(activation_data <= 0)),
            "saturation_fraction": float(np.mean(np.abs(activation_data) > 0.99)),
            "variance_per_unit": np.var(activation_data, axis=0).tolist(),
            "mean_activation": float(np.mean(activation_data[activation_data > 0])),
            "sparsity": float(np.mean(activation_data == 0))
        }
    
    @staticmethod
    def compute_weight_stats(weight_data: np.ndarray) -> Dict[str, Any]:
        """Compute weight-specific statistics."""
        basic_stats = StatisticalMeasures.distribution_stats(weight_data)
        
        return {
            **basic_stats,
            "spectral_norm": float(np.linalg.norm(weight_data, ord=2)),
            "frobenius_norm": float(np.linalg.norm(weight_data)),
            "effective_rank": float(np.linalg.matrix_rank(weight_data)),
            "condition_number": float(np.linalg.cond(weight_data)),
            "symmetry": float(np.mean(np.abs(weight_data - weight_data.T)))
            if weight_data.ndim == 2 and weight_data.shape[0] == weight_data.shape[1]
            else None
        }