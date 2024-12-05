# monitoring/model/base.py
from typing import Any, Dict

import torch.nn as nn


class MonitoringMixin:
    """
    Mixin class to add monitoring capabilities to models.

    Usage:
        class MyModel(nn.Module, MonitoringMixin):
            def __init__(self):
                super().__init__()
                self.initialize_monitoring()
    """

    def initialize_monitoring(self):
        """Initialize monitoring-related attributes."""
        self._monitor = None
        self._monitoring_enabled = False

        # Register hooks for common monitoring points
        self._monitoring_hooks = {}

    def attach_monitor(self, monitor) -> None:
        """Attach a monitor to the model."""
        self._monitor = monitor
        self._monitoring_enabled = True

    def detach_monitor(self) -> None:
        """Detach the current monitor."""
        self._monitor = None
        self._monitoring_enabled = False

    def get_monitoring_data(self) -> Dict[str, Any]:
        """
        Get model-specific monitoring data.
        Override this method to provide custom monitoring data.
        """
        return {}

    def _forward_monitor_hook(self, module: nn.Module) -> None:
        """Monitor forward pass data."""
        if not self._monitoring_enabled or not self._monitor:
            return

        monitoring_data = self.get_monitoring_data()
        if monitoring_data:
            self._monitor.track_model_data(monitoring_data)


class AttentionMonitoringMixin(MonitoringMixin):
    """Mixin for monitoring attention-based models."""

    def get_monitoring_data(self) -> Dict[str, Any]:
        data = super().get_monitoring_data()

        # Add attention-specific monitoring
        if hasattr(self, "attention_weights"):
            data["attention"] = {
                "weights": self.attention_weights,
                "patterns": self.attention_patterns if hasattr(self, "attention_patterns") else None,
            }

        return data


class MultiModalMonitoringMixin(MonitoringMixin):
    """Mixin for monitoring multi-modal models."""

    def get_monitoring_data(self) -> Dict[str, Any]:
        data = super().get_monitoring_data()

        # Add modality-specific monitoring
        modality_data = {}
        for name, module in self.named_modules():
            if hasattr(module, "modality"):
                modality_data[module.modality] = {
                    "name": name,
                    "output": getattr(module, "last_output", None),
                    "embedding": getattr(module, "last_embedding", None),
                }

        data["modalities"] = modality_data
        return data
