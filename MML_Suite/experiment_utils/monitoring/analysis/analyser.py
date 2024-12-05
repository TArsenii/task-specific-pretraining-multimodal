from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .stats import StatisticalMeasures


class MonitoringAnalyser:
    """Enhanced analyzer with specific analysis methods."""

    def analyze_gradients(
        self, run_id: Optional[str] = None, layers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze gradient flow patterns."""
        results = {}
        run_ids = [run_id] if run_id else list(self.runs.keys())

        for rid in run_ids:
            grad_group = self.runs[rid]["gradients"]
            epochs = sorted(
                [
                    int(k.split("_")[1])
                    for k in grad_group.keys()
                    if k.startswith("epoch_")
                ]
            )

            epoch_results = {}
            for epoch in epochs:
                if (
                    self.config.start_epoch is not None
                    and epoch < self.config.start_epoch
                ):
                    continue
                if self.config.end_epoch is not None and epoch > self.config.end_epoch:
                    break

                epoch_key = f"epoch_{epoch}"
                epoch_grads = grad_group[epoch_key]

                layer_results = {}
                for layer_name, layer_data in epoch_grads.items():
                    if layers and not any(pattern in layer_name for pattern in layers):
                        continue

                    # Compute comprehensive gradient statistics
                    layer_results[layer_name] = (
                        StatisticalMeasures.compute_gradient_stats(layer_data[:])
                    )

                epoch_results[epoch] = layer_results

            results[rid] = epoch_results

        return results

    def analyze_activations(
        self, run_id: Optional[str] = None, layers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze activation patterns."""
        results = {}
        run_ids = [run_id] if run_id else list(self.runs.keys())

        for rid in run_ids:
            act_group = self.runs[rid]["activations"]
            epochs = sorted(
                [
                    int(k.split("_")[1])
                    for k in act_group.keys()
                    if k.startswith("epoch_")
                ]
            )

            epoch_results = {}
            for epoch in epochs:
                if (
                    self.config.start_epoch is not None
                    and epoch < self.config.start_epoch
                ):
                    continue
                if self.config.end_epoch is not None and epoch > self.config.end_epoch:
                    break

                epoch_key = f"epoch_{epoch}"
                epoch_acts = act_group[epoch_key]

                layer_results = {}
                for layer_name, layer_data in epoch_acts.items():
                    if layers and not any(pattern in layer_name for pattern in layers):
                        continue

                    # Compute comprehensive activation statistics
                    layer_results[layer_name] = (
                        StatisticalMeasures.compute_activation_stats(layer_data[:])
                    )

                epoch_results[epoch] = layer_results

            results[rid] = epoch_results

        return results

    def analyze_weights(
        self, run_id: Optional[str] = None, layers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze weight evolution."""
        results = {}
        run_ids = [run_id] if run_id else list(self.runs.keys())

        for rid in run_ids:
            weight_group = self.runs[rid]["weights"]
            epochs = sorted(
                [
                    int(k.split("_")[1])
                    for k in weight_group.keys()
                    if k.startswith("epoch_")
                ]
            )

            epoch_results = {}
            for epoch in epochs:
                if (
                    self.config.start_epoch is not None
                    and epoch < self.config.start_epoch
                ):
                    continue
                if self.config.end_epoch is not None and epoch > self.config.end_epoch:
                    break

                epoch_key = f"epoch_{epoch}"
                epoch_weights = weight_group[epoch_key]

                layer_results = {}
                for layer_name, layer_data in epoch_weights.items():
                    if layers and not any(pattern in layer_name for pattern in layers):
                        continue

                    # Compute comprehensive weight statistics
                    layer_results[layer_name] = (
                        StatisticalMeasures.compute_weight_stats(layer_data[:])
                    )

                epoch_results[epoch] = layer_results

            results[rid] = epoch_results

        return results

    def get_temporal_evolution(
        self, metric: str, run_id: str, layer: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Get temporal evolution of a metric."""
        valid_metrics = {"gradients", "activations", "weights"}
        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of {valid_metrics}")

        metric_group = self.runs[run_id][metric]
        epochs = sorted(
            [
                int(k.split("_")[1])
                for k in metric_group.keys()
                if k.startswith("epoch_")
            ]
        )

        evolution = defaultdict(list)
        for epoch in epochs:
            if self.config.start_epoch is not None and epoch < self.config.start_epoch:
                continue
            if self.config.end_epoch is not None and epoch > self.config.end_epoch:
                break

            epoch_key = f"epoch_{epoch}"
            epoch_data = metric_group[epoch_key]

            for layer_name, layer_data in epoch_data.items():
                if layer and layer not in layer_name:
                    continue

                # Get appropriate statistical measure based on metric type
                if metric == "gradients":
                    stats = StatisticalMeasures.compute_gradient_stats(layer_data[:])
                elif metric == "activations":
                    stats = StatisticalMeasures.compute_activation_stats(layer_data[:])
                else:  # weights
                    stats = StatisticalMeasures.compute_weight_stats(layer_data[:])

                evolution[layer_name].append({"epoch": epoch, "stats": stats})

        return dict(evolution)

    def get_summary_statistics(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for all monitored metrics."""
        run_ids = [run_id] if run_id else list(self.runs.keys())
        summary = {}

        for rid in run_ids:
            summary[rid] = {
                "gradients": self.analyze_gradients(run_id=rid),
                "activations": self.analyze_activations(run_id=rid),
                "weights": self.analyze_weights(run_id=rid),
            }

            # Add overall training statistics
            summary[rid]["training_duration"] = {
                "epochs": len(self.runs[rid]["gradients"].keys()),
                "start_time": self.run_metadata[rid].get("start_time"),
                "end_time": self.run_metadata[rid].get("end_time"),
            }

        return summary
