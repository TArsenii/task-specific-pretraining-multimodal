import importlib
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from experiment_utils.logging import get_logger
from experiment_utils.printing import get_console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .base_config import BaseConfig

logger = get_logger()
console = get_console()


@dataclass
class MetricConfig(BaseConfig):
    """
    Configuration for metrics with support for groups and documentation.
    """

    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    groups: Dict[str, List[str]] = field(default_factory=dict)
    _docs_cache: Dict[str, str] = field(default_factory=dict, init=False)

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        metrics = "\n".join([f'- {k}: {v["function"]}' for k, v in self.metrics.items()])
        return "\n".join(
            [
                f"Metrics: {metrics}",
                f"Groups: {self.groups}",
            ]
        )

    def __repr__(self) -> str:
        """Return a string representation of the configuration."""
        return f"MetricConfig(metrics={self.metrics}, groups={self.groups})"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricConfig":
        """Create MetricConfig from a dictionary."""

        return cls(metrics=data.get("metrics", {}), groups=data.get("groups", {}))

    def __post_init__(self):
        """Validate metrics configuration and display summary."""
        self._validate_metrics()
        self._validate_groups()
        self._cache_docs()
        self._display_metrics_summary()

    def _validate_metrics(self) -> None:
        invalid_metrics = []
        for metric_name, metric_info in self.metrics.items():
            try:
                if "function" not in metric_info:
                    raise ValueError("Missing 'function' specification")

                if "level" not in metric_info:
                    metric_info["level"] = "batch"  # Default to batch-level
                elif metric_info["level"] not in ["batch", "epoch", "both"]:
                    raise ValueError(f"Invalid level: {metric_info['level']}")

                # Rest of validation logic remains the same
                if not isinstance(metric_info["function"], str) or "." not in metric_info["function"]:
                    raise ValueError(f"Invalid function path: {metric_info['function']}")

                if "kwargs" in metric_info and not isinstance(metric_info["kwargs"], dict):
                    raise ValueError("kwargs must be a dictionary")

                module_path, func_name = metric_info["function"].rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                    getattr(module, func_name)
                except (ImportError, AttributeError) as e:
                    raise ValueError(f"Could not import metric function: {str(e)}")

            except Exception as e:
                invalid_metrics.append((metric_name, str(e)))
                logger.error(f"Validation error for metric '{metric_name}': {str(e)}")

        if invalid_metrics:
            console.print(Panel("[red]⚠️ Metric Configuration Errors[/]", expand=False))
            for metric, error in invalid_metrics:
                console.print(f"[red]✗[/] {metric}: {error}")

    def _display_metrics_summary(self) -> None:
        console.rule("[heading]Initialising Metric Configuration[/]")
        metrics_table = Table(title="Configured Metrics", title_style="bold white", expand=True)
        metrics_table.add_column("Metric Name", style="cyan")
        metrics_table.add_column("Function", style="green")
        metrics_table.add_column("Parameters", style="yellow")
        metrics_table.add_column("Level", style="blue")
        metrics_table.add_column("Groups", style="magenta")

        metric_groups = defaultdict(list)
        for group, metrics in self.groups.items():
            for metric in metrics:
                metric_groups[metric].append(group)

        for metric_name, metric_info in self.metrics.items():
            kwargs_str = str(metric_info.get("kwargs", {})) if metric_info.get("kwargs") else "None"
            groups_str = ", ".join(metric_groups[metric_name]) or "None"
            level_str = metric_info.get("level", "batch")
            metrics_table.add_row(metric_name, metric_info["function"], kwargs_str, level_str, groups_str)

        console.print(metrics_table)

        if self.groups:
            groups_table = Table(title="Metric Groups")
            groups_table.add_column("Group Name", style="cyan")
            groups_table.add_column("Metrics", style="green")

            for group_name, metrics in self.groups.items():
                groups_table.add_row(group_name, ", ".join(metrics))

            console.print("\n", groups_table)

    def _validate_groups(self) -> None:
        """Validate metric groups configuration."""
        all_metrics = set(self.metrics.keys())
        invalid_groups = []

        for group_name, metrics in self.groups.items():
            unknown_metrics = set(metrics) - all_metrics
            if unknown_metrics:
                invalid_groups.append((group_name, unknown_metrics))
                logger.error(f"Unknown metrics in group '{group_name}': {unknown_metrics}")

        if invalid_groups:
            console.print(Panel("[red]⚠️ Group Configuration Errors[/]", expand=False))
            for group, unknown in invalid_groups:
                console.print(f"[red]✗[/] {group}: Unknown metrics: {unknown}")

    def _cache_docs(self) -> None:
        """Cache documentation for all metric functions."""
        for metric_name, metric_info in self.metrics.items():
            try:
                module_path, func_name = metric_info["function"].rsplit(".", 1)
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
                self._docs_cache[metric_name] = inspect.getdoc(func) or "No documentation available"
            except Exception as e:
                self._docs_cache[metric_name] = f"Documentation unavailable: {str(e)}"

    def add_metric(
        self,
        name: str,
        function_path: str,
        kwargs: Optional[Dict[str, Any]] = None,
        groups: Optional[List[str]] = None,
    ) -> None:
        """
        Add a new metric to the configuration.

        Args:
            name: Name of the metric
            function_path: Full import path to the metric function
            kwargs: Optional parameters for the metric function
            groups: Optional list of groups to add the metric to
        """
        metric_info = {"function": function_path}
        if kwargs:
            metric_info["kwargs"] = kwargs

        self.metrics[name] = metric_info

        if groups:
            for group in groups:
                if group not in self.groups:
                    self.groups[group] = []
                if name not in self.groups[group]:
                    self.groups[group].append(name)

        self._cache_docs()
        logger.info(f"Added metric: {name} with function {function_path}")
        self._display_metrics_summary()

    def add_group(self, name: str, metrics: List[str]) -> None:
        """Add a new metric group."""
        unknown_metrics = set(metrics) - set(self.metrics.keys())
        if unknown_metrics:
            raise ValueError(f"Unknown metrics in group: {unknown_metrics}")

        self.groups[name] = metrics
        logger.info(f"Added metric group: {name} with metrics {metrics}")
        self._display_metrics_summary()

    def get_group_metrics(self, group: str) -> Dict[str, Dict[str, Any]]:
        """Get all metrics configuration for a specific group."""
        if group not in self.groups:
            raise ValueError(f"Unknown group: {group}")

        return {metric: self.metrics[metric] for metric in self.groups[group]}

    def get_metric_docs(self, metric_name: Optional[str] = None) -> str:
        """
        Get documentation for metrics.

        Args:
            metric_name: Optional specific metric name. If None, returns all docs.
        """
        if metric_name:
            if metric_name not in self.metrics:
                raise ValueError(f"Unknown metric: {metric_name}")
            return self._docs_cache[metric_name]

        # Generate comprehensive documentation
        docs = []

        # Metrics documentation
        docs.append("# Metrics Documentation\n")
        for metric_name, doc in self._docs_cache.items():
            docs.append(f"## {metric_name}\n")
            docs.append(f"Function: `{self.metrics[metric_name]['function']}`\n")
            if self.metrics[metric_name].get("kwargs"):
                docs.append("\nParameters:\n")
                for k, v in self.metrics[metric_name]["kwargs"].items():
                    docs.append(f"- {k}: {v}\n")
            docs.append("\nDescription:\n")
            docs.append(f"{doc}\n\n")

        # Groups documentation
        if self.groups:
            docs.append("# Metric Groups\n")
            for group_name, metrics in self.groups.items():
                docs.append(f"## {group_name}\n")
                docs.append("Included metrics:\n")
                for metric in metrics:
                    docs.append(f"- {metric}\n")
                docs.append("\n")

        return "".join(docs)

    def display_docs(self, metric_name: Optional[str] = None) -> None:
        """Display formatted documentation in the console."""
        docs = self.get_metric_docs(metric_name)
        console.print(Markdown(docs))

    def items(self):
        """Return metrics items for compatibility with MetricRecorder."""
        return self.metrics.items()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {"metrics": self.metrics, "groups": self.groups}
