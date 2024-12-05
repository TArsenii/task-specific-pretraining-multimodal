import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .printing import get_console

console = get_console()


FONT_SIZE_TITLE = 26
FONT_SIZE_LABELS = 26
FONT_SIZE_TICKS = 24
FONT_SIZE_LEGEND = 22
FONT_SIZE_LEGEND_TITLE = 24
FONT_SIZE = 20


class ExperimentVisualiser:
    """Visualization module for multimodal experiment analysis."""

    # Okabe-Ito color scheme
    COLORS = [
        "#E69F00",  # orange
        "#56B4E9",  # light blue
        "#009E73",  # green
        "#F0E442",  # yellow
        "#0072B2",  # dark blue
        "#D55E00",  # red
        "#CC79A7",  # pink
        "#000000",  # black
    ]

    def __init__(self, output_dir: Path):
        """
        Initialize visualizer with output directory and style settings.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()

    def _setup_style(self):
        """Set up plotting style with LaTeX and Okabe-Ito colors."""
        sns.set_context("paper")
        sns.set_style("darkgrid")
        sns.set_palette(self.COLORS)

        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "axes.labelsize": FONT_SIZE_LABELS,
                "font.size": FONT_SIZE,
                "legend.fontsize": FONT_SIZE_LEGEND,
                "legend.title_fontsize": FONT_SIZE_LEGEND_TITLE,
                "xtick.labelsize": FONT_SIZE_TICKS,
                "ytick.labelsize": FONT_SIZE_TICKS,
                "figure.titlesize": FONT_SIZE_TITLE,
            }
        )

    def _save_figure(self, fig: Figure, name: str) -> Path:
        """
        Save figure and return the path.

        Args:
            fig: Figure to save
            name: Base name for the file

        Returns:
            Path to the saved figure
        """
        output_path = self.output_dir / f"{name}.pdf"
        os.makedirs(Path(output_path).parent, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        fig.savefig(str(output_path).replace("pdf", "png"), bbox_inches="tight", dpi=300)
        plt.close(fig)  # Close figure to free memory
        return output_path

    def plot_performance_distribution(
        self,
        df: pd.DataFrame,
        metric: str,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 8),
    ) -> Path:
        """Create violin plot with overlaid box plot and individual points."""
        fig, ax = plt.subplots(figsize=figsize)

        # Create violin plot
        sns.violinplot(data=df, x="Modality Availability", y=metric, ax=ax, inner=None, alpha=0.3)

        # Add box plot
        sns.boxplot(
            data=df,
            x="Modality Availability",
            y=metric,
            ax=ax,
            width=0.2,
            color="white",
            showfliers=False,
        )

        # Add individual points
        sns.stripplot(
            data=df,
            x="Modality Availability",
            y=metric,
            ax=ax,
            size=4,
            alpha=0.5,
            jitter=0.2,
        )

        # Styling
        ax.set_xlabel("Modality Availability", fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel(metric, fontsize=FONT_SIZE_LABELS)
        if title:
            ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=20)

        # Rotate x-labels if needed
        plt.xticks(rotation=45, ha="right")

        return self._save_figure(fig, f"performance_distribution_{metric}")

    def plot_paired_differences(
        self,
        df: pd.DataFrame,
        metric: str,
        reference_condition: str = "Full",
        figsize: Tuple[float, float] = (12, 8),
    ) -> Path:
        """Create paired difference plot showing impact of removing modalities."""
        fig, ax = plt.subplots(figsize=figsize)

        conditions = df["Modality Availability"].unique()
        differences = []
        labels = []

        reference_data = df[df["Modality Availability"] == reference_condition][metric].values

        # Collect differences and labels
        for condition in conditions:
            if condition != reference_condition and metric != "loss":
                condition_data = df[df["Modality Availability"] == condition][metric].values
                diff = condition_data - reference_data
                differences.append(diff)
                labels.append(f"{reference_condition} vs {condition}")

        # Calculate number of conditions for palette
        n_conditions = len(differences)
        # Select only the needed colors from the palette
        colors = self.COLORS[1 : n_conditions + 1]

        # Create box plot of differences
        sns.boxplot(
            data=differences,
            orient="h",
            ax=ax,
            palette=colors,  # Use subset of colors
            showfliers=False,
        )

        # Add individual points
        for i, diff in enumerate(differences):
            sns.stripplot(
                data=diff,
                orient="h",
                ax=ax,
                size=4,
                alpha=0.5,
                jitter=0.2,
                color=colors[i],  # Use matching color from subset
            )

        # Add reference line at 0
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Properly set ticks and labels
        ax.set_yticks(range(len(labels)))  # Set ticks explicitly
        ax.set_yticklabels(labels, fontsize=FONT_SIZE_LABELS)

        ax.set_xlabel(f"Difference in {metric}", fontsize=FONT_SIZE_LABELS)
        ax.set_title(f"Performance Impact Relative to {reference_condition}", fontsize=FONT_SIZE_TITLE, pad=20)

        return self._save_figure(fig, f"paired_differences_{metric}")

    def plot_metric_comparison(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        dataset: str,
        figsize: Tuple[float, float] = (10, 8),
        skip_metrics: List[str] = None,
    ) -> Path:
        """Create a plot showing performance across multiple metrics with averages as points."""
        if skip_metrics is None:
            skip_metrics = []
        self._setup_style()
        # Melt the DataFrame to get metrics in a single column
        melted_df = df.melt(
            id_vars=["split", "Modality Availability"], value_vars=metrics, var_name="Metric", value_name="Value"
        )

        # Calculate average values for each modality-metric combination
        avg_df = (
            melted_df.groupby(["Modality Availability", "Metric"], as_index=False)["Value"]
            .mean()
            .rename(columns={"Value": "Average Value"})
        )

        categories = list(avg_df["Modality Availability"].unique())
        console.print(categories)
        categories = sorted(categories, key=lambda x: len(x), reverse=True)
        console.print(categories)

        avg_df["Modality Availability"] = pd.Categorical(
            avg_df["Modality Availability"], categories=categories, ordered=True
        )

        # Define unique colors and markers for each modality
        modalities = avg_df["Modality Availability"].unique()
        colors = self.COLORS
        markers = ["o"]  # Variety of markers
        color_marker_dict = {
            modality: (colors[i % len(colors)], markers[i % len(markers)]) for i, modality in enumerate(modalities)
        }

        fig, ax = plt.subplots(figsize=figsize)
        avg_df = avg_df[~avg_df["Metric"].isin(skip_metrics)]

        for modality in categories:
            color, marker = color_marker_dict[modality]

            # Filter data for each modality
            modality_data = avg_df[avg_df["Modality Availability"] == modality]

            # Plot each modality's average points with a unique marker and color
            sns.scatterplot(
                data=modality_data,
                x="Metric",
                y="Average Value",
                ax=ax,
                color=color,
                alpha=0.85,
                marker=marker,
                edgecolor="black",
                s=150,  # Size of the points
                label=modality,
            )

        # Set y-axis limit and labels
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=FONT_SIZE_LABELS)
        ax.set_xticklabels(
            [self._format_metric_name(m) for m in metrics if m not in skip_metrics], fontsize=FONT_SIZE_TICKS
        )

        # Add legend below the plot
        ax.legend(title="Modality Availability", loc="lower center", ncol=len(modalities), fontsize=FONT_SIZE_LEGEND)

        # Add title and layout adjustments
        plt.suptitle(rf"\textbf{{{dataset}}}")
        plt.title(r"\textbf{Metrics Per Modality Availabilty Condition}", fontsize=FONT_SIZE_TITLE)
        # plt.tight_layout()
        return self._save_figure(fig, f"{dataset}_metric_comparison")

    def _format_metric_name(self, metric: str) -> str:
        metric = metric.lower()
        metric_map = {
            "accuracy": "Accuracy",
            "balanced_accuracy": "Bal.\nAcc.",
            "recall_micro": r"$\begin{array}{c}\mathrm{Recall} \\ \mathrm{Micro}\end{array}$",
            "recall_macro": r"$\begin{array}{c}\mathrm{Recall} \\ \mathrm{Macro}\end{array}$",
            "recall_weighted": r"$\begin{array}{c}\mathrm{Recall} \\ \mathrm{Weighted}\end{array}$",
            "precision_micro": r"$\begin{array}{c}\mathrm{Prec.} \\ \mathrm{Micro}\end{array}$",
            "precision_macro": r"$\begin{array}{c}\mathrm{Prec.} \\ \mathrm{Macro}\end{array}$",
            "precision_weighted": r"$\begin{array}{c}\mathrm{Precision} \\ \mathrm{Weighted}\end{array}$",
            "f1_micro": r"$\begin{array}{c}\mathrm{F1} \\ \mathrm{Micro}\end{array}$",
            "f1_macro": r"$\begin{array}{c}\mathrm{F1} \\ \mathrm{Macro}\end{array}$",
            "f1_weighted": r"$\begin{array}{c}\mathrm{F1} \\ \mathrm{Weighted}\end{array}$",
        }

        return metric_map.get(metric, metric)

    def plot_significance_matrix(
        self, analysis_results: Dict, metric: str, figsize: Tuple[float, float] = (8, 6)
    ) -> Path:
        """Create heatmap showing statistical significance and effect sizes."""
        friedman_test = analysis_results[metric].get("friedman_test")
        pairwise_tests = analysis_results[metric].get("pairwise_tests", {})

        conditions = sorted(list(set(cond1 for comp in pairwise_tests.keys() for cond1 in comp.split("_vs_"))))
        condition_index = {cond: idx for idx, cond in enumerate(conditions)}
        n_conditions = len(conditions)

        # Initialize matrices
        p_values = np.full((n_conditions, n_conditions), np.nan)
        effect_sizes = np.full((n_conditions, n_conditions), np.nan)

        # Populate matrices symmetrically
        for comparison, test_results in pairwise_tests.items():
            cond1, cond2 = comparison.split("_vs_")
            i, j = condition_index[cond1], condition_index[cond2]

            p_values[i, j] = test_results["p_value"]
            p_values[j, i] = test_results["p_value"]
            effect_sizes[i, j] = abs(test_results["cohens_d"])
            effect_sizes[j, i] = abs(test_results["cohens_d"])

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot pairwise p-values heatmap
        sns.heatmap(
            p_values,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            ax=ax1,
            xticklabels=conditions,
            yticklabels=conditions,
            mask=np.isnan(p_values),
        )
        ax1.set_title("Pairwise p-values", fontsize=FONT_SIZE_LABELS)

        # Rotate labels
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

        # Plot pairwise effect sizes heatmap
        sns.heatmap(
            effect_sizes,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=ax2,
            xticklabels=conditions,
            yticklabels=conditions,
            mask=np.isnan(effect_sizes),
        )
        ax2.set_title("Cohen's |d|", fontsize=FONT_SIZE_LABELS)

        # Rotate labels
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

        # Add Friedman test result if available
        if friedman_test:
            fig.suptitle(
                f"Statistical Analysis for {metric}\n" f"Friedman Test p-value: {friedman_test['p_value']:.3f}",
                fontsize=FONT_SIZE_TITLE,
                y=1.05,
            )

        plt.tight_layout()

        return self._save_figure(fig, f"significance_matrix_{metric}")

    def plot_forest_significance(
        self, analysis_results: Dict, metric: str, figsize: Tuple[float, float] = (8, 10)
    ) -> Path:
        """Create a forest plot showing statistical significance and effect sizes."""
        pairwise_tests = analysis_results[metric].get("pairwise_tests", {})

        # Extract conditions and prepare data for plotting
        comparisons = []
        effect_sizes = []
        p_values = []

        for comparison, test_results in pairwise_tests.items():
            cond1, cond2 = comparison.split("_vs_")
            comparisons.append(f"{cond1} vs {cond2}")
            effect_sizes.append(test_results["cohens_d"])
            p_values.append(test_results["p_value"])

        # Sort by effect size magnitude
        sorted_indices = np.argsort(np.abs(effect_sizes))
        comparisons = [comparisons[i] for i in sorted_indices]
        effect_sizes = [effect_sizes[i] for i in sorted_indices]
        p_values = [p_values[i] for i in sorted_indices]

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        y_positions = np.arange(len(comparisons))

        # Plot effect sizes as points with lines (optional CI, here assumed to be absent)
        ax.errorbar(effect_sizes, y_positions, xerr=None, fmt="o", color="black", label="Cohen's d")

        # Add p-values as text annotations next to each point
        for i, (p_val, effect_size) in enumerate(zip(p_values, effect_sizes)):
            ax.text(effect_size, i, f"p={p_val:.3f}", va="center", ha="left", fontsize=10)

        # Set labels and title
        ax.set_yticks(y_positions)
        ax.set_yticklabels(comparisons)
        ax.invert_yaxis()  # Invert y-axis to have the largest effect size at the top
        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.set_title(f"Pairwise Significance and Effect Sizes for {metric}")

        plt.tight_layout()
        return self._save_figure(fig, f"forest_significance_{metric}")

    def plot_validation_over_epochs(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        split,
    ) -> List[Path]:
        """Plot metrics across training epochs."""
        file_paths = []

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 8))

            for condition in df["Modality Availability"].unique():
                condition_data = df[df["Modality Availability"] == condition]
                ax.plot(condition_data["epoch"], condition_data[metric], label=condition, alpha=0.8)

            ax.set_xlabel("Epoch", fontsize=FONT_SIZE_LABELS)
            ax.set_ylabel(metric, fontsize=FONT_SIZE_LABELS)
            ax.set_title(f"{metric} Across Training", fontsize=FONT_SIZE_TITLE, pad=20)
            ax.legend(fontsize=FONT_SIZE_LEGEND)

            file_paths.append(self._save_figure(fig, f"{split}/across_training_{metric}"))

        return file_paths

    def create_all_visualizations(
        self,
        results: pd.DataFrame,
        metrics: List[str],
        reference_condition: str,
        analysis_results: dict,
        dataset: str,
        skip_metrics: List[str] = None,
    ) -> Dict[str, Dict[str, Path]]:
        """Create all visualizations for the experiment results."""
        visualization_paths = {
            "distributions": {},
            "paired_differences": {},
            "significance_matrices": {},
            "metric_comparison": None,
        }

        # console.start_task("Metrics", total=len(metrics))

        # # Create individual metric visualizations
        # for metric in metrics:
        #     if metric == "loss":
        #         console.update_task("Metrics", advance =1)
        #         continue

        #     console.print(f"[bold green]{metric}[/] - Plotting performance distributions")
        #     # Performance distributions
        #     viz_path = self.plot_performance_distribution(
        #         df=results,
        #         metric=metric,
        #         title=f"Performance Distribution for {metric}"
        #     )
        #     visualization_paths["distributions"][metric] = viz_path

        #     console.print(f"[bold green]{metric}[/] - Plotting pairwise differences")
        #     # Paired differences
        #     viz_path = self.plot_paired_differences(
        #         df=results,
        #         metric=metric,
        #         reference_condition=reference_condition,
        #     )
        #     visualization_paths["paired_differences"][metric] = viz_path

        #     # console.print(f"[bold green]{metric}[/] - Plotting Significance Matrices")
        #     # # Significance matrices
        #     # viz_path = self.plot_forest_significance(
        #     #     analysis_results=analysis_results,
        #     #     metric=metric
        #     # )
        #     # visualization_paths["significance_matrices"][metric] = viz_path
        #     console.update_task("Metrics", advance =1)
        # console.complete_task("Metrics")
        # # Create multi-metric comparison

        viz_path = self.plot_metric_comparison(
            df=results, metrics=[m for m in metrics if m != "loss"], dataset=dataset, skip_metrics=skip_metrics
        )
        visualization_paths["metric_comparison"] = viz_path

        return visualization_paths
