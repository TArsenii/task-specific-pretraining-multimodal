import json
import os
import subprocess
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from .logging import get_logger
from .printing import get_console

ModelName = str
PathLike = Union[str, Path]


@dataclass
class ExperimentReport:
    """
    A utility class for storing and managing the final results of an experiment.

    This class provides a structured way to store experiment results and metadata,
    with methods for serialization and deserialization.
    """

    model_size_mb: Union[float, Dict[ModelName, float]]
    model_parameter_count: Union[int, Dict[ModelName, int]]
    batch_size: int
    optimizer_info: Dict[str, Any]
    # criterion_info: Dict[str, Any]
    confusion_matrices_path: PathLike = field(default=None)
    train_dataset_size: int = field(default=-1)
    validation_dataset_size: int = field(default=-1)
    test_dataset_size: int = field(default=-1)
    epochs: int = field(default=1)
    avg_training_time: Optional[float] = field(default=None)
    avg_inference_time: Optional[float] = field(default=None)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate input data after initialization."""
        # self._validate_paths()
        # self._validate_numeric_values()
        pass

    def _validate_numeric_values(self) -> None:
        """Validate numeric attributes."""
        for attr in [
            "train_dataset_size",
            "validation_dataset_size",
            "test_dataset_size",
            "epochs",
        ]:
            value = getattr(self, attr)
            if not isinstance(value, int) or value < -1:
                raise ValueError(f"{attr} must be a non-negative integer or -1")

        for attr in ["avg_training_time", "avg_inference_time"]:
            value = getattr(self, attr)
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                raise ValueError(f"{attr} must be a non-negative number or None")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ExperimentReport to a dictionary."""
        return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in self.__dict__.items()}

    def to_json(self, fp: PathLike) -> None:
        """
        Serialize the ExperimentReport to a JSON file.

        Args:
            fp (PathLike): The file path to write the JSON to.

        Raises:
            IOError: If there's an error writing to the file.
        """
        try:
            with open(fp, "w") as f:
                json.dump({k: str(v) for k, v in self.to_dict().items()}, f, indent=2)
        except IOError as e:
            raise IOError(f"Error writing to JSON file: {e}")

    def to_yaml(self, fp: PathLike) -> None:
        """
        Serialize the ExperimentReport to a YAML file.

        Args:
            fp (PathLike): The file path to write the YAML to.

        Raises:
            IOError: If there's an error writing to the file.
        """
        try:
            with open(fp, "w") as f:
                yaml.dump(self.to_dict(), f)
        except IOError as e:
            raise IOError(f"Error writing to YAML file: {e}")

    def to_text(self, fp: PathLike) -> None:
        """
        Serialize the ExperimentReport to a plain text file.

        Args:
            fp (PathLike): The file path to write the text to.

        Raises:
            IOError: If there's an error writing to the file.
        """
        try:
            with open(fp, "w") as f:
                for key, value in self.to_dict().items():
                    f.write(f"{key}: {value}\n")
        except IOError as e:
            raise IOError(f"Error writing to text file: {e}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentReport":
        """
        Create an ExperimentReport instance from a dictionary.

        Args:
            d (Dict[str, Any]): A dictionary containing ExperimentReport attributes.

        Returns:
            ExperimentReport: An instance of ExperimentReport.

        Raises:
            ValueError: If the dictionary contains invalid data.
        """
        try:
            if "timestamp" in d:
                d["timestamp"] = datetime.fromisoformat(d["timestamp"])
            return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data in dictionary: {e}")

    @classmethod
    def from_json(cls, fp: PathLike) -> "ExperimentReport":
        """
        Create an ExperimentReport instance from a JSON file.

        Args:
            fp (PathLike): The file path to read the JSON from.

        Returns:
            ExperimentReport: An instance of ExperimentReport.

        Raises:
            IOError: If there's an error reading the file.
            ValueError: If the JSON data is invalid.
        """
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        except IOError as e:
            raise IOError(f"Error reading JSON file: {e}")

    @classmethod
    def from_yaml(cls, fp: PathLike) -> "ExperimentReport":
        """
        Create an ExperimentReport instance from a YAML file.

        Args:
            fp (PathLike): The file path to read the YAML from.

        Returns:
            ExperimentReport: An instance of ExperimentReport.

        Raises:
            IOError: If there's an error reading the file.
            ValueError: If the YAML data is invalid.
        """
        try:
            with open(fp, "r") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file: {e}")
        except IOError as e:
            raise IOError(f"Error reading YAML file: {e}")


logger = get_logger()
console = get_console()


class LatexReport:
    """Handles LaTeX document creation and compilation."""

    def __init__(self, title: str, author: str = "Experiment Report"):
        self.content = []
        self.preamble = [
            "\\documentclass{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{graphicx}",
            "\\usepackage{booktabs}",
            "\\usepackage{float}",
            "\\usepackage{subcaption}",
            "\\usepackage{geometry}",
            "\\usepackage{hyperref}",
            "\\geometry{margin=2.5cm}",
            f"\\title{{{title}}}",
            f"\\author{{{author}}}",
            "\\date{\\today}",
        ]

    def add_section(self, title: str):
        """Add a section to the report."""
        self.content.append(f"\\section{{{title}}}")

    def add_subsection(self, title: str):
        """Add a subsection to the report."""
        self.content.append(f"\\subsection{{{title}}}")

    def add_figure(self, path: str, caption: str, label: str, width: str = "0.8\\textwidth"):
        """Add a figure to the report."""
        self.content.extend(
            [
                "\\begin{figure}[H]",
                "\\centering",
                f"\\includegraphics[width={width}]{{{path}}}",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\end{figure}",
            ]
        )

        console.print(f"[green]✓[/] Added figure: {path}")
        logger.info(f"Added figure: {path}")

    def add_table(self, df: pd.DataFrame, caption: str, label: str):
        """Add a table to the report using pandas dataframe."""
        # First convert to latex
        latex_table = df.to_latex(
            index=True,
            float_format=lambda x: "{:.4f}".format(x),
            caption=caption,
            label=f"tab:{label}",  # Adding 'tab:' prefix for consistent label formatting
            escape=True,
        )

        # Add centering command by inserting after the begin{table} line
        latex_table = latex_table.replace(r"\begin{table}", r"\begin{table}[h]" + "\n" + r"\centering")

        self.content.append(latex_table)

    def compile(self, output_path: Path) -> Path:
        """Compile the LaTeX document to PDF."""
        tex_content = "\n".join(self.preamble + ["\\begin{document}", "\\maketitle", *self.content, "\\end{document}"])
        tex_file = output_path.with_suffix(".tex")
        with open(tex_file, "w") as f:
            f.write(tex_content)

        # Compile twice for proper references

        console.print(f"Compiling LaTeX report to PDF ({output_path.parent})...")
        for _ in range(2):
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    f"{output_path.parent}",
                    f"{tex_file}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # Clean up extra latex files
        for ext in [".aux", ".log", ".out"]:
            extra_file = output_path.with_suffix(ext)
            console.print(f"Cleaning up {extra_file}")

            if extra_file.exists():
                extra_file.unlink()
                console.print(f"[green]✓[/] Deleted: {extra_file}")
            else:
                console.print(f"[yellow]✗[/] File not found: {extra_file}")
        return output_path.with_suffix(".pdf")

    def add_text(self, text: str):
        """Add plain text to the report."""
        self.content.append(text)


class SubReport:
    """Base class for individual report components."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the sub-report content."""
        raise NotImplementedError


class MetricsReport(SubReport):
    """Handles metric visualization and reporting."""

    def __init__(
        self,
        output_dir: Path,
        plot_fn: Optional[Callable] = None,
        metric_keys: Optional[List[str]] = None,
    ):
        super().__init__(output_dir)
        self.plot_fn = plot_fn
        self.metric_keys = metric_keys

    def generate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        dfs = []

        confusion_matrices = defaultdict(lambda: defaultdict(list))

        for key, metrics_data in data["metrics_history"].items():
            if len(metrics_data) == 0:
                continue

            if isinstance(metrics_data, dict):
                metrics_data = [metrics_data]
            confusion_keys = [k for k in metrics_data[0].keys() if "ConfusionMatrix" in k]

            for k in confusion_keys:
                for i, metrics in enumerate(metrics_data):
                    confusion_matrices[key][k].append(metrics[k])

                for data in metrics_data:
                    data.pop(k)

            if key == "test":
                df = pd.DataFrame(metrics_data, index=[0])
            else:
                df = pd.DataFrame(metrics_data)
            df["split"] = key
            dfs.append(df)

        metrics_df = pd.concat(dfs, ignore_index=True)

        # Generate plots
        if self.plot_fn:
            plot_path = self.output_dir / "metrics_plot.pdf"
            self.plot_fn(metrics_df, plot_path)
        else:
            plot_path = None

        train = metrics_df[metrics_df["split"] == "train"].reset_index()
        train["Epoch"] = range(1, len(train) + 1)
        validation = metrics_df[metrics_df["split"] == "validation"].reset_index()
        validation["Epoch"] = range(1, len(validation) + 1)
        test = metrics_df[metrics_df["split"] == "test"].reset_index()

        split_train = {}
        split_validation = {}
        split_test = {}

        if len(train) > 0:
            train.to_json(self.output_dir / "train_metrics.json", orient="records")
            split_train = split_missing_conditions(train)
        if len(validation) > 0:
            validation.to_json(self.output_dir / "validation_metrics.json", orient="records")
            split_validation = split_missing_conditions(validation)
        if len(test) > 0:
            test.to_json(self.output_dir / "test_metrics.json", orient="records")
            split_test = split_missing_conditions(test)

        return {
            "metrics_df": {
                "train": split_train,
                "validation": split_validation,
                "test": split_test,
            },
            "confusion_matrices": confusion_matrices,
            "plot_path": plot_path,
        }


def split_missing_conditions(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # 1 Drop columns with any NaN values
    df = df.dropna(how="any", axis=1)

    ignore_columns = ["split", "loss"]

    unique_conditions = set([x.split("_")[-1] for x in df.columns if x not in ignore_columns])
    # sort the conditions by length and then alphabetically
    unique_conditions = sorted(unique_conditions, key=lambda x: (len(x), x))

    # 2 Split the dataframe by condition
    split_dfs = {}
    for condition in unique_conditions:
        matching_columns = [x for x in df.columns if x.endswith(f"_{condition}")]
        split_df = df[["split", "loss"] + matching_columns]
        split_df = split_df.rename(columns={x: x.replace(f"_{condition}", "") for x in matching_columns})
        split_dfs[condition] = split_df
    return split_dfs


class EmbeddingVisualizationReport(SubReport):
    """Handles embedding visualization."""

    def __init__(
        self,
        output_dir: Path,
        visualization_fn: Callable,
        preprocessing_fn: Optional[Callable] = lambda _: _,
    ):
        super().__init__(output_dir)
        self.visualization_fn = visualization_fn
        self.preprocessing_fn = preprocessing_fn

    def generate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        embeddings = data["embeddings"]

        if self.preprocessing_fn:
            embeddings = self.preprocessing_fn(embeddings)

        plot_path = self.output_dir / "embeddings.pdf"
        metadata = self.visualization_fn(embeddings, plot_path)

        return {"plot_path": plot_path, "metadata": metadata}


class ModelReport(SubReport):
    """Handles model information reporting."""

    def generate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        _data = data.get("model_info", {})

        if len(_data) == 0:
            logger.warning("No model information found in the experiment data.")
            console.warning("No model information found in the experiment data.")
            return {}

        model_info = {
            "parameters": _data["parameters"],
            "size": _data["size"],
            "architecture": _data["architecture"],
        }

        return model_info


class TimingReport(SubReport):
    """Handles timing statistics."""

    def generate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        timing_data = data.get("timing_history", {})
        timing_data = {k: v for k, v in timing_data.items() if len(v) > 0}

        timing_data = {k: np.mean(v) for k, v in timing_data.items()}

        timing_df = pd.DataFrame(timing_data, index=[0])
        csv_path = self.output_dir / "timing.csv"
        timing_df.to_csv(csv_path, index=False)

        summary = {f"{x}_time": timing_df[x].item() for x in timing_df.columns}

        return {"timing_df": timing_df, "summary": summary, "csv_path": csv_path}


class ExperimentReportGenerator:
    """Main report generator that coordinates sub-reports and generates final PDF."""

    def __init__(self, output_dir: Path, config: Any, subreports: Dict[str, SubReport]):
        self.output_dir = Path(output_dir)
        self.config = config
        self.subreports = subreports

    def generate_report(self, experiment_data: Dict[str, Any]) -> Path:
        """Generate complete experiment report."""
        # Generate all sub-reports
        report_components = {}
        for name, subreport in self.subreports.items():
            try:
                report_components[name] = subreport.generate(experiment_data)
                logger.info(f"Generated {name} report")
            except Exception as e:
                logger.error(f"Error generating {name} report: {e}")
                console.print(f"[red]✗[/] Failed to generate {name} report due to {traceback.format_exc()}")
                exit(1)

        # Create LaTeX report
        latex_report = LatexReport(
            title=f"Experiment Report: {self.config.experiment.name.replace('_', ' ')}",
            author=f"Run ID: {self.config.experiment.run_id}",
        )

        # Add configuration section
        latex_report.add_section("Experiment Configuration")

        import re

        def escape_latex(text, is_code_block=False):
            """
            Escape special characters in a LaTeX string.
            """
            special_chars = {
                "\\": r"\\textbackslash ",
                "&": r"\&",
                "%": r"\%",
                "$": r"\$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
                "~": r"\textasciitilde ",
                "^": r"\textasciicircum ",
            }

            # Replace special characters
            for char, escaped_char in special_chars.items():
                text = text.replace(char, escaped_char)

            # Handle line breaks differently for code blocks vs inline text
            if is_code_block:
                text = re.sub(r"\n+", r"\\\\ ", text)  # Double backslash for LaTeX line break in code block
            else:
                text = re.sub(r"\n+", r"\\par ", text)  # Paragraph break for inline sections

            return text

        formatted_latex = escape_latex(str(self.config))

        latex_report.add_text(formatted_latex)

        # Add metrics section
        if "metrics" in report_components:
            latex_report.add_section("Performance Metrics")
            metrics_data = report_components["metrics"]

            for k, m_dfs in metrics_data["metrics_df"].items():
                for cond, cond_df in m_dfs.items():
                    latex_report.add_table(
                        cond_df,
                        f"{k}-{cond}-Metrics",
                        f"tab:{k}_{cond}_metrics",
                    )
                    if metrics_data["plot_path"] and os.path.exists(metrics_data["plot_path"]):
                        latex_report.add_figure(
                            metrics_data["plot_path"],
                            "Metrics Evolution",
                            "fig:metrics",
                        )
            # Add confusion matrices as plots
            for split in metrics_data["confusion_matrices"]:
                np.save(
                    self.config.logging.metrics_path / f"confusion_matrices_{split}.npy",
                    metrics_data["confusion_matrices"][split],
                )
                console.print(
                    f"[green]✓[/] Saved confusion matrices for {split} to {self.config.logging.metrics_path / f'confusion_matrices_{split}.npy'}"
                )
                # confusion_plot = plot_confusion_matrices(
                #     metrics_data["confusion_matrices"][split],
                #     self.config.logging.metrics_path / f"confusion_matrices_{split}.pdf",
                # )

                # latex_report.add_figure(
                #     confusion_plot,
                #     f"Confusion Matrices ({split})",
                #     label=f"fig:confusion_{split}",
                #     width="0.9\\textwidth",
                # )

        # Add embedding visualization
        if "embeddings" in report_components and os.path.exists(report_components["embeddings"]["plot_path"]):
            latex_report.add_section("Embedding Visualization")
            latex_report.add_figure(
                report_components["embeddings"]["plot_path"],
                "Embedding Space Visualization",
                "fig:embeddings",
            )

        # Add timing information
        if "timing" in report_components:
            latex_report.add_section("Timings")
            timing_data = report_components["timing"]
            latex_report.add_table(timing_data["timing_df"], "Training and Inference Timing", "tab:timing")

        # Add model information
        if "model" in report_components:
            latex_report.add_section("Model Information")
            model_data = report_components["model"]
            latex_report.add_text(
                escape_latex(
                    f"Model Architecture: {model_data['architecture']}\n"
                    f"Model Size: {model_data['size']:.2f} MB\n"
                    f"Model Parameters: {model_data['parameters']:.2E}"
                )
            )
        # Compile report
        pdf_path = self.output_dir / "experiment_report.pdf"
        final_path = latex_report.compile(pdf_path)

        # Create ExperimentReport instance
        experiment_report = ExperimentReport(
            model_size_mb=report_components["model"]["size"],
            model_parameter_count=report_components["model"]["parameters"],
            batch_size=",".join(f"{k}:{v.batch_size}" for k, v in self.config.data.datasets.items()),
            optimizer_info=self.config.training.optimizer,
            # criterion_info=self.config.training.criterion_kwargs,
            train_dataset_size=len(experiment_data["metrics_history"]["train"]),
            validation_dataset_size=len(experiment_data["metrics_history"]["validation"]),
            test_dataset_size=len(experiment_data["metrics_history"].get("test", [])),
            epochs=self.config.training.epochs,
            avg_training_time=report_components["timing"]["summary"].get("train_time", "-1.0"),
            # avg_validation_time=report_components["timing"]["summary"].get("validation_time", "-1.0"),
            avg_inference_time=report_components["timing"]["summary"].get("test_time", "-1.0"),
        )

        # Save experiment report in different formats
        experiment_report.to_json(self.output_dir / "experiment_report.json")
        experiment_report.to_yaml(self.output_dir / "experiment_report.yaml")

        return final_path

    def __str__(self) -> str:
        """Return string representation of the ExperimentReportGenerator."""
        subreport_names = ", ".join(sorted(self.subreports.keys()))
        return f"ExperimentReportGenerator(" f"output_dir='{self.output_dir}', " f"subreports=[{subreport_names}])"


# def plot_confusion_matrices(data: Dict[str, NDArray], save_to: Union[str, Path], vertical: bool = False) -> Path:
#     """
#     Plot confusion matrices using seaborn with LaTeX formatting and colorblind-friendly colors.
#     Uses a shared colorbar for all matrices.

#     Parameters
#     ----------
#     data : Dict[str, NDArray]
#         Dictionary mapping model names to their confusion matrices
#     save_to : Union[str, Path]
#         Path where the figure should be saved
#     vertical : bool, optional
#         If True, stack matrices vertically. If False, arrange horizontally

#     Returns
#     -------
#     Figure
#         The matplotlib figure object containing the plots
#     """
#     # Set up LaTeX rendering
#     plt.rcParams.update(
#         {
#             "text.usetex": True,
#             "font.family": "serif",
#             "font.serif": ["Computer Modern Roman"],
#             "text.latex.preamble": r"\usepackage{amsmath}",
#         }
#     )

#     # Okabe-Ito color palette (colorblind friendly)
#     colors = [
#         "#E69F00",
#         "#56B4E9",
#         "#009E73",
#         "#F0E442",
#         "#0072B2",
#         "#D55E00",
#         "#CC79A7",
#         "#000000",
#     ]

#     # Calculate layout
#     n_matrices = len(data)
#     if vertical:
#         n_rows = n_matrices
#         n_cols = 1
#         fig_width = 12
#         fig_height = 4 * n_matrices + 0.5  # Extra space for colorbar
#     else:
#         n_rows = 1
#         n_cols = n_matrices
#         fig_width = min(14, 4.5 * n_matrices)  # Scale width by number of matrices, max two-column width
#         fig_height = 4.5  # Fixed height plus space for colorbar

#     # Create figure with space for colorbar
#     fig = plt.figure(figsize=(fig_width, fig_height))

#     # Create a gridspec layout
#     gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.15)

#     # Create subplot for matrices
#     matrix_axes = gs[0].subgridspec(n_rows, n_cols, hspace=0.4, wspace=0.2)

#     # Create normalized matrices and find global min/max for consistent scale
#     # normalized_matrices = {}
#     # for name, matrix in data.items():
#     #     normalized_matrices[name] = (
#     # matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]
#     # )

#     # Plot each confusion matrix
#     for idx, (name, matrix) in enumerate(data.items()):
#         if vertical:
#             ax = fig.add_subplot(matrix_axes[idx, 0])
#         else:
#             ax = fig.add_subplot(matrix_axes[0, idx])

#         # Create heatmap without colorbar
#         sns.heatmap(
#             matrix,
#             ax=ax,
#             cmap=sns.color_palette(colors, as_cmap=True),
#             square=True,
#             annot=True,
#             cbar=False,
#             vmin=0,
#             vmax=1,
#             fmt=".2E",
#         )

#         # Customize appearance
#         ax.set_title(f"{name.replace('_', ' ')}", pad=10)
#         ax.set_xlabel("Predicted Label")
#         ax.set_ylabel("True Label")

#         # Set tick positions and labels
#         n_classes = matrix.shape[0]
#         tick_positions = np.arange(n_classes) + 0.5

#         if n_classes <= 10:
#             ax.set_xticks(tick_positions)
#             ax.set_yticks(tick_positions)
#             ax.set_xticklabels([f"${i}$" for i in range(n_classes)])
#             ax.set_yticklabels([f"${i}$" for i in range(n_classes)])

#     # # Add horizontal colorbar at the bottom
#     # cbar_ax = fig.add_subplot(gs[1])
#     # norm = mpl.colors.Normalize(vmin=0, vmax=1)
#     # cbar = mpl.colorbar.ColorbarBase(
#     #     cbar_ax,
#     #     cmap=sns.color_palette(colors, as_cmap=True),
#     #     norm=norm,
#     #     orientation="horizontal",
#     #     label="Normalized Count",
#     # )

#     # Save figure
#     save_path = Path(save_to)
#     fig.savefig(save_path, bbox_inches="tight", dpi=300)

#     return save_path
