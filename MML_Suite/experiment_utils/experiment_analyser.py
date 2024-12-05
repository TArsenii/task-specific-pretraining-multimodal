import json
import re
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare
from rich.table import Table
from .experiment_visualiser import ExperimentVisualiser
from .logging import get_logger
from .printing import get_console

console = get_console()
logger = get_logger()


class Mode(Enum):
    DICT = "dict"
    DATAFRAME = "df"

    def __str__(self) -> str:
        return str(self.value)


class Split(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"

    def __str__(self) -> str:
        return str(self.value)

    def resolve_split_metrics(self, metrics_root: PathLike) -> PathLike:
        return metrics_root / f"{self.value}_metrics.json"


class ExperimentAnalyser:
    def __init__(self, experiment_root: PathLike, confidence_level: float = 0.95):
        self.experiment_root = Path(experiment_root)
        self.metrics_root = self.experiment_root / "metrics"
        self.confidence_level = confidence_level
        self.visualiser = ExperimentVisualiser(self.metrics_root / "visualisations")
        self.console = get_console()
        self.logger = get_logger()

    def _load_metrics_from_json(self, json_fp: PathLike, mode: Mode = Mode.DATAFRAME) -> Union[pd.DataFrame, Dict]:
        """Load metrics from a JSON file."""
        self.logger.debug(f"Loading metrics from {json_fp} - Mode = {mode}")
        self.console.print(f"[cyan]Loading metrics {json_fp} - Mode: {mode}[/]")

        try:
            with open(json_fp, "r") as json_f:
                data = json.load(json_f)
        except FileNotFoundError as err:
            self.console.print(f"[bold red]✘ Failed to load {json_fp} due to {err}[/]")
            self.logger.error(f"Failed to load {json_fp} due to {err}")
            raise err

        return pd.DataFrame(data) if mode == Mode.DATAFRAME else data

    def _split_modality_availability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split the dataframe by modality availability."""
        ignore_columns = ["split", "loss", "epoch"]

        unique_availabilities = set([x.split("_")[-1] for x in df.columns if x not in ignore_columns])
        self.console.print(f"Available modalities: {', '.join(unique_availabilities)}")

        dfs = []
        for availability in unique_availabilities:
            matching_columns = [x for x in df.columns if x.endswith(f"_{availability}")]
            split_df = df[["split"] + (["epoch"] if "epoch" in df.columns else []) + ["loss"] + matching_columns].copy()
            split_df["Modality Availability"] = availability

            split_df = split_df.rename(columns={x: x.replace(f"_{availability}", "") for x in matching_columns})
            dfs.append(split_df)

        return pd.concat(dfs)

    def _find_run_directories(self) -> List[Path]:
        """Find all run directories in the metrics root."""
        if not self.metrics_root.exists():
            raise FileNotFoundError(f"Metrics root {self.metrics_root} does not exist")

        directories = [path for path in self.metrics_root.iterdir() if path.is_dir() and re.match(r"\d+", path.name)]

        self.console.print(f"Found {len(directories)} run directories")
        return directories

    def _process_train_data(self, run_directories: List[Path]) -> pd.DataFrame:
        """Process training data with epoch information."""
        run_data = []

        for run_dir in run_directories:
            metrics_file = run_dir / "train_metrics.json"
            df = self._load_metrics_from_json(metrics_file)
            if "epoch" not in df.columns:
                df["epoch"] = range(len(df))
            run_data.append(df)

        combined_df = pd.concat(run_data)
        return self._split_modality_availability(combined_df)

    def _process_validation_data(self, run_directories: List[Path]) -> pd.DataFrame:
        """Process validation data with epoch information."""
        run_data = []

        for run_dir in run_directories:
            metrics_file = run_dir / "validation_metrics.json"
            df = self._load_metrics_from_json(metrics_file)
            if "epoch" not in df.columns:
                df["epoch"] = range(len(df))
            run_data.append(df)

        combined_df = pd.concat(run_data)
        return self._split_modality_availability(combined_df)

    def _process_test_data(self, run_directories: List[Path]) -> pd.DataFrame:
        """Process test data (no epochs)."""
        run_data = []

        for run_dir in run_directories:
            metrics_file = run_dir / "test_metrics.json"
            df = self._load_metrics_from_json(metrics_file)
            if "index" in df.columns:
                df = df.drop(columns=["index"])
            run_data.append(df)

        combined_df = pd.concat(run_data)
        split_df = self._split_modality_availability(combined_df)

        # Calculate aggregate statistics
        agg_df = split_df.groupby(["split", "Modality Availability"]).agg(["mean", "std"]).reset_index()
        return split_df, agg_df

    def analyze_results(self, df: pd.DataFrame, metric_columns: List[str]) -> Dict:
        """Analyze experimental results with statistical measures."""
        results = {}

        for metric in metric_columns:
            metric_results = {}
            conditions = df["Modality Availability"].unique()

            # Calculate statistics for each condition
            for condition in conditions:
                condition_data = df[df["Modality Availability"] == condition][metric]

                # Basic statistics
                mean = np.mean(condition_data)
                std = np.std(condition_data, ddof=1)
                n = len(condition_data)

                # Confidence interval
                confidence_interval = stats.t.interval(
                    self.confidence_level,
                    df=n - 1,
                    loc=mean,
                    scale=stats.sem(condition_data),
                )

                metric_results[condition] = {
                    "mean": mean,
                    "std": std,
                    "n": n,
                    "sem": stats.sem(condition_data),
                    "ci_lower": confidence_interval[0],
                    "ci_upper": confidence_interval[1],
                    "raw_data": condition_data.values,
                }

            # Perform pairwise significance tests
            metric_results["pairwise_tests"] = self._compute_pairwise_tests(conditions, metric_results)

            # Perform Friedman Test for repeated measures across conditions
            if metric != "loss":
                try:
                    # Gather metric values across conditions for the Friedman test
                    condition_data = [df[df["Modality Availability"] == cond][metric].values for cond in conditions]
                    statistic, p_value = friedmanchisquare(*condition_data)

                    metric_results["friedman_test"] = {
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": p_value < (1 - self.confidence_level),
                    }
                except Exception as e:
                    self.logger.error(f"Failed to perform Friedman test on {metric}: {e}")
                    metric_results["friedman_test"] = None

            results[metric] = metric_results

        return results

    def _compute_pairwise_tests(self, conditions: np.ndarray, metric_results: Dict) -> Dict:
        """Compute pairwise statistical tests between conditions."""
        pairwise_tests = {}

        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i + 1 :]:
                t_stat, p_value = stats.ttest_ind(metric_results[cond1]["raw_data"], metric_results[cond2]["raw_data"])

                # Calculate Cohen's d
                n1, n2 = metric_results[cond1]["n"], metric_results[cond2]["n"]
                s1, s2 = metric_results[cond1]["std"], metric_results[cond2]["std"]

                # Pooled standard deviation
                s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                cohens_d = (metric_results[cond1]["mean"] - metric_results[cond2]["mean"]) / s_pooled

                pairwise_tests[f"{cond1}_vs_{cond2}"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                    "significant": p_value < (1 - self.confidence_level),
                }

        return pairwise_tests

    def _format_p_value(self, p_value: float) -> str:
        """Format p-value into a readable string with appropriate inequality symbols.

        Args:
            p_value: The p-value to format

        Returns:
            Formatted string representing the p-value
        """
        if p_value < 0.001:
            return "p < 0.001"
        elif p_value < 0.01:
            return "p < 0.01"
        elif p_value < 0.05:
            return "p < 0.05"
        elif p_value > 0.99:
            return "p > 0.99"
        elif p_value > 0.95:
            return "p > 0.95"
        else:
            return f"p = {p_value:.3f}"

    def format_summary_table(self, results: Dict) -> Table:
        """Format analysis results into a Rich table."""
        table = Table(title="Metrics Summary", show_header=True, header_style="bold magenta", show_lines=True)

        # Add columns
        table.add_column("Metric", style="cyan")
        table.add_column("Condition", style="green")
        table.add_column("Mean", justify="right")
        table.add_column("Std Dev", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("Statistical Test", justify="right")
        table.add_column("Significant", justify="center")

        for metric, metric_results in results.items():
            if metric == "loss":  # Skip loss metric
                continue

            friedman_test = metric_results.get("friedman_test", {})
            for condition, stats in metric_results.items():
                if condition not in ["pairwise_tests", "friedman_test"]:
                    table.add_row(
                        metric,
                        condition,
                        f"{stats['mean']:.4f}",
                        f"{stats['std']:.4f}",
                        f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]",
                        self._format_p_value(friedman_test["p_value"]) if friedman_test else "N/A",
                        "✓" if friedman_test and friedman_test.get("significant", False) else "✗",
                    )

        return table

    def format_pairwise_table(self, results: Dict) -> Table:
        """Format pairwise comparison results into a Rich table."""
        table = Table(title="Pairwise Comparisons", show_header=True, header_style="bold magenta", show_lines=True)

        table.add_column("Metric", style="cyan")
        table.add_column("Comparison", style="green")
        table.add_column("t-statistic", justify="right")
        table.add_column("Statistical Test", justify="right")
        table.add_column("Cohen's d", justify="right")
        table.add_column("Significant", justify="center")

        for metric, metric_results in results.items():
            if metric == "loss":  # Skip loss metric
                continue

            pairwise_tests = metric_results.get("pairwise_tests", {})
            for comparison, test_results in pairwise_tests.items():
                table.add_row(
                    metric,
                    comparison,
                    f"{test_results['t_statistic']:.4f}",
                    self._format_p_value(test_results["p_value"]),
                    f"{test_results['cohens_d']:.4f}",
                    "✓" if test_results["significant"] else "✗",
                )

        return table

    def generate_latex_summary_table(self, results: Dict) -> str:
        """Generate LaTeX code for the summary table."""
        latex = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Metrics Summary}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Metric & Condition & Mean & Std Dev & 95\\% CI & Statistical Test & Significant \\\\",
            "\\midrule",
        ]

        for metric, metric_results in results.items():
            if metric == "loss":
                continue

            friedman_test = metric_results.get("friedman_test", {})
            for condition, stats in metric_results.items():
                if condition not in ["pairwise_tests", "friedman_test"]:
                    row = [
                        metric,
                        condition,
                        f"{stats['mean']:.4f}",
                        f"{stats['std']:.4f}",
                        f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]",
                        self._format_p_value(friedman_test["p_value"]).replace("<", "$<$").replace(">", "$>$")
                        if friedman_test
                        else "N/A",
                        "Yes" if friedman_test and friedman_test.get("significant", False) else "No",
                    ]
                    latex.append(" & ".join(row) + " \\\\")

        latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

        return "\n".join(latex)

    def generate_latex_pairwise_table(self, results: Dict) -> str:
        """Generate LaTeX code for the pairwise comparison table."""
        latex = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Pairwise Comparisons}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Metric & Comparison & t-statistic & Statistical Test & Cohen's d & Significant \\\\",
            "\\midrule",
        ]

        for metric, metric_results in results.items():
            if metric == "loss":
                continue

            pairwise_tests = metric_results.get("pairwise_tests", {})
            for comparison, test_results in pairwise_tests.items():
                row = [
                    metric,
                    comparison.replace("_", "\\_"),
                    f"{test_results['t_statistic']:.4f}",
                    self._format_p_value(test_results["p_value"]).replace("<", "$<$").replace(">", "$>$"),
                    f"{test_results['cohens_d']:.4f}",
                    "Yes" if test_results["significant"] else "No",
                ]
                latex.append(" & ".join(row) + " \\\\")

        latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

        return "\n".join(latex)

    def process_experiment(self, split: Split, reference_condition: str, dataset: str) -> Dict:
        """Process experiment data for a given split."""
        run_directories = self._find_run_directories()

        match split:
            case Split.TRAIN:
                df = self._process_train_data(run_directories)
                metrics = [col for col in df.columns if col not in ["split", "epoch", "Modality Availability"]]
                visuals = self.visualiser.plot_validation_over_epochs(df=df, metrics=metrics, split=Split.TRAIN)
                return {
                    "data": df,
                    "analysis": self.analyze_results(df, metrics),
                    "visualisations": visuals,
                }

            case Split.VALIDATION:
                df = self._process_validation_data(run_directories)
                metrics = [col for col in df.columns if col not in ["split", "epoch", "Modality Availability"]]
                visuals = self.visualiser.plot_validation_over_epochs(df=df, metrics=metrics, split=Split.VALIDATION)
                return {
                    "data": df,
                    "analysis": self.analyze_results(df, metrics),
                    "visualisations": visuals,
                }

            case Split.TEST:
                df, agg_df = self._process_test_data(run_directories)
                metrics = [col for col in df.columns if col not in ["split", "Modality Availability"]]
                analysis_results = self.analyze_results(df, metrics)

                visuals = self.visualiser.create_all_visualizations(
                    results=df,
                    metrics=metrics,
                    reference_condition=reference_condition,
                    analysis_results=analysis_results,
                    dataset=dataset,
                    skip_metrics=[
                        "recall_micro",
                        "recall_macro",
                        "precision_micro",
                        "precision_macro",
                        "f1_micro",
                        "f1_macro",
                        "balanced_accuracy",
                    ],
                )

                return {
                    "data": df,
                    "aggregate": agg_df,
                    "analysis": analysis_results,
                    "visualisations": visuals,
                }

    def format_analysis_results(self, results: Dict) -> str:
        """Format analysis results into a readable summary."""
        summary = []

        for metric, metric_results in results.items():
            if metric == "loss":
                continue

            summary.append(f"\nResults for metric: {metric}")
            summary.append("-" * 50)

            # Print Friedman test results if available
            if "friedman_test" in metric_results and metric_results["friedman_test"]:
                friedman = metric_results["friedman_test"]
                summary.append("\nOverall Statistical Test (Friedman):")
                summary.append(f"Statistic: {friedman['statistic']:.4f}")
                summary.append(f"p-value: {friedman['p_value']:.4f}")
                summary.append(f"Significant: {friedman['significant']}")

            # Pairwise comparison results
            summary.append("\nPairwise Comparisons:")
            for comparison, test_results in metric_results["pairwise_tests"].items():
                summary.append(f"\n{comparison}")
                summary.append(f"t-statistic: {test_results['t_statistic']:.4f}")
                summary.append(f"p-value: {test_results['p_value']:.4f}")
                summary.append(f"Cohen's d: {test_results['cohens_d']:.4f}")
                summary.append(f"Significant: {test_results['significant']}")

            # Print condition-specific statistics
            summary.append("\nCondition Statistics:")
            for condition, stats in metric_results.items():
                if condition not in ["pairwise_tests", "friedman_test"]:
                    summary.append(f"\nCondition: {condition}")
                    summary.append(f"Mean: {stats['mean']:.4f}")
                    summary.append(f"Standard Deviation: {stats['std']:.4f}")
                    summary.append(f"Standard Error: {stats['sem']:.4f}")
                    summary.append(f"95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
                    summary.append(f"Sample Size: {stats['n']}")

        return "\n".join(summary)

    def create_latex_tables(self, results: Dict) -> Dict[str, str]:
        """Create all LaTeX tables for the results."""
        return {
            "summary": self.generate_latex_summary_table(results),
            "pairwise": self.generate_latex_pairwise_table(results),
        }

    def create_rich_tables(self, results: Dict) -> Dict[str, Table]:
        """Create all Rich tables for the results."""
        return {"summary": self.format_summary_table(results), "pairwise": self.format_pairwise_table(results)}
