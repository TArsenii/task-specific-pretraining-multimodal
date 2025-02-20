from collections.abc import Set
import glob
import json
import os
import re
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats
from statsmodels.stats.multitest import multipletests
from modalities import Modality


def get_run_data(root: str | Path | PathLike, debug: bool = False) -> List[Path]:
    """
    Find and return all run directories in the metrics folder.

    Args:
        root: Root directory path containing the metrics folder
        debug: If True, prints debug information

    Returns:
        List of Path objects for each run directory
    """
    runs_pattern = re.compile(r"(\d+)")

    run_paths = [
        Path(path)
        for path in glob.glob(os.path.join(root / "metrics", "*"))
        if os.path.isdir(path) and runs_pattern.match(os.path.basename(path))
    ]

    if debug:
        print(f"Found {len(run_paths)} runs in {root}.")

    return run_paths


def extract_modality_availability(s: str) -> str:
    """
    Extract modality from a string in the format "*_<modality>".

    Args:
        s: Input string containing modality information

    Returns:
        Extracted modality string
    """
    split = s.rsplit("_", 1)
    return split[-1]


def extract_metric(s: str) -> str:
    """
    Extract metric name from a string in the format "**<metric>_<modality>".

    Args:
        s: Input string containing metric information

    Returns:
        Extracted metric name
    """
    split = s.rsplit("_", 1)
    return split[0]


def load_test_metrics(fp: str | Path | PathLike, extract_key: Optional[str] = None) -> DataFrame:
    """
    Load test metrics from a JSON file into a DataFrame.

    Args:
        fp: File path to the JSON metrics file

    Returns:
        DataFrame containing test metrics
    """

    with open(fp, "r") as f:
        metrics = json.load(f)

    metrics = metrics[0] if isinstance(metrics, list) else metrics

    if extract_key:
        metrics = metrics[extract_key]

    keys_to_remove = []

    for metric in metrics:
        if "ConfusionMatrix" in metric:
            keys_to_remove.append(metric)

    if len(keys_to_remove) > 0:
        for i in range(0, len(keys_to_remove)):
            del metrics[keys_to_remove[i]]

    metrics = pd.DataFrame([metrics])

    if "index" in metrics.columns:
        metrics = metrics.drop(columns=["index"])
    if "split" in metrics.columns:
        metrics = metrics.drop(columns=["split"])
    return metrics.reset_index(drop=True)


def calculate_within_modality_stats(
    run_data: DataFrame, metrics_to_test: Dict[str, float], baseline_modality: str = "ATV"
) -> Tuple[DataFrame, DataFrame]:
    """
    Calculate statistical significance within modalities against chance and baseline.

    Args:
        run_data: DataFrame with MultiIndex (Metric, Modalities Available)
        metrics_to_test: Dictionary mapping metric patterns to their chance levels
        baseline_modality: Modality to use as baseline for comparisons

    Returns:
        Tuple containing two DataFrames:
            - DataFrame with chance comparison results
            - DataFrame with baseline comparison results
    """
    chance_results: List[Dict[str, Any]] = []
    baseline_results: List[Dict[str, Any]] = []

    unique_metrics = run_data.index.get_level_values("Metric").unique()
    unique_modalities = run_data.index.get_level_values("Modalities Available").unique()

    baseline_data = {
        metric: run_data.loc[(metric, baseline_modality)]
        for metric in unique_metrics
        if (metric, baseline_modality) in run_data.index
    }

    for metric in unique_metrics:
        should_test_chance = False
        chance_level = None
        for metric_pattern, chance_value in metrics_to_test.items():
            if metric_pattern in metric:
                should_test_chance = True
                chance_level = chance_value
                break

        for modality in unique_modalities:
            if (metric, modality) not in run_data.index:
                continue

            runs = run_data.loc[(metric, modality)].values

            if should_test_chance:
                t_stat_chance, p_val_chance = stats.ttest_1samp(runs, chance_level)
                d_chance = np.mean(runs - chance_level) / np.std(runs) if np.std(runs) != 0 else 0

                chance_results.append(
                    {
                        "Metric": metric,
                        "Modalities Available": modality,
                        "t_statistic": t_stat_chance,
                        "p_value": p_val_chance,
                        "cohens_d": d_chance,
                        "compared_to": "chance",
                    }
                )

            if modality != baseline_modality and metric in baseline_data:
                baseline_runs = baseline_data[metric]
                t_stat_base, p_val_base = stats.ttest_ind(runs, baseline_runs)

                n1, n2 = len(runs), len(baseline_runs)
                var1, var2 = np.var(runs, ddof=1), np.var(baseline_runs, ddof=1)
                pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                d_base = (np.mean(runs) - np.mean(baseline_runs)) / pooled_se if pooled_se != 0 else 0

                baseline_results.append(
                    {
                        "Metric": metric,
                        "Modalities Available": modality,
                        "t_statistic": t_stat_base,
                        "p_value": p_val_base,
                        "cohens_d": d_base,
                        "compared_to": baseline_modality,
                    }
                )

    chance_df = pd.DataFrame(chance_results)
    baseline_df = pd.DataFrame(baseline_results)

    if not chance_df.empty:
        _, p_corrected, _, _ = multipletests(chance_df["p_value"], method="fdr_bh")
        chance_df["p_value_corrected"] = p_corrected

    if not baseline_df.empty:
        _, p_corrected, _, _ = multipletests(baseline_df["p_value"], method="fdr_bh")
        baseline_df["p_value_corrected"] = p_corrected

    return chance_df, baseline_df


def load_all_test_metrics(
    files: List[str | Path | PathLike],
    test_metrics_name: str = "test_metrics.json",
    remove_prefix: Optional[str] = None,
    drop_loss: bool = True,
    round: Optional[int] = 6,
    format: Literal["standard", "grouped"] = "standard",
    baseline_modality: str = "ATV",
    extract_key: Optional[str] = None,
    metrics_to_test: Dict[str, float] = {
        "Has0_Accuracy": 0.5,
        "Has0_F1": 0.5,
        "Non0_Accuracy": 0.5,
        "Non0_F1": 0.5,
    },
) -> DataFrame | Tuple[DataFrame, DataFrame]:
    """
    Load and process test metrics from multiple files.

    Args:
        files: List of file paths containing test metrics
        test_metrics_name: Name of the metrics JSON file
        remove_prefix: Prefix to remove from metric names
        drop_loss: Whether to drop loss columns
        round: Number of decimal places to round to
        format: Output format ('standard' or 'grouped')
        baseline_modality: Modality to use as baseline
        metrics_to_test: Dictionary of metrics to test against chance

    Returns:
        Either a single DataFrame or tuple of DataFrames depending on format
    """
    dfs = [load_test_metrics(fp / test_metrics_name, extract_key=extract_key) for fp in files]
    print(f"Loaded {len(dfs)} files.")

    df = pd.concat(dfs, ignore_index=True)
    if "ConfusionMatrix" in df.columns:
        df = df.drop(columns=["ConfusionMatrix"])
        print("Dropped ConfusionMatrix columns.")

    if drop_loss:
        df = df.drop(columns=["loss"])
        print("Dropped loss columns.")

    return df

    df = df.T
    print(df.shape)

    modalities_series = df.index.map(extract_modality_availability)
    metrics_series = df.index.map(extract_metric)
    if remove_prefix:
        metrics_series = metrics_series.str.replace(remove_prefix, "")

    df.index = pd.MultiIndex.from_tuples(
        list(zip(metrics_series, modalities_series)), names=["Metric", "Modalities Available"]
    )

    return df

    stats_results = calculate_stats(df)

    try:
        chance_df, baseline_df = calculate_within_modality_stats(
            run_data=df, metrics_to_test=metrics_to_test, baseline_modality=baseline_modality
        )
    except Exception as e:
        print(f"\nError in calculate_within_modality_stats: {str(e)}")
        raise

    def get_significance_stars(p_value: float) -> Literal["", "***", "**", "*", "ns"]:
        if pd.isna(p_value):
            return ""
        if p_value <= 0.001:
            return "***"
        elif p_value <= 0.01:
            return "**"
        elif p_value <= 0.05:
            return "*"
        return "ns"

    if format == "standard":
        result_df = pd.DataFrame(
            {
                "Value": df.mean(axis=1),
                "Std": stats_results["basic_stats"]["std"],
                "CI_lower": stats_results["confidence_intervals"]["ci_lower"],
                "CI_upper": stats_results["confidence_intervals"]["ci_upper"],
            }
        )

        if not chance_df.empty:
            result_df = result_df.merge(
                chance_df[["Metric", "Modalities Available", "p_value_corrected"]],
                on=["Metric", "Modalities Available"],
                how="left",
                suffixes=("", "_chance"),
            )
            result_df["vs_chance_significance"] = result_df["p_value_corrected"].apply(get_significance_stars)

        if not baseline_df.empty:
            result_df = result_df.merge(
                baseline_df[["Metric", "Modalities Available", "p_value_corrected"]],
                on=["Metric", "Modalities Available"],
                how="left",
                suffixes=("", "_baseline"),
            )
            result_df["vs_baseline_significance"] = result_df["p_value_corrected"].apply(get_significance_stars)

    else:
        if not chance_df.empty:
            chance_df["significance"] = chance_df["p_value_corrected"].apply(get_significance_stars)
            chance_df = chance_df.merge(
                df.mean(axis=1).reset_index(), on=["Metric", "Modalities Available"], how="left"
            )

        if not baseline_df.empty:
            baseline_df["significance"] = baseline_df["p_value_corrected"].apply(get_significance_stars)
            baseline_df = baseline_df.merge(
                df.mean(axis=1).reset_index(), on=["Metric", "Modalities Available"], how="left"
            )
        chance_df = chance_df.rename(columns={0: "Value"})
        baseline_df = baseline_df.rename(columns={0: "Value"})

        numeric_cols = ["Value", "p_value", "cohens_d", "t_statistic"]
        baseline_df[numeric_cols] = baseline_df[numeric_cols].round(round)
        chance_df[numeric_cols] = chance_df[numeric_cols].round(round)
        return chance_df, baseline_df

    if round is not None:
        numeric_cols = ["Value", "p_value", "cohens_d", "t_statistic"]
        result_df[numeric_cols] = result_df[numeric_cols].round(round)

    return result_df


def load_full_modality_metrics(
    files: List[str | Path | PathLike],
    test_metrics_name: str = "test_metrics.json",
    drop_loss: bool = True,
    remove_prefix: Optional[str] = None,
    round: Optional[int] = 6,
    full_modality_name: str = "ATV",
) -> DataFrame:
    """
    Load metrics for the full modality configuration.

    Args:
        files: List of file paths containing metrics
        test_metrics_name: Name of the metrics JSON file
        drop_loss: Whether to drop loss columns
        remove_prefix: Prefix to remove from metric names
        round: Number of decimal places to round to
        full_modality_name: Name of the full modality configuration

    Returns:
        DataFrame containing processed metrics for the full modality
    """
    df = pd.concat([load_test_metrics(fp / test_metrics_name) for fp in files], ignore_index=True)

    if drop_loss:
        df = df.drop(columns=["loss"])

    df = df[[col for col in df.columns if full_modality_name in col]]
    df = df.T

    modalities_series = df.index.map(extract_modality_availability)
    metrics_series = df.index.map(extract_metric)

    if remove_prefix:
        metrics_series = metrics_series.str.replace(remove_prefix, "")

    df.index = pd.MultiIndex.from_tuples(
        list(zip(metrics_series, modalities_series)), names=["Metric", "Modalities Available"]
    )

    stats_results = calculate_stats(df)

    stats_df = pd.DataFrame(
        {"std": stats_results["basic_stats"]["std"]},
        index=df.index,
    )

    result_df = pd.DataFrame(
        {
            "Value": df.mean(axis=1),
            "Std": stats_df["std"],
        }
    )

    if round is not None:
        numeric_cols = ["Value", "Std"]
        result_df[numeric_cols] = result_df[numeric_cols].round(round)

    return result_df


def calculate_stats(df: DataFrame) -> Dict[str, DataFrame]:
    """
    Calculate comprehensive statistics for metrics across modalities.

    Args:
        df: DataFrame with MultiIndex (Metric, Modalities Available)

    Returns:
        Dictionary containing basic statistics and confidence intervals
    """
    run_cols = df.columns

    basic_stats = pd.DataFrame(
        {
            "mean": df[run_cols].mean(axis=1),
            "std": df[run_cols].std(axis=1),
            "min": df[run_cols].min(axis=1),
            "max": df[run_cols].max(axis=1),
            "median": df[run_cols].median(axis=1),
            "q25": df[run_cols].quantile(0.25, axis=1),
            "q75": df[run_cols].quantile(0.75, axis=1),
            "sem": df[run_cols].apply(lambda x: stats.sem(x), axis=1),
            "n_runs": df[run_cols].notna().sum(axis=1),
        }
    ).round(4)

    ci_data = df[run_cols].apply(
        lambda x: stats.t.interval(confidence=0.95, df=len(x) - 1, loc=np.mean(x), scale=stats.sem(x)), axis=1
    )

    ci_df = pd.DataFrame(
        {
            "ci_lower": [x[0] for x in ci_data],
            "ci_upper": [x[1] for x in ci_data],
        }
    ).round(4)

    return {"basic_stats": basic_stats, "confidence_intervals": ci_df}


def pivot_data_to_modalities_available(
    df: DataFrame,
    index: str = "Metric",
    values: List[str] = ["Value", "p_value", "significance", "cohens_d", "t_statistic"],
) -> DataFrame:
    """
    Pivot data to show modalities as columns.

    Args:
        df: Input DataFrame to pivot

    Returns:
        Pivoted DataFrame with modalities as columns
    """
    df = df.pivot(index=index, columns=["Modalities Available"], values=values).swaplevel(axis=1).sort_index(axis=1)

    return df


def write_latex_to_file(df: DataFrame, file_name: str | Path | PathLike) -> None:
    """
    Write DataFrame to LaTeX file with specific formatting.

    Args:
        df: DataFrame to convert to LaTeX
        file_name: Path to save the LaTeX file
    """
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].apply(lambda x: f"{x:.3f}")

    for col in df.columns:
        try:
            df[col] = df[col].apply(lambda x: f"{x:.3f}")
        except ValueError:
            pass

    n_cols = len(df.columns)
    df1 = df.iloc[:, : n_cols // 2]
    df2 = df.iloc[:, n_cols // 2 :]

    def make_table(dataframe: DataFrame) -> str:
        return dataframe.to_latex(
            escape=True,
            longtable=False,
            multicolumn=True,
            multicolumn_format="c",
            column_format="|l|" + "c|" * (len(dataframe.columns)),
        )

    table_template = (
        "\\afterpage{\n"
        "\\clearpage\n"
        "\\begin{landscape}\n"
        "\\begin{table}\n"
        "\\setlength\\tabcolsep{4pt}\n"
        "\\fontsize{12}{14}\\selectfont\n"
        "\\resizebox{1.5\\textwidth}{!}{\n"
        "{content}"
        "}\n\\end{table}\n"
        "\\end{landscape}\n"
        "\\clearpage}\n"
    )

    latex_content = table_template.format(content=make_table(df1)) + table_template.format(content=make_table(df2))

    Path(file_name).write_text(latex_content)
    print(f"Saved LaTeX table to {file_name}")


def load_validation_metrics(fp: str | Path | PathLike) -> DataFrame:
    """
    Load validation metrics from a JSON file.

    Args:
        fp: File path to the validation metrics JSON file

    Returns:
        DataFrame containing validation metrics
    """
    metrics = pd.read_json(fp / "validation_metrics.json")
    if "index" in metrics.columns:
        metrics = metrics.drop(columns=["index"])
    if "split" in metrics.columns:
        metrics = metrics.drop(columns=["split"])
    return metrics.reset_index(drop=True)


def load_all_validation_metrics(
    files: List[str | Path | PathLike],
    round: Optional[int] = 4,
    remove_prefix: Optional[str] = None,
    drop_loss: bool = True,
    drop_columns: Optional[List[str]] = None,
) -> DataFrame:
    """
    Load, combine and analyze validation metrics across epochs.

    Args:
        files: List of file paths containing validation metrics
        groupby_columns: Columns to group metrics by (default: ["Epoch"])
        metrics_to_drop: Columns to exclude from analysis
        round: Number of decimal places to round to

    Returns:
        DataFrame with combined statistics and raw data per group
    """
    dfs = [load_validation_metrics(fp) for fp in files]
    combined_df = pd.concat(dfs, ignore_index=True)

    if drop_loss:
        combined_df = combined_df.drop(columns=["loss"])

    if drop_columns:
        combined_df = combined_df.drop(columns=drop_columns)

    combined_df = combined_df.groupby("Epoch").mean()

    if round is not None:
        combined_df = combined_df.round(round)

    df = combined_df.T

    modalities_series = df.index.map(extract_modality_availability)
    metrics_series = df.index.map(extract_metric)

    if remove_prefix:
        metrics_series = metrics_series.str.replace(remove_prefix, "")

    df.index = pd.MultiIndex.from_tuples(
        list(zip(metrics_series, modalities_series)), names=["Metric", "Modalities Available"]
    )

    df = pd.DataFrame(df).T
    df = df.swaplevel(axis=1).sort_index(axis=1)
    return df


def split_validation_metrics_by_available_modalities(idf: DataFrame, modalities: List[str]) -> Dict[str, DataFrame]:
    metrics = list(set([metric for _, metric in idf.columns]))
    validation_df = {}
    for metric in metrics:
        # Initialize a dictionary to hold data for each modality
        metric_data = {}
        # print(df.columns)
        for modality in modalities:
            data = idf[(modality, metric)]
            # Assign the Series to the corresponding modality
            metric_data[modality] = data
        # Create a DataFrame where columns are modalities
        df = pd.DataFrame(metric_data)
        # Optionally, you can set an appropriate index name if needed
        df.index.name = "Index"  # Replace 'Index' with your actual index name if applicable

        validation_df[metric] = df

    return validation_df


def load_confusion_matrices(
    root: str | Path | PathLike, split: Literal["train", "test", "validation"] = "test"
) -> Dict[str, np.ndarray]:
    fp = f"confusion_matrices_{split}.npy"
    fp = root / fp
    print(f"Loading confusion matrices from {fp}")
    return np.load(root / fp, allow_pickle=True).item()


def load_all_confusion_matrices(
    files: list[os.PathLike], split: Literal["train", "test", "valid"]
) -> Dict[int, Dict[str, List[np.ndarray]]]:
    all_matrices = defaultdict(lambda: defaultdict(list))
    for i, fp in enumerate(files, 1):
        matrices = load_confusion_matrices(fp, split)
        for k, v in matrices.items():
            all_matrices[i][k].extend(v)

    return all_matrices


def compute_mean_confusion_matrix_per_epoch(run_confusion_matrices: Dict[int, List[np.ndarray]]) -> List[np.ndarray]:
    """
    Computes the mean confusion matrix for each epoch across multiple runs.

    Parameters:
    - run_confusion_matrices (Dict[int, List[np.ndarray]]):
        A dictionary where each key is a run ID and each value is a list of confusion matrices
        (as numpy arrays) for each epoch.

    Returns:
    - List[np.ndarray]:
        A list where each element is the mean confusion matrix for that epoch across all runs.
    """

    if not run_confusion_matrices:
        raise ValueError("The input dictionary is empty.")

    # Extract all lists of confusion matrices
    all_runs = run_confusion_matrices
    run_data = [run_confusion_matrices[i] for i in all_runs.keys()]

    # Compute the mean confusion matrix for each epoch
    mean_cms = []

    for epoch_cm in zip(*run_data):
        mean_cm = np.mean(epoch_cm, axis=0).astype(int)
        mean_cms.append(mean_cm)

    return mean_cms


def load_embeddings(
    root: str | Path | PathLike,
    modality: Modality,
    include_reconstructions: bool = False,
    embeddings_dir: str = "embeddings",
) -> np.ndarray:
    """
    Loads embeddings for a specific modality from a given root directory.

    Parameters:
        root (str | Path | PathLike): The root directory containing the embeddings files.
        modality (Modality): The modality for which embeddings are being loaded.
        include_reconstructions (bool, optional): Whether to include reconstructed embeddings.
            If True, returns a 3D array with ground truth and reconstructed embeddings stacked along
            the third dimension.

    Returns:
        np.ndarray: A numpy array containing the embeddings. The shape is:
            - (n_samples, n_features) if `include_reconstructions` is False.
            - (n_samples, n_features, 2) if `include_reconstructions` is True.
    """
    # File path for the embeddings
    embeddings_fp = root / embeddings_dir / f"{modality}_embeddings.npy"
    embeddings = np.load(embeddings_fp)

    # Optionally load reconstructed embeddings
    if include_reconstructions:
        reconstructed_fp = root / embeddings_dir / f"{modality}_reconstructions.npy"
        reconstructed = np.load(reconstructed_fp)
        if embeddings.shape != reconstructed.shape:
            raise ValueError("Embeddings and reconstructed embeddings must have the same shape.")
        # Stack along a new axis to create 3D array
        embeddings = np.stack([embeddings, reconstructed], axis=2)

    return embeddings


def load_all_embeddings(
    files: List[str | Path | PathLike], modalities: Set[Modality], include_reconstructions: bool = False
) -> Dict[Modality, np.ndarray]:
    """
    Loads embeddings for multiple modalities across multiple root directories.

    Parameters:
        files (List[str | Path | PathLike]): A list of root directories containing embeddings files.
        modalities (Set[Modality]): A set of modalities to load embeddings for.
        include_reconstructions (bool, optional): Whether to include reconstructed embeddings.
            If True, returns a dictionary with 3D arrays (n_samples, n_features, 2) for each modality.

    Returns:
        Dict[Modality, np.ndarray]: A dictionary where keys are modalities and values are numpy arrays
        containing the embeddings for each modality.
    """
    embeddings = {}

    # Iterate over each root directory and load embeddings for each modality
    for root in files:
        for modality in modalities:
            embeddings[modality] = load_embeddings(root, modality, include_reconstructions)

    return embeddings
