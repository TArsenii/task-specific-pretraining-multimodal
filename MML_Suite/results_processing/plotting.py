import colorsys
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.colors import LinearSegmentedColormap
from modalities import Modality, add_modality
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

add_modality("video")
# Define Okabe-Ito color palette
OKABE_ITO_COLORS: Dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# modality to okabe-ito colour
MODALITY_COLOURS: Dict[Modality, str] = {
    Modality.AUDIO: OKABE_ITO_COLORS["orange"],
    Modality.VIDEO: OKABE_ITO_COLORS["sky_blue"],
    Modality.TEXT: OKABE_ITO_COLORS["bluish_green"],
    Modality.IMAGE: OKABE_ITO_COLORS["yellow"],
}


def generate_alternative_hex(modality: Modality | str, hue_shift: float = 0.3) -> str:
    """
    Generate an alternative color by shifting the hue.

    Args:
        modality_colour (str): Original hex color (e.g., "#64c8fa").
        hue_shift (float): Amount to shift the hue (in fractions of 1, e.g., 0.5 for 180°).

    Returns:
        str: Alternative hex color (e.g., "#fa6498").
    """

    if isinstance(modality, Modality):
        modality_colour = MODALITY_COLOURS[modality]
    else:
        modality_colour = modality
    # Remove the '#' prefix and convert hex to RGB
    modality_colour = modality_colour.lstrip("#")
    rgb = tuple(int(modality_colour[i : i + 2], 16) for i in (0, 2, 4))

    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*[x / 255 for x in rgb])

    # Shift hue
    h = (h + hue_shift) % 1.0  # Wrap around to stay in [0, 1]

    # Convert back to RGB
    rgb = colorsys.hls_to_rgb(h, l, s)
    rgb = tuple(int(x * 255) for x in rgb)  # Scale back to [0, 255]

    # Return as hex
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


# Define the colors for the colormap from yellow to blue
CMAP_COLOURS = [
    OKABE_ITO_COLORS["yellow"],  # Start at yellow
    OKABE_ITO_COLORS["orange"],  # Transition to orange
    OKABE_ITO_COLORS["vermillion"],  # Transition to vermillion
    OKABE_ITO_COLORS["sky_blue"],  # Transition to sky blue
    OKABE_ITO_COLORS["blue"],  # End at blue
]
CMAP_COLOURS.reverse()  # Reverse the order to go from yellow to blue
enhanced_cmap = LinearSegmentedColormap.from_list("OkabeIto_Yellow_Blue_Enhanced", CMAP_COLOURS, N=256)
# (Optional) Register the colormap
mpl.colormaps.register(name="OkabeIto_Yellow_Blue_Enhanced", cmap=enhanced_cmap)


okabe_ito_palette = [
    OKABE_ITO_COLORS["orange"],
    OKABE_ITO_COLORS["vermillion"],
    OKABE_ITO_COLORS["sky_blue"],
    OKABE_ITO_COLORS["bluish_green"],
    OKABE_ITO_COLORS["blue"],
    OKABE_ITO_COLORS["yellow"],
    OKABE_ITO_COLORS["reddish_purple"],
    OKABE_ITO_COLORS["black"],
]


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Computer Modern Roman"
sns.set_theme(style="darkgrid")
sns.set_context("paper")
F_SIZES: Dict[str, int] = {
    "title": 32,
    "xlabel": 24,
    "ylabel": 24,
    "xticks": 20,
    "yticks": 20,
    "legend": 20,
    "annot_size": 16,
    "annot_color": "black",
    "colorbar_size": 16,
}


def prettify_title(title: str) -> str:
    return r"\textbf{" + title.replace("_", " ").title() + r"}".capitalize()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    label_map: Dict[int, str],
    title: str,
    *,
    output_root: Optional[str | Path | os.PathLike] = None,
    font_sizes: Optional[Dict[str, int]] = None,
    fig_size: tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    grid_params: Optional[Dict[str, Any]] = None,
    annotation_format: str = "d",
    ylim: Optional[tuple[float, float]] = None,
    title_mapping_fn: Optional[callable] = None,
    show: bool = True,
) -> Path:
    """
    Enhanced confusion matrix visualization using seaborn heatmap.

    Args:
        conf_matrix (np.ndarray): Square confusion matrix
        label_map (dict): Mapping from integer indices to string labels
        title (str): Title for the plot
        output_root (Optional[str | Path]): Directory to save output files
        font_sizes (Optional[Dict[str, int]]): Font sizes for different elements
            Keys:
                - 'title': Font size for the plot title
                - 'xlabel': Font size for the x-axis label
                - 'ylabel': Font size for the y-axis label
                - 'xticks': Font size for the x-axis tick labels
                - 'yticks': Font size for the y-axis tick labels
                - 'annot_size': Font size for the annotations within the heatmap
                - 'annot_color': Color for the annotations within the heatmap
                - 'colorbar_size': Font size for the colorbar tick labels
        fig_size (tuple): Figure dimensions (width, height)
        cmap (str): Colormap for heatmap
        grid_params (Optional[Dict]): Parameters for grid customization
        annotation_format (str): Format string for annotations
        ylim (Optional[tuple]): Y-axis limits (min, max)
        title_mapping_fn (Optional[callable]): Function to transform title text

    """
    try:
        # Default font sizes
        f_sizes = F_SIZES.copy()
        if font_sizes:
            f_sizes.update(font_sizes)

        # Extract annotation size and color
        annot_size = f_sizes.get("annot_size", 12)
        annot_color = f_sizes.get("annot_color", "black")

        # Extract colorbar font size
        colorbar_size = f_sizes.get("colorbar_size", 12)

        # Default grid parameters
        default_grid = {"axis": "y", "linestyle": "--", "alpha": 0.6}
        grid_params = {**default_grid, **(grid_params or {})}

        # Convert indices to labels
        labels = [label_map[i] for i in range(len(conf_matrix))]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=fig_size)

        # Create heatmap with custom annotation properties
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt=annotation_format,
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            annot_kws={"size": annot_size, "color": annot_color, "fontweight": "bold"},
            ax=ax,
            linewidths=1,
            linecolor="black",
            # cbar_kws={"label": "Count", "fontsize": colorbar_size},
        )

        output_title = title

        # Apply title mapping function if provided
        if title_mapping_fn:
            title = title_mapping_fn(title)
        ax.set_title(title, fontsize=f_sizes.get("title", 16))
        ax.set_xlabel("Predicted", fontsize=f_sizes.get("xlabel", 14))
        ax.set_ylabel("Actual", fontsize=f_sizes.get("ylabel", 14))
        ax.tick_params(axis="x", labelsize=f_sizes.get("xticks", 12))
        ax.tick_params(axis="y", labelsize=f_sizes.get("yticks", 12))

        if ylim:
            ax.set_ylim(ylim)

        # Customize grid
        ax.grid(**grid_params)

        # Adjust layout
        plt.tight_layout()

        # Customize colorbar tick label size
        if ax.collections:
            colorbar = ax.collections[0].colorbar
            if colorbar:
                colorbar.ax.tick_params(labelsize=colorbar_size)

        # Save if output directory specified
        if output_root:
            output_path = Path(output_root) / "confusion_matrices"
            output_path.mkdir(parents=True, exist_ok=True)

            base_filename = Path(output_title.lower().replace(" ", "_").replace("-", "").replace("+", ""))
            fig.savefig(output_path / f"{base_filename}.png", dpi=300, bbox_inches="tight")
            fig.savefig(output_path / f"{base_filename}.pdf", dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        if output_root:
            return output_path / f"{base_filename}"
    finally:
        plt.cla()
        plt.close("all")


def plot_modality_available_results_with_significance(
    df,
    baseline_df: pd.DataFrame,
    *,
    modalities=["A", "AT", "ATV", "AV", "T", "TV", "V"],
    output_root: Optional[str | Path | os.PathLike] = None,
    title_mapping_fn=prettify_title,
    font_sizes=None,
    fig_size=(10, 8),
    baseline: bool = False,
    dataset_name: str = "mosei",
    bar_colour: str = "#F0E442",
    show: bool = True,
) -> None:
    """
    Creates Seaborn bar charts for each metric with LaTeX text symbols for significance.

    Parameters:
    - df (pd.DataFrame): MultiIndex DataFrame with metrics and modalities.
    - baseline_df (pd.DataFrame): DataFrame with baseline values (Full modality availability).
    - title_mapping_fn (function): Function to prettify titles (default: None).
    - font_sizes (dict): Dictionary to configure font sizes for title, labels, ticks, and legend.
    """
    try:
        # Default font sizes

        global F_SIZES

        f_sizes = F_SIZES.copy()
        f_sizes.update(font_sizes or {})
        font_sizes = f_sizes
        metrics = df.index
        if baseline:
            if "ATV" in modalities:
                modalities.remove("ATV")
        significance_levels = {
            "***": r"$p < 0.001$",
            "**": r"$p < 0.01$",
            "*": r"$p < 0.05$",
            "ns": r"$p \ge 0.05$",
        }

        caption_msg = r"""\large{* Significance levels are with respect to the full modality availability (ATV) results.}
    - - - \large{$\rightarrow$ Full modality performance}"""

        for metric in metrics:
            # Prepare data for Seaborn
            data = {
                "Modality": modalities,
                "Value": df.loc[metric, [(mod, "Value") for mod in modalities]].values,
                "Significance": df.loc[metric, [(mod, "significance") for mod in modalities]].values,
            }
            plot_df = pd.DataFrame(data)

            baseline_h_line = baseline_df.loc[metric]["Value"] if baseline else None

            # Create the barplot
            plt.figure(figsize=fig_size)
            ax = sns.barplot(data=plot_df, x="Modality", y="Value", color=bar_colour, edgecolor="black")
            if baseline_h_line is not None:
                ax.axhline(y=baseline_h_line, color="r", linestyle="--", label="Baseline")

            def format_significance(significance):
                if significance == "ns":
                    return r"$\mathbf{p \ge 0.05}$"  # Use \mathbf instead of \textbf
                else:
                    return significance_levels[significance]

            # Add significance markers
            for i, row in plot_df.iterrows():
                value = row["Value"]
                formatted = format_significance(row["Significance"])

                formatted = rf"""\begin{{center}}{formatted}\\{value:.2f}\end{{center}}"""

                ax.text(
                    i,
                    row["Value"] + 0.02,
                    formatted,
                    ha="center",
                    va="bottom",
                    fontsize=font_sizes["yticks"],
                )

            title = "{dataset_name} | {metric} Test Performance for Varying Modality Availability".format(
                dataset_name=dataset_name.upper(),
                metric=metric,
            )
            # Customize the plot
            prettified_title: str = title_mapping_fn(title) if title_mapping_fn else title
            caption = caption_msg
            plt.figtext(0.05, -0.05, caption, wrap=True, horizontalalignment="left", fontsize=font_sizes["xlabel"])
            # plt.figtext(0.2, -0.05, other_caption_msg, wrap=True, horizontalalignment="center", fontsize=font_sizes["xlabel"])

            ## Adjust the plot to make room for the caption
            plt.subplots_adjust(bottom=0.2)

            ax.set_title(prettified_title, fontsize=font_sizes["title"])
            ax.set_xlabel("Modality Combination", fontsize=font_sizes["xlabel"])
            ax.set_ylabel("Value", fontsize=font_sizes["ylabel"])
            ax.tick_params(axis="x", labelsize=font_sizes["xticks"])
            ax.tick_params(axis="y", labelsize=font_sizes["yticks"])
            if metric.lower() not in ["mae", "mse", "rmse", "loss"]:
                ax.set_ylim(0, 1.1)
            else:
                min_val = plot_df["Value"].min() * 0.9  ## Make sure the min value is visible
                max_val = plot_df["Value"].max() * 1.1  ## Make sure the max value is visible
                ax.set_ylim(min_val, max_val)

            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            if output_root:
                os.makedirs(f"{output_root}/plots", exist_ok=True)
                plt.savefig(f"{output_root}/plots/{metric}.png", dpi=300, bbox_inches="tight")
                plt.savefig(f"{output_root}/plots/{metric}.pdf", dpi=300, bbox_inches="tight")

            if show:
                plt.show()
    finally:
        plt.cla()
        plt.close("all")


def plot_validation_loss(
    df_with_loss: pd.DataFrame,
    title_mapping_fn=prettify_title,
    title: str = "Loss Curve",
    dataset_name: Optional[str] = None,
    loss_column_name: str = "loss",
    output_root: Optional[str | Path | os.PathLike] = None,
    fig_size: Tuple[int, int] = (10, 8),
    line_colour: str = "##0072B2",
    font_sizes: Optional[Dict[str, int]] = None,
    show: bool = True,
) -> None:
    """
    Plot the validation loss curve.

    Args:
    - df_with_loss (pd.DataFrame): DataFrame with loss values.
    - title (str): Title for the plot.
    - dataset_name (str): Name of the dataset.
    - loss_column_name (str): Name of the loss column.
    - output_root (Optional[str | Path]): Directory to save output files.
    """
    f_sizes = F_SIZES.copy()
    f_sizes.update(font_sizes or {})

    x = df_with_loss.index
    y = df_with_loss[loss_column_name].squeeze()

    plt.figure(figsize=fig_size)
    sns.lineplot(data=df_with_loss, x=x, y=y, marker="o", color=line_colour)

    plt.xlabel("Epoch", fontsize=f_sizes["xlabel"])
    plt.ylabel("Loss", fontsize=f_sizes["ylabel"])

    if dataset_name:
        title = f"{dataset_name} | {title}"

    title = title_mapping_fn(title) if title_mapping_fn else title

    plt.title(title, fontsize=f_sizes["title"])
    plt.xticks(fontsize=f_sizes["xticks"])

    plt.yticks(fontsize=f_sizes["yticks"])
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    if output_root:
        os.makedirs(f"{output_root}/plots", exist_ok=True)
        plt.savefig(f"{output_root}/plots/loss_curve.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_root}/plots/loss_curve.pdf", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.cla()
    plt.close("all")


def plot_validation_metric(
    df: pd.DataFrame,
    title_mapping_fn: Callable[[str], str],
    title: str,
    dataset_name: Optional[str] = None,
    output_root: Optional[str] = None,
    fig_size: Tuple[int, int] = (10, 8),
    font_sizes: Optional[Dict[str, int]] = None,
    show: bool = True,
) -> None:
    """
    Plot validation metrics for different modalities using Seaborn.

    Args:
        df (pd.DataFrame): DataFrame where each column represents a modality and each row corresponds to an epoch or iteration.
                           The index should represent the x-axis (e.g., epochs).
        title_mapping_fn (Callable[[str], str]): Function to transform the plot title.
        title (str): Base title for the plot.
        dataset_name (Optional[str], optional): Name of the dataset. Defaults to None.
        output_root (Optional[str | Path | os.PathLike], optional): Directory to save the plot images. Defaults to None.
        fig_size (Tuple[int, int], optional): Size of the figure in inches. Defaults to (10, 8).
        font_sizes (Optional[Dict[str, int]], optional): Dictionary to specify font sizes for various plot elements.
            Expected keys: 'title', 'xlabel', 'ylabel', 'xticks', 'yticks'. Defaults to None.

    Returns:
        None
    """
    try:
        sns.set_palette(sns.color_palette(okabe_ito_palette))

        # Update default font sizes with any provided by the user
        if font_sizes:
            F_SIZES.update(font_sizes)

        # Reset index to use as x-axis if not already a column
        if df.index.name is None:
            df = df.reset_index()
            x_label = "Index"
        else:
            df = df.reset_index()
            x_label = df.columns[0]

        # Ensure the x-axis starts from epoch 1
        if isinstance(df[x_label].iloc[0], (int, float)):
            df[x_label] = df[x_label] - df[x_label].min() + 1

        # Determine x-axis ticks
        total_epochs = df.shape[0]
        if total_epochs <= 20:
            x_ticks = df[x_label].tolist()
        else:
            x_ticks = [1]  # Always include epoch 1
            for epoch in df[x_label].tolist()[1:]:
                if epoch % 2 == 0 or epoch % 5 == 0:
                    x_ticks.append(epoch)
            # Ensure the last epoch is included
            if df[x_label].iloc[-1] not in x_ticks:
                x_ticks.append(df[x_label].iloc[-1])

        # Remove duplicates and sort
        x_ticks = sorted(list(set(x_ticks)))

        # Melt the DataFrame to long format for Seaborn
        df_melted = df.melt(id_vars=x_label, var_name="Modality", value_name="Value")

        # Initialize the matplotlib figure
        plt.figure(figsize=fig_size)

        # Create the line plot with Seaborn
        sns.lineplot(
            data=df_melted,
            x=x_label,
            y="Value",
            hue="Modality",
            marker="o",
        )

        # Set y-axis limits from 0.0 to 1.0
        plt.ylim(0.0, 1.05)

        # Set x-axis ticks
        plt.xticks(ticks=x_ticks, fontsize=F_SIZES["xticks"])

        # Set labels with specified font sizes
        plt.xlabel(x_label, fontsize=F_SIZES["xlabel"])
        plt.ylabel(title, fontsize=F_SIZES["ylabel"])

        # Modify the title
        if dataset_name:
            full_title = f"{dataset_name} | {title}"
        else:
            full_title = title

        if title_mapping_fn:
            full_title = title_mapping_fn(full_title)

        plt.title(full_title, fontsize=F_SIZES["title"])

        # Set y-ticks font size
        plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=F_SIZES["yticks"])

        # Enhance the grid
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        # Adjust layout for better spacing, especially to accommodate legend below
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Adjust legend placement
        if "Modality" in df_melted.columns:
            plt.legend(
                title="Modality",
                title_fontsize=F_SIZES["legend"],
                fontsize=F_SIZES["legend"],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.1),
                ncol=min(len(df.columns), 7),  # Adjust ncols as needed, here limited to 5
            )
        else:
            plt.legend().remove()

        # Save the plot if output_root is provided
        if output_root:
            plots_dir = Path(output_root) / "validation_plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            # Sanitize the title to create a valid filename
            sanitized_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in full_title)
            plt.savefig(plots_dir / f"{sanitized_title}.png", dpi=300, bbox_inches="tight")
            plt.savefig(plots_dir / f"{sanitized_title}.pdf", dpi=300, bbox_inches="tight")

        if show:
            plt.show()
    finally:
        plt.cla()
        plt.close("all")


def create_video_from_pngs(
    image_paths: List[str | Path | os.PathLike], output_path, fps=24, linger_time_seconds=3, size=None, codec="mp4v"
) -> None:
    """
    Creates a video from a list of PNG images using OpenCV, lingering on each image for a specified time.

    Parameters:
    - image_paths (list of str): List of file paths to PNG images.
    - output_path (str): Path where the output video will be saved (e.g., 'output.mp4').
    - fps (int, optional): Frames per second for the video. Default is 2.
    - linger_time_seconds (float, optional): Time in seconds each image should linger. Default is 2 seconds.
    - size (tuple, optional): Desired video size as (width, height). If None, uses the size of the first image.
    - codec (str, optional): Codec to use for the video. Default is 'mp4v'.

    Returns:
    - None
    """
    if not image_paths:
        raise ValueError("The list of image paths is empty.")

    # Ensure all files exist
    for img_path in image_paths:
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

    # Read the first image to get the size
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        raise ValueError(f"Failed to read image: {image_paths[0]}")

    height, width, layers = first_image.shape
    if size is not None:
        width, height = size

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Calculate number of frames each image should linger
    frames_per_image = max(1, int(fps * linger_time_seconds))
    total_frames = len(image_paths) * frames_per_image

    # Initialize tqdm progress bar
    with tqdm(total=total_frames, desc="Creating Video", unit="frame") as pbar:
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                tqdm.write()(f"Warning: Skipping unreadable image {img_path}")
                continue
            if size is not None:
                img = cv2.resize(img, size)
            # Write the same image multiple times to linger
            for _ in range(frames_per_image):
                video.write(img)
                pbar.update(1)

    video.release()
    tqdm.write(f"Video saved to {output_path}")


def plot_embeddings(
    embedding_data: Dict[Modality, np.ndarray],
    method: Literal["PCA", "t-SNE", "UMAP"],
    *,
    output_root: Optional[str | Path] = None,
    labels_data: Optional[Dict["Modality", np.ndarray]] = None,
    seed: int = 42,
    figsize: tuple = (10, 8),
    font_sizes: Optional[Dict[str, int]] = None,
    show: bool = True,
    marker_scale: int = 50,
    title_mapping_fn: Optional[Callable[[str], str]] = None,
    title: Optional[str] = None,
) -> None:
    match method:
        case "PCA":
            _compute_pca(embedding_data, output_root, labels_data, seed, figsize, font_sizes, show)
        case "t-SNE":
            _compute_tsne(
                embedding_data,
                output_root,
                labels_data=labels_data,
                seed=seed,
                figsize=figsize,
                font_sizes=font_sizes,
                show=show,
                marker_scale=marker_scale,
                title_mapping_fn=title_mapping_fn,
                title=title,
            )
        case "UMAP":
            _compute_umap(embedding_data, output_root, labels_data, seed, figsize, font_sizes, show)
        case _:
            raise ValueError(f"Invalid method: {method}")


def _compute_pca(
    embeddings: Dict["Modality", np.ndarray],
    output_root: Optional[str | Path] = None,
    labels_data: Optional[Dict["Modality", np.ndarray]] = None,
    seed: int = 42,
    figsize: tuple = (8, 8),
    font_sizes: Optional[Dict[str, int]] = None,
    show: bool = True,
) -> None:
    """
    Compute and plot PCA embeddings for each modality's ground truth and (optionally) reconstructed embeddings.

    Args:
        embeddings (Dict[Modality, np.ndarray]): A dictionary where each key is a Modality, and the value is an ndarray.
            The ndarray shape is (n_samples, n_features) for ground truth or
            (n_samples, n_features, 2) if it includes ground truth and reconstructed embeddings.
        output_root (Optional[str | Path]): Directory to save the plots. Defaults to None.
        labels_data (Optional[Dict[Modality, np.ndarray]]): A dictionary with the same keys as embeddings, where each value is
            either (n_samples,) for ground truth or (n_samples, 2) if it includes labels for reconstructed embeddings.
        seed (int): Random seed for PCA. Defaults to 42.
        figsize (tuple): Size of the plot. Defaults to (8, 8).
        font_sizes (Optional[Dict[str, int]]): Font sizes for various plot elements. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
    """
    try:
        # Initialize font sizes
        global F_SIZES
        f_sizes = F_SIZES.copy()
        f_sizes.update(font_sizes or {})

        # Prepare PCA inputs
        pca_input = []
        plot_labels = []
        modalities = []
        annotation_labels = []

        for modality, data in embeddings.items():
            if data.ndim == 2:  # Only ground truth embeddings
                pca_input.append(data)
                plot_labels.extend([f"{modality} (Ground Truth)"] * data.shape[0])
                modalities.extend([modality] * data.shape[0])
                if labels_data and modality in labels_data:
                    annotation_labels.extend(labels_data[modality])
                else:
                    annotation_labels.extend([None] * data.shape[0])
            elif data.ndim == 3 and data.shape[2] == 2:  # Ground truth and reconstructions
                pca_input.append(data[:, :, 0])  # Ground truth
                pca_input.append(data[:, :, 1])  # Reconstructed
                plot_labels.extend([f"{modality} (Ground Truth)"] * data.shape[0])
                plot_labels.extend([f"{modality} (Reconstructed)"] * data.shape[0])
                modalities.extend([modality] * data.shape[0] * 2)
                if labels_data and modality in labels_data:
                    annotation_labels.extend(labels_data[modality][:, 0])
                    annotation_labels.extend(labels_data[modality][:, 1])
                else:
                    annotation_labels.extend([None] * data.shape[0] * 2)
            else:
                raise ValueError("Embeddings must be either 2D or 3D with size (n_samples, n_features[, 2]).")

        # Concatenate embeddings for PCA
        pca_input = np.vstack(pca_input)

        # Compute PCA
        pca = PCA(n_components=2, random_state=seed)
        pca_result = pca.fit_transform(pca_input)

        # Prepare Data for Seaborn
        plot_data = {
            "PCA-1": pca_result[:, 0],
            "PCA-2": pca_result[:, 1],
            "Modality": plot_labels,
        }

        # Extend palette for reconstructions
        extended_palette = {}
        for modality, color in MODALITY_COLOURS.items():
            extended_palette[f"{modality} (Ground Truth)"] = color
            extended_palette[f"{modality} (Reconstructed)"] = generate_alternative_hex(modality=modality, hue_shift=0.3)

        # Create Plot
        plt.figure(figsize=figsize)
        sns.scatterplot(
            x="PCA-1",
            y="PCA-2",
            hue="Modality",
            data=plot_data,
            palette=extended_palette,
            style="Modality",
            markers=True,
        )

        # Remove axis labels since PCA's absolute values are not important
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks([])
        plt.yticks([])
        plt.title("PCA Embeddings by Modality", fontsize=f_sizes.get("title", 16))
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=f_sizes.get("legend", 12))
        plt.tight_layout()

        # Save the plot if output_root is provided
        contains_reconstructions = any(data.ndim == 3 for data in embeddings.values())
        if output_root:
            f_name_no_ext = "pca_embeddings_with_reconstructions" if contains_reconstructions else "pca_embeddings"
            os.makedirs(f"{output_root}/plots", exist_ok=True)
            plt.savefig(f"{output_root}/plots/{f_name_no_ext}.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{output_root}/plots/{f_name_no_ext}.pdf", dpi=300, bbox_inches="tight")

        # Show the plot if required
        if show:
            plt.show()
    finally:
        plt.cla()
        plt.close("all")


def _compute_tsne(
    embeddings: Dict[Modality, np.ndarray],
    output_root: Optional[str | Path] = None,
    *,
    labels_data: Optional[Dict[Modality, np.ndarray]] = None,
    seed: int = 42,
    figsize: tuple = (8, 8),
    font_sizes: Optional[Dict[str, int]] = None,
    show: bool = True,
    marker_scale: int = 50,
    title: str = "T-SNE Embeddings by Modality",
    title_mapping_fn: Optional[Callable[[str], str]] = None,
) -> None:
    """
    Compute and plot T-SNE embeddings for each modality's ground truth and (optionally) reconstructed embeddings.

    Args:
        embeddings (Dict[Modality, np.ndarray]): A dictionary where each key is a Modality, and the value is an ndarray.
            The ndarray shape is (n_samples, n_features) for ground truth or
            (n_samples, n_features, 2) if it includes ground truth and reconstructed embeddings.
        output_root (Optional[str | Path]): Directory to save the plots. Defaults to None.
        labels_data (Optional[Dict[Modality, np.ndarray]]): A dictionary with the same keys as embeddings, where each value is
            either (n_samples,) for ground truth or (n_samples, 2) if it includes labels for reconstructed embeddings.
        seed (int): Random seed for T-SNE. Defaults to 42.
        figsize (tuple): Size of the plot. Defaults to (8, 8).
        font_sizes (Optional[Dict[str, int]]): Font sizes for various plot elements. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
    """
    try:
        # Initialize font sizes
        global F_SIZES
        f_sizes = F_SIZES.copy()
        f_sizes.update(font_sizes or {})

        # Prepare T-SNE inputs
        tsne_input = []
        plot_labels = []
        modalities = []
        annotation_labels = []

        for modality, data in embeddings.items():
            if data.ndim == 2:  # Only ground truth embeddings
                tsne_input.append(data)
                plot_labels.extend([f"{modality} (Ground Truth)"] * data.shape[0])
                modalities.extend([modality] * data.shape[0])
                if labels_data and modality in labels_data:
                    annotation_labels.extend(labels_data[modality])
                else:
                    annotation_labels.extend([None] * data.shape[0])
            elif data.ndim == 3 and data.shape[2] == 2:  # Ground truth and reconstructions
                tsne_input.append(data[:, :, 0])  # Ground truth
                tsne_input.append(data[:, :, 1])  # Reconstructed
                plot_labels.extend([f"{modality} (Ground Truth)"] * data.shape[0])
                plot_labels.extend([f"{modality} (Reconstructed)"] * data.shape[0])
                modalities.extend([modality] * data.shape[0] * 2)
                if labels_data and modality in labels_data:
                    annotation_labels.extend(labels_data[modality][:, 0])
                    annotation_labels.extend(labels_data[modality][:, 1])
                else:
                    annotation_labels.extend([None] * data.shape[0] * 2)
            else:
                raise ValueError("Embeddings must be either 2D or 3D with size (n_samples, n_features[, 2]).")

        # Concatenate embeddings for T-SNE
        tsne_input = np.vstack(tsne_input)

        # Compute T-SNE
        print("Computing T-SNE embeddings...")
        tsne = TSNE(n_components=2, random_state=seed)
        tsne_result = tsne.fit_transform(tsne_input)
        print("T-SNE embeddings computed.")
        # Prepare Data for Seaborn
        plot_data = {
            "TSNE-1": tsne_result[:, 0],
            "TSNE-2": tsne_result[:, 1],
            "Modality": plot_labels,
        }

        # Extend palette for reconstructions
        extended_palette = {}
        for modality, color in MODALITY_COLOURS.items():
            extended_palette[f"{modality} (Ground Truth)"] = color
            extended_palette[f"{modality} (Reconstructed)"] = generate_alternative_hex(color)
        print(f"Extended palette: {extended_palette}")

        n_legend_cols = len(embeddings)

        # Create Plot
        plt.figure(figsize=figsize)
        sns.scatterplot(
            x="TSNE-1",
            y="TSNE-2",
            hue="Modality",
            data=plot_data,
            palette=extended_palette,
            style="Modality",
            markers=True,
            s=marker_scale,
        )

        # Remove axis labels since T-SNE scale is arbitrary
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks([])
        plt.yticks([])

        title = title_mapping_fn(title) if title_mapping_fn else title

        plt.title(title, fontsize=f_sizes.get("title", 16))
        plt.legend(
            bbox_to_anchor=(0.5, -0.125), loc="lower center", fontsize=f_sizes.get("legend", 12), ncols=n_legend_cols
        )
        plt.tight_layout()  # Adjust the layout
        plt.subplots_adjust(bottom=0.3)
        # Save the plot if output_root is provided
        contains_reconstructions = any(data.ndim == 3 for data in embeddings.values())
        if output_root:
            f_name_no_ext = "tsne_embeddings_with_reconstructions" if contains_reconstructions else "tsne_embeddings"
            os.makedirs(f"{output_root}/plots", exist_ok=True)
            plt.savefig(f"{output_root}/plots/{f_name_no_ext}.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{output_root}/plots/{f_name_no_ext}.pdf", dpi=300, bbox_inches="tight")

        # Show the plot if required
        if show:
            plt.show()
    finally:
        plt.cla()
        plt.close("all")


def _compute_umap(
    embeddings: Dict["Modality", np.ndarray],
    output_root: Optional[str | Path] = None,
    labels_data: Optional[Dict["Modality", np.ndarray]] = None,
    seed: int = 42,
    figsize: tuple = (8, 8),
    font_sizes: Optional[Dict[str, int]] = None,
    show: bool = True,
) -> None:
    """
    Compute and plot UMAP embeddings for each modality's ground truth and (optionally) reconstructed embeddings.

    Args:
        embeddings (Dict[Modality, np.ndarray]): A dictionary where each key is a Modality, and the value is an ndarray.
            The ndarray shape is (n_samples, n_features) for ground truth or
            (n_samples, n_features, 2) if it includes ground truth and reconstructed embeddings.
        output_root (Optional[str | Path]): Directory to save the plots. Defaults to None.
        labels_data (Optional[Dict[Modality, np.ndarray]]): A dictionary with the same keys as embeddings, where each value is
            either (n_samples,) for ground truth or (n_samples, 2) if it includes labels for reconstructed embeddings.
        seed (int): Random seed for UMAP. Defaults to 42.
        figsize (tuple): Size of the plot. Defaults to (8, 8).
        font_sizes (Optional[Dict[str, int]]): Font sizes for various plot elements. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
    """
    try:
        # Initialize font sizes
        global F_SIZES
        f_sizes = F_SIZES.copy()
        f_sizes.update(font_sizes or {})

        # Prepare UMAP inputs
        umap_input = []
        plot_labels = []
        modalities = []
        annotation_labels = []

        for modality, data in embeddings.items():
            if data.ndim == 2:  # Only ground truth embeddings
                umap_input.append(data)
                plot_labels.extend([f"{modality} (Ground Truth)"] * data.shape[0])
                modalities.extend([modality] * data.shape[0])
                if labels_data and modality in labels_data:
                    annotation_labels.extend(labels_data[modality])
                else:
                    annotation_labels.extend([None] * data.shape[0])
            elif data.ndim == 3 and data.shape[2] == 2:  # Ground truth and reconstructions
                umap_input.append(data[:, :, 0])  # Ground truth
                umap_input.append(data[:, :, 1])  # Reconstructed
                plot_labels.extend([f"{modality} (Ground Truth)"] * data.shape[0])
                plot_labels.extend([f"{modality} (Reconstructed)"] * data.shape[0])
                modalities.extend([modality] * data.shape[0] * 2)
                if labels_data and modality in labels_data:
                    annotation_labels.extend(labels_data[modality][:, 0])
                    annotation_labels.extend(labels_data[modality][:, 1])
                else:
                    annotation_labels.extend([None] * data.shape[0] * 2)
            else:
                raise ValueError("Embeddings must be either 2D or 3D with size (n_samples, n_features[, 2]).")

        # Concatenate embeddings for UMAP
        umap_input = np.vstack(umap_input)

        # Compute UMAP
        reducer = umap.UMAP(n_components=2, random_state=seed)
        umap_result = reducer.fit_transform(umap_input)

        # Prepare Data for Seaborn
        plot_data = {
            "UMAP-1": umap_result[:, 0],
            "UMAP-2": umap_result[:, 1],
            "Modality": plot_labels,
        }

        # Extend palette for reconstructions
        extended_palette = {}
        for modality, color in OKABE_ITO_COLORS.items():
            extended_palette[f"{modality} (Ground Truth)"] = color
            extended_palette[f"{modality} (Reconstructed)"] = generate_alternative_hex(color)

        # Create Plot
        plt.figure(figsize=figsize)
        sns.scatterplot(
            x="UMAP-1",
            y="UMAP-2",
            hue="Modality",
            data=plot_data,
            palette=extended_palette,
            style="Modality",
            markers=True,
        )

        # Remove axis labels since UMAP's absolute values are not important
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks([])
        plt.yticks([])
        plt.title("UMAP Embeddings by Modality", fontsize=f_sizes.get("title", 16))
        plt.legend(bbox_to_anchor=(0.05, 1), loc="upper left", fontsize=f_sizes.get("legend", 12))
        plt.tight_layout()

        # Save the plot if output_root is provided
        contains_reconstructions = any(data.ndim == 3 for data in embeddings.values())
        if output_root:
            f_name_no_ext = "umap_embeddings_with_reconstructions" if contains_reconstructions else "umap_embeddings"
            os.makedirs(f"{output_root}/plots", exist_ok=True)
            plt.savefig(f"{output_root}/plots/{f_name_no_ext}.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{output_root}/plots/{f_name_no_ext}.pdf", dpi=300, bbox_inches="tight")

        # Show the plot if required
        if show:
            plt.show()
    finally:
        plt.cla()
        plt.close("all")
