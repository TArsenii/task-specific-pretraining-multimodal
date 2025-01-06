import os
import re
import warnings
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
import h5py
import torch
from modalities import Modality
from numpy import ndarray
from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, Linear, Module, init

from .printing import get_console

console = get_console()

PARAMETER_SIZE_BYTES: int = 4  # Size of a float parameter in bytes


def hdf5_to_dict(f: h5py.File) -> Dict[str, Any]:
    """
    Convert an HDF5 file to a dictionary.

    Args:
    - f (h5py.File): HDF5 file object.

    Returns:
    - Dict[str, Any]: Dictionary containing the contents of the HDF5 file.
    """
    return {k: f[k][()] for k in f.keys()}


def format_path_with_env(path_template, **kwargs):
    # Find environment variables in the path template
    env_vars = re.findall(r"\$(\w+)", path_template)

    # Replace each environment variable with its value or a default
    for var in env_vars:
        default_value = kwargs.get(var.lower(), ".")  # Fetch from kwargs or use empty string
        path_template = path_template.replace(f"${var}", os.getenv(var, default_value))

    # Format with additional variables like exp_name and run
    return path_template


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def gpu_memory() -> str:
    if torch.cuda.is_available():
        return (
            f"Allocated:\t{torch.cuda.memory_allocated()/1e9:.2f}GB\nCached:\t{torch.cuda.memory_reserved()/1e9:.2f}GB"
        )
    else:
        raise Exception("gpu_memory function called, but gpu is not available")


def to_gpu_safe(x: Tensor | Dict[str | Modality, Tensor | Any]) -> Tensor | Dict[str | Modality, Tensor | Any]:
    """
    Safely move a tensor or dictionary containing tensors to the GPU. If the GPU is not available, fallsback to the CPU with a warning.

    Args:
        x (Tensor | Dict[str  |  Modality, Tensor  |  Any]): The input(s) to move to the GPU (if available)

    Returns:
        Tensor | Dict[str | Modality, Tensor | Any]: The input(s) moved to the GPU (if available)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("CUDA device not available, using CPU as a fallback")
        console.info("[bold yellow]\u26a0[/] [yellow]CUDA device not available, using CPU as a fallback[/]")

    if isinstance(x, Tensor):
        return x.to(device)

    return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in x.items()}


def kaiming_init(module: Module) -> None:
    if isinstance(module, (Conv2d, Linear)):
        init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


def clean_checkpoints(
    checkpoints_dir: str,
    keep_epochs: List[int] = None,
    keep_best: bool = True,
    keep_last: bool = True,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Clean model checkpoints in a directory.

    Args:
    - checkpoints_dir (str): Directory containing checkpoint files.
    - keep_epochs (List[int]): List of epoch numbers to keep.
    - keep_best (bool): Whether to keep the best checkpoint. Defaults to True.
    - keep_last (bool): Whether to keep the last checkpoint. Defaults to True.
    - dry_run (bool): If True, only print what would be done without actually deleting. Defaults to False.
    - print_fn (Callable): Function to use for printing. Defaults to built-in print.

    Returns:
    - Optional[str]: Path to the last kept checkpoint, or None if no checkpoints were kept.
    """
    if not os.path.exists(checkpoints_dir):
        console.print(f"Directory {checkpoints_dir} does not exist.")
        return None

    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
    kept_checkpoints = []
    last_checkpoint = None

    epoch_pattern = re.compile(r"epoch_(\d+)\.pth$")
    if len(checkpoints) == 0:
        return None
    console.print(f"Cleaning checkpoints in {checkpoints_dir}...")
    for checkpoint in sorted(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x))):
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint)

        if keep_best and checkpoint.endswith("best.pth"):
            kept_checkpoints.append(checkpoint)
            console.print(f"Keeping best checkpoint: {checkpoint}")
            continue

        match = epoch_pattern.search(checkpoint)
        if match:
            epoch = int(match.group(1))
            if keep_epochs is not None and epoch in keep_epochs:
                kept_checkpoints.append(checkpoint)
                console.print(f"Keeping checkpoint for epoch {epoch}: {checkpoint}")
            else:
                if not dry_run:
                    os.remove(checkpoint_path)
                console.print(f"{'Would remove' if dry_run else 'Removing'} checkpoint: {checkpoint}")
        else:
            if not dry_run:
                os.remove(checkpoint_path)
            console.print(f"{'Would remove' if dry_run else 'Removing'} non-conforming checkpoint: {checkpoint}")

        last_checkpoint = checkpoint

    if keep_last and last_checkpoint and last_checkpoint not in kept_checkpoints:
        kept_checkpoints.append(last_checkpoint)
        console.print(f"Keeping last checkpoint: {last_checkpoint}")
        if not dry_run and os.path.exists(os.path.join(checkpoints_dir, last_checkpoint)):
            os.rename(
                os.path.join(checkpoints_dir, last_checkpoint),
                os.path.join(checkpoints_dir, last_checkpoint.replace(".pth", "_last.pth")),
            )

    return os.path.join(checkpoints_dir, kept_checkpoints[-1]) if kept_checkpoints else None


def safe_detach(tensor: Tensor | ndarray, to_np: bool = True) -> Tensor | ndarray:
    """
    Safely detaches PyTorch tensors and optionally converts them to numpy arrays. If the input is already a numpy array then it returns the input.

    Args:
        tensor: Input tensor or numpy array
        to_np: If True, converts PyTorch tensor to numpy array

    Returns:
        Detached tensor or numpy array
    """
    assert isinstance(tensor, Tensor) or isinstance(
        tensor, ndarray
    ), f"Expected tensor or numpy array, got {type(tensor)}"
    match isinstance(tensor, Tensor):
        case True:
            if to_np:
                return tensor.detach().cpu().numpy()
            return tensor.detach()
        case False:
            return tensor


def prepare_metrics_for_json(metrics_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert metrics to JSON-serializable format, handling numpy types.
    """

    def convert_value(v):
        if isinstance(
            v,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(v)
        elif isinstance(v, (np.float64, np.float16, np.float32, np.float64)):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v

    return [{k: convert_value(v) for k, v in epoch_metrics.items()} for epoch_metrics in metrics_list]
