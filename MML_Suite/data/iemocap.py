from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import h5py
import numpy as np
import torch
from data.base_dataset import MultimodalBaseDataset
from experiment_utils.logging import get_logger
from experiment_utils.utils import hdf5_to_dict
from modalities import Modality, add_modality
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

logger = get_logger()
add_modality("video")


class IEMOCAP(MultimodalBaseDataset):
    """Dataset class for the IEMOCAP multimodal emotion recognition task."""

    VALID_SPLITS: List[str] = ["trn", "val", "tst"]
    NUM_CLASSES: int = 4  # IEMOCAP 4-class classification [happy, sad, angry, neutral]
    AVAILABLE_MODALITIES: Dict[str, Modality] = {
        "audio": Modality.AUDIO,
        "video": Modality.VIDEO,
        "text": Modality.TEXT,
    }

    def __init__(
        self,
        data_fp: str | Path | PathLike,
        split: str,
        selected_patterns: List[str],
        cv_no: int = 1,
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
        target_modality: Modality | str = Modality.MULTIMODAL,
        *,
        target_dir_fp_fmt: str = "target/{cv_no}",
        norm_method: Literal["trn", "utt"] = "trn",
        audio_type: Literal["comparE", "comparE_raw"] = "comparE",
        video_type: Literal["denseface"] = "denseface",
        text_type: Literal["bert", "bert_large"] = "bert_large",
        in_memory: bool = False,
    ) -> None:
        """
        Initialize the IEMOCAP dataset.

        Args:
            data_root (str | Path | PathLike): Path to the root data directory.
            cv_no (int): Cross-validation fold number (1-10).
            split (str): Dataset split, one of "trn", "val", or "tst".
            selected_patterns (List[str]): Selected patterns for the dataset.
            missing_patterns (Optional[Dict[str, Dict[str, float]]]): Dictionary defining missing modality patterns.
            target_dir_fp_fmt (str): Format for target directory paths.
            norm_method (Literal["trn", "utt"]): Normalization method ("trn" or "utt").
            audio_type (Literal["comparE", "comparE_raw"]): Type of audio features.
            video_type (Literal["denseface"]): Type of video features.
            text_type (Literal["bert", "bert_large"]): Type of text features.
            in_memory (bool): Whether to load data into memory.
        """
        # Set up missing patterns
        m_patterns = missing_patterns or {
            "atv": {Modality.AUDIO: 1.0, Modality.TEXT: 1.0, Modality.VIDEO: 1.0},
            "at": {Modality.AUDIO: 1.0, Modality.TEXT: 1.0, Modality.VIDEO: 0.0},
            "av": {Modality.AUDIO: 1.0, Modality.TEXT: 0.0, Modality.VIDEO: 1.0},
            "tv": {Modality.AUDIO: 0.0, Modality.TEXT: 1.0, Modality.VIDEO: 1.0},
            "a": {Modality.AUDIO: 1.0, Modality.TEXT: 0.0, Modality.VIDEO: 0.0},
            "t": {Modality.AUDIO: 0.0, Modality.TEXT: 1.0, Modality.VIDEO: 0.0},
            "v": {Modality.AUDIO: 0.0, Modality.TEXT: 0.0, Modality.VIDEO: 1.0},
        }
        super().__init__(split=split, selected_patterns=selected_patterns, missing_patterns=m_patterns, _id=cv_no)
        assert 1 <= cv_no <= 10, "Cross-validation fold number must be in [1, 10]."

        root = Path(data_fp)
        cv_root = root / target_dir_fp_fmt.format(cv_no=cv_no)

        logger.info(f"Loading IEMOCAP dataset from {cv_root}")
        # Process target modality
        if isinstance(target_modality, str):
            target_modality = Modality.from_str(target_modality)
        assert isinstance(
            target_modality, Modality
        ), f"Invalid modality provided, must be a Modality instance, not {type(target_modality)}"
        assert (
            target_modality in self.AVAILABLE_MODALITIES.values() or target_modality == Modality.MULTIMODAL
        ), f"Invalid target modality provided, must be one of {list(self.AVAILABLE_MODALITIES.values())}"
        self.target_modality = target_modality

        self.norm_method = norm_method
        self.all_A = h5py.File(root / "A" / f"{audio_type}.h5", "r")
        self.all_T = h5py.File(root / "T" / f"{text_type}.h5", "r")
        self.all_V = h5py.File(root / "V" / f"{video_type}.h5", "r")

        if audio_type == "comparE":
            self.mean_std = h5py.File(root / "A" / "comparE_mean_std.h5", "r")
            self.mean = torch.from_numpy(self.mean_std[str(cv_no)]["mean"][()]).unsqueeze(0).float()
            self.std = torch.from_numpy(self.mean_std[str(cv_no)]["std"][()]).unsqueeze(0).float()
        elif audio_type == "comparE_raw":
            self.mean, self.std = self._calc_mean_std()
        else:
            raise ValueError(f"Invalid audio type: {audio_type}")

        self.in_memory = in_memory
        if self.in_memory:
            self.all_A = hdf5_to_dict(self.all_A)
            self.all_T = hdf5_to_dict(self.all_T)
            self.all_V = hdf5_to_dict(self.all_V)

        labels_path = cv_root / f"{split}_label.npy"
        int_to_name_path = cv_root / f"{split}_int2name.npy"
        self.labels = np.argmax(np.load(labels_path), axis=1)
        self.int_to_name = np.load(int_to_name_path)
        self.manual_collate = True
        self.num_samples = len(self.labels)

        # Set up pattern-specific indices for validation/test
        if split != "trn":
            self.pattern_indices = {pattern: list(range(self.num_samples)) for pattern in self.selected_patterns}
        self.masks = self._initialise_missing_masks(self.missing_patterns, len(self))

        logger.info(
            f"Initialized {self.__class__.__name__} dataset:"
            f"\n  Split: {split}"
            f"\n  Target Modality: {target_modality}"
            f"\n  Samples: {self.num_samples}"
            f"\n  In-memory: {self.in_memory}"
            f"\n  CV Number: {cv_no}"
            f"\n  Patterns: {', '.join(self.selected_patterns)}"
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples if self.split == "trn" else self.num_samples * len(self.selected_patterns)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single data sample.

        Args:
            idx (int): Index of the data sample.

        Returns:
            Dict[str, Any]: Data sample containing label, patterns, and modalities.
        """
        _data = super().__getitem__(idx)

        pattern_name, idx = _data.pop("pattern"), _data.pop("sample_idx")
        self.current_pattern = pattern_name
        # pattern_name, idx = self._get_pattern_and_sample_idx(idx)
        int_to_name = self.int_to_name[idx]
        int_to_name = int_to_name[0].decode()
        label = torch.tensor(self.labels[idx])

        sample = {"label": label, "pattern_name": pattern_name, "missing_mask": {}, "sample_idx": idx, **_data}

        modality_loaders = {
            "audio": (lambda: self._load_audio(int_to_name), Modality.AUDIO),
            "video": (lambda: self._load_video(int_to_name), Modality.VIDEO),
            "text": (lambda: self._load_text(int_to_name), Modality.TEXT),
        }
        sample = self.get_samples(sample=sample, modality_loaders=modality_loaders)
        return sample

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for batching with pattern-aware handling.

        Args:
            batch (List[Dict[str, Any]]): List of individual samples.

        Returns:
            Dict[str, Any]: Batched data.
        """
        return self._collate_eval_batch(batch) if self.split != "trn" else self._collate_train_batch(batch)

    def _collate_train_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate training batch with mixed patterns.

        Args:
            batch (List[Dict[str, Any]]): List of training samples.

        Returns:
            Dict[str, Any]: Collated batch.
        """
        collated = {
            "label": torch.stack([b["label"] for b in batch]),
            "pattern_name": [b["pattern_name"] for b in batch],
        }

        for mod_enum in self.AVAILABLE_MODALITIES.values():
            sequences = [b.get(str(mod_enum)) for b in batch if str(mod_enum) in b]
            collated[f"{mod_enum}_missing_index"] = torch.stack(
                [torch.tensor(b[f"{mod_enum}_missing_index"]) for b in batch]
            )
            collated[str(mod_enum)] = pad_sequence(sequences, batch_first=True, padding_value=0) if sequences else None

        return collated

    def _collate_eval_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate evaluation batch with pattern-specific grouping.

        Args:
            batch (List[Dict[str, Any]]): List of evaluation samples.

        Returns:
            Dict[str, Any]: Collated batch grouped by patterns.
        """
        return self._collate_train_batch(batch)
        # pattern_groups = {}
        # for b in batch:
        #     pattern = b["pattern_name"]
        #     pattern_groups.setdefault(pattern, []).append(b)

        # return {pattern: self._collate_train_batch(group) for pattern, group in pattern_groups.items()}

    def _load_audio(self, int_to_name: str) -> Tensor:
        """
        Load and normalize audio features.

        Args:
            idx (int): Sample index.
            int_to_name (str): Corresponding name from the int_to_name mapping.

        Returns:
            Tensor: Normalized audio features.
        """
        A = torch.from_numpy(self.all_A[int_to_name][()]).float()
        A = self._normalize_on_utt(A) if self.norm_method == "utt" else self._normalize_on_train(A)
        return A

    def _load_text(self, int_to_name: str) -> Tensor:
        """
        Load text features.

        Args:
            idx (int): Sample index.
            int_to_name (str): Corresponding name from the int_to_name mapping.

        Returns:
            Tensor: Text features.
        """
        return torch.from_numpy(self.all_T[int_to_name][()]).float()

    def _load_video(self, int_to_name: str) -> Tensor:
        """
        Load video features.

        Args:
            idx (int): Sample index.
            int_to_name (str): Corresponding name from the int_to_name mapping.

        Returns:
            Tensor: Video features.
        """
        return torch.from_numpy(self.all_V[int_to_name][()]).float()

    def _normalize_on_train(self, features: Tensor) -> Tensor:
        """
        Normalize features using training statistics.

        Args:
            features (Tensor): Input features.

        Returns:
            Tensor: Normalized features.
        """
        return (features - self.mean) / self.std

    def _normalize_on_utt(self, features: Tensor) -> Tensor:
        """
        Normalize features at the utterance level.

        Args:
            features (Tensor): Input features.

        Returns:
            Tensor: Normalized features.
        """
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        return (features - mean_f) / std_f

    def _calc_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and standard deviation for audio features.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and standard deviation arrays.
        """
        assert hasattr(self, "all_A"), "Audio features not loaded."
        utt_ids = list(self.all_A.keys())
        feats = np.array([self.all_A[utt_id] for utt_id in utt_ids])
        _feats = feats.reshape(-1, feats.shape[2])
        mean = np.mean(_feats, axis=0)
        std = np.std(_feats, axis=0)
        std[std == 0.0] = 1.0
        return mean, std
