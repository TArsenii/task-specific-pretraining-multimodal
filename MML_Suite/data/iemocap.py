from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import h5py
import numpy as np
import torch
from data.base_dataset import MultimodalBaseDataset
from experiment_utils.logging import get_logger
from experiment_utils.utils import hdf5_to_dict
from modalities import Modality
from torch import Tensor

logger = get_logger()


class IEMOCAP(MultimodalBaseDataset):
    """Dataset class for the IEMOCAP multimodal emotion recognition task."""

    VALID_SPLIT: List[str] = ["trn", "val", "tst"]
    NUM_CLASSES: int = 4
    AVAILABLE_MODALITIES: Dict[str, Modality] = {
        "audio": Modality.AUDIO,
        "video": Modality.VIDEO,
        "text": Modality.TEXT,
    }

    def __init__(
        self,
        data_root: str | Path | PathLike,
        cv_no: int,
        split: str,
        selected_patterns: List[str],
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
        *,
        target_dir_fp_fmt: str = "target/{cv_no}",
        norm_method: Literal["trn", "utt"] = "utt",
        audio_type: Literal["compareE", "comparE_raw"] = "compareE",
        video_type: Literal["denseface"] = "denseface",
        text_type: Literal["bert", "bert_large"] = "bert",
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
            audio_type (Literal["compareE", "comparE_raw"]): Type of audio features.
            video_type (Literal["denseface"]): Type of video features.
            text_type (Literal["bert", "bert_large"]): Type of text features.
            in_memory (bool): Whether to load data into memory.
        """
        m_patterns = missing_patterns or {
            "atv": {"audio": 1.0, "text": 1.0, "video": 1.0},
            "at": {"audio": 1.0, "text": 1.0, "video": 0.0},
            "av": {"audio": 1.0, "text": 0.0, "video": 1.0},
            "tv": {"audio": 0.0, "text": 1.0, "video": 1.0},
            "a": {"audio": 1.0, "text": 0.0, "video": 0.0},
            "t": {"audio": 0.0, "text": 1.0, "video": 0.0},
            "v": {"audio": 0.0, "text": 0.0, "video": 1.0},
        }
        super().__init__(split=split, selected_patterns=selected_patterns, missing_patterns=m_patterns)
        assert 1 <= cv_no <= 10, "Cross-validation fold number must be in [1, 10]."

        root = Path(data_root)
        cv_root = root / target_dir_fp_fmt.format(cv_no=cv_no)

        logger.info(f"Loading IEMOCAP dataset from {cv_root}")

        self.norm_method = norm_method
        self.all_A = h5py.File(data_root / "A" / f"{audio_type}.h5", "r")
        self.all_T = h5py.File(data_root / "T" / f"{text_type}.h5", "r")
        self.all_V = h5py.File(data_root / "V" / f"{video_type}.h5", "r")

        if audio_type == "compareE":
            self.mean_std = h5py.File(data_root / "A" / "compareE_mean_std.h5", "r")
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
        int_to_name_path = cv_root / "int2name.npy"
        self.labels = np.argmax(np.load(labels_path), axis=1)
        self.int_to_name = np.load(int_to_name_path)
        self.manual_collate = True
        self.num_samples = len(self.labels)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single data sample.

        Args:
            idx (int): Index of the data sample.

        Returns:
            Dict[str, Any]: Data sample containing label, patterns, and modalities.
        """
        pattern_name, idx = self._get_pattern_and_sample_idx(idx)
        int_to_name = self.int_to_name[idx]
        int_to_name = int_to_name[0].decode()
        label = torch.tensor(self.labels[idx])

        sample = {
            "label": label,
            "pattern_name": pattern_name,
            "missing_mask": {},
            "sample_idx": idx,
        }

        modality_loaders = {
            "audio": (lambda idx: self._load_audio(idx, int_to_name), Modality.AUDIO),
            "video": (lambda idx: self._load_video(idx, int_to_name), Modality.VIDEO),
            "text": (lambda idx: self._load_text(idx, int_to_name), Modality.TEXT),
        }
        sample = self.get_sample_and_apply_mask(pattern_name, sample, modality_loaders)
        return sample

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for batching.

        Args:
            batch (List[Dict[str, Any]]): List of individual samples.

        Returns:
            Dict[str, Any]: Batched data.
        """
        A = [sample[Modality.AUDIO] for sample in batch]
        T = [sample[Modality.TEXT] for sample in batch]
        V = [sample[Modality.VIDEO] for sample in batch]

        A_orig = [sample[f"{Modality.AUDIO}_orig"] for sample in batch]
        T_orig = [sample[f"{Modality.TEXT}_orig"] for sample in batch]
        V_orig = [sample[f"{Modality.VIDEO}_orig"] for sample in batch]

        lengths = torch.tensor([len(a) for a in A]).long()

        A = torch.nn.utils.rnn.pad_sequence(A, batch_first=True, padding_value=0.0)
        T = torch.nn.utils.rnn.pad_sequence(T, batch_first=True, padding_value=0.0)
        V = torch.nn.utils.rnn.pad_sequence(V, batch_first=True, padding_value=0.0)
        label = torch.tensor([sample["label"] for sample in batch]).long()
        int_to_name = [sample["int_to_name"] for sample in batch]

        return {
            Modality.AUDIO: A,
            Modality.VIDEO: V,
            Modality.TEXT: T,
            f"{Modality.AUDIO}_orig": A_orig,
            f"{Modality.TEXT}_orig": T_orig,
            f"{Modality.VIDEO}_orig": V_orig,
            "label": label,
            "lengths": lengths,
            "int_to_name": int_to_name,
        }

    def _load_audio(self, idx: int, int_to_name: str) -> Tensor:
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

    def _load_text(self, idx: int, int_to_name: str) -> Tensor:
        """
        Load text features.

        Args:
            idx (int): Sample index.
            int_to_name (str): Corresponding name from the int_to_name mapping.

        Returns:
            Tensor: Text features.
        """
        return torch.from_numpy(self.all_T[int_to_name][()]).float()

    def _load_video(self, idx: int, int_to_name: str) -> Tensor:
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
