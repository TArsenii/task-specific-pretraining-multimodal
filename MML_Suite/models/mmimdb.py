from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
from data.mmimdb import MMIMDb as MMIMDbDataset
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.utils import safe_detach
from modalities import Modality
from models.gates import GatedBiModalNetwork
from models.maxout import MaxOut
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    Linear,
    Module,
    Sequential,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class MLPGenreClassifier(Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.bn_one = BatchNorm1d(input_size)
        self.fc_one = MaxOut(input_size, hidden_size, use_bias=False)
        self.dropout_one = Dropout(p=0.5)

        self.bn_two = BatchNorm1d(hidden_size)
        self.fc_two = MaxOut(hidden_size, hidden_size, use_bias=False)
        self.dropout_two = Dropout(p=0.5)

        self.bn_three = BatchNorm1d(hidden_size)
        self.fc_three = Linear(hidden_size, output_size)

    def forward(self, tensor):
        tensor = self.bn_one(tensor)
        tensor = self.fc_one(tensor)
        tensor = self.dropout_one(tensor)

        tensor = self.bn_two(tensor)
        tensor = self.fc_two(tensor)
        tensor = self.dropout_two(tensor)
        tensor = self.bn_three(tensor)
        tensor = self.fc_three(tensor)

        return tensor


class MMIMDbModalityEncoder(Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(MMIMDbModalityEncoder, self).__init__()
        self.net = Sequential(
            BatchNorm1d(input_dim),
            Linear(input_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GMUModel(Module):
    def __init__(
        self,
        image_encoder: MMIMDbModalityEncoder,
        text_encoder: MMIMDbModalityEncoder,
        gated_bimodal_network: GatedBiModalNetwork,
        classifier: MLPGenreClassifier,
        binary_threshold: float = 0.5,
    ):
        super(GMUModel, self).__init__()
        self.image_model = image_encoder
        self.text_model = text_encoder
        self.gmu = gated_bimodal_network
        self.mm_mlp = classifier
        self.binary_threshold = binary_threshold

    def flatten_parameters(self):
        pass

    def get_encoder(self, modality: Modality) -> MMIMDbModalityEncoder:
        if modality == Modality.IMAGE:
            return self.image_model
        if modality == Modality.TEXT:
            return self.text_model

    def freeze_irrelevant_parameters(self, modality: Modality) -> None:
        if modality == Modality.IMAGE:
            self.text_model.requires_grad_(False)
            self.image_model.requires_grad_(True)
            self.gmu.requires_grad_(True)
            self.mm_mlp.requires_grad_(True)
        if modality == Modality.TEXT:
            self.text_model.requires_grad_(True)
            self.image_model.requires_grad_(False)
            self.gmu.requires_grad_(True)
            self.mm_mlp.requires_grad_(True)
        if Modality == Modality.MULTIMODAL:
            self.text_model.requires_grad_(True)
            self.image_model.requires_grad_(True)
            self.gmu.requires_grad_(True)
            self.mm_mlp.requires_grad_(True)

    def get_relevant_parameters(self, modality: Modality) -> Dict[str, Tensor]:
        if modality == Modality.IMAGE:
            encoder = self.image_model
            prefix = "iamge_model."

        if modality == Modality.TEXT:
            encoder = self.text_model
            prefix = "text_model."
        if modality == Modality.MULTIMODAL:
            return self.state_dict()

        encoder_params = {f"{prefix}{name}": param for name, param in encoder.state_dict().items()}

        gmu_params = {f"gmu.{name}": param for name, param in self.gmu.state_dict().items()}

        # Get classifier parameters
        classifier_params = {f"mm_mlp.{name}": param for name, param in self.mm_mlp.state_dict().items()}

        # Combine encoder and classifier parameters
        return {**encoder_params, **gmu_params, **classifier_params}

    def forward(
        self,
        I: Tensor,
        T: Tensor,
        *,
        is_embd_I: bool = False,
        is_embd_T: bool = False,
    ) -> Tensor:
        assert not all((I is None, T is None)), "At least one modality must be provided"
        assert not all((is_embd_I, is_embd_T)), "Cannot both be embeddings"

        image = self.image_model(I) if not is_embd_I else I
        text = self.text_model(T) if not is_embd_T else T

        z = self.gmu(image, text)
        logits = self.mm_mlp(z)

        return logits

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: Optimizer,
        criterion: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
    ) -> dict[str, Any]:
        I, T, labels, miss_type = (
            batch[Modality.IMAGE],
            batch[Modality.TEXT],
            batch["label"],
            batch["pattern_name"],
        )

        I, T, labels = (
            I.to(device),
            T.to(device),
            labels.to(device),
        )

        I = I.float()
        T = T.float()

        self.train()
        optimizer.zero_grad()

        logits = self.forward(I=I, T=T)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = safe_detach(torch.nn.functional.sigmoid(logits))
        predictions = (predictions > self.binary_threshold).astype(int)

        labels = safe_detach(labels)

        miss_type = np.array(miss_type)
        metric_recorder.update_all(predictions=predictions, targets=labels, m_types=miss_type)
        return {"loss": loss.item()}

    def validation_step(
        self,
        batch: Dict[str, Any],
        criterion: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        return_test_info: bool = False,
    ):
        self.eval()

        if return_test_info:
            all_predictions = []
            all_labels = []
            all_miss_types = []

        with torch.no_grad():
            I, T, labels, miss_type = (
                batch[Modality.IMAGE],
                batch[Modality.TEXT],
                batch["label"],
                batch["pattern_name"],
            )

            I, T, labels = (
                I.to(device),
                T.to(device),
                labels.to(device),
            )

            I = I.float()
            T = T.float()

            logits = self.forward(I=I, T=T)
            loss = criterion(logits, labels)
            predictions = safe_detach(torch.nn.functional.sigmoid(logits))
            predictions = (predictions > 0.5).astype(int)

            labels = safe_detach(labels)
            if return_test_info:
                all_predictions.append(predictions)
                all_labels.append(labels)
                all_miss_types.append(miss_type)
            miss_type = np.array(miss_type)
            metric_recorder.update_all(predictions=predictions, targets=labels, m_types=miss_type)
        self.train()

        if return_test_info:
            return {
                "loss": loss.item(),
                "predictions": all_predictions,
                "labels": all_labels,
                "miss_types": all_miss_types,
            }

        return {"loss": loss.item()}

    def get_embeddings(self, dataloader: DataLoader, device: torch.device) -> Dict[Modality, np.ndarray]:
        """
        Extract embeddings from the model for a given dataloader.

        Args:
            dataloader (DataLoader): DataLoader for extracting embeddings.
            device (torch.device): Device to perform computations.

        Returns:
            Dict[Modality, np.ndarray]: Extracted embeddings for audio and image.
        """
        embeddings = defaultdict(list)
        self.to(device)
        self.eval()

        for batch in dataloader:
            with torch.no_grad():
                I, T, miss_type = (
                    batch[Modality.IMAGE],
                    batch[Modality.TEXT],
                    batch["pattern_name"],
                )

                # Filter to full modality availability
                miss_type = np.array(miss_type)
                I = I[miss_type == MMIMDbDataset.get_full_modality()]
                T = T[miss_type == MMIMDbDataset.get_full_modality()]

                I, T = I.to(device), T.to(device)
                I = I.float()
                T = T.float()

                image_embedding = self.image_model(I)
                text_embedding = self.text_model(T)

                embeddings[Modality.IMAGE].append(safe_detach(image_embedding))
                embeddings[Modality.TEXT].append(safe_detach(text_embedding))

        return embeddings

    def __str__(self):
        return str(self.image_model) + "\n" + str(self.text_model) + "\n" + str(self.gmu) + "\n" + str(self.mm_mlp)
