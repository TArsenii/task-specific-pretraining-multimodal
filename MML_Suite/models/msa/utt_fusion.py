from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.utils import safe_detach
from experiment_utils.printing import get_console
from modalities import Modality
from models.msa.networks.classifier import FcClassifier
from models.msa.networks.lstm import LSTMEncoder
from models.msa.networks.textcnn import TextCNN
from torch.nn import Module
from torch.optim import Optimizer
import torch.nn.functional as F

console = get_console()


class UttFusionModel(Module):
    def __init__(
        self,
        netA: LSTMEncoder,
        netV: LSTMEncoder,
        netT: TextCNN,
        netC: FcClassifier,
        clip: float = 0.5,
        *,
        pretrained_path: Optional[str] = None,
    ):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(UttFusionModel, self).__init__()

        self.netA = netA
        self.netV = netV
        self.netT = netT
        self.netC = netC

        self.clip = clip

        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path, weights_only=True))

    def get_encoder(self, modality: Modality | str):
        if isinstance(modality, str):
            modality = Modality.from_str(modality)
        match modality:
            case Modality.AUDIO:
                return self.netA
            case Modality.VIDEO:
                return self.netV
            case Modality.TEXT:
                return self.netT
            case _:
                raise ValueError(f"Unknown modality: {modality}")

    ## new forward function takes in values rather than from self. Missing data is expected to be applied PRIOR to calling forward
    ## C-MAM generated features should be passed in in-place of the original feature(s)
    def forward(
        self,
        A: torch.Tensor = None,
        V: torch.Tensor = None,
        T: torch.Tensor = None,
        is_embd_A: bool = False,
        is_embd_V: bool = False,
        is_embd_T: bool = False,
        device: torch.device = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        assert not all((A is None, V is None, T is None)), "At least one of A, V, L must be provided"
        assert not all([is_embd_A, is_embd_V, is_embd_T]), "Cannot have all embeddings as True"

        if A is not None:
            batch_size = A.size(0)
        elif V is not None:
            batch_size = V.size(0)
        else:
            batch_size = T.size(0)
        if A is None:
            A = torch.zeros(batch_size, 1, self.input_size_a).to(device)
        if V is None:
            V = torch.zeros(
                batch_size,
                1,
                self.input_size_v,
            ).to(device)
        if T is None:
            T = torch.zeros(batch_size, 50, self.input_size_t).to(device)

        a_embd = self.netA(A) if not is_embd_A else A
        v_embd = self.netV(V) if not is_embd_V else V
        t_embd = self.netT(T) if not is_embd_T else T
        fused = torch.cat([a_embd, v_embd, t_embd], dim=-1)
        logits = self.netC(fused)
        return logits

    def validation_step(
        self,
        batch: Dict[str, Any],
        criterion: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        return_test_info: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        self.eval()

        if return_test_info:
            all_predictions = []
            all_labels = []
            all_miss_types = []

        with torch.no_grad():
            (
                A,
                V,
                T,
                labels,
                miss_type,
            ) = (
                batch[Modality.AUDIO],
                batch[Modality.VIDEO],
                batch[Modality.TEXT],
                batch["label"],
                batch["pattern_name"],
            )

            A, V, T, labels = (
                A.to(device),
                V.to(device),
                T.to(device),
                labels.to(device),
            )

            A = A.float()
            V = V.float()
            T = T.float()

            logits = self.forward(A, V, T)
            predictions = logits.argmax(dim=-1)

            if return_test_info:
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels)
                all_miss_types.append(miss_type)

            labels = labels.squeeze()
            logits = logits.squeeze()

            loss = criterion(logits, labels)

            labels = safe_detach(labels)
            predictions = safe_detach(predictions).squeeze()

            metrics = {}
            miss_types = np.array(miss_type)
            for m_type in set(miss_type):
                mask = miss_types == m_type
                mask_labels = labels[mask]
                mask_preds = predictions[mask]
                metric_recorder.update(mask_preds, mask_labels, m_type)

        if return_test_info:
            return {
                "loss": loss.item(),
                **metrics,
                "predictions": all_predictions,
                "labels": all_labels,
                "miss_types": all_miss_types,
            }
        return {"loss": loss.item(), **metrics}

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: Optimizer,
        criterion: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        **kwargs,
    ) -> dict:
        """
        Perform a single training step.

        Args:
            batch (tuple): Tuple containing (audio, video, text, labels, mask, lengths).
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            criterion (torch.nn.Module): The loss function.
            device (torch.device): The device to run the computations on.

        Returns:
            dict: A dictionary containing the loss and other metrics.
        """

        A, V, T, labels, _miss_type = (
            batch[Modality.AUDIO],
            batch[Modality.VIDEO],
            batch[Modality.TEXT],
            batch["label"],
            batch["pattern_name"],
        )

        (
            A,
            V,
            T,
            labels,
        ) = (
            A.to(device),
            V.to(device),
            T.to(device),
            labels.to(device),
        )

        A = A.float()
        V = V.float()
        T = T.float()

        self.train()
        logits = self.forward(A, V, T, device=device)

        predictions = logits.argmax(dim=-1)

        optimizer.zero_grad()
        labels = labels.squeeze()
        logits = logits.squeeze()
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.netA.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.netV.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.netT.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.netC.parameters(), self.clip)

        optimizer.step()

        labels = safe_detach(labels)
        predictions = safe_detach(predictions).squeeze()
        _miss_type = np.array(_miss_type)
        for m_type in set(_miss_type):
            mask = _miss_type == m_type
            mask_labels = labels[mask]
            mask_preds = predictions[mask]
            metric_recorder.update(mask_preds, mask_labels, m_type)

        return {
            "loss": loss.item(),
        }

    def flatten_parameters(self):
        self.netA.rnn.flatten_parameters()
        self.netV.rnn.flatten_parameters()
