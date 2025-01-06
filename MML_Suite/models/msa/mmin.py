from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from experiment_utils.utils import safe_detach
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from modalities import Modality
from models.mixins import MonitoringMixin
from models.msa.networks.autoencoder import ResidualAE
from models.msa.networks.classifier import FcClassifier
from models.msa.networks.lstm import LSTMEncoder
from models.msa.networks.textcnn import TextCNN
from models.msa.utt_fusion import UttFusionModel
from models.protocols import MultimodalModelProtocol
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


class MMIN(Module, MonitoringMixin, MultimodalModelProtocol):
    def __init__(
        self,
        netA: LSTMEncoder,
        netV: LSTMEncoder,
        netT: TextCNN,
        netAE: ResidualAE,
        netC: FcClassifier,
        *,
        clip: Optional[float] = None,
        share_weight: bool = False,
        pretrained_model: Optional[UttFusionModel] = None,
    ) -> None:
        super(MMIN, self).__init__()
        self.netA: LSTMEncoder = netA
        self.netV: LSTMEncoder = netV
        self.netT: TextCNN = netT
        self.netAE: ResidualAE = netAE

        ae_input_dim = self.netA.hidden_size + self.netV.hidden_size + self.netT.hidden_size

        if share_weight:
            self.netAE_cycle = self.netAE
        else:
            self.netAE_cycle = ResidualAE(
                self.netAE._layers, self.netAE.n_blocks, ae_input_dim, dropout=0.0, use_bn=False
            )

        self.netC = netC
        self.pretrained_model: UttFusionModel = pretrained_model
        self.pretrained_model.load_pretrained()  # Load the pre-trained model, but will quit if the model does not have a pretrained path
        self.pretrained_model.eval()
        self.clip = clip

    def train(self):
        """
        Sets the module in training, overriding the default behavior of PyTorch's `train` method.
        This is necessary to ensure that the pre-trained module is not also set to training mode.
        """
        self.netA.train()
        self.netV.train()
        self.netT.train()
        self.netAE.train()
        self.netAE_cycle.train()
        self.netC.train()

    def eval(self):
        self.netA.eval()
        self.netV.eval()
        self.netT.eval()
        self.netAE.eval()

    def forward(
        self, A: Tensor, V: Tensor, T: Tensor, A_reverse: Tensor, V_reverse: Tensor, T_reverse: Tensor
    ) -> Dict[str, Tensor]:
        A_feat_miss = self.netA(A)
        V_feat_miss = self.netV(V)
        T_feat_miss = self.netT(T)

        feat_fusion_miss = torch.cat([A_feat_miss, V_feat_miss, T_feat_miss], dim=-1)

        recon_fusion, latent = self.netAE(feat_fusion_miss)
        recon_cycle, latent_cycle = self.netAE_cycle(recon_fusion)

        logits = self.netC(latent)

        if self.training:
            with torch.no_grad():
                embd_A = self.pretrained_model.netA(A_reverse)
                embd_V = self.pretrained_model.netV(V_reverse)
                embd_T = self.pretrained_model.netT(T_reverse)
                embds = torch.cat([embd_A, embd_V, embd_T], dim=-1)

        return {
            str(Modality.AUDIO): A_feat_miss,
            str(Modality.VIDEO): V_feat_miss,
            str(Modality.TEXT): T_feat_miss,
            "fusion": feat_fusion_miss,
            "embds": embds,
            "recon_fusion": recon_fusion,
            "recon_cycle": recon_cycle,
            "latent": latent,
            "latent_cycle": latent_cycle,
            "logits": logits,
        }

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: Optimizer,
        loss_functions: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        **kwargs,
    ) -> Dict[str, Any]:
        A, V, T, A_reverse, V_reverse, T_reverse, labels, miss_types = (
            batch[str(Modality.AUDIO)],
            batch[str(Modality.VIDEO)],
            batch[str(Modality.TEXT)],
            batch[f"{str(Modality.AUDIO)}_reverse"],
            batch[f"{str(Modality.VIDEO)}_reverse"],
            batch[f"{str(Modality.TEXT)}_reverse"],
            batch["label"],
            batch["pattern_name"],
        )

        A, V, T, A_reverse, V_reverse, T_reverse, labels = (
            A.to(device),
            V.to(device),
            T.to(device),
            A_reverse.to(device),
            V_reverse.to(device),
            T_reverse.to(device),
            labels.to(device),
        )
        miss_types = np.array(miss_types)

        self.train()
        optimizer.zero_grad()

        forward_results = self(A, V, T, A_reverse, V_reverse, T_reverse)
        predictions = forward_results["logits"].argmax(dim=1)

        loss_ce = loss_functions(forward_results["logits"], labels, key="cross_entropy")
        loss_mse = loss_functions(forward_results["fusion"], forward_results["recon_fusion"], key="mse")
        loss_cycle = loss_functions(
            safe_detach(forward_results["fusion"], to_np=False), forward_results["recon_cycle"], key="cycle"
        )

        loss = loss_ce + loss_mse + loss_cycle
        loss.backward()

        ## Clip gradients excluding the pre-trained module
        for parameter in self.parameters():
            if parameter.requires_grad:
                torch.nn.utils.clip_grad_norm_(parameter, self.clip)

        optimizer.step()
        labels = safe_detach(labels)
        predictions = safe_detach(predictions)
        for m_type in set(miss_types):
            mask = miss_types == m_type
            mask_preds = predictions[mask]
            mask_labels = labels[mask]
            metric_recorder.update(predictions=mask_preds, targets=mask_labels, modality=m_type)
        return {
            "loss": loss.item(),
            "losses": {"ce": loss_ce.item(), "mse": loss_mse.item(), "cycle": loss_cycle.item()},
        }

    def validation_step(
        self,
        batch: Dict[str, Any],
        loss_functions: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        **kwargs,
    ) -> Dict[str, Any]:
        with torch.no_grad():
            A, V, T, A_reverse, V_reverse, T_reverse, labels, miss_types = (
                batch[str(Modality.AUDIO)],
                batch[str(Modality.VIDEO)],
                batch[str(Modality.TEXT)],
                batch[str(Modality.AUDIO) + "_reverse"],
                batch[str(Modality.VIDEO) + "_reverse"],
                batch[str(Modality.TEXT) + "_reverse"],
                batch["label"],
                batch["pattern_name"],
            )

            A, V, T, A_reverse, V_reverse, T_reverse, labels = (
                A.to(device),
                V.to(device),
                T.to(device),
                A_reverse.to(device),
                V_reverse.to(device),
                T_reverse.to(device),
                labels.to(device),
            )
            miss_types = np.array(miss_types)

            self.eval()

            forward_results = self(A, V, T, A_reverse, V_reverse, T_reverse)
            predictions = forward_results["logits"].argmax(dim=1)

            loss_ce = loss_functions(forward_results["logits"], labels, key="cross_entropy")
            loss_mse = loss_functions(forward_results["fusion"], forward_results["recon_fusion"], key="mse")
            loss_cycle = loss_functions(
                safe_detach(forward_results["fusion"], to_np=False), forward_results["recon_cycle"], key="cycle"
            )

            loss = loss_ce + loss_mse + loss_cycle

            labels = safe_detach(labels)
            predictions = safe_detach(predictions)
            for m_type in set(miss_types):
                mask = miss_types == m_type
                mask_preds = predictions[mask]
                mask_labels = labels[mask]
                metric_recorder.update(predictions=mask_preds, targets=mask_labels, modality=m_type)
            return {
                "loss": loss.item(),
                "losses": {"ce": loss_ce.item(), "mse": loss_mse.item(), "cycle": loss_cycle.item()},
            }

    # def get_embeddings(
    #     self, dataloader: DataLoader, device: torch.device
    # ) -> Tuple[Dict[Modality, np.ndarray]] | Dict[Modality, np.ndarray]:
    #     """
    #     Get embeddings for all samples in a dataloader.

    #     Args:
    #         dataloader (DataLoader): DataLoader for the dataset.
    #         device (torch.device): Computation device.

    #     Returns:
    #         Dict[Modality, np.ndarray]: Dictionary of embeddings for each modality.
    #     """
    #     self.eval()
    #     embeddings = defaultdict(list)
    #     reconstructions = defaultdict(list)
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             A, V, T = (
    #                 batch[Modality.AUDIO].to(device).float(),
    #                 batch[Modality.VIDEO].to(device).float(),
    #                 batch[Modality.TEXT].to(device).float(),
    #             )
    #             a_embd = self.netA(A)
    #             v_embd = self.netV(V)
    #             t_embd = self.netT(T)

    #             feat_fusion_miss = torch.cat([a_embd, v_embd, t_embd], dim=-1)

    #             recon_fusion, latent = self.netAE(feat_fusion_miss)
    #             a_recon = recon_fusion[:, : self.netA.hidden_size]
    #             v_recon = recon_fusion[:, self.netA.hidden_size : self.netA.hidden_size + self.netV.hidden_size]
    #             t_recon = recon_fusion[:, -self.netT.hidden_size :]

    #             for mod, embd in zip([Modality.AUDIO, Modality.VIDEO, Modality.TEXT], [a_embd, v_embd, t_embd]):
    #                 if embd is not None:
    #                     embeddings[mod].append(safe_detach(embd))

    #             for mod, recon in zip([Modality.AUDIO, Modality.VIDEO, Modality.TEXT], [a_recon, v_recon, t_recon]):
    #                 reconstructions[mod].append(safe_detach(recon))

    #     embeddings: Dict[Modality, np.ndarray] = {mod: np.concatenate(embds) for mod, embds in embeddings.items()}
    #     reconstructions: Dict[Modality, np.ndarray] = {
    #         mod: np.concatenate(recons) for mod, recons in reconstructions.items()
    #     }
    #     return (embeddings, reconstructions)
