from __future__ import annotations
import numpy as np
import torch
from torch.optim import Optimizer
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from typing import Any, Dict, Optional

from modalities import Modality
from experiment_utils import safe_detach
from experiment_utils.loss import LossFunctionGroup
from models.msa import LSTMEncoder, TextCNN, ResidualAE, FcClassifier, UttFusionModel
from experiment_utils.metric_recorder import MetricRecorder


class MMIN(Module):
    def __init__(
        self,
        netA: LSTMEncoder,
        netV: LSTMEncoder,
        netT: TextCNN,
        netAE: ResidualAE,
        netC: FcClassifier,
        *,
        share_weight: bool = False,
        pretrained_module: Optional[UttFusionModel] = None,
    ):
        super(MMIN, self).__init__()
        ## loss names
        ## model names

        self.netA = netA
        self.netV = netV
        self.netT = netT
        self.netAE = netAE

        ae_input_dim = self.netA.hidden_size + self.netV.hidden_size + self.netT.hidden_size

        if share_weight:
            self.netAE_cycle = self.netAE
        else:
            self.netAE_cycle = ResidualAE(
                self.netAE._layers, self.netAE.n_blocks, ae_input_dim, dropout=0.0, use_bn=False
            )

        self.netC = netC
        self.pretrained_module: UttFusionModel = pretrained_module

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

        logits, _ = self.netC(latent)

        if self.training:
            with torch.no_grad():
                embd_A = self.pretrained_module.netA(A_reverse)
                embd_V = self.pretrained_module.netV(V_reverse)
                embd_T = self.pretrained_module.netT(T_reverse)
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
    ) -> Dict:
        A, V, T, A_reverse, V_reverse, T_reverse, labels, miss_types = (
            batch[Modality.AUDIO],
            batch[Modality.VIDEO],
            batch[Modality.TEXT],
            batch[str(Modality.AUDIO) + "_reverse"],
            batch[str(Modality.VIDEO) + "_reverse"],
            batch[str(Modality.TEXT) + "_reverse"],
            batch["label"],
            batch["miss_type"],
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

        loss_ce = loss_functions["ce"](forward_results["logits"], labels)
        loss_mse = loss_functions["mse"](forward_results["fusion"], forward_results["recon_fusion"])
        loss_cycle = loss_functions["cycle"](
            safe_detach(forward_results["fusion"], to_np=False), forward_results["recon_cycle"]
        )

        loss = loss_ce + loss_mse + loss_cycle
        loss.backward()

        ## Clip gradients excluding the pre-trained module
        for parameter in self.parameters():
            if torch.requires_grad_(parameter):
                torch.nn.utils.clip_grad_norm_(parameter, self.clip)

        optimizer.step()
        labels = safe_detach(labels)
        predictions = safe_detach(predictions)
        for m_type in set(miss_types):
            mask = miss_types == m_type
            mask_preds = predictions[mask]
            mask_labels = labels[mask]
            metric_recorder.update(predictions=mask_preds, targets=mask_labels, modality=m_type)
        return {"loss": loss.item()}

    def validation_step(
        self,
        batch: Dict[str, Any],
        loss_functions: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        **kwargs,
    ) -> Dict:
        with torch.no_grad():
            A, V, T, A_reverse, V_reverse, T_reverse, labels, miss_types = (
                batch[Modality.AUDIO],
                batch[Modality.VIDEO],
                batch[Modality.TEXT],
                batch[str(Modality.AUDIO) + "_reverse"],
                batch[str(Modality.VIDEO) + "_reverse"],
                batch[str(Modality.TEXT) + "_reverse"],
                batch["label"],
                batch["miss_type"],
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
            predictions = F.softmax(forward_results["logits"], dim=-1)

            loss_ce = loss_functions["ce"](forward_results["logits"], labels)
            loss_mse = loss_functions["mse"](forward_results["fusion"], forward_results["recon_fusion"])
            loss_cycle = loss_functions["cycle"](
                safe_detach(forward_results["fusion"], to_np=False), forward_results["recon_cycle"]
            )

            loss = loss_ce + loss_mse + loss_cycle

            labels = safe_detach(labels)
            predictions = safe_detach(predictions)
            for m_type in set(miss_types):
                mask = miss_types == m_type
                mask_preds = predictions[mask]
                mask_labels = labels[mask]
                metric_recorder.update(predictions=mask_preds, targets=mask_labels, modality=m_type)
            return {"loss": loss.item()}
