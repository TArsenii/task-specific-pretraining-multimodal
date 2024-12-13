from typing import Any, Dict

import numpy as np
import torch
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.utils import safe_detach
from modalities import Modality
from models.mixins import MonitoringMixin
from models.msa.networks.autoencoder import ResidualAE, ResidualXE
from models.msa.networks.classifier import FcClassifier
from models.msa.networks.transformer import Transformer
from models.protocols import MultimodalModelProtocol
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


class RedCore(Module, MonitoringMixin, MultimodalModelProtocol):
    feature_dim: int = 32  # Magic number
    lambda_one: float = 0.0008  # Another magic number

    def __init__(
        self,
        netA: Transformer,
        netV: Transformer,
        netT: Transformer,
        netAE: ResidualAE,
        netC: FcClassifier,
        netAT_V: ResidualXE,
        netAV_T: ResidualXE,
        netVT_A: ResidualAE,
        netCls_A: FcClassifier,
        netCls_V: FcClassifier,
        netCls_T: FcClassifier,
        share_weight: bool = False,
        loss_beta: float = 0.95,
        interval_i: int = 2,
        eta: float = 0.001,
        eta_ext: float = 1.5,
        clip: float = 1.0,
    ) -> None:
        super(RedCore, self).__init__()
        self.netA = netA
        self.netA.initialize_parameters()
        self.netV = netV
        self.netV.initialize_parameters()
        self.netT = netT
        self.netT.initialize_parameters()

        self.netAE = netAE
        self.netC = netC
        self.netAT_V = netAT_V
        self.netAV_T = netAV_T
        self.netVT_A = netVT_A
        self.netCls_A = netCls_A
        self.netCls_V = netCls_V
        self.netCls_T = netCls_T
        ae_input_dim = self.netA.embd_width + self.netV.embd_width + self.netT.embd_width

        if share_weight:
            self.netAE_cycle = self.netAE
        else:
            self.netAE_cycle = ResidualAE(
                self.netAE._layers, self.netAE.n_blocks, ae_input_dim, dropout=0.0, use_bn=False
            )

        self._loss_A = 0.0
        self._loss_V = 0.0
        self._loss_T = 0.0
        self._loss_beta = loss_beta
        self._beta = np.array([1.0, 1.0, 1.0])
        self._iter_count = 0
        self._interval_i = interval_i

        self._eta = eta
        self._eta_ext = eta_ext
        self.clip = clip

    def train(self) -> None:
        super().train()

    def eval(self) -> None:
        super().eval()

    def forward(
        self,
        A: Tensor,
        V: Tensor,
        T: Tensor,
        A_missing_index: Tensor,
        V_missing_index: Tensor,
        T_missing_index: Tensor,
    ) -> Dict[str, Tensor]:
        feature_A_miss, fmu_A, flog_var_A = self.netA.forward(A)
        feature_V_miss, fmu_V, flog_var_V = self.netV.forward(V)
        feature_T_miss, fmu_T, flog_var_T = self.netT.forward(T)

        feature_fusion_miss = torch.cat([feature_A_miss, feature_V_miss, feature_T_miss], dim=-1)
        recon_fusion, latent = self.netAE.forward(feature_fusion_miss)
        recon_cycle, latent_cycle = self.netAE_cycle.forward(recon_fusion)

        gen_A, _latent_A = self.netVT_A.forward(torch.cat([feature_V_miss, feature_T_miss], dim=-1))
        gen_V, _latent_V = self.netAT_V.forward(torch.cat([feature_A_miss, feature_T_miss], dim=-1))
        gen_T, _latent_T = self.netAV_T.forward(torch.cat([feature_A_miss, feature_V_miss], dim=-1))

        batch_size = feature_A_miss.shape[0]
        feature_A_r = (
            A_missing_index.reshape(batch_size, 1) * feature_A_miss
            - (A_missing_index.reshape(batch_size, 1) - 1) * gen_A
        )

        feature_V_r = (
            V_missing_index.reshape(batch_size, 1) * feature_V_miss
            - (V_missing_index.reshape(batch_size, 1) - 1) * gen_V
        )

        feature_T_r = (
            T_missing_index.reshape(batch_size, 1) * feature_T_miss
            - (T_missing_index.reshape(batch_size, 1) - 1) * gen_T
        )

        ## In the official implementation this is not commented, but they are not used?
        # feature_A_re = A_missing_index.reshape(batch_size, 1) * (feature_A_miss - gen_a)
        # feature_V_re = V_missing_index.reshape(batch_size, 1) * (feature_V_miss - gen_v)
        # feature_T_re = T_missing_index.reshape(batch_size, 1) * (feature_T_miss - gen_t)

        feature_fusion_r = torch.cat([feature_A_r, feature_V_r, feature_T_r], dim=-1)
        logits = self.netC.forward(feature_fusion_r)

        logits_a = self.netCls_A.forward(feature_A_r)
        logits_v = self.netCls_V.forward(feature_V_r)
        logits_t = self.netCls_T.forward(feature_T_r)

        return {
            "logits": logits,
            "fusion": feature_fusion_miss,
            "recon_fusion": recon_fusion,
            "recon_cycle": recon_cycle,
            "latent": latent,
            "latent_cycle": latent_cycle,
            "feature_A_miss": feature_A_miss,
            "feature_V_miss": feature_V_miss,
            "feature_T_miss": feature_T_miss,
            "gen_A": gen_A,
            "logits_A": logits_a,
            "fmu_A": fmu_A,
            "flog_var_A": flog_var_A,
            "gen_V": gen_V,
            "logits_V": logits_v,
            "fmu_V": fmu_V,
            "flog_var_V": flog_var_V,
            "gen_T": gen_T,
            "logits_T": logits_t,
            "fmu_T": fmu_T,
            "flog_var_T": flog_var_T,
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
        A, V, T, missing_index_A, missing_index_A, missing_index_T, labels, miss_types = (
            batch[Modality.AUDIO],
            batch[Modality.VIDEO],
            batch[Modality.TEXT],
            batch[str(Modality.AUDIO) + "_missing_index"],
            batch[str(Modality.VIDEO) + "_missing_index"],
            batch[str(Modality.TEXT) + "_missing_index"],
            batch["label"],
            batch["pattern_name"],
        )

        A, V, T, missing_index_A, missing_index_A, missing_index_T, labels = (
            A.to(device),
            V.to(device),
            T.to(device),
            missing_index_A.to(device),
            missing_index_A.to(device),
            missing_index_T.to(device),
            labels.to(device),
        )
        miss_types = np.array(miss_types)

        self.train()
        optimizer.zero_grad()

        forward_results = self.forward(A, V, T, missing_index_A, missing_index_A, missing_index_T)
        batch_size = missing_index_A.shape[0]

        logits = forward_results["logits"]

        index_A = missing_index_A.reshape(batch_size, 1)
        index_V = missing_index_A.reshape(batch_size, 1)
        index_T = missing_index_T.reshape(batch_size, 1)

        ## Below handles the * cross_entropy_weight variable too, just inside the loss_function call
        ## Each one is equivalent to ``cross_entropy_weight * cross_entropy_loss(logits, labels)``
        loss_ce = loss_functions("cross_entropy", logits, labels)
        loss_ce_A = loss_functions("cross_entropy", forward_results["logits_A"], labels)
        loss_ce_V = loss_functions("cross_entropy", forward_results["logits_V"], labels)
        loss_ce_T = loss_functions("cross_entropy", forward_results["logits_T"], labels)

        fmu_A = forward_results["fmu_A"]
        flog_var_A = forward_results["flog_var_A"]

        fmu_V = forward_results["fmu_V"]
        flog_var_V = forward_results["flog_var_V"]

        fmu_T = forward_results["fmu_T"]
        flog_var_T = forward_results["flog_var_T"]

        KLD_feature_A: Tensor = (
            -1.0
            * self.lambda_one
            * torch.sum((1.0 + flog_var_A - fmu_A.pow(2) - flog_var_A.exp()) * index_A)
            / batch_size
        )

        KLD_feature_V: Tensor = (
            -1.0
            * self.lambda_one
            * torch.sum((1.0 + flog_var_V - fmu_V.pow(2) - flog_var_V.exp()) * index_V)
            / batch_size
        )

        KLD_feature_T: Tensor = (
            -1.0
            * self.lambda_one
            * torch.sum((1.0 + flog_var_T - fmu_T.pow(2) - flog_var_T.exp()) * index_T)
            / batch_size
        )

        batch_size_A = sum(missing_index_A)
        batch_size_V = sum(missing_index_A)
        batch_size_T = sum(missing_index_T)

        feature_A_miss, gen_A = forward_results["feature_A_miss"], forward_results["gen_A"]
        feature_V_miss, gen_V = forward_results["feature_V_miss"], forward_results["gen_V"]
        feature_T_miss, gen_T = forward_results["feature_T_miss"], forward_results["gen_T"]

        loss_mse_A = (
            loss_functions(
                "mse",
                1.0,
                gen_A * index_A,
                feature_A_miss * index_A,
            )
            / batch_size_A
        )

        loss_mse_V = (
            loss_functions(
                "mse",
                1.0,
                gen_V * index_V,
                feature_V_miss * index_V,
            )
            / batch_size_V
        )

        loss_mse_T = (
            loss_functions(
                "mse",
                1.0,
                gen_T * index_T,
                feature_T_miss * index_T,
            )
            / batch_size_T
        )

        loss_update_A = loss_mse_A if loss_mse_A.item() != 0.0 else self._loss_A
        loss_update_V = loss_mse_V if loss_mse_V.item() != 0.0 else self._loss_V
        loss_update_T = loss_mse_T if loss_mse_T.item() != 0.0 else self._loss_T

        self._loss_A = (1.0 - self._loss_beta) * self._loss_A + self._loss_beta * loss_update_A
        self._loss_V = (1.0 - self._loss_beta) * self._loss_V + self._loss_beta * loss_update_V
        self._loss_T = (1.0 - self._loss_beta) * self._loss_T + self._loss_beta * loss_update_T

        loss_ATV = self._loss_A + self._loss_V + self._loss_T
        loss_ATV_avg = loss_ATV / 3.0
        ra = (loss_ATV_avg - loss_ATV) / loss_ATV_avg

        if self._iter_count % 500 == 0:
            self._eta = self._eta * self._eta_ext

        if self._iter_count % self._interval_i == 0:
            self._beta = self._beta * self._eta * ra
            self._beta[0] = max(0.1, self._beta[0])
            self._beta[1] = max(0.1, self._beta[1])
            self._beta[2] = max(0.1, self._beta[2])
            self._beta = self.beta / (sum(self._beta**2) ** (0.5))

        self._iter_count += 1
        mse_weight = loss_functions["mse"].weight

        loss_mse: Tensor = mse_weight * (
            self._beta[0] * loss_mse_A + self._beta[1] * loss_mse_V + self._beta[2] * loss_mse_T
        )
        loss = loss_ce + KLD_feature_A + KLD_feature_V + KLD_feature_T + loss_ce_A + loss_ce_V + loss_ce_T + loss_mse
        loss.backward()
        ## Clip grad norm
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        optimizer.step()

        predictions = logits.argmax(dim=1)
        labels = safe_detach(labels)
        predictions = safe_detach(predictions)
        for m_type in set(miss_types):
            mask = miss_types == m_type
            mask_preds = predictions[mask]
            mask_labels = labels[mask]
            metric_recorder.update(predictions=mask_preds, targets=mask_labels, modality=m_type)
        return {
            "loss": loss.item(),
            "losses": {
                "ce": loss_ce.item(),
                "mse": loss_mse.item(),
                "KLD_A": KLD_feature_A.item(),
                "KLD_V": KLD_feature_V.item(),
                "KLD_T": KLD_feature_T.item(),
                "ce_A": loss_ce_A.item(),
                "ce_V": loss_ce_V.item(),
                "ce_T": loss_ce_T.item(),
                "mse_A": loss_mse_A.item(),
                "mse_V": loss_mse_V.item(),
                "mse_T": loss_mse_T.item(),
            },
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
            A, V, T, missing_index_A, missing_index_A, missing_index_T, labels, miss_types = (
                batch[Modality.AUDIO],
                batch[Modality.VIDEO],
                batch[Modality.TEXT],
                batch[str(Modality.AUDIO) + "_missing_index"],
                batch[str(Modality.VIDEO) + "_missing_index"],
                batch[str(Modality.TEXT) + "_missing_index"],
                batch["label"],
                batch["pattern_name"],
            )

        A, V, T, missing_index_A, missing_index_A, missing_index_T, labels = (
            A.to(device),
            V.to(device),
            T.to(device),
            missing_index_A.to(device),
            missing_index_A.to(device),
            missing_index_T.to(device),
            labels.to(device),
        )
        miss_types = np.array(miss_types)

        self.eval()()

        forward_results = self.forward(A, V, T, missing_index_A, missing_index_A, missing_index_T)
        batch_size = missing_index_A.shape[0]

        logits = forward_results["logits"]

        index_A = missing_index_A.reshape(batch_size, 1)
        index_V = missing_index_A.reshape(batch_size, 1)
        index_T = missing_index_T.reshape(batch_size, 1)

        ## Below handles the * cross_entropy_weight variable too, just inside the loss_function call
        ## Each one is equivalent to ``cross_entropy_weight * cross_entropy_loss(logits, labels)``
        loss_ce = loss_functions("cross_entropy", logits, labels)
        loss_ce_A = loss_functions("cross_entropy", forward_results["logits_A"], labels)
        loss_ce_V = loss_functions("cross_entropy", forward_results["logits_V"], labels)
        loss_ce_T = loss_functions("cross_entropy", forward_results["logits_T"], labels)

        fmu_A = forward_results["fmu_A"]
        flog_var_A = forward_results["flog_var_A"]

        fmu_V = forward_results["fmu_V"]
        flog_var_V = forward_results["flog_var_V"]

        fmu_T = forward_results["fmu_T"]
        flog_var_T = forward_results["flog_var_T"]

        KLD_feature_A: Tensor = (
            -1.0
            * self.lambda_one
            * torch.sum((1.0 + flog_var_A - fmu_A.pow(2) - flog_var_A.exp()) * index_A)
            / batch_size
        )

        KLD_feature_V: Tensor = (
            -1.0
            * self.lambda_one
            * torch.sum((1.0 + flog_var_V - fmu_V.pow(2) - flog_var_V.exp()) * index_V)
            / batch_size
        )

        KLD_feature_T: Tensor = (
            -1.0
            * self.lambda_one
            * torch.sum((1.0 + flog_var_T - fmu_T.pow(2) - flog_var_T.exp()) * index_T)
            / batch_size
        )

        batch_size_A = sum(missing_index_A)
        batch_size_V = sum(missing_index_A)
        batch_size_T = sum(missing_index_T)

        feature_A_miss, gen_A = forward_results["feature_A_miss"], forward_results["gen_A"]
        feature_V_miss, gen_V = forward_results["feature_V_miss"], forward_results["gen_V"]
        feature_T_miss, gen_T = forward_results["feature_T_miss"], forward_results["gen_T"]

        loss_mse_A = (
            loss_functions(
                "mse",
                1.0,
                gen_A * index_A,
                feature_A_miss * index_A,
            )
            / batch_size_A
        )

        loss_mse_V = (
            loss_functions(
                "mse",
                1.0,
                gen_V * index_V,
                feature_V_miss * index_V,
            )
            / batch_size_V
        )

        loss_mse_T = (
            loss_functions(
                "mse",
                1.0,
                gen_T * index_T,
                feature_T_miss * index_T,
            )
            / batch_size_T
        )

        loss_update_A = loss_mse_A if loss_mse_A.item() != 0.0 else self._loss_A
        loss_update_V = loss_mse_V if loss_mse_V.item() != 0.0 else self._loss_V
        loss_update_T = loss_mse_T if loss_mse_T.item() != 0.0 else self._loss_T

        self._loss_A = (1.0 - self._loss_beta) * self._loss_A + self._loss_beta * loss_update_A
        self._loss_V = (1.0 - self._loss_beta) * self._loss_V + self._loss_beta * loss_update_V
        self._loss_T = (1.0 - self._loss_beta) * self._loss_T + self._loss_beta * loss_update_T

        loss_ATV = self._loss_A + self._loss_V + self._loss_T
        loss_ATV_avg = loss_ATV / 3.0
        ra = (loss_ATV_avg - loss_ATV) / loss_ATV_avg

        if self._iter_count % 500 == 0:
            self._eta = self._eta * self._eta_ext

        if self._iter_count % self._interval_i == 0:
            self._beta = self._beta * self._eta * ra
            self._beta[0] = max(0.1, self._beta[0])
            self._beta[1] = max(0.1, self._beta[1])
            self._beta[2] = max(0.1, self._beta[2])
            self._beta = self.beta / (sum(self._beta**2) ** (0.5))

        self._iter_count += 1
        mse_weight = loss_functions["mse"].weight

        loss_mse: Tensor = mse_weight * (
            self._beta[0] * loss_mse_A + self._beta[1] * loss_mse_V + self._beta[2] * loss_mse_T
        )
        loss = loss_ce + KLD_feature_A + KLD_feature_V + KLD_feature_T + loss_ce_A + loss_ce_V + loss_ce_T + loss_mse

        predictions = logits.argmax(dim=1)
        labels = safe_detach(labels)
        predictions = safe_detach(predictions)
        for m_type in set(miss_types):
            mask = miss_types == m_type
            mask_preds = predictions[mask]
            mask_labels = labels[mask]
            metric_recorder.update(predictions=mask_preds, targets=mask_labels, modality=m_type)
        return {
            "loss": loss.item(),
            "losses": {
                "ce": loss_ce.item(),
                "mse": loss_mse.item(),
                "KLD_A": KLD_feature_A.item(),
                "KLD_V": KLD_feature_V.item(),
                "KLD_T": KLD_feature_T.item(),
                "ce_A": loss_ce_A.item(),
                "ce_V": loss_ce_V.item(),
                "ce_T": loss_ce_T.item(),
                "mse_A": loss_mse_A.item(),
                "mse_V": loss_mse_V.item(),
                "mse_T": loss_mse_T.item(),
            },
        }

    # def get_embeddings(self, dataloader: DataLoader, device: torch.device) -> Dict[Modality, np.ndarray]:
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

    #             for mod, embd in zip([Modality.AUDIO, Modality.VIDEO, Modality.TEXT], [a_embd, v_embd, t_embd]):
    #                 if embd is not None:
    #                     embeddings[mod].append(safe_detach(embd))

    #     embeddings: Dict[Modality, np.ndarray] = {mod: np.concatenate(embds) for mod, embds in embeddings.items()}
    #     return embeddings
