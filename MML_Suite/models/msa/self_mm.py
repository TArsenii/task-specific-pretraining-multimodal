# self supervised multimodal multi-task learning network

from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from data import MOSI
from experiment_utils.logging import LoggerSingleton, get_logger
from experiment_utils.managers import CenterManager, FeatureManager, LabelManager
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.printing import EnhancedConsole, get_console
from experiment_utils.utils import safe_detach
from modalities import Modality
from models.mixins import MultimodalMonitoringMixin
from models.msa.networks.avsubset import AuViSubNet
from models.msa.networks.bert_text_encoder import BertTextEncoder
from models.protocols import MultimodalModelProtocol
from torch import Tensor
from torch.nn import Dropout, Linear, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

DEFAULT_TEXT_LENGTH: int = 50
console: EnhancedConsole = get_console()
logger: LoggerSingleton = get_logger()


class Self_MM(Module, MultimodalMonitoringMixin, MultimodalModelProtocol):
    def __init__(
        self,
        audio_encoder: AuViSubNet,
        video_encoder: AuViSubNet,
        text_encoder: BertTextEncoder,
        metric_recorder: MetricRecorder,
        *,
        need_data_aligned: bool,
        audio_out: int,
        video_out: int,
        text_out: int,
        post_fusion_dropout: float,
        post_fusion_dim: int,
        post_text_dropout: float,
        post_text_dim: int,
        post_audio_dropout: float,
        post_audio_dim: int,
        post_video_dropout: float,
        post_video_dim: int,
        feature_manager: FeatureManager,
        labels_manager: LabelManager,
        center_manager: CenterManager,
        H: float = 3.0,
        update_every: int = 1,
    ):
        super(Self_MM, self).__init__()

        # Storing parameters
        self.need_data_aligned = need_data_aligned

        # audio-vision subnets
        self.audio_model = audio_encoder
        self.video_model = video_encoder
        self.text_model = text_encoder
        # post-fusion layers
        self.post_fusion_dropout = Dropout(p=post_fusion_dropout)
        self.post_fusion_layer_1 = Linear(text_out + video_out + audio_out, post_fusion_dim)
        self.post_fusion_layer_2 = Linear(post_fusion_dim, post_fusion_dim)
        self.post_fusion_layer_3 = Linear(post_fusion_dim, 1)

        # classify layers for text, audio, and video
        self.post_text_dropout = Dropout(p=post_text_dropout)
        self.post_text_layer_1 = Linear(text_out, post_text_dim)
        self.post_text_layer_2 = Linear(post_text_dim, post_text_dim)
        self.post_text_layer_3 = Linear(post_text_dim, 1)

        self.post_audio_dropout = Dropout(p=post_audio_dropout)
        self.post_audio_layer_1 = Linear(audio_out, post_audio_dim)
        self.post_audio_layer_2 = Linear(post_audio_dim, post_audio_dim)
        self.post_audio_layer_3 = Linear(post_audio_dim, 1)

        self.post_video_dropout = Dropout(p=post_video_dropout)
        self.post_video_layer_1 = Linear(video_out, post_video_dim)
        self.post_video_layer_2 = Linear(post_video_dim, post_video_dim)
        self.post_video_layer_3 = Linear(post_video_dim, 1)

        self.metric_recorder = metric_recorder
        self.feature_manager = feature_manager
        self.labels_manager = labels_manager
        self.center_manager = center_manager
        self.update_every = update_every
        self.saved_labels = {}
        self.H = H

    def post_init_with_dataloaders(self, dataloaders: DataLoader | Dict[str, DataLoader]):
        dataloader = dataloaders if isinstance(dataloaders, DataLoader) else dataloaders["train"]
        self.feature_manager.set_num_samples(dataloader.dataset.__len__())
        self.labels_manager.set_num_samples(dataloader.dataset.__len__())
        console.start_task("Label Initialisation", total=len(dataloader))
        for batch in dataloader:
            labels = batch["label"].view(-1)
            indexes = batch["sample_idx"].view(-1)
            self.labels_manager.init_labels(indexes=indexes, labels=labels)
            console.update_task("Label Initialisation", advance=1)
        console.complete_task("Label Initialisation")

    def get_encoder(self, modality: Modality):
        match modality:
            case Modality.AUDIO:
                return self.audio_model
            case Modality.VIDEO:
                return self.video_model
            case Modality.TEXT:
                return self.text_model
            case _:
                raise ValueError(f"Unknown modality ({modality}) provided")

    def flatten_parameters(self):
        pass

    def forward(
        self,
        A: Tensor,
        V: Tensor,
        T: Tensor,
        *,
        is_embd_A: bool = False,
        is_embd_V: bool = False,
        is_embd_T: bool = False,
        device: torch.device = None,
    ) -> Tensor:
        audio, audio_lengths = A
        video, video_lengths = V
        text = T

        if A is not None:
            batch_size = audio.size(0)
        elif V is not None:
            batch_size = video.size(0)
        else:
            batch_size = text.size(0)
        if A is None:
            audio, audio_lengths = torch.zeros(batch_size, 1, self.input_size_a).to(device), 0
        if V is None:
            video, video_lengths = torch.zeros(batch_size, 1, self.input_size_v, 0).to(device)
        if T is None:
            text, text_lengths = torch.zeros(batch_size, 50, self.input_size_t).to(device), 0
        else:
            mask_len = torch.sum(text[:, 1, :], dim=1, keepdim=True)
            text_lengths = safe_detach(mask_len.squeeze().int(), to_np=False)
            # change any 0 length to some set length
            text_lengths[text_lengths == 0] = DEFAULT_TEXT_LENGTH

        text = self.text_model(text)[:, 0, :] if not is_embd_T else text

        if not self.need_data_aligned:
            audio = self.audio_model(audio, text_lengths) if not is_embd_A else audio
            video = self.video_model(video, text_lengths) if not is_embd_V else video
        else:
            audio = self.audio_model(audio, audio_lengths) if not is_embd_A else audio
            video = self.video_model(video, video_lengths) if not is_embd_V else video

        # fusion
        fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio

        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = {
            "predictions": {
                Modality.MULTIMODAL: output_fusion,
                Modality.AUDIO: output_audio,
                Modality.VIDEO: output_video,
                Modality.TEXT: output_text,
            },
            "features": {
                Modality.MULTIMODAL: fusion_h,
                Modality.AUDIO: audio_h,
                Modality.VIDEO: video_h,
                Modality.TEXT: text_h,
            },
            "features_pre_activation": {
                Modality.AUDIO: audio,
                Modality.VIDEO: video,
                Modality.TEXT: text,
            },
        }

        return res

    def train_step(
        self,
        batch: Dict[str | Modality, Any],
        optimizer: Optimizer,
        criterion: Module,  # Necessary for the main driver code, but not necessary within this function
        device: torch.device,
        epoch: int,  # TODO: Add this to the main driver code, kwargs when not necessary
    ) -> Dict[str, Any]:
        if epoch % self.update_every == 0:
            optimizer.zero_grad()

        A, V, T, labels, miss_types, indexes = (
            batch[Modality.AUDIO],
            batch[Modality.VIDEO],
            batch[Modality.TEXT],
            batch["label"],
            batch["pattern_name"],
            batch["sample_idx"].view(-1),
        )

        A = A.to(device)

        V = V.to(device)

        T = T.to(device)

        labels = labels.to(device)

        if self.need_data_aligned:
            A_lengths, V_lengths = batch["audio_length"].to(device), batch["video_length"].to(device)
        else:
            A_lengths, V_lengths = 0, 0

        outputs = self.forward(
            (A, A_lengths),
            (V, V_lengths),
            T,
        )

        loss = 0.0
        for modality in [Modality.MULTIMODAL, Modality.AUDIO, Modality.VIDEO, Modality.TEXT]:
            if modality in outputs["predictions"]:
                loss += self._weighted_loss(
                    outputs["predictions"][modality],
                    self.labels_manager.get_labels(modality=modality, indexes=indexes),
                    indexes=indexes,
                    modality=modality,
                )
        loss.backward()

        predictions = outputs["predictions"][Modality.MULTIMODAL]
        features = outputs["features"]

        features = {k: safe_detach(v, to_np=False) for k, v in features.items()}

        if epoch > 1:
            self._update_labels(features=features, current_epoch=epoch, indexes=indexes)

        # self.feature_manager.update(features=features, indexes=indexes)
        self._update_features(features=features, indexes=indexes)
        self._update_centers()

        if epoch % self.update_every == 0:
            optimizer.step()

        # calculate metrics and return loss and metrics
        miss_types = np.array(miss_types)
        for m_type in miss_types:
            mask = miss_types == m_type
            mask_preds = predictions[mask].view(-1)
            mask_labels = labels[mask]
            self.metric_recorder.update(predictions=mask_preds, targets=mask_labels, modality=m_type)

        return {"loss": loss.item()}

    def validation_step(self, batch, criterion, device, return_test_info: bool = False) -> Dict[str, Any]:
        self.eval()
        if return_test_info:
            all_predictions = []
            all_labels = []
            all_miss_types = []

        A, V, T, labels, miss_types = (
            batch[Modality.AUDIO],
            batch[Modality.VIDEO],
            batch[Modality.TEXT],
            batch["label"],
            batch["pattern_name"],
        )

        A = A.to(device)
        V = V.to(device)
        T = T.to(device)

        labels = labels.to(device)

        if self.need_data_aligned:
            A_lengths, V_lengths = batch["audio_length"].to(device), batch["video_length"].to(device)
        else:
            A_lengths, V_lengths = 0, 0

        outputs = self.forward(
            (A, A_lengths),
            (V, V_lengths),
            T,
        )
        miss_types = np.array(miss_types)

        predictions = outputs["predictions"][Modality.MULTIMODAL]
        loss = self._weighted_loss(predictions, labels)

        if return_test_info:
            all_predictions.append(safe_detach(predictions, to_np=True))
            all_labels.append(labels)
            all_miss_types.append(miss_types)

        for m_type in miss_types:
            mask = miss_types == m_type
            mask_preds = predictions[mask].view(-1)
            mask_labels = labels[mask]
            self.metric_recorder.update(predictions=mask_preds, targets=mask_labels, modality=m_type)

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
        console = get_console()
        console.print("Getting embeddings...")
        console.start_task("Embeddings", len(dataloader))
        embeddings = defaultdict(list)
        self.to(device)
        self.eval()

        for batch in dataloader:
            with torch.no_grad():
                A, V, T, labels, miss_types = (
                    batch[Modality.AUDIO],
                    batch[Modality.VIDEO],
                    batch[Modality.TEXT],
                    batch["label"],
                    batch["pattern_name"],
                )
                miss_types = np.array(miss_types)

                A = A[miss_types == MOSI.get_full_modality()].to(device)
                V = V[miss_types == MOSI.get_full_modality()].to(device)
                T = T[miss_types == MOSI.get_full_modality()].to(device)

                labels = labels.to(device)

                if self.need_data_aligned:
                    A_lengths, V_lengths = batch["audio_length"].to(device), batch["video_length"].to(device)
                else:
                    A_lengths, V_lengths = 0, 0

                outputs = self.forward(
                    (A, A_lengths),
                    (V, V_lengths),
                    T,
                )

                audio_features, video_features, text_features = (
                    outputs["features_pre_activation"][Modality.AUDIO],
                    outputs["features_pre_activation"][Modality.VIDEO],
                    outputs["features_pre_activation"][Modality.TEXT],
                )

                embeddings[Modality.AUDIO].append(safe_detach(audio_features, to_np=True))
                embeddings[Modality.VIDEO].append(safe_detach(video_features, to_np=True))
                embeddings[Modality.TEXT].append(safe_detach(text_features, to_np=True))

            console.update_task("Embeddings", advance=1)

        console.complete_task("Embeddings")
        console.print(
            f"Gathered {len(embeddings[Modality.AUDIO])} audio embeddings, {len(embeddings[Modality.VIDEO])} video embeddings and {len(embeddings[Modality.TEXT])} text embeddings"
        )
        embeddings: Dict[Modality, np.ndaray] = {k: np.concatenate(v, axis=0) for k, v in embeddings.items()}
        return embeddings

    def _weighted_loss(self, y_pred, y_true, indexes=None, modality: Modality = Modality.MULTIMODAL) -> Tensor:
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        weighted = (
            torch.ones_like(y_pred)
            if modality == Modality.MULTIMODAL
            else torch.tanh(
                torch.abs(
                    self.labels_manager.get_labels(modality=modality, indexes=indexes)
                    - self.labels_manager.get_labels(modality=Modality.MULTIMODAL, indexes=indexes)
                )
            )
        )
        return torch.mean(weighted * torch.abs(y_pred - y_true))

    def _update_features(self, features, indexes) -> None:
        self.feature_manager.update(features=features, indexes=indexes)

    def _update_centers(self) -> None:
        for modality in [Modality.MULTIMODAL, Modality.AUDIO, Modality.VIDEO, Modality.TEXT]:
            labels = self.labels_manager[modality]
            self.center_manager.update(features=self.feature_manager.feature_maps, labels=labels)

    def _update_labels(self, features, current_epoch, indexes) -> None:
        def update_single_label(f, modality, delta_f, indexes) -> None:
            d_sp = torch.norm(f - self.center_manager.get_center(modality=modality, polarity="pos"), dim=-1)
            d_sn = torch.norm(f - self.center_manager.get_center(modality=modality, polarity="neg"), dim=-1)
            delta_s = (d_sn - d_sp) / (d_sp) + 1e-8
            alpha = delta_s / (delta_f + 1e-8)
            new_labels = 0.5 * alpha * self.labels_manager.get_labels(Modality.MULTIMODAL, indexes=indexes) + 0.5 * (
                self.labels_manager.get_labels(Modality.MULTIMODAL, indexes=indexes) + delta_s - delta_f
            )
            new_labels = torch.clamp(new_labels, min=-self.H, max=self.H)
            new_labels = (current_epoch - 1) / (current_epoch + 1) * self.labels_manager.get_labels(
                modality=modality, indexes=indexes
            ) + 2 / (current_epoch + 1) * new_labels
            self.labels_manager.update_labels(modality=modality, indexes=indexes, new_labels=new_labels)

        d_fp = torch.norm(
            features[Modality.MULTIMODAL]
            - self.center_manager.get_center(modality=Modality.MULTIMODAL, polarity="pos"),
            dim=-1,
        )
        d_fn = torch.norm(
            features[Modality.MULTIMODAL]
            - self.center_manager.get_center(modality=Modality.MULTIMODAL, polarity="neg"),
            dim=-1,
        )
        delta_f = (d_fn - d_fp) / (d_fp + 1e-8)

        update_single_label(f=features[Modality.AUDIO], modality=Modality.AUDIO, delta_f=delta_f, indexes=indexes)
        logger.info("Updated audio labels")
        update_single_label(f=features[Modality.VIDEO], modality=Modality.VIDEO, delta_f=delta_f, indexes=indexes)
        logger.info("Updated video labels")
        update_single_label(f=features[Modality.TEXT], modality=Modality.TEXT, delta_f=delta_f, indexes=indexes)
        logger.info("Updated text labels")
