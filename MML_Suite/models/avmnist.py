from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from data.avmnist import AVMNIST as AVMNISTDataset
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.printing import get_console
from experiment_utils.utils import safe_detach
from modalities import Modality
from models.conv import ConvBlock, ConvBlockArgs
from torch import Tensor
from torch.nn import (
    Dropout,
    Flatten,
    Identity,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from torch.nn.functional import softmax
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .mixins import MultimodalMonitoringMixin

console = get_console()


class MNISTAudio(Module):
    """
    Audio encoder for the AVMNIST dataset using a convolutional architecture.
    """

    def __init__(
        self,
        conv_block_one_args: ConvBlockArgs,
        conv_block_two_args: ConvBlockArgs,
        hidden_dim: int,
        *,
        conv_batch_norm: bool = True,
        max_pool_kernel_size: Union[int, Tuple[int, int]] = (2, 2),
    ) -> None:
        """
        Initialize the audio encoder.

        Args:
            conv_block_one_args (ConvBlockArgs): Arguments for the first convolution block.
            conv_block_two_args (ConvBlockArgs): Arguments for the second convolution block.
            hidden_dim (int): Dimension of the hidden layer.
            conv_batch_norm (bool): Whether to use batch normalization.
            max_pool_kernel_size (Union[int, Tuple[int, int]]): Kernel size for max pooling.
        """
        super().__init__()
        conv_block = ConvBlock(
            conv_block_one_args=conv_block_one_args,
            conv_block_two_args=conv_block_two_args,
            batch_norm=conv_batch_norm,
        )
        conv_block_out_dim = 24064  # Precomputed based on architecture

        self.hidden_dim = hidden_dim
        self.net = Sequential(
            conv_block,
            MaxPool2d(kernel_size=max_pool_kernel_size),
            Flatten(),
            Linear(conv_block_out_dim, hidden_dim),
        )

    def get_embedding_size(self) -> int:
        """
        Get the size of the embedding layer.

        Returns:
            int: Embedding size.
        """
        return self.hidden_dim

    def forward(self, audio: Tensor) -> Tensor:
        """
        Forward pass for the audio encoder.

        Args:
            audio (Tensor): Input audio data.

        Returns:
            Tensor: Encoded audio features.
        """
        audio = audio.unsqueeze(1)  # Add channel dimension
        return self.net(audio)

    def __str__(self) -> str:
        return str(self.net)


class MNISTImage(Module):
    """
    Image encoder for the AVMNIST dataset using a convolutional architecture.
    """

    def __init__(
        self,
        conv_block_one_one_args: ConvBlockArgs,
        conv_block_one_two_args: ConvBlockArgs,
        conv_block_two_one_args: ConvBlockArgs,
        conv_block_two_two_args: ConvBlockArgs,
        hidden_dim: int,
        *,
        conv_batch_norm: bool = True,
        max_pool_kernel_size: Union[int, Tuple[int, int]] = (2, 2),
    ) -> None:
        """
        Initialize the image encoder.

        Args:
            conv_block_one_one_args (ConvBlockArgs): Args for the first layer of the first block.
            conv_block_one_two_args (ConvBlockArgs): Args for the second layer of the first block.
            conv_block_two_one_args (ConvBlockArgs): Args for the first layer of the second block.
            conv_block_two_two_args (ConvBlockArgs): Args for the second layer of the second block.
            hidden_dim (int): Dimension of the hidden layer.
            conv_batch_norm (bool): Whether to use batch normalization.
            max_pool_kernel_size (Union[int, Tuple[int, int]]): Kernel size for max pooling.
        """
        super().__init__()
        conv_block_one = ConvBlock(
            conv_block_one_args=conv_block_one_one_args,
            conv_block_two_args=conv_block_one_two_args,
            batch_norm=conv_batch_norm,
        )
        conv_block_two = ConvBlock(
            conv_block_one_args=conv_block_two_one_args,
            conv_block_two_args=conv_block_two_two_args,
            batch_norm=conv_batch_norm,
        )

        conv_block_out_dim = 3136  # Precomputed based on architecture
        self.hidden_dim = hidden_dim
        self.net = Sequential(
            conv_block_one,
            MaxPool2d(kernel_size=max_pool_kernel_size),
            conv_block_two,
            MaxPool2d(kernel_size=max_pool_kernel_size),
            Flatten(),
            Linear(conv_block_out_dim, hidden_dim),
        )

    def get_embedding_size(self) -> int:
        """
        Get the size of the embedding layer.

        Returns:
            int: Embedding size.
        """
        return self.hidden_dim

    def forward(self, image: Tensor) -> Tensor:
        """
        Forward pass for the image encoder.

        Args:
            image (Tensor): Input image data.

        Returns:
            Tensor: Encoded image features.
        """
        return self.net(image)

    def __str__(self) -> str:
        return str(self.net)


class AVMNIST(Module, MultimodalMonitoringMixin):
    """
    Multimodal model for the AVMNIST dataset, fusing audio and image encoders.
    """

    def __init__(
        self,
        audio_encoder: MNISTAudio,
        image_encoder: MNISTImage,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
        fusion_fn: str = "concat",
    ) -> None:
        """
        Initialize the AVMNIST model.

        Args:
            audio_encoder (MNISTAudio): Audio encoder.
            image_encoder (MNISTImage): Image encoder.
            hidden_dim (int): Dimension of the hidden layers.
            dropout (float): Dropout rate for the fusion layers.
            fusion_fn (str): Fusion function for combining modalities.
        """
        super().__init__()
        self.audio_encoder = audio_encoder
        self.image_encoder = image_encoder

        self.embd_size_A = audio_encoder.get_embedding_size()
        self.embd_size_I = image_encoder.get_embedding_size()

        fc_fusion = Linear(self.embd_size_A + self.embd_size_I, hidden_dim)
        fc_intermediate = Linear(hidden_dim, hidden_dim // 2)
        fc_out = Linear(hidden_dim // 2, AVMNISTDataset.NUM_CLASSES)

        self.net = Sequential(
            fc_fusion,
            ReLU(),
            Dropout(dropout) if dropout > 0 else Identity(),
            fc_intermediate,
            ReLU(),
            fc_out,
        )

        match fusion_fn.lower():
            case "concat":
                self.fusion_fn = partial(torch.cat, dim=1)
            case _:
                raise ValueError(f"Unknown fusion function: {fusion_fn}")

    def forward(
        self,
        A: Optional[Tensor] = None,
        I: Optional[Tensor] = None,
        *,
        is_embd_A: bool = False,
        is_embd_I: bool = False,
    ) -> Tensor:
        """
        Perform a forward pass through the model.

        Args:
            A (Optional[Tensor]): Audio input or embedding.
            I (Optional[Tensor]): Image input or embedding.
            is_embd_A (bool): Whether the audio input is pre-embedded.
            is_embd_I (bool): Whether the image input is pre-embedded.

        Returns:
            Tensor: Logits for classification.
        """
        assert not all((A is None, I is None)), "At least one of A, I must be provided"
        assert not all([is_embd_A, is_embd_I]), "Cannot have all embeddings as True"

        A = A if A is not None else torch.zeros(I.size(0), self.embd_size_A)
        I = I if I is not None else torch.zeros(A.size(0), self.embd_size_I)

        audio = self.audio_encoder(A) if not is_embd_A else A
        image = self.image_encoder(I) if not is_embd_I else I
        fused = self.fusion_fn((audio, image))
        return self.net(fused)

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: Optimizer,
        criterion: Module,
        device: torch.device,
        metric_recorder: MetricRecorder,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform a training step.

        Args:
            batch (Dict[str, Any]): Batch of data.
            optimizer (Optimizer): Optimizer for training.
            criterion (Module): Loss function.
            device (torch.device): Device to run training on.
            metric_recorder (MetricRecorder): Metric recorder for performance tracking.

        Returns:
            Dict[str, Any]: Dictionary containing the training loss.
        """
        A, I, labels, miss_type = (
            batch[Modality.AUDIO].to(device).float(),
            batch[Modality.IMAGE].to(device).float(),
            batch["label"].to(device),
            batch["pattern_name"],
        )

        self.train()
        optimizer.zero_grad()
        logits = self.forward(A=A, I=I)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = softmax(logits, dim=1).argmax(dim=1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
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
    ) -> Dict[str, Any]:
        """
        Perform a validation step.

        Args:
            batch (Dict[str, Any]): Batch of data.
            criterion (LossFunctionGroup): Loss function group.
            device (torch.device): Device for validation.
            metric_recorder (MetricRecorder): Metric recorder for performance tracking.
            return_test_info (bool): Whether to return additional test information.

        Returns:
            Dict[str, Any]: Validation results, including loss and optionally predictions.
        """
        self.eval()
        with torch.no_grad():
            A, I, labels, miss_type = (
                batch[Modality.AUDIO].to(device).float(),
                batch[Modality.IMAGE].to(device).float(),
                batch["label"].to(device),
                batch["pattern_name"],
            )

            logits = self.forward(A=A, I=I)
            loss = criterion(logits, labels)
            predictions = softmax(logits, dim=1).argmax(dim=1).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            miss_type = np.array(miss_type)

            metric_recorder.update_all(predictions=predictions, targets=labels, m_types=miss_type)

            if return_test_info:
                return {
                    "loss": loss.item(),
                    "predictions": predictions,
                    "labels": labels,
                    "miss_types": miss_type,
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
        console.print("Getting embeddings...")
        embeddings = defaultdict(list)
        self.to(device)
        self.eval()

        for batch in dataloader:
            with torch.no_grad():
                A, I, miss_type = (
                    batch[Modality.AUDIO],
                    batch[Modality.IMAGE],
                    batch["pattern_name"],
                )

                # Filter to full modality availability
                miss_type = np.array(miss_type)
                A = A[miss_type == AVMNISTDataset.get_full_modality()]
                I = I[miss_type == AVMNISTDataset.get_full_modality()]

                A = A.to(device).float()
                I = I.to(device).float()

                audio_embedding = self.audio_encoder(A)
                image_embedding = self.image_encoder(I)

                embeddings[Modality.AUDIO].append(safe_detach(audio_embedding, to_np=True))
                embeddings[Modality.IMAGE].append(safe_detach(image_embedding, to_np=True))

        return embeddings
