"""Multimodal Transformer for cross-modal interaction and fusion."""

from typing import Any, Dict, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from experiment_utils.utils import to_gpu_safe
from modalities import Modality
from torch import Tensor
from torch.nn import BCELoss, Linear, Module, ModuleDict
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .networks import GatedTransformer, LanguageEmbeddingLayer, SeqEncoder


class MultModalTransformer(Module):
    """Multimodal Transformer for cross-modal feature learning and fusion.

    This model implements cross-modal attention between different modalities
    (text, audio, video) using gated transformers and domain-invariant encoders.
    It supports both BERT and traditional word embeddings for text processing.

    Args:
        orig_dim_a: Original dimension of audio features
        orig_dim_t: Original dimension of text features
        orig_dim_v: Original dimension of video features
        attention_dim: Dimension of attention and hidden states
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        attention_dropout: General attention dropout rate
        attention_dropout_a: Audio-specific attention dropout
        attention_dropout_v: Video-specific attention dropout
        relu_dropout: Dropout rate for ReLU layers
        residual_dropout: Dropout rate for residual connections
        output_dropout: Dropout rate for output layer
        attention_mask: Whether to use attention masking
        a_ksize: Kernel size for audio CNN
        t_ksize: Kernel size for text CNN
        v_ksize: Kernel size for video CNN
        output_dim: Dimension of final output
        div_dropout: Dropout rate for domain-invariant encoder
        use_bert: Whether to use BERT for text embedding
    """

    def __init__(
        self,
        *,
        orig_dim_a: int,
        orig_dim_t: int,
        orig_dim_v: int,
        attention_dim: int,
        num_heads: int = 5,
        num_layers: int = 5,
        attention_dropout: float = 0.1,
        attention_dropout_a: float = 0.0,
        attention_dropout_v: float = 0.0,
        relu_dropout: float = 0.1,
        embd_dropout: float = 0.25,
        residual_dropout: float = 0.1,
        output_dropout: float = 0.0,
        attention_mask: bool = True,
        a_ksize: int = 3,
        t_ksize: int = 3,
        v_ksize: int = 3,
        output_dim: int,
        div_dropout: float = 0.0,
        use_bert: bool = True,
        proj_type: Literal["linear", "cnn", "lstm", "gru"] = "cnn",
        word2id: Dict[str, int] = None,
        lambda_d: float = 0.1,
        use_discriminator: bool = True,
        clip_grad_norm: float = 0.8,
    ) -> None:
        super().__init__()

        # Configuration
        self.use_bert = use_bert
        self.orig_dim_a = orig_dim_a
        self.orig_dim_t = orig_dim_t
        self.orig_dim_v = orig_dim_v

        # Dimensions
        self.attention_dim = attention_dim
        self.embedding_dim = attention_dim
        self.fused_dim = 4 * attention_dim
        self.output_dim = output_dim

        # Architecture parameters
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Dropout rates
        self.attention_dropout = attention_dropout
        self.attention_dropout_a = attention_dropout_a
        self.attention_dropout_v = attention_dropout_v
        self.relu_dropout = relu_dropout
        self.residual_dropout = residual_dropout
        self.output_dropout = output_dropout
        self.div_dropout = div_dropout
        self.embd_dropout = embd_dropout

        # Other parameters
        self.attention_mask = attention_mask
        self.a_ksize = a_ksize
        self.t_ksize = t_ksize
        self.v_ksize = v_ksize

        self.proj_type = proj_type
        self.lambda_d = lambda_d

        if use_discriminator:
            self.criterion_disc = BCELoss()
            self.lambda_d = self.lambda_d
        self.use_discriminator = use_discriminator
        self.clip_grad_norm = clip_grad_norm
        # Initialize components
        self._init_embeddings(word2id=word2id)
        self._init_encoders()
        self._init_fusion_layers()

    def _init_embeddings(self, word2id: Dict[str, int] = None) -> None:
        """Initialize text embedding layer."""
        self.text_embedding = LanguageEmbeddingLayer(
            use_bert=self.use_bert,
            word2id=word2id if not self.use_bert else None,
            embedding_dim=self.orig_dim_t if not self.use_bert else None,
            bert_model_name="bert-base-uncased",
        )

    def _init_encoders(self) -> None:
        """Initialize sequence encoder and cross-modal interaction networks."""
        self.sequence_encoder = SeqEncoder(
            orig_dim_a=self.orig_dim_a,
            orig_dim_t=self.orig_dim_t,
            orig_dim_v=self.orig_dim_v,
            attention_dim=self.attention_dim,
            num_enc_layers=self.num_layers,
            proj_type=self.proj_type,
            a_ksize=self.a_ksize,
            t_ksize=self.t_ksize,
            v_ksize=self.v_ksize,
        )

        # Initialize cross-modal interaction networks
        self.modality_interaction = ModuleDict(
            {
                Modality.TEXT + Modality.VIDEO: self._get_network(layers=self.num_layers),
                Modality.TEXT + Modality.AUDIO: self._get_network(layers=self.num_layers),
            }
        )

    def _init_fusion_layers(self) -> None:
        """Initialize fusion and output projection layers."""
        # Projection layers for feature fusion
        self.projection_one = Linear(self.fused_dim, self.fused_dim)
        self.projection_two = Linear(self.fused_dim, self.fused_dim)
        self.output_layer = Linear(self.fused_dim, self.output_dim)

    def _get_network(self, _type: Modality = Modality.TEXT, layers: int = 2) -> GatedTransformer:
        """Create a gated transformer network for cross-modal interaction.

        Args:
            _type: Modality type (unused, kept for compatibility)
            layers: Number of transformer layers

        Returns:
            Configured GatedTransformer instance
        """
        return GatedTransformer(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            layers=layers,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.residual_dropout,
            embed_dropout=self.embd_dropout,
            attention_mask=self.attention_mask,
            div_dropout=self.div_dropout,
        )

    def _apply_sequence_pooling(
        self, tensor: Tensor, lengths: Tensor, mode: Literal["max", "mean", "avg"] = "mean"
    ) -> Tensor:
        """Apply pooling over sequence dimension.

        Args:
            tensor: Input tensor of shape (batch_size, seq_len, embedding_dim)
            lengths: Sequence lengths of shape (batch_size,)
            mode: Pooling mode ('max', 'mean', or 'avg')

        Returns:
            Pooled tensor of shape (batch_size, embedding_dim)
        """
        if mode == "max":
            return tensor.max(dim=1)[0]

        if mode in ["mean", "avg"]:
            batch_size, seq_len, embed_dim = tensor.size()
            mask = (torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(tensor.device)) < lengths.unsqueeze(1)

            mask = mask.unsqueeze(2).expand(-1, -1, embed_dim)
            return (tensor * mask.float()).sum(dim=1) / lengths.unsqueeze(1).float()

        raise ValueError(f"Unsupported pooling mode: {mode}")

    def forward(
        self,
        audio_input: Tensor,
        video_input: Tensor,
        text_input: Union[Tensor, Tuple[Tensor, ...]],
        is_embedded_A: bool,
        is_embedded_V: bool,
        is_embedded_T: bool,
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """Forward pass through the multimodal transformer.

        Args:
            audio_input: Audio features
            video_input: Video features
            text_input: Text features or tuple of (input_ids, lengths, attention_mask, etc.)
            is_embedded_A: Whether audio input is already embedded
            is_embedded_V: Whether video input is already embedded
            is_embedded_T: Whether text input is already embedded

        Returns:
            Dictionary containing:
                - predictions: Model predictions/outputs
                - features: Dictionary of intermediate features including:
                    - discriminator_predictions: Predictions from domain discriminator
                    - discriminator_labels: True labels for discriminator
                    - text_hidden: Text modality hidden states
                    - audio_hidden: Audio modality hidden states
                    - video_hidden: Video modality hidden states
                    - cross_modal_features: Cross-modal attention features
        """

        def _pad_sequences(audio: Tensor, video: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            """Add padding to sequences for alignment."""
            padding = (0, 0, 0, 0, 1, 1)
            padded_audio = F.pad(audio, padding, "constant", 0.0)
            padded_video = F.pad(video, padding, "constant", 0.0)
            return padded_audio, padded_video, lengths + 2

        # Process text input based on BERT usage
        if self.use_bert:
            (text_tokens, lengths, bert_input, bert_type_ids, bert_mask) = text_input

            text_embeddings = self.text_embedding(
                sentences=text_tokens, bert_sent=bert_input, bert_sent_type=bert_type_ids, bert_sent_mask=bert_mask
            )

            # Pad sequences for alignment
            audio_input, video_input, lengths = _pad_sequences(audio_input, video_input, lengths)
        else:
            text_embeddings = self.text_embedding(
                sentences=text_input, bert_sent=None, bert_sent_type=None, bert_sent_mask=None
            )

        # Encode sequences
        encoded_features = self.sequence_encoder(
            input_t=text_embeddings, input_v=video_input, input_a=audio_input, lengths=lengths
        )

        # Extract features for each modality
        text_seq, text_hidden = encoded_features[Modality.TEXT]
        video_seq, video_hidden = encoded_features[Modality.VIDEO]
        audio_seq, audio_hidden = encoded_features[Modality.AUDIO]

        # Get attention mask
        attention_mask = bert_mask if self.use_bert else None

        # Process cross-modal interactions
        audio_text_out = self.modality_interaction[Modality.TEXT + Modality.AUDIO](
            text_seq, audio_seq, text_hidden, audio_hidden, lengths, attention_mask
        )

        video_text_out = self.modality_interaction[Modality.TEXT + Modality.VIDEO](
            text_seq, video_seq, text_hidden, video_hidden, lengths, attention_mask
        )

        # Unpack outputs
        last_a2t, last_t2a, disc_pred_ta, disc_true_ta = audio_text_out
        last_v2t, last_t2v, disc_pred_tv, disc_true_tv = video_text_out

        # Combine discriminator outputs
        disc_predictions = torch.cat((disc_pred_ta, disc_pred_tv), dim=0)
        disc_labels = torch.cat((disc_true_ta, disc_true_tv), dim=0)

        # Extract first tokens (assume they contain global information)
        batch_indices = torch.arange(last_a2t.size(1))
        last_a2t = last_a2t.permute(1, 0, 2)[batch_indices, 0]
        last_t2a = last_t2a.permute(1, 0, 2)[batch_indices, 0]
        last_v2t = last_v2t.permute(1, 0, 2)[batch_indices, 0]
        last_t2v = last_t2v.permute(1, 0, 2)[batch_indices, 0]

        # Concatenate all features
        fused_features = torch.cat([last_a2t, last_t2a, last_v2t, last_t2v], dim=1)

        # Apply residual projection block
        projected_features = self.projection_two(
            F.dropout(F.relu(self.projection_one(fused_features)), p=self.output_dropout, training=self.training)
        )
        projected_features = projected_features + fused_features

        # Final output projection
        output = self.output_layer(projected_features)

        # Collect features
        features = {
            "discriminator_predictions": disc_predictions,
            "discriminator_labels": disc_labels,
            "text_hidden": text_hidden,
            "audio_hidden": audio_hidden,
            "video_hidden": video_hidden,
            "cross_modal_features": {
                "audio_to_text": last_a2t,
                "text_to_audio": last_t2a,
                "video_to_text": last_v2t,
                "text_to_video": last_t2v,
                "fused": projected_features,
            },
        }

        return {"predictions": output, "features": features}

    def train_step(
        self, batch: Dict[str, Any], criterion: Module, optimizer: Optimizer, device: torch.device, **kwargs
    ) -> Dict[str, Any]:
        A, V, T, lengths, labels = (
            batch[Modality.AUDIO],
            batch[Modality.VIDEO],
            batch[Modality.TEXT],  ## can contain BERT inputs
            batch["lengths"],
            batch["labels"],
        )

        A = A.to(device)
        V = V.to(device)
        T = to_gpu_safe(T)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = self(A, V, T, is_embedded_A=False, is_embedded_V=False, is_embedded_T=False)
        predictions = outputs["predictions"]
        features = outputs["features"]

        loss = criterion(predictions, labels)

        if self.use_discriminator:
            disc_predictions = features["discriminator_predictions"]
            disc_labels = features["discriminator_labels"]
            disc_loss = self.criterion_disc(disc_predictions, disc_labels)
            loss += self.lambda_d * disc_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        optimizer.step()

        ## Metrics

        return {"loss": loss.item()}

    def validation_step(self, batch: Dict[str, Any], criterion: Module, device: torch.device) -> Dict[str, Any]:
        pass

    def get_embeddings(self, dataloader: DataLoader, device: torch.device) -> Dict[Modality, np.ndarray]:
        pass
