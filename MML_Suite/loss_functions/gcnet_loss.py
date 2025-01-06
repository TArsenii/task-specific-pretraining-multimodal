"""
Custom loss functions for the GCNet model implementing masked variants of standard losses
to handle missing modalities and utterance masking in conversational data.
"""

from typing import List, Tuple
import torch
from torch.nn import Module, MSELoss, NLLLoss
import torch.nn.functional as F
from torch import Tensor


class MaskedReconLoss(Module):
    """
    Masked reconstruction loss for multimodal features.

    Computes MSE loss only on missing modalities, using masks to identify which
    modalities are present/missing for each sample. Handles audio, text, and
    visual modalities separately.
    """

    def __init__(self) -> None:
        super(MaskedReconLoss, self).__init__()
        self.loss = MSELoss(reduction="none")

    def forward(
        self,
        recon_input: List[Tensor],
        target_input: List[Tensor],
        input_mask: List[Tensor],
        umask: Tensor,
        adim: int,
        tdim: int,
        vdim: int,
    ) -> Tensor:
        """
        Compute masked reconstruction loss across modalities.

        Args:
            recon_input: List containing reconstructed features [seqlen, batch, dim]
            target_input: List containing target features [seqlen, batch, dim]
            input_mask: List containing modality masks [seqlen, batch, 3]
                       (1: modality present, 0: modality missing)
            umask: Utterance mask [batch, seqlen]
            adim: Audio feature dimension
            tdim: Text feature dimension
            vdim: Visual feature dimension

        Returns:
            Weighted sum of reconstruction losses across modalities
        """
        assert len(recon_input) == 1, "Expected single tensor in recon_input list"

        # Reshape inputs for processing
        recon, target, mask = self._prepare_inputs(recon_input[0], target_input[0], input_mask[0], umask)

        # Split features by modality
        modality_features = self._split_modalities(recon, target, mask, adim, tdim)
        A_rec, L_rec, V_rec, A_full, L_full, V_full, A_miss_idx, L_miss_idx, V_miss_idx = modality_features

        # Compute per-modality losses
        loss_audio = self._compute_modality_loss(A_rec, A_full, A_miss_idx, umask, adim)
        loss_text = self._compute_modality_loss(L_rec, L_full, L_miss_idx, umask, tdim)
        loss_visual = self._compute_modality_loss(V_rec, V_full, V_miss_idx, umask, vdim)

        # Combine modality losses
        total_loss = (loss_audio + loss_text + loss_visual) / torch.sum(umask)
        return total_loss

    def _prepare_inputs(
        self, recon: Tensor, target: Tensor, mask: Tensor, umask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Reshape inputs for processing."""
        recon = torch.reshape(recon, (-1, recon.size(2)))  # [seqlen*batch, dim]
        target = torch.reshape(target, (-1, target.size(2)))  # [seqlen*batch, dim]
        mask = torch.reshape(mask, (-1, mask.size(2)))  # [seqlen*batch, 3]
        umask = torch.reshape(umask, (-1, 1))  # [seqlen*batch, 1]
        return recon, target, mask, umask

    def _split_modalities(
        self, recon: Tensor, target: Tensor, mask: Tensor, adim: int, tdim: int
    ) -> Tuple[Tensor, ...]:
        """Split features and masks by modality."""
        # Split reconstructed features
        A_rec = recon[:, :adim]
        L_rec = recon[:, adim : adim + tdim]
        V_rec = recon[:, adim + tdim :]

        # Split target features
        A_full = target[:, :adim]
        L_full = target[:, adim : adim + tdim]
        V_full = target[:, adim + tdim :]

        # Prepare modality masks
        A_miss_idx = torch.reshape(mask[:, 0], (-1, 1))
        L_miss_idx = torch.reshape(mask[:, 1], (-1, 1))
        V_miss_idx = torch.reshape(mask[:, 2], (-1, 1))

        return A_rec, L_rec, V_rec, A_full, L_full, V_full, A_miss_idx, L_miss_idx, V_miss_idx

    def _compute_modality_loss(self, rec: Tensor, target: Tensor, miss_idx: Tensor, umask: Tensor, dim: int) -> Tensor:
        """Compute loss for a single modality."""
        loss = self.loss(rec * umask, target * umask) * -1 * (miss_idx - 1)
        return torch.sum(loss) / dim


class MaskedCELoss(Module):
    """
    Masked Cross Entropy Loss for classification tasks.

    Applies utterance mask to compute loss only on valid utterances.
    Used for emotion classification on IEMOCAP dataset.
    """

    def __init__(self) -> None:
        super(MaskedCELoss, self).__init__()
        self.loss = NLLLoss(reduction="sum")

    def forward(self, pred: Tensor, target: Tensor, umask: Tensor) -> Tensor:
        """
        Compute masked cross entropy loss.

        Args:
            pred: Predictions [batch*seq_len, n_classes]
            target: Ground truth labels [batch*seq_len]
            umask: Utterance mask [batch, seq_len]

        Returns:
            Masked cross entropy loss
        """
        # Reshape inputs
        umask = umask.view(-1, 1)  # [batch*seq_len, 1]
        target = target.view(-1, 1)  # [batch*seq_len, 1]

        # Apply softmax and compute loss
        pred = F.log_softmax(pred, 1)  # [batch*seqlen, n_classes]
        loss = self.loss(pred * umask, (target * umask).squeeze().long()) / torch.sum(umask)

        return loss


class MaskedMSELoss(Module):
    """
    Masked Mean Squared Error Loss.

    Used for regression tasks on CMU-MOSI and CMU-MOSEI datasets.
    Applies utterance mask to compute loss only on valid utterances.
    """

    def __init__(self) -> None:
        super(MaskedMSELoss, self).__init__()
        self.loss = MSELoss(reduction="sum")

    def forward(self, pred: Tensor, target: Tensor, umask: Tensor) -> Tensor:
        """
        Compute masked MSE loss.

        Args:
            pred: Predictions [batch*seq_len]
            target: Ground truth values [batch*seq_len]
            umask: Utterance mask [batch*seq_len]

        Returns:
            Masked MSE loss
        """
        # Reshape inputs to 2D
        pred = pred.view(-1, 1)  # [batch*seq_len, 1]
        target = target.view(-1, 1)  # [batch*seq_len, 1]
        umask = umask.view(-1, 1)  # [batch*seq_len, 1]

        # Compute masked loss
        loss = self.loss(pred * umask, target * umask) / torch.sum(umask)

        return loss
