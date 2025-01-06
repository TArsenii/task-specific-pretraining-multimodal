"""
Implementation of Matching Attention mechanism from GCNet paper.

Reference:
    Title: GCNet: Graph Completion Network for Incomplete Multimodal Learning in Conversation
    Authors: Lian, Zheng and Chen, Lan and Sun, Licai and Liu, Bin and Tao, Jianhua
    Journal: IEEE Transactions on pattern analysis and machine intelligence (2022)
    Code: https://github.com/zeroQiaoba/GCNet
"""

from typing import Literal, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module


class MatchingAttention(Module):
    """
    Implements various forms of matching-based attention mechanisms.

    This module supports different types of attention mechanisms for matching
    between a memory bank and candidate features:
    - dot: Direct dot product attention
    - general: Linear transform of candidate before dot product
    - general2: Linear transform with bias and mask handling
    - concat: Concatenation-based attention

    Args:
        mem_dim: Dimension of memory bank features
        cand_dim: Dimension of candidate features
        alpha_dim: Dimension of attention space (required for concat attention)
        att_type: Type of attention mechanism to use
            Options: "general", "dot", "general2", "concat"
            Default: "general"

    Raises:
        AssertionError: If incompatible attention type and dimension combinations are provided
    """

    def __init__(
        self,
        mem_dim: int,
        cand_dim: int,
        alpha_dim: Optional[int] = None,
        att_type: Literal["general", "dot", "general2", "concat"] = "general",
    ) -> None:
        super(MatchingAttention, self).__init__()

        # Validate inputs
        if att_type == "concat" and alpha_dim is None:
            raise ValueError("alpha_dim must be provided for concat attention")
        if att_type == "dot" and mem_dim != cand_dim:
            raise ValueError("mem_dim must equal cand_dim for dot attention")

        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type

        # Initialize appropriate transformation layers
        if att_type == "general":
            self.transform = Linear(cand_dim, mem_dim, bias=False)
        elif att_type == "general2":
            self.transform = Linear(cand_dim, mem_dim, bias=True)
        elif att_type == "concat":
            self.transform = Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = Linear(alpha_dim, 1, bias=False)

    def forward(self, memory_bank: Tensor, candidate: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Compute attention between memory bank and candidate features.

        Args:
            memory_bank: Memory bank features of shape (seq_len, batch, mem_dim)
            candidate: Candidate features of shape (batch, cand_dim)
            mask: Optional attention mask of shape (batch, seq_len)

        Returns:
            Tuple containing:
            - attended_memory: Attention-pooled memory features (batch, mem_dim)
            - attention_weights: Attention weights (batch, 1, seq_len)
        """
        # Create default mask if none provided
        if mask is None:
            mask = torch.ones(memory_bank.size(1), memory_bank.size(0), device=memory_bank.device)

        if self.att_type == "dot":
            attention_weights = self._dot_attention(memory_bank, candidate)
        elif self.att_type == "general":
            attention_weights = self._general_attention(memory_bank, candidate)
        elif self.att_type == "general2":
            attention_weights = self._general2_attention(memory_bank, candidate, mask)
        else:  # concat
            attention_weights = self._concat_attention(memory_bank, candidate)

        # Compute attention-pooled memory features
        attended_memory = torch.bmm(attention_weights, memory_bank.transpose(0, 1))[:, 0, :]  # [batch, mem_dim]

        return attended_memory, attention_weights

    def _dot_attention(self, memory_bank: Tensor, candidate: Tensor) -> Tensor:
        """Compute dot product attention."""
        M_ = memory_bank.permute(1, 2, 0)  # [batch, mem_dim, seq_len]
        x_ = candidate.unsqueeze(1)  # [batch, 1, cand_dim]
        return F.softmax(torch.bmm(x_, M_), dim=2)  # [batch, 1, seq_len]

    def _general_attention(self, memory_bank: Tensor, candidate: Tensor) -> Tensor:
        """Compute general attention with linear transform."""
        M_ = memory_bank.permute(1, 2, 0)  # [batch, mem_dim, seq_len]
        x_ = self.transform(candidate).unsqueeze(1)  # [batch, 1, mem_dim]
        return F.softmax(torch.bmm(x_, M_), dim=2)  # [batch, 1, seq_len]

    def _general2_attention(self, memory_bank: Tensor, candidate: Tensor, mask: Tensor) -> Tensor:
        """Compute general attention with bias and mask handling."""
        # Prepare inputs
        M_ = memory_bank.permute(1, 2, 0)  # [batch, mem_dim, seq_len]
        x_ = self.transform(candidate).unsqueeze(1)  # [batch, 1, mem_dim]

        # Apply mask to memory bank
        mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
        M_ = M_ * mask_  # [batch, mem_dim, seq_len]

        # Compute masked attention
        alpha_ = torch.bmm(x_, M_) * mask.unsqueeze(1)  # [batch, 1, seq_len]
        alpha_ = torch.tanh(alpha_)
        alpha_ = F.softmax(alpha_, dim=2)

        # Normalize masked attention
        alpha_masked = alpha_ * mask.unsqueeze(1)
        alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)
        return alpha_masked / alpha_sum  # [batch, 1, seq_len]

    def _concat_attention(self, memory_bank: Tensor, candidate: Tensor) -> Tensor:
        """Compute concatenation-based attention."""
        M_ = memory_bank.transpose(0, 1)  # [batch, seq_len, mem_dim]
        x_ = candidate.unsqueeze(1).expand(-1, memory_bank.size(0), -1)
        M_x_ = torch.cat([M_, x_], 2)  # [batch, seq_len, mem_dim+cand_dim]

        # Transform and compute attention weights
        mx_a = F.tanh(self.transform(M_x_))  # [batch, seq_len, alpha_dim]
        return F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # [batch, 1, seq_len]
