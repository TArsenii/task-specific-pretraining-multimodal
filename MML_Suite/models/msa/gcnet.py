from typing import List, Literal, Tuple

import torch
import torch.nn.functional as F
from models.mixins import MultimodalMonitoringMixin
from models.msa.networks.graph_utils import batch_graphify, utterance_to_conversation
from models.msa.networks.matching_attention import MatchingAttention
from models.protocols import MultimodalModelProtocol
from torch import Tensor
from torch.nn import GRU, LSTM, Linear, Module
from torch_geometric.nn import GraphConv, RGCNConv


class GraphNetwork(Module, MultimodalMonitoringMixin, MultimodalModelProtocol):
    """
    A graph neural network module that processes conversation data using both graph convolutions
    and attention mechanisms.

    This network combines RGCN (Relational Graph Convolutional Network) with traditional sequence
    modeling approaches to capture both structural and temporal dependencies in conversation data.

    Args:
        num_features: Number of input features
        num_relations: Number of different types of relations in the graph
        time_attention: Whether to use temporal attention mechanism
        hidden_size: Size of hidden layers (default: 64)
        dropout: Dropout rate (default: 0.5)
    """

    def __init__(
        self, num_features: int, num_relations: int, time_attention: bool, hidden_size: int = 64, dropout: float = 0.5
    ) -> None:
        super(GraphNetwork, self).__init__()

        self.num_features = num_features
        self.num_relations = num_relations
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.time_attn = time_attention

        # Graph convolution layers
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations)
        self.conv2 = GraphConv(hidden_size, hidden_size)

        # Modal fusion parameters
        D_h = num_features + hidden_size
        self.grufusion = LSTM(input_size=D_h, hidden_size=D_h, num_layers=2, bidirectional=True, dropout=dropout)

        # Attention mechanisms
        self.matchatt = MatchingAttention(2 * D_h, 2 * D_h, att_type="general2")
        self.linear = Linear(2 * D_h, D_h)

    def forward(
        self, features: Tensor, edge_index: Tensor, edge_type: Tensor, seq_lengths: List[int], umask: Tensor
    ) -> Tensor:
        """
        Forward pass of the graph network.

        Args:
            features: Input node features
            edge_index: Graph edge indices
            edge_type: Edge type indicators
            seq_lengths: Sequence lengths for each conversation
            umask: Utterance mask

        Returns:
            Tensor of shape [seqlen, batch, D_h] containing processed features
        """
        # Graph convolution operations
        out = self.conv1(features, edge_index, edge_type)
        out = self.conv2(out, edge_index)
        outputs = torch.cat([features, out], dim=-1)

        # Reshape to conversation format
        outputs = outputs.reshape(-1, outputs.size(1))
        outputs = utterance_to_conversation(outputs, seq_lengths)
        outputs = outputs.reshape(outputs.size(0), outputs.size(1), 1, -1)

        # Sequence processing
        seqlen, batch = outputs.size(0), outputs.size(1)
        outputs = torch.reshape(outputs, (seqlen, batch, -1))
        outputs = self.grufusion(outputs)[0]

        # Apply attention if enabled
        if self.time_attn:
            hidden = self._apply_attention(outputs, umask)
        else:
            hidden = F.relu(self.linear(outputs))

        return hidden

    def _apply_attention(self, outputs: Tensor, umask: Tensor) -> Tensor:
        """
        Applies temporal attention to the sequence outputs.

        Args:
            outputs: Sequence outputs to attend to
            umask: Utterance mask for attention

        Returns:
            Attended sequence outputs
        """
        alpha = []
        att_emotions = []

        for t in outputs:
            att_em, alpha_ = self.matchatt(outputs, t, mask=umask)
            att_emotions.append(att_em.unsqueeze(0))
            alpha.append(alpha_[:, 0, :])

        att_emotions = torch.cat(att_emotions, dim=0)
        return F.relu(self.linear(att_emotions))


class GraphModel(Module):
    """
    Complete graph-based model for multimodal sentiment analysis.

    This model combines sequential modeling (LSTM/GRU) with graph-based processing
    to capture both temporal and structural dependencies in conversation data.

    Args:
        base_model: Type of base sequential model ('LSTM' or 'GRU')
        adim: Audio feature dimension
        tdim: Text feature dimension
        vdim: Visual feature dimension
        D_e: Emotion embedding dimension
        graph_hidden_size: Hidden size for graph networks
        n_speakers: Number of speakers in conversations
        window_past: Past context window size
        window_future: Future context window size
        n_classes: Number of output classes
        dropout: Dropout rate (default: 0.5)
        time_attn: Whether to use temporal attention (default: True)
    """

    def __init__(
        self,
        base_model: Literal["LSTM", "GRU"],
        adim: int,
        tdim: int,
        vdim: int,
        D_e: int,
        graph_hidden_size: int,
        n_speakers: int,
        window_past: int,
        window_future: int,
        n_classes: int,
        dropout: float = 0.5,
        time_attn: bool = True,
    ) -> None:
        super(GraphModel, self).__init__()

        self.base_model = base_model
        self.n_speakers = n_speakers
        self.window_past = window_past
        self.window_future = window_future
        self.time_attn = time_attn

        # Sequential context encoder
        input_size = adim + tdim + vdim
        self.lstm = LSTM(input_size=input_size, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru = GRU(input_size=input_size, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        # Graph networks for temporal and speaker relations
        self.graph_net_temporal = GraphNetwork(2 * D_e, 3, time_attn, graph_hidden_size, dropout)
        self.graph_net_speaker = GraphNetwork(2 * D_e, n_speakers**2, time_attn, graph_hidden_size, dropout)

        # Output layers
        D_h = 2 * D_e + graph_hidden_size
        self.smax_fc = Linear(D_h, n_classes)
        self.linear_rec = Linear(D_h, input_size)

    def forward(
        self, inputfeats: List[Tensor], qmask: Tensor, umask: Tensor, seq_lengths: List[int]
    ) -> Tuple[Tensor, List[Tensor], Tensor]:
        """
        Forward pass of the complete model.

        Args:
            inputfeats: List of input features
            qmask: Speaker mask
            umask: Utterance mask
            seq_lengths: Sequence lengths

        Returns:
            Tuple containing:
            - log_prob: Classification logits
            - rec_outputs: Reconstruction outputs
            - hidden: Hidden representations
        """
        # Sequential modeling
        outputs = self._apply_sequential_model(inputfeats[0])

        # Graph processing
        hidden = self._process_graphs(outputs, qmask, umask, seq_lengths)

        # Output generation
        log_prob = self.smax_fc(hidden)
        rec_outputs = [self.linear_rec(hidden)]

        return log_prob, rec_outputs, hidden

    def _apply_sequential_model(self, inputs: Tensor) -> Tensor:
        """
        Applies the selected sequential model (LSTM/GRU) to inputs.
        """
        if self.base_model == "LSTM":
            outputs, _ = self.lstm(inputs)
        else:
            outputs, _ = self.gru(inputs)
        return outputs.unsqueeze(2)

    def _process_graphs(self, outputs: Tensor, qmask: Tensor, umask: Tensor, seq_lengths: List[int]) -> Tensor:
        """
        Processes the sequence through temporal and speaker-based graphs.
        """
        # Temporal graph processing
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(
            outputs, qmask, seq_lengths, self.n_speakers, self.window_past, self.window_future, "temporal"
        )
        assert len(edge_type_mapping) == 3
        hidden1 = self.graph_net_temporal(features, edge_index, edge_type, seq_lengths, umask)

        # Speaker graph processing
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(
            outputs, qmask, seq_lengths, self.n_speakers, self.window_past, self.window_future, "speaker"
        )
        assert len(edge_type_mapping) == self.n_speakers**2
        hidden2 = self.graph_net_speaker(features, edge_index, edge_type, seq_lengths, umask)

        return hidden1 + hidden2
