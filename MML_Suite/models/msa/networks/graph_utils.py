from typing import Dict, List, Literal, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable


def edge_perms(seq_length: int, window_past: int, window_future: int) -> List[Tuple[int, int]]:
    """
    Constructs edges between nodes considering past and future context windows.

    This function creates edge connections between nodes in a sequence, taking into account
    specified context windows for both past and future connections. A value of -1 for
    either window indicates unlimited context in that direction.

    Args:
        seq_length: Length of the sequence
        window_past: Number of past timesteps to consider (-1 for unlimited)
        window_future: Number of future timesteps to consider (-1 for unlimited)

    Returns:
        List of tuples containing (source_node, target_node) pairs representing edges

    Example:
        >>> edges = edge_perms(3, 1, 1)
        >>> print(edges)  # Shows all valid connections within ±1 timestep window
    """
    all_perms: Set[Tuple[int, int]] = set()
    array = np.arange(seq_length)

    for current_idx in range(seq_length):
        # Determine effective range based on window settings
        if window_past == -1 and window_future == -1:
            effective_range = array
        elif window_past == -1:
            effective_range = array[: min(seq_length, current_idx + window_future + 1)]
        elif window_future == -1:
            effective_range = array[max(0, current_idx - window_past) :]
        else:
            effective_range = array[
                max(0, current_idx - window_past) : min(seq_length, current_idx + window_future + 1)
            ]

        # Add edges for current node
        perms = {(current_idx, target_idx) for target_idx in effective_range}
        all_perms.update(perms)

    return list(all_perms)


def batch_graphify(
    features: Tensor,
    qmask: Tensor,
    lengths: List[int],
    n_speakers: int,
    window_past: int,
    window_future: int,
    graph_type: Literal["temporal", "speaker"],
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, int]]:
    """
    Prepares data for GCN network by creating a graph representation of conversations.

    This function converts sequential conversation data into a graph format where nodes
    represent utterances and edges represent relationships (temporal or speaker-based).

    Args:
        features: Input features tensor [Time, Batch, ?, Features]
        qmask: Speaker mask tensor [Batch, Time]
        lengths: List of conversation lengths
        n_speakers: Number of speakers (must be 1 or 2)
        window_past: Past context window size (-1 for unlimited)
        window_future: Future context window size (-1 for unlimited)
        graph_type: Type of graph to construct ('temporal' or 'speaker')

    Returns:
        Tuple containing:
        - node_features: Tensor of node features
        - edge_index: Tensor of edge indices
        - edge_type: Tensor of edge types
        - edge_type_mapping: Dictionary mapping edge types to indices

    Raises:
        AssertionError: If n_speakers > 2 or graph_type is invalid
    """
    # Define edge type categories
    order_types = ["past", "now", "future"]
    assert n_speakers <= 2, "Number of speakers must be <= 2"
    speaker_types = ["00"] if n_speakers == 1 else ["00", "01", "10", "11"]

    # Validate and setup graph type
    assert graph_type in ["temporal", "speaker"], "Invalid graph type"
    merge_types = {"temporal": set(order_types), "speaker": set(speaker_types)}[graph_type]

    # Create edge type mapping
    edge_type_mapping = {edge_type: idx for idx, edge_type in enumerate(merge_types)}

    # Convert qmask to numpy for processing
    qmask = qmask.cpu().data.numpy().astype(int)

    # Initialize lists for graph construction
    node_features: List[Tensor] = []
    edge_index: List[List[int]] = []
    edge_type: List[int] = []
    length_sum = 0
    batch_size = features.size(1)

    # Process each conversation in the batch
    for batch_idx in range(batch_size):
        # Extract and reshape node features
        conv_length = lengths[batch_idx]
        batch_features = features[:conv_length, batch_idx, :, :]
        batch_features = torch.reshape(batch_features, (-1, batch_features.size(-1)))
        node_features.append(batch_features)

        # Generate edges for current conversation
        conv_edges = edge_perms(conv_length, window_past, window_future)
        shifted_edges = [(src + length_sum, tgt + length_sum) for src, tgt in conv_edges]

        # Process each edge
        for (src, tgt), (shifted_src, shifted_tgt) in zip(conv_edges, shifted_edges):
            edge_index.append([shifted_src, shifted_tgt])

            # Determine temporal relationship
            if tgt > src:
                order = "past"
            elif tgt == src:
                order = "now"
            else:
                order = "future"

            # Determine speaker relationship
            speaker_src = qmask[batch_idx, src]
            speaker_tgt = qmask[batch_idx, tgt]
            speaker_pattern = f"{speaker_tgt}{speaker_src}"

            # Assign edge type
            edge_type_name = speaker_pattern if graph_type == "speaker" else order
            edge_type.append(edge_type_mapping[edge_type_name])

        length_sum += conv_length

    # Convert lists to tensors
    node_features_tensor = torch.cat(node_features, dim=0)
    edge_index_tensor = torch.tensor(edge_index).transpose(0, 1)
    edge_type_tensor = torch.tensor(edge_type)

    return node_features_tensor, edge_index_tensor, edge_type_tensor, edge_type_mapping


def utterance_to_conversation(outputs: Tensor, seq_lengths: List[int]) -> Tensor:
    """
    Converts utterance-level features to conversation-level format with padding.

    Args:
        outputs: Tensor of shape [num_utterance, dim]
        seq_lengths: List of sequence lengths for each conversation

    Returns:
        Tensor of shape [seqlen, batch, dim] with appropriate padding
    """
    conv_lengths = torch.tensor(seq_lengths)
    start_zero = conv_lengths.new_zeros(1)
    max_len = max(seq_lengths)

    # Calculate starting indices for each conversation
    start_indices = torch.cumsum(torch.cat((start_zero, conv_lengths[:-1])), 0)

    # Pad and stack conversations
    padded_conversations = [
        pad(outputs.narrow(0, start, length), max_len)
        for start, length in zip(start_indices.tolist(), conv_lengths.tolist())
    ]

    return torch.stack(padded_conversations, 0).transpose(0, 1)


def pad(tensor: Union[Tensor, Variable], length: int) -> Tensor:
    """
    Pads a tensor or Variable to a specified length along the first dimension.

    Args:
        tensor: Input tensor or Variable to pad
        length: Desired length after padding

    Returns:
        Padded tensor of specified length
    """
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        return var

    if length > tensor.size(0):
        return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
    return tensor
