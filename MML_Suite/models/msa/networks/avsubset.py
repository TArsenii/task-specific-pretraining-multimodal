from torch import Tensor
from torch.nn import LSTM, Dropout, Linear, Module
from torch.nn.utils.rnn import pack_padded_sequence


class AuViSubNet(Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        out_size: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        """
        super(AuViSubNet, self).__init__()
        self.rnn = LSTM(
            in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True
        )
        self.dropout = Dropout(dropout)
        self.linear_1 = Linear(hidden_size, out_size)

    def forward(self, x: Tensor, lengths: Tensor):
        """
        x: (batch_size, sequence_len, in_size)
        """
        packed_sequence = pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1
