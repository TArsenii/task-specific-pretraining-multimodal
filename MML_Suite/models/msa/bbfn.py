from models.mixins import MultimodalMonitoringMixin
from torch.nn import Module


class BBFN(Module, MultimodalMonitoringMixin):
    def __init__(self):
        super(BBFN, self).__init__()

    def forward(self, A, V, T, is_embd_A, is_embd_V, is_embd_T):
        pass

    def train_step(self, batch, criterion, optimizer, device):
        pass

    def validation_step(self, batch, criterion, device):
        pass
