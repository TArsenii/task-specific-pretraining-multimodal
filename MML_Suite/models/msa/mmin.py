from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import Module, Linear
import torch.nn.functional as F
from typing import Dict

from modalities import Modality

from models.msa import LSTMEncoder, TextCNN, ResidualAE, FcClassifier, UttFusionModel


class MMIN(Module):
    def __init__(
        self,
        netA: LSTMEncoder,
        netV: LSTMEncoder,
        netT: TextCNN,
        netAE: ResidualAE,
        netC: FcClassifier,
        *,
        share_weight: bool = False,
    ):
        super(MMIN, self).__init__()
        ## loss names
        ## model names

        self.netA = netA
        self.netV = netV
        self.netT = netT
        self.netAE = netAE

        ae_input_dim = self.netA.hidden_size + self.netV.hidden_size + self.netT.hidden_size

        if share_weight:
            self.netAE_cycle = self.netAE
        else:
            self.netAE_cycle = ResidualAE(
                self.netAE._layers, self.netAE.n_blocks, ae_input_dim, dropout=0.0, use_bn=False
            )
        
        self.netC = netC
        self.pretrained_module: UttFusionModel = None
        

    def forward(
        self, A: Tensor, V: Tensor, T: Tensor, A_reverse: Tensor, V_reverse: Tensor, T_reverse: Tensor
    ) -> Dict[Modality, Tensor]:
        A_feat_miss = self.netA(A)
        V_feat_miss = self.netV(V)
        T_feat_miss = self.netT(T)
        
        feat_fusion_miss = torch.cat([A_feat_miss, V_feat_miss, T_feat_miss], dim=-1)
        
        recon_fusion, latent = self.netAE(feat_fusion_miss)
        recon_cycle, latent_cycle = self.netAE_cycle(recon_fusion)
        
        logits, _ = self.netC(latent)
        predictions = F.softmax(logits, dim=-1)
        
        if self.training:
            with torch.no_grad():
                embd_A = self.pretrained_module.netA(A_reverse)
                embd_V = self.pretrained_module.netV(V_reverse)
                embd_T = self.pretrained_module.netT(T_reverse)
                embds = torch.cat([embd_A, embd_V, embd_T], dim=-1)
                
        # losses + backward + optimizer
                
        
        return {}
    
    def train_step():
        pass
    
    def validation_step():
        pass
