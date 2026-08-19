import typing as T
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import LayerNorm
import numpy as np

from .distformer_block import Trunk, Trunk2

import torch.nn.functional as F
from .openfold.model.ipa_openfold import StructureModule
from .openfold.model.ipa_openfold_dist_pure import StructureModuleDist

import torch.utils.checkpoint as checkpoint

class DistFormer(nn.Module):
    def __init__(self, cs=[512,640,768], cz=[192,224,256], num_layers=[2,2,2]):
        super().__init__()
        
        self.c_s = cs
        self.c_z = cz
        self.num_layers = num_layers

        self.stage1 = Trunk(self.c_s[0],self.c_z[0], self.num_layers[0],4,None)        
        self.dist_4bin = nn.Linear(self.c_z[0], 4).float()   

        self.stage2 = Trunk2(self.c_s[1],self.c_z[1], self.num_layers[1],16,None,last_cs=self.c_s[0],last_cz=self.c_z[0])
        self.dist_16bin = nn.Linear(self.c_z[1], 16).float()   

        self.stage3 = Trunk2(self.c_s[2],self.c_z[2], self.num_layers[2],64,None,last_cs=self.c_s[1],last_cz=self.c_z[1])
        self.dist_64bin = nn.Linear(self.c_z[2], 64).float()   

        self.recycle_bins = 15
        self.recycle_s_norm = nn.LayerNorm(self.c_s[0])
        self.recycle_z_norm = nn.LayerNorm(self.c_z[0])
        self.recycle_disto = nn.Embedding(self.recycle_bins, self.c_z[0])
        self.recycle_disto.weight[0].detach().zero_()
        self.recycle_disto_bins = torch.Tensor(list(np.linspace(3.375,21.375,14)))




    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
        

    def forward(
        self,
        seq_repr, 
        pair_repr,
        aa_tokens,
        af2_aa_tokens,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles_lcl: T.Optional[int] = 0,        
        return_all: T.Optional[bool] = False,
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            aa (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles_lcl (int): Number of recycles to perform. If not provided, will use the default number of recycles.
                which is 0
            return_all (bool): If True, returns all intermediate outputs. If False, returns only the final output.
            
        """ 

        device = seq_repr.device        
        b,crp_len,_ = seq_repr.shape

        recycle_s = torch.zeros(b,crp_len,self.c_s[0], device=device)
        recycle_z = torch.zeros(b,crp_len,crp_len,self.c_z[0], device=device)
        recycle_bins = torch.zeros(*recycle_z.shape[:-1], device=device, dtype=torch.int64)
        recycle_s_gbl = checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False)
        recycle_z_gbl = checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z + self.recycle_disto(recycle_bins.detach()) , use_reentrant=False)

        s_s0, s_z0 = self.stage1(seq_repr,pair_repr,aa_tokens,af2_aa_tokens,mask,residx,masking_pattern,num_recycles_lcl,recycle_s_in=recycle_s_gbl,recycle_z_in=recycle_z_gbl)        
        
        dist4 = checkpoint.checkpoint(self.custom(self.dist_4bin), s_z0, use_reentrant=False)       
        dist4 = F.log_softmax(dist4,dim=-1)

        cs0_feats = checkpoint.checkpoint(self.custom(self.stage1.cs_mpr), s_s0, use_reentrant=False)
        cz0_feats = checkpoint.checkpoint(self.custom(self.stage1.cz_mpr), s_z0, use_reentrant=False)
        
        s_s1, s_z1 = self.stage2(seq_repr, pair_repr,s_s0,s_z0,aa_tokens,af2_aa_tokens,mask,residx,masking_pattern,num_recycles_lcl,plm_add=False)
        dist16 = checkpoint.checkpoint(self.custom(self.dist_16bin), s_z1, use_reentrant=False)       
        dist16 = F.log_softmax(dist16,dim=-1)

        cs1_feats = checkpoint.checkpoint(self.custom(self.stage2.cs_mpr), s_s1, use_reentrant=False)
        cz1_feats = checkpoint.checkpoint(self.custom(self.stage2.cz_mpr), s_z1, use_reentrant=False)
        
        s_s2, s_z2 = self.stage3(seq_repr, pair_repr,s_s1,s_z1,aa_tokens,af2_aa_tokens,mask,residx,masking_pattern,num_recycles_lcl,plm_add=False)
        dist64 = checkpoint.checkpoint(self.custom(self.dist_64bin), s_z2, use_reentrant=False)       
        dist64 = F.log_softmax(dist64,dim=-1)

        cs2_feats = checkpoint.checkpoint(self.custom(self.stage3.cs_mpr), s_s2, use_reentrant=False)
        cz2_feats = checkpoint.checkpoint(self.custom(self.stage3.cz_mpr), s_z2, use_reentrant=False)
        

        if return_all:
            return [dist4, dist16, dist64], [cs0_feats, cs1_feats, cs2_feats], [cz0_feats, cz1_feats, cz2_feats]
        else:
            return [dist64], [cs2_feats], [cz2_feats]
        