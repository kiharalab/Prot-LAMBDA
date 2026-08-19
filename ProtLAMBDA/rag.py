# This is basically esmfold trunk
# no longer aa pred, aa is induced
# same model as trunk1v1, multi-axis attention


import typing as T
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch import nn
from torch.nn import LayerNorm
import numpy as np


from distformer_block import Trunk, Trunk2
from distogram_fusion_inpainting import DistogramFusionModel

import torch.nn.functional as F


from torch import nn

from .openfold.model.ipa_openfold import StructureModule
from .openfold.model.ipa_openfold_dist_pure import StructureModuleDist

import torch.utils.checkpoint as checkpoint

class LambdaFold(nn.Module):
    def __init__(self, dist_quantizer, params,
                    pretrained_model,
                    saplm_pth,
                    old_weight_pth):
        super().__init__()

        self.params = params
        self.c_s = params['cs']        
        self.c_z = params['cz']        

        self.saplm = SAPLM(pretrained_model,embed_dim=pretrained_model.embed_dim,params={'mdl_id': 0, 'backbone': 'esm2_t33_650M_UR50D', 'train_mode': 'masked', 'mask_ratio': 0.75, 'logit_weight': 1.0, 'ss8_weight': 1.0, 'cmap_weight': 1.0, 'frozen_backbone': 22, 'pair_repr': 'cdist_L2', 'ss_mode': 3, 'ss_conv_flt_len': 1, 'cm_pred_layer': 'CNN', 'cm_mlp_dim': 128, 'cm_conv_flt_len': 1, 'ss_dropout': 0.75, 'cm_dropout': 0.05, 'ss_batchnorm': True, 'cm_batchnorm': True, 'dist_feature_layer': 32, 'pos_embd': False, 'pos_embd_range': 128, 'lr': 0.001, 'c_s': 2560, 'text': '30k experiment, saplm, 22 lyr frzn, 10 heads, grad accum iter 64', 'croplen': 256})
        self.saplm.load_state_dict(torch.load(saplm_pth,map_location='cpu')['model'])
        self.saplm.eval();

        for param in self.saplm.parameters():
            param.requires_grad = False
        
        
        self.trunk0 = Trunk(self.c_s[0],self.c_z[0], params['num_layers'][0],4,None)        
        self.dist_4bin = nn.Linear(self.c_z[0], 4).float()   
        self.structure_module0 = StructureModule(
                        c_s=384,
                        c_z=128,
                        c_ipa=16,
                        c_resnet=128,
                        no_heads_ipa=12,
                        no_qk_points=4,
                        no_v_points=8,
                        dropout_rate=0.1,
                        no_blocks=8,
                        no_transition_layers=1,
                        no_resnet_blocks=2,
                        no_angles=7,
                        trans_scale_factor=10,
                        epsilon=1e-8,
                        inf=1e5
                )

        self.recycle_bins = 15
        self.recycle_s_norm = nn.LayerNorm(self.c_s[0])
        self.recycle_z_norm = nn.LayerNorm(self.c_z[0])
        self.recycle_disto = nn.Embedding(self.recycle_bins, self.c_z[0])
        self.recycle_disto.weight[0].detach().zero_()

        self.recycle_disto_bins = torch.Tensor(list(np.linspace(3.375,21.375,14)))


        self.trunk1 = Trunk2(self.c_s[1],self.c_z[1], params['num_layers'][0],16,None)        
        self.dist_16bin = nn.Linear(self.c_z[1], 16).float()   

        

        
        self.trunk2 = Trunk2(self.c_s[2],self.c_z[2], params['num_layers'][0],64,None,last_cs=self.c_s[1],last_cz=self.c_z[1])
        self.dist_64bin = nn.Linear(self.c_z[2], 64).float()   



        #for param in self.parameters():
        #    param.requires_grad = False

        self.structure_module_dist = StructureModuleDist(
                        dist_quantizer,
                        c_s=384,
                        c_z=128,
                        c_ipa=16,
                        c_resnet=128,
                        no_heads_ipa=12,
                        no_qk_points=4,
                        no_v_points=8,
                        dropout_rate=0.1,
                        no_blocks=8,
                        no_transition_layers=1,
                        no_resnet_blocks=2,
                        no_angles=7,
                        trans_scale_factor=10,
                        epsilon=1e-8,
                        inf=1e5
                )      
        




        self.structure_module2 = StructureModule(
                        c_s=384,
                        c_z=128,
                        c_ipa=16,
                        c_resnet=128,
                        no_heads_ipa=12,
                        no_qk_points=4,
                        no_v_points=8,
                        dropout_rate=0.1,
                        no_blocks=8,
                        no_transition_layers=1,
                        no_resnet_blocks=2,
                        no_angles=7,
                        trans_scale_factor=10,
                        epsilon=1e-8,
                        inf=1e5
                )
        
        
        #self.structure_module2.load_state_dict(torch.load('/net/kihara/home/nibtehaz/dev_struc_emb_new/struc_emb_mbzuai/experiments/pretrained_structure_module/model_state_dict.pt',map_location='cpu'))
        #self.structure_module2.load_state_dict(self.structure_module0.state_dict(), strict=True) 


        self.dist_quantizer = torch.Tensor(dist_quantizer)
        self.dist_crrect = DistogramFusionModel()

        if old_weight_pth is not None:
            try:
                self.load_state_dict(torch.load(old_weight_pth,map_location='cpu')['model'], True)
                print(f"Loaded weights from {old_weight_pth}")
            except:
                self.load_state_dict(torch.load(old_weight_pth,map_location='cpu'), True)


        self.structure_module2.load_state_dict(self.structure_module0.state_dict(), strict=True) 


    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
        
    #repr

    def forward_full(
        self,
        X_minibatch,
        msk_prb,
        aa_tokens,
        af2_aa_tokens,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 0,
        num_recycles_lcl: T.Optional[int] = 0,        
        dist_in=None,
        dist_msk=None,
        chunk_size=None
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
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """ 

        device = X_minibatch.device


        with torch.no_grad():
            seq_repr, pair_repr = self.saplm.get_repr(X_minibatch,masking_ratio=msk_prb,chunk_size=chunk_size)

            seq_repr = seq_repr[:,1:-1,:]
            pair_repr = pair_repr[:,1:-1,1:-1,:]
        
            b,crp_len,_ = seq_repr.shape

            recycle_s = torch.zeros(b,crp_len,self.c_s[0], device=device)
            recycle_z = torch.zeros(b,crp_len,crp_len,self.c_z[0], device=device)
            recycle_bins = torch.zeros(*recycle_z.shape[:-1], device=device, dtype=torch.int64)
            recycle_s_gbl = checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False)
            #recycle_z_gbl = checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z + self.recycle_disto(recycle_bins.detach()) , use_reentrant=False)
            recycle_z_gbl = checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False)

            
            s_s0, s_z0 = self.trunk0(seq_repr,pair_repr,aa_tokens,af2_aa_tokens,mask,residx,masking_pattern,num_recycles_lcl,recycle_s_in=recycle_s_gbl,recycle_z_in=recycle_z_gbl)
            
            dist4 = checkpoint.checkpoint(self.custom(self.dist_4bin), s_z0, use_reentrant=False)       
            dist4 = F.log_softmax(dist4,dim=-1)
            cs0_feats = checkpoint.checkpoint(self.custom(self.trunk0.cs_mpr), s_s0, use_reentrant=False)
            cz0_feats = checkpoint.checkpoint(self.custom(self.trunk0.cz_mpr), s_z0, use_reentrant=False)
            struc_out0 = self.structure_module0(cs0_feats,cz0_feats,None,None, af2_aa_tokens,mask.to(torch.float32).to(cs0_feats.device))
        
        
            s_s1, s_z1 = self.trunk1(seq_repr, pair_repr,s_s0,s_z0,aa_tokens,af2_aa_tokens,mask,residx,masking_pattern,num_recycles_lcl,plm_add=False)

            dist16 = checkpoint.checkpoint(self.custom(self.dist_16bin), s_z1, use_reentrant=False)       
            dist16 = F.log_softmax(dist16,dim=-1)
            cs1_feats = checkpoint.checkpoint(self.custom(self.trunk1.cs_mpr), s_s1, use_reentrant=False)
            cz1_feats = checkpoint.checkpoint(self.custom(self.trunk1.cz_mpr), s_z1, use_reentrant=False)
            struc_out1 = self.structure_module0(cs1_feats,cz1_feats,None,None, af2_aa_tokens,mask.to(torch.float32).to(cs1_feats.device))

            s_s2, s_z2 = self.trunk2(seq_repr, pair_repr,s_s1,s_z1,aa_tokens,af2_aa_tokens,mask,residx,masking_pattern,num_recycles_lcl,plm_add=False)

            dist64 = checkpoint.checkpoint(self.custom(self.dist_64bin), s_z2, use_reentrant=False)       
            dist64 = F.log_softmax(dist64,dim=-1)
            cs2_feats = checkpoint.checkpoint(self.custom(self.trunk2.cs_mpr), s_s2, use_reentrant=False)
            cz2_feats = checkpoint.checkpoint(self.custom(self.trunk2.cz_mpr), s_z2, use_reentrant=False)
            struc_out2 = self.structure_module0(cs2_feats,cz2_feats,None,None, af2_aa_tokens,mask.to(torch.float32).to(cs2_feats.device))



            cur_pos = {'positions':struc_out2['positions'][7]}

            cb_coords = (cur_pos['positions'][:,:,4,:] * torch.unsqueeze(af2_aa_tokens!=7,dim=-1) + cur_pos['positions'][:,:,1,:] * torch.unsqueeze(af2_aa_tokens==7,dim=-1))
            pred_struc_dist = torch.bucketize(torch.cdist(cb_coords,cb_coords),self.dist_quantizer.to(cb_coords.device)).long()

            residue_index = torch.arange(struc_out2['single'].shape[1], device=struc_out2['single'].device)[None,:].expand(struc_out2['single'].shape[0],-1)

        rfnd_dist = self.dist_crrect(pred_struc_dist, dist_in, dist_msk,residue_index,mask)
        
        rfnd_dist_true = torch.argmax(rfnd_dist, dim=-1)

        probs = torch.softmax(rfnd_dist, dim=-1)
        rfnd_dist_sft = torch.sum(probs * torch.arange(rfnd_dist.shape[-1], device=rfnd_dist.device), dim=-1)        # soft version        

        rfnd_dist = torch.argmax(rfnd_dist, dim=-1)

        rfnd_dist = dist_msk * dist_in + (1-dist_msk) * rfnd_dist

        rfnd_dist = rfnd_dist.long()

        cur_pos = {'positions':struc_out2['positions'][7]}
    
        struc_out3 = self.structure_module_dist(struc_out2['single'],rfnd_dist,struc_out2['t'],cur_pos,af2_aa_tokens,mask.to(torch.float32),dist_msk=None)

        struc_out4 = self.structure_module2(cs2_feats,cz2_feats,struc_out3['t'],struc_out3['single'],af2_aa_tokens,mask.to(torch.float32))


        return rfnd_dist_true,rfnd_dist_sft,rfnd_dist, [struc_out0, struc_out1, struc_out2, struc_out3, struc_out4]
       
        
    def forward_dxy(
        self,
        cs2_feats,
        cz2_feats,
        dist_in,
        dist_msk=None,
        af2_aa_tokens=None,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 0,
        num_recycles_lcl: T.Optional[int] = 0,        
        gt_dist=None,
        strct_copy=True,
        tmplt_in=None,
        tmplt_mask=None
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            dist_in (torch.Tensor): Tensor containing distance information.
            af2_aa_tokens (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """ 

        device = cs2_feats.device


        with torch.no_grad():
            
            struc_out = self.structure_module0(cs2_feats,cz2_feats,None,None, af2_aa_tokens,mask.to(torch.float32).to(cs2_feats.device))

            cur_pos = {'positions':struc_out['positions'][7]}

            cb_coords = (cur_pos['positions'][:,:,4,:] * torch.unsqueeze(af2_aa_tokens!=7,dim=-1) + cur_pos['positions'][:,:,1,:] * torch.unsqueeze(af2_aa_tokens==7,dim=-1))
            pred_struc_dist = torch.bucketize(torch.cdist(cb_coords,cb_coords),self.dist_quantizer.to(cb_coords.device)).long()

            residue_index = torch.arange(struc_out['single'].shape[1], device=struc_out['single'].device)[None,:].expand(struc_out['single'].shape[0],-1)

        rfnd_dist = self.dist_crrect(pred_struc_dist, dist_in, dist_msk,residue_index,mask)

        return rfnd_dist


    def forward_xy(
        self,
        cs2_feats,
        cz2_feats,
        dist_in,
        dist_msk=None,
        af2_aa_tokens=None,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 0,
        num_recycles_lcl: T.Optional[int] = 0,        
        gt_dist=None,
        strct_copy=True,
        tmplt_in=None,
        tmplt_mask=None
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            dist_in (torch.Tensor): Tensor containing distance information.
            af2_aa_tokens (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """ 

        device = cs2_feats.device


        with torch.no_grad():
            
            struc_out0 = self.structure_module0(cs2_feats,cz2_feats,None,None, af2_aa_tokens,mask.to(torch.float32).to(cs2_feats.device))

            cur_pos = {'positions':struc_out0['positions'][7]}

            cb_coords = (cur_pos['positions'][:,:,4,:] * torch.unsqueeze(af2_aa_tokens!=7,dim=-1) + cur_pos['positions'][:,:,1,:] * torch.unsqueeze(af2_aa_tokens==7,dim=-1))
            pred_struc_dist = torch.bucketize(torch.cdist(cb_coords,cb_coords),self.dist_quantizer.to(cb_coords.device)).long()

            residue_index = torch.arange(struc_out0['single'].shape[1], device=struc_out0['single'].device)[None,:].expand(struc_out0['single'].shape[0],-1)

            rfnd_dist = self.dist_crrect(pred_struc_dist, dist_in, dist_msk,residue_index,mask)
            rfnd_dist = torch.argmax(rfnd_dist, dim=-1)
            rfnd_dist = dist_msk * dist_in + (1-dist_msk) * rfnd_dist
            rfnd_dist = rfnd_dist.long()
    
        struc_out1 = self.structure_module_dist(struc_out0['single'],rfnd_dist,struc_out0['t'],cur_pos,af2_aa_tokens,mask.to(torch.float32),dist_msk=None)

        return struc_out0, struc_out1
    
    def forward(
        self,
        cs2_feats,
        cz2_feats,
        dist_in,
        dist_msk=None,
        af2_aa_tokens=None,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 0,
        num_recycles_lcl: T.Optional[int] = 0,        
        gt_dist=None,
        strct_copy=True,
        tmplt_in=None,
        tmplt_mask=None
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            dist_in (torch.Tensor): Tensor containing distance information.
            af2_aa_tokens (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """ 

        device = cs2_feats.device


        with torch.no_grad():
            
            struc_out0 = self.structure_module0(cs2_feats,cz2_feats,None,None, af2_aa_tokens,mask.to(torch.float32).to(cs2_feats.device))

            cur_pos = {'positions':struc_out0['positions'][7]}

            cb_coords = (cur_pos['positions'][:,:,4,:] * torch.unsqueeze(af2_aa_tokens!=7,dim=-1) + cur_pos['positions'][:,:,1,:] * torch.unsqueeze(af2_aa_tokens==7,dim=-1))
            pred_struc_dist = torch.bucketize(torch.cdist(cb_coords,cb_coords),self.dist_quantizer.to(cb_coords.device)).long()

            residue_index = torch.arange(struc_out0['single'].shape[1], device=struc_out0['single'].device)[None,:].expand(struc_out0['single'].shape[0],-1)

            rfnd_dist = self.dist_crrect(pred_struc_dist, dist_in, dist_msk,residue_index,mask)
            rfnd_dist = torch.argmax(rfnd_dist, dim=-1)
            rfnd_dist = dist_msk * dist_in + (1-dist_msk) * rfnd_dist
            rfnd_dist = rfnd_dist.long()
    
            struc_out1 = self.structure_module_dist(struc_out0['single'],rfnd_dist,struc_out0['t'],cur_pos,af2_aa_tokens,mask.to(torch.float32),dist_msk=None)

        struc_out2 = self.structure_module2(cs2_feats,cz2_feats,struc_out1['t'],struc_out1['single'],af2_aa_tokens,mask.to(torch.float32))

        return struc_out0, struc_out2

    def get_all_pred(
        self,
        X_minibatch,
        msk_prb,
        aa_tokens,
        af2_aa_tokens,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 0,
        num_recycles_lcl: T.Optional[int] = 0,        
        gt_dist=None,
        strct_copy=True
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
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """ 

        device = X_minibatch.device


        with torch.no_grad():
            seq_repr, pair_repr = self.saplm.get_repr(X_minibatch,masking_ratio=msk_prb)

            seq_repr = seq_repr[:,1:-1,:]
            pair_repr = pair_repr[:,1:-1,1:-1,:]
        
            b,crp_len,_ = seq_repr.shape

            recycle_s = torch.zeros(b,crp_len,self.c_s[0], device=device)
            recycle_z = torch.zeros(b,crp_len,crp_len,self.c_z[0], device=device)
            recycle_bins = torch.zeros(*recycle_z.shape[:-1], device=device, dtype=torch.int64)
            recycle_s_gbl = checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False)
            #recycle_z_gbl = checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z + self.recycle_disto(recycle_bins.detach()) , use_reentrant=False)
            recycle_z_gbl = checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False)

            
            s_s0, s_z0 = self.trunk0(seq_repr,pair_repr,aa_tokens,af2_aa_tokens,mask,residx,masking_pattern,num_recycles_lcl,recycle_s_in=recycle_s_gbl,recycle_z_in=recycle_z_gbl)            
            dist4 = checkpoint.checkpoint(self.custom(self.dist_4bin), s_z0, use_reentrant=False)       
            dist4 = F.log_softmax(dist4,dim=-1)
            cs0_feats = checkpoint.checkpoint(self.custom(self.trunk0.cs_mpr), s_s0, use_reentrant=False)
            cz0_feats = checkpoint.checkpoint(self.custom(self.trunk0.cz_mpr), s_z0, use_reentrant=False)
            struc_out0 = self.structure_module0(cs0_feats,cz0_feats,None,None, af2_aa_tokens,mask.to(torch.float32).to(cs0_feats.device))
        
        
            s_s1, s_z1 = self.trunk1(seq_repr, pair_repr,s_s0,s_z0,aa_tokens,af2_aa_tokens,mask,residx,masking_pattern,num_recycles_lcl,plm_add=False)
            dist16 = checkpoint.checkpoint(self.custom(self.dist_16bin), s_z1, use_reentrant=False)       
            dist16 = F.log_softmax(dist16,dim=-1)
            cs1_feats = checkpoint.checkpoint(self.custom(self.trunk1.cs_mpr), s_s1, use_reentrant=False)
            cz1_feats = checkpoint.checkpoint(self.custom(self.trunk1.cz_mpr), s_z1, use_reentrant=False)
            struc_out1 = self.structure_module0(cs1_feats,cz1_feats,None,None, af2_aa_tokens,mask.to(torch.float32).to(cs1_feats.device))

            s_s2, s_z2 = self.trunk2(seq_repr, pair_repr,s_s1,s_z1,aa_tokens,af2_aa_tokens,mask,residx,masking_pattern,num_recycles_lcl,plm_add=False)
            dist64 = checkpoint.checkpoint(self.custom(self.dist_64bin), s_z2, use_reentrant=False)       
            dist64 = F.log_softmax(dist64,dim=-1)
            cs2_feats = checkpoint.checkpoint(self.custom(self.trunk2.cs_mpr), s_s2, use_reentrant=False)
            cz2_feats = checkpoint.checkpoint(self.custom(self.trunk2.cz_mpr), s_z2, use_reentrant=False)
            struc_out2 = self.structure_module0(cs2_feats,cz2_feats,None,None, af2_aa_tokens,mask.to(torch.float32).to(cs2_feats.device))

        return [dist4,dist16,dist64], [struc_out0,struc_out1,struc_out2], [cs0_feats,cs1_feats,cs2_feats], [cz0_feats,cz1_feats,cz2_feats]