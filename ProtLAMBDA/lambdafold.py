import typing as T
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import LayerNorm
import numpy as np

import torch.nn.functional as F
from .openfold.model.ipa_openfold import StructureModule
from .openfold.model.dataset.af2_util import residue_constants
from .utils import prepare_pdb_file
import torch.utils.checkpoint as checkpoint


class LambdaFold(nn.Module):
    def __init__(self, 
                prot_lambda_mdl,
                distformer_mdl):
        super().__init__()

        
        self.prot_lambda = prot_lambda_mdl
        self.prot_lambda.eval();

        self.distformer = distformer_mdl
        self.distformer.eval();

        self.structure_module = StructureModule(
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
        

        self.lddt_head = nn.Sequential(
                LayerNorm(384),
                nn.Linear(384, 128),
                nn.ReLU(),
                nn.Linear(128, 100),
            )



    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
        
    #repr

    def forward(self,seq, masking_ratio=1.00, chunk_size=None, return_all=False):
        """
        Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
                    seq str : The input sequence.
                    masking_ratio (float): The ratio of tokens to mask.
                    chunk_size (int): The size of the chunks to process.
        """

        
        
        ## add padding
        padding = 16 - len(seq)%16
        seq = seq+'<pad>'*padding        
        seqs = [[f"input", seq]]
        
        seq_repr, pair_repr = self.prot_lambda.get_repr(seqs)

        seq_repr = seq_repr[:,1:-1,:]
        pair_repr = pair_repr[:,1:-1,1:-1,:]
                
        aa_tokens = self.prot_lambda.esm2_backbone.batch_converter(seqs)[:,1:-1].to(seq_repr.device)
        af2_aa_tokens = torch.Tensor([residue_constants.restype_order.get(residue_constants.tok_mapper[bb], residue_constants.restype_num) for bb in torch.flatten(aa_tokens)]).to(torch.long).reshape(aa_tokens.shape).to(aa_tokens.device)
        valid_seq_masks = (aa_tokens!=1)*1

        
        dist_outs, css, czs = self.distformer(seq_repr, pair_repr, aa_tokens, af2_aa_tokens,mask=valid_seq_masks, return_all=return_all)

        struc_outs = []
        for i in range(len(css)):
            struc_out = self.structure_module(css[i],czs[i],None,None, af2_aa_tokens,valid_seq_masks.to(torch.float32).to(css[i].device))                
            struc_outs.append(struc_out)

        plddt = self.lddt_head(struc_outs[-1]['single'])
        plddt = torch.log_softmax(plddt,dim=-1)
        plddt = plddt[0,:-padding,:].cpu().detach().numpy()
        


        ## fix padding

        for i,dist_out in enumerate(dist_outs):            
            dist_outs[i] = dist_out[0,:-padding,:-padding,:].cpu().detach().numpy()

        pdb_outs = []
        for struc_out in struc_outs:
            for ky in struc_out:                
                if ky!='t':
                    struc_out[ky] = struc_out[ky].to('cpu')

            pdb_outs.append(prepare_pdb_file(struc_out, af2_aa_tokens, plddt))
        
            

        return dist_outs, struc_outs, plddt, pdb_outs