# This is basically esmfold trunk
# no longer aa pred, aa is induced
# same model as trunk1v1, multi-axis attention


import typing as T
from dataclasses import dataclass
from einops import rearrange

import math
import torch
import torch.nn as nn
from torch import nn
from torch.nn import LayerNorm
#from torchvision.ops.stochastic_depth import StochasticDepth

import json
import torch.nn.functional as F
from .openfold.model.triangular_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from .openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from torch import nn

from .esm import Dropout, Attention, PairToSequence, ResidueMLP


from .esm.multihead_attention import MultiheadAttention 
from .esm.modules import ESM1bLayerNorm


import torch.utils.checkpoint as checkpoint

def _get_relative_position_index(height: int, width: int) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid([torch.arange(height), torch.arange(width)]))
    coords_flat = torch.flatten(coords, 1)
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += height - 1
    relative_coords[:, :, 1] += width - 1
    relative_coords[:, :, 0] *= 2 * width - 1
    return relative_coords.sum(-1)



class GridAttention(nn.Module):
    def __init__(self, embed_dim, attention_heads):
        super().__init__()

        self.row_attn = MultiheadAttention(
            embed_dim,
            attention_heads,
            add_bias_kv=False,
            add_zero_attn=False,
            use_rotary_embeddings=True
        )

        self.col_attn = MultiheadAttention(
            embed_dim,
            attention_heads,
            add_bias_kv=False,
            add_zero_attn=False,
            use_rotary_embeddings=True
        )
        
        self.row_attn_layer_norm = ESM1bLayerNorm(embed_dim)
        self.col_attn_layer_norm = ESM1bLayerNorm(embed_dim)


       
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
    
    def custom_attn(self, module):
        def custom_forward_attn(*inputs):
            out,_ = module(query=inputs[0],
                            key=inputs[0],
                            value=inputs[0],
                            key_padding_mask=inputs[1],
                            need_weights=False,
                            need_head_weights=False,
                            attn_mask=None)
            return out
        return custom_forward_attn

    def forward(self, x, pad_msk):

        def add_func_my(x1, x2):
            return x1 + x2
        
        ## ESM MHA requires (tokens,batch,embed)

        b,e,l,w = x.shape
        if pad_msk is not None:            
            pad_msk = torch.unsqueeze(pad_msk,dim=1)
            pad_msk = torch.flatten(pad_msk.repeat(1,w,1),end_dim=1)

        pad_msk = rearrange(pad_msk, 'b (t ts)-> (b ts) t', ts=8)
        pad_msk = 1 - pad_msk

        x_r = rearrange(x, 'b e x y -> x (b y) e')      # I am assuming this level gets (B E X Y)
        x_r = rearrange(x_r, '(t ts) b c -> t (b ts) c', ts=8)
        #x_r = self.row_attn_layer_norm(x_r)
        x_r = checkpoint.checkpoint(self.custom(self.row_attn_layer_norm), x_r, use_reentrant=False)
        
        '''
        x_r,_ = self.row_attn(query=x_r,
                            key=x_r,
                            value=x_r,
                            key_padding_mask=pad_msk,
                            need_weights=False,
                            need_head_weights=False,
                            attn_mask=None)
        '''
        x_r = checkpoint.checkpoint(self.custom_attn(self.row_attn), x_r,pad_msk, use_reentrant=False)
        x_r = rearrange(x_r, 't (b ts) c -> (t ts) b c', ts=8)
        x = rearrange(x_r, 'x (b y) e -> b e x y',b=b)

        x_c = rearrange(x, 'b e x y -> y (b x) e')        
        x_c = rearrange(x_c, '(t ts) b c -> t (b ts) c', ts=8)
        #x_c = self.col_attn_layer_norm(x_c)
        x_c = checkpoint.checkpoint(self.custom(self.col_attn_layer_norm), x_c, use_reentrant=False)        
        '''
        x_c,_ = self.col_attn(query=x_c,
                            key=x_c,
                            value=x_c,
                            key_padding_mask=pad_msk,
                            need_weights=False,
                            need_head_weights=False,
                            attn_mask=None)
        '''
        x_c = checkpoint.checkpoint(self.custom_attn(self.col_attn), x_c,pad_msk, use_reentrant=False)
        x_c = rearrange(x_c, 't (b ts) c -> (t ts) b c', ts=8)
        x = rearrange(x_c, 'y (b x) e -> b e x y',b=b)

        return x


class GridAttentionLayer(nn.Module):
    def __init__(self, embed_dim, attention_heads, mlp_ratio,attention_dropout,mlp_dropout):
        super().__init__()

        self.attn = GridAttention(embed_dim,attention_heads)

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio,  embed_dim),
        )

        self.attn_drp = nn.Dropout(attention_dropout)

        self.mlp_drp = nn.Dropout(mlp_dropout)        
       
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
    
   
    def forward(self, x, pad_msk):

        # x should be (b,e,x,y)

        x = x + self.attn_drp(self.attn(x,pad_msk))
        #x = x + self.mlp_drp(self.mlp(x.permute(0,2,3,1)).permute(0,3,1,2))
        x = x + self.mlp_drp(
            checkpoint.checkpoint(self.custom(self.mlp), x.permute(0,2,3,1), use_reentrant=False).permute(0,3,1,2)
        )        
        
        return x


class GridAttentionBlock(nn.Module):
    def __init__(self, nblocks, embed_dim, attention_heads, mlp_ratio,attention_dropout,mlp_dropout):

        super().__init__()
        self.lyrs = nn.ModuleList()

        for i in range(nblocks):
            self.lyrs.append(
                GridAttentionLayer(embed_dim, attention_heads, mlp_ratio,attention_dropout,mlp_dropout)
            )

    def forward(self,x,pad_msk):
        #nans = []
        #nans.append(torch.isnan(x).any())
        for i in range(len(self.lyrs)):
            x = self.lyrs[i](x,pad_msk)
            #nans.append(torch.isnan(x).any())

        #print(nans)
        return x



class RelativePositionalMultiHeadAttention(nn.Module):    
    def __init__(
        self,
        feat_dim: int,
        head_dim: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()

        if feat_dim % head_dim != 0:
            raise ValueError(f"feat_dim: {feat_dim} must be divisible by head_dim: {head_dim}")

        self.n_heads = feat_dim // head_dim
        self.head_dim = head_dim
        self.size = int(math.sqrt(max_seq_len))
        self.max_seq_len = max_seq_len

        self.to_qkv = nn.Linear(feat_dim, self.n_heads * self.head_dim * 3)
        self.scale_factor = feat_dim**-0.5

        self.merge = nn.Linear(self.head_dim * self.n_heads, feat_dim)
        self.relative_position_bias_table = nn.parameter.Parameter(
            torch.empty(((2 * self.size - 1) * (2 * self.size - 1), self.n_heads), dtype=torch.float32),
        )

        # initialize with truncated normal the bias
        self.register_buffer("relative_position_index", _get_relative_position_index(self.size, self.size))
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_positional_bias(self) -> torch.Tensor:
        bias_index = self.relative_position_index.view(-1)  # type: ignore
        relative_bias = self.relative_position_bias_table[bias_index].view(self.max_seq_len, self.max_seq_len, -1)  # type: ignore
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        return relative_bias.unsqueeze(0)

    def forward(self, x, mask):
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, G, P, D].
            mask : mask should also be [B,G,P], 1 if good, 0 is padding
        Returns:
            Tensor: Output tensor with expected layout of [B, G, P, D].
        """
        B, G, P, D = x.shape
        H, DH = self.n_heads, self.head_dim

        mask = (mask[:,:,None,None,:]*mask[:,:,None,:,None]).repeat(1,1,H,1,1)
        mask = mask.float()
        ###mask[mask<0.5] = float('-inf')

        #print('x',torch.isnan(x).any(),torch.isinf(x).any())#,x)

        #print('mask',torch.isnan(mask).any(),torch.isinf(mask).any())#,mask)

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)

        #print('q',torch.isnan(q).any(),torch.isinf(q).any())#,q)
        #print('k',torch.isnan(k).any(),torch.isinf(k).any())#,k)
        #print('v',torch.isnan(v).any(),torch.isinf(v).any())#,v)

        k = k * self.scale_factor

        #print('scaled k',torch.isnan(k).any(),torch.isinf(k).any())#,k)

        dot_prod = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        #print('dot_prod',torch.isnan(dot_prod).any(),torch.isinf(dot_prod).any())#,dot_prod)
        pos_bias = self.get_relative_positional_bias()

        

        ####dot_prod = F.softmax(dot_prod + pos_bias + mask.detach(), dim=-1)
        ####dot_prod = torch.nan_to_num(dot_prod)
        
        dot_prod = torch.exp(dot_prod+pos_bias) * mask
        dot_prod = dot_prod/(torch.sum(dot_prod,dim=-1,keepdims=True)+1e-4)
        #dot_prod = torch.nan_to_num(dot_prod)


        #print('softmax dot_prod',torch.isnan(dot_prod).any(),torch.isinf(dot_prod).any())#,dot_prod)

        out = torch.einsum("B G H I J, B G H J D -> B G H I D", dot_prod, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, G, P, D)
        #print('out',torch.isnan(out).any(),torch.isinf(out).any())#,out)

        out = self.merge(out)
        return out

class SwapAxes(nn.Module):
    """Permute the axes of a tensor."""

    def __init__(self, a: int, b: int) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        res = torch.swapaxes(x, self.a, self.b)
        return res

class WindowPartition(nn.Module):
    """
    Partition the input tensor into non-overlapping windows.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, p):
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
            p (int): Number of partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, H/P * W/P, P*P, C].
        """
        B, C, H, W = x.shape
        P = p
        # chunk up H and W dimensions
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1)
        # colapse P * P dimension
        x = x.reshape(B, (H // P) * (W // P), P * P, C)
        return x

class WindowDepartition(nn.Module):
    """
    Departition the input tensor of non-overlapping windows into a feature volume of layout [B, C, H, W].
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, p: int, h_partitions: int, w_partitions: int):
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, (H/P * W/P), P*P, C].
            p (int): Number of partitions.
            h_partitions (int): Number of vertical partitions.
            w_partitions (int): Number of horizontal partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """
        B, G, PP, C = x.shape
        P = p
        HP, WP = h_partitions, w_partitions
        # split P * P dimension into 2 P tile dimensionsa
        x = x.reshape(B, HP, WP, P, P, C)
        # permute into B, C, HP, P, WP, P
        x = x.permute(0, 5, 1, 3, 2, 4)
        # reshape into B, C, H, W
        x = x.reshape(B, C, HP * P, WP * P)
        return x


class PartitionAttentionLayer(nn.Module):    
    def __init__(
        self,
        in_channels: int,
        head_dim: int,
        # partitioning parameters
        partition_size: int,
        partition_type: str,
        # grid size needs to be known at initialization time
        # because we need to know hamy relative offsets there are in the grid
        grid_size,
        mlp_ratio: int,
        activation_layer,
        norm_layer,
        attention_dropout: float,
        mlp_dropout: float,
        p_stochastic_dropout: float,
    ) -> None:
        super().__init__()

        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.partition_type = partition_type

        
        
        self.p = partition_size
        
        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()

        

        self.attn_norm = norm_layer(in_channels)
        self.attn_layer = RelativePositionalMultiHeadAttention(in_channels, head_dim, partition_size**2)        

        self.attn_drp = nn.Dropout(attention_dropout)

        # pre-normalization similar to transformer layers
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * mlp_ratio),
            activation_layer(),
            nn.Linear(in_channels * mlp_ratio, in_channels),            
        )

        self.mlp_drp = nn.Dropout(mlp_dropout)

        # layer scale factors
        
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def custom2inp(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0],inputs[1])
            return inputs
        return custom_forward

    def forward(self, x, mask):
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
            mask : Input tensor with expected layout of [B, 1, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """

        # Undefined behavior if H or W are not divisible by p
        # https://github.com/google-research/maxvit/blob/da76cf0d8a6ec668cc31b399c4126186da7da944/maxvit/models/maxvit.py#L766

        
        gh, gw = x.shape[2]//self.p, x.shape[3]//self.p

        x = self.partition_op(x, self.p)
        mask = torch.squeeze(self.partition_op(mask, self.p),dim=-1)

        res = x
        ###x = self.attn_norm(x)
        x = checkpoint.checkpoint(self.custom(self.attn_norm), x, use_reentrant=False)               
        
        #x = res + self.attn_layer(x ,mask)
        x = res + checkpoint.checkpoint(self.custom2inp(self.attn_layer), x ,mask, use_reentrant=False)
        # self.attn_drp(
        #        checkpoint.checkpoint(self.custom2inp(self.attn_layer), x ,mask, use_reentrant=False)
        #    )

        #x = x + self.mlp_layer(x)
        x = x + checkpoint.checkpoint(self.custom(self.mlp_layer), x, use_reentrant=False)            
        # self.mlp_drp(
        #        checkpoint.checkpoint(self.custom(self.mlp_layer), x, use_reentrant=False)            
        #    )
        ###x = self.departition_swap(x)
        x = self.departition_op(x, self.p, gh, gw)

        return x


class Concat3rdAxis(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(x,y):

        return torch.cat([x,
                         y],
                    dim=3)

class MultiAxisAttention(nn.Module):

    def __init__(
        self,        
        n_channels,        
        activation_layer= nn.GELU,
        head_dim=8,
        mlp_ratio=4,
        partition_size=8,
        grid_size=(128,128),
        flip = False
    ) -> None:
        super().__init__()

        
        # attention layers, block -> grid
        self.local_attention = PartitionAttentionLayer(
            in_channels=n_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="window",
            grid_size=grid_size,
            mlp_ratio=4,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            p_stochastic_dropout=0.0,
        )

        self.grid_attention = GridAttentionBlock(
                                            2,
                                            n_channels,
                                            n_channels//head_dim,
                                            4,
                                            attention_dropout=0.0,
                                            mlp_dropout=0.0)
        
        '''PartitionAttentionLayer(
            in_channels=n_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="grid",
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            p_stochastic_dropout=0.0,
        )'''


        self.flip = flip

        self.cat = Concat3rdAxis()

        self.mixer = nn.Sequential(
            LayerNorm(n_channels*2),
            nn.Linear(n_channels*2, n_channels),
            nn.GELU(),
            nn.Linear(n_channels,n_channels),            
        ).float()

        
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
     

    def custom_cat(self, module):
        def custom_forward_cat(*inputs):
            inputs = module(inputs[0],inputs[1])
            return inputs
        return custom_forward_cat

    


    def forward(self, x, mask, mask0):        
        #x1 = self.window_attention(x).permute(0,2,3,1)
        #x2 = self.grid_attention(x).permute(0,2,3,1)

        
        def concat_func_3rd(x1, x2):
            return torch.cat([x1, x2], dim=3)

        ## probably (b,c,x,y)
        #x1 = checkpoint.checkpoint(self.custom(self.local_attention), x, use_reentrant=False).permute(0,2,3,1)
        #print('MultiAxisAttention',torch.isnan(x).any(),torch.isinf(x).any())
        
        
        if self.flip:
            x1 = self.local_attention(torch.roll(x, shifts=(4, 4), dims=(2, 3)),torch.roll(mask, shifts=(4, 4), dims=(2, 3)))
            x1 = torch.roll(x1, shifts=(-4, -4), dims=(2, 3)).permute(0,2,3,1)

        else:
            x1 = self.local_attention(x,mask).permute(0,2,3,1)
        
        
        
        #x2 = checkpoint.checkpoint(self.custom(self.global_attention), x, use_reentrant=False).permute(0,2,3,1)
        x2 = self.grid_attention(x,mask0).permute(0,2,3,1)
        


        #x = torch.cat([x1,
        #               x2],
        #            dim=3)

        #x = checkpoint.checkpoint(self.custom_cat(self.cat), x1,x2, use_reentrant=False)
        x = checkpoint.checkpoint(concat_func_3rd, x2, x2, use_reentrant=False)
        


        #x = self.mixer(x)
        x = checkpoint.checkpoint(self.custom(self.mixer), x, use_reentrant=False)
        

        x = x.permute(0,3,1,2)     

        return x



class SequenceToPair(nn.Module):
    def __init__(self, sequence_state_dim, inner_dim, pairwise_state_dim, maxvit_depth=1):
        super().__init__()

        self.layernorm = nn.LayerNorm(sequence_state_dim)
        

        self.maxvit_depth = maxvit_depth
        
        self.proj = nn.Linear(sequence_state_dim, inner_dim, bias=True)
        self.o_proj = nn.Linear(inner_dim, pairwise_state_dim, bias=True)

        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)
        
        if maxvit_depth==1:

            self.proj_ = nn.Linear(sequence_state_dim, inner_dim, bias=True)
            self.o_proj_ = nn.Linear(inner_dim, pairwise_state_dim, bias=True)

            torch.nn.init.zeros_(self.proj_.bias)
            torch.nn.init.zeros_(self.o_proj_.bias)

            self.proj = nn.Linear(sequence_state_dim, inner_dim*2, bias=True)
            self.o_proj = nn.Linear(inner_dim*2, pairwise_state_dim, bias=True)

            torch.nn.init.zeros_(self.proj.bias)
            torch.nn.init.zeros_(self.o_proj.bias)

            
            
            self.sequence_to_pair_attn = MultiAxisAttention(inner_dim,activation_layer=nn.GELU,head_dim=8,mlp_ratio=4,partition_size=8,grid_size=(256,256))#,
             
        elif maxvit_depth==2:

            
            

            self.sequence_to_pair_attn_0 = MultiAxisAttention(inner_dim,activation_layer=nn.GELU,head_dim=8,mlp_ratio=4,partition_size=8,grid_size=(256,256))
            self.sequence_to_pair_attn_1 = MultiAxisAttention(inner_dim,activation_layer=nn.GELU,head_dim=8,mlp_ratio=4,partition_size=8,grid_size=(256,256),flip=True)
            #self.sequence_to_pair_attn_2 = MultiAxisAttention(inner_dim,activation_layer=nn.GELU,head_dim=8,mlp_ratio=4,partition_size=8,grid_size=(256,256))
            #self.sequence_to_pair_attn_3 = MultiAxisAttention(inner_dim,activation_layer=nn.GELU,head_dim=8,mlp_ratio=4,partition_size=8,grid_size=(256,256))


    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, sequence_state, valid_mask_0):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim

        Output:
          pairwise_state: B x L x L x pairwise_state_dim

        Intermediate state:
          B x L x L x 2*inner_dim
        """

        #assert len(sequence_state.shape) == 3
        
        '''valid_mask = []
        #for ii in range(len(valid_mask_0)):
        #    valid_mask.append(valid_mask_0[ii] * torch.unsqueeze(valid_mask_0[ii],-1))            

        #valid_mask = torch.stack(valid_mask)'''
        valid_mask = (valid_mask_0[:,None,:]*valid_mask_0[:,:,None])
                
        #s = self.layernorm(sequence_state)
        s = checkpoint.checkpoint(self.custom(self.layernorm), sequence_state, use_reentrant=False)        
        
        #s = self.proj(s)
        if self.maxvit_depth==1:
            #s = self.proj_(s)
            s = checkpoint.checkpoint(self.custom(self.proj_), s, use_reentrant=False)
        elif self.maxvit_depth==2:
            s = checkpoint.checkpoint(self.custom(self.proj), s, use_reentrant=False)
        s = s.transpose(1, 2)
        

        s2 = torch.flatten(s,end_dim=1)    
        s2 = s2.unsqueeze(-1)        
        s2 = torch.cdist(s2,s2)    
        s2 = s2.reshape(list(s.size())+[list(s.size())[-1]])        
        
        ###s2 = s2 * torch.unsqueeze(valid_mask, dim=1)
        ###s2 = s2.to(torch.float32)
        
        #s2 = s2.permute(0,2,3,1)
        #s2 = s2.permute(0,3,1,2)

        if self.maxvit_depth==1:
            s2 = s2 + self.sequence_to_pair_attn(s2,torch.unsqueeze(valid_mask, dim=1),valid_mask_0)

        elif self.maxvit_depth==2:
            s2 = s2 + self.sequence_to_pair_attn_0(s2,torch.unsqueeze(valid_mask, dim=1),valid_mask_0)
            s2 = s2 + self.sequence_to_pair_attn_1(s2,torch.unsqueeze(valid_mask, dim=1),valid_mask_0)
        #s2 = self.sequence_to_pair_attn_2(s2,torch.unsqueeze(valid_mask, dim=1),valid_mask_0)
        #s2 = self.sequence_to_pair_attn_3(s2,torch.unsqueeze(valid_mask, dim=1),valid_mask_0)
        
        s2 = s2.permute(0,2,3,1) 

        #x = self.o_proj(s2)
        if self.maxvit_depth==1:
            #x = self.o_proj_(s2)
            x = checkpoint.checkpoint(self.custom(self.o_proj_), s2, use_reentrant=False)
        elif self.maxvit_depth==2:
            x = checkpoint.checkpoint(self.custom(self.o_proj), s2, use_reentrant=False)
        
        return x



class TriangularSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        sequence_state_dim,
        pairwise_state_dim,
        sequence_head_width,
        pairwise_head_width,
        dropout=0,
        maxvit_depth=1,
        **__kwargs,
    ):
        super().__init__()

        #assert sequence_state_dim % sequence_head_width == 0
        #assert pairwise_state_dim % pairwise_head_width == 0
        sequence_num_heads = sequence_state_dim // sequence_head_width
        pairwise_num_heads = pairwise_state_dim // pairwise_head_width
        #assert sequence_state_dim == sequence_num_heads * sequence_head_width
        #assert pairwise_state_dim == pairwise_num_heads * pairwise_head_width
        #assert pairwise_state_dim % 2 == 0

        self.sequence_state_dim = sequence_state_dim
        self.pairwise_state_dim = pairwise_state_dim

        self.layernorm_1 = nn.LayerNorm(sequence_state_dim).float()

        self.cat = Concat3rdAxis()

        self.sequence_to_pair = SequenceToPair(
            sequence_state_dim, pairwise_state_dim // 2, pairwise_state_dim, maxvit_depth
        ).float()
                                    
        
        self.pair_to_sequence = PairToSequence(pairwise_state_dim, sequence_num_heads).float()

        self.seq_attention = Attention(
            sequence_state_dim, sequence_num_heads, sequence_head_width, gated=True
        ).float()
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pairwise_state_dim,
            pairwise_state_dim,
        ).float()
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pairwise_state_dim,
            pairwise_state_dim,
        ).float()
        

        self.tri_att_start = TriangleAttentionStartingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        ).float()  # type: ignore
        self.tri_att_end = TriangleAttentionEndingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        ).float()  # type: ignore


        self.mixer = nn.Sequential(
            LayerNorm(pairwise_state_dim*2),
            nn.Linear(pairwise_state_dim*2, pairwise_state_dim),
            nn.GELU(),
            nn.Linear(pairwise_state_dim,pairwise_state_dim),            
        ).float()

        self.mlp_seq = ResidueMLP(sequence_state_dim, 4 * sequence_state_dim, dropout=dropout).float()
        self.mlp_pair = ResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim, dropout=dropout).float()

        assert dropout < 0.4
        self.drop = nn.Dropout(dropout).float()
        self.row_drop = Dropout(dropout * 2, 2).float()
        self.col_drop = Dropout(dropout * 2, 1).float()

        torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.bias)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.bias)

        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)
        torch.nn.init.zeros_(self.pair_to_sequence.linear.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.bias)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-1].bias)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-1].bias)


    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
        

    def custom_msk(self, module):
        def custom_forward_msk(*inputs):
            inputs = module(inputs[0],mask=inputs[1])
            return inputs
        return custom_forward_msk
    
    def custom_msk_chunk(self, module):
        def custom_forward_msk_chunk(*inputs):
            inputs = module(inputs[0],mask=inputs[1],chunk_size=32)
            return inputs
        return custom_forward_msk_chunk

    def custom_msk_bias(self, module):
        def custom_forward_msk_bias(*inputs):
            inputs = module(inputs[0],mask=inputs[1],bias=inputs[2])
            return inputs
        return custom_forward_msk_bias

    def custom_cat(self, module):
        def custom_forward_cat(*inputs):
            inputs = module(inputs[0],inputs[1])
            return inputs
        return custom_forward_cat

    def forward(self, sequence_state, pairwise_state, mask=None, chunk_size=None, **__kwargs):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
          mask: B x L boolean tensor of valid positions

        Output:
          sequence_state: B x L x sequence_state_dim
          pairwise_state: B x L x L x pairwise_state_dim
        """
        #assert len(sequence_state.shape) == 3
        #assert len(pairwise_state.shape) == 4
        #if mask is not None:
        #    assert len(mask.shape) == 2

        def concat_func_3rd(x1, x2):
            return torch.cat([x1, x2], dim=3)

        #sequence_state = sequence_state.double()
        #pairwise_state = pairwise_state.double()

        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]
        #assert sequence_state_dim == self.sequence_state_dim
        #assert pairwise_state_dim == self.pairwise_state_dim
        #assert batch_dim == pairwise_state.shape[0]
        #assert seq_dim == pairwise_state.shape[1]
        #assert seq_dim == pairwise_state.shape[2]

        # Update sequence state        
        bias = checkpoint.checkpoint(self.custom(self.pair_to_sequence), pairwise_state, use_reentrant=False)
        # Self attention with bias + mlp.        
        y = checkpoint.checkpoint(self.custom(self.layernorm_1), sequence_state, use_reentrant=False)
        #y = self.seq_attention(y, mask=mask, bias=bias)
        y = checkpoint.checkpoint(self.custom_msk_bias(self.seq_attention), y,mask,bias, use_reentrant=False)
        
        sequence_state = sequence_state + self.drop(y)
        sequence_state = checkpoint.checkpoint(self.custom(self.mlp_seq), sequence_state, use_reentrant=False)

        # Update pairwise state        
        #pairwise_state = checkpoint.checkpoint(self.custom_cat(self.cat), pairwise_state,self.sequence_to_pair(sequence_state, mask), use_reentrant=False)
        #pairwise_state = checkpoint.checkpoint(concat_func_3rd, pairwise_state, self.sequence_to_pair(sequence_state, mask), use_reentrant=False)
        pairwise_state = pairwise_state + checkpoint.checkpoint(self.custom(self.mixer), 
                                               checkpoint.checkpoint(concat_func_3rd, pairwise_state, self.sequence_to_pair(sequence_state, mask),use_reentrant=False), use_reentrant=False)

        #pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state, mask)

        # Axial attention with triangular bias.
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None

        pairwise_state = pairwise_state + self.row_drop(
            checkpoint.checkpoint(self.custom_msk(self.tri_mul_out), pairwise_state.float(),tri_mask, use_reentrant=False)
        )

        pairwise_state = pairwise_state + self.col_drop(
            checkpoint.checkpoint(self.custom_msk(self.tri_mul_in), pairwise_state,tri_mask, use_reentrant=False)
        )
        
        pairwise_state = pairwise_state + self.row_drop(
            checkpoint.checkpoint(self.custom_msk_chunk(self.tri_att_start), pairwise_state,tri_mask, use_reentrant=False)
        )

        pairwise_state = pairwise_state + self.col_drop(
            checkpoint.checkpoint(self.custom_msk_chunk(self.tri_att_end), pairwise_state,tri_mask, use_reentrant=False)
        )

        # MLP over pairs.
        #pairwise_state = self.mlp_pair(pairwise_state)
        pairwise_state = checkpoint.checkpoint(self.custom(self.mlp_pair), pairwise_state, use_reentrant=False)

        return sequence_state, pairwise_state
    


class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)
          mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """

        assert residue_index.dtype == torch.long
        if mask is not None:
            assert residue_index.shape == mask.shape

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0
        #print('diff',torch.isnan(diff).any(),torch.isinf(diff).any())
        output = self.embedding(diff)
        #print('output',torch.isnan(output).any(),torch.isinf(output).any())
        return output


class Trunk(nn.Module):
    def __init__(self, cs,cz, num_layers,distogram_bins,first_training_pth):
        super().__init__()

        
        self.esm_feats = 1280 #512
        self.pair_feats = 512

        self.c_s = cs
        self.c_z = cz

        
        self.distogram_bins = distogram_bins

        self.first_stage = False



        if self.distogram_bins==4:
            self.first_stage = True
        
            self.esm_s_mlp = nn.Sequential(
                LayerNorm(self.esm_feats),
                nn.Linear(self.esm_feats, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
            ).float()

            #self.esm_s_mlp = LayerNorm(self.esm_feats)
                

            self.esm_z_mlp = nn.Sequential(
                LayerNorm(self.pair_feats),
                nn.Linear(self.pair_feats, self.c_z),
                nn.ReLU(),
                nn.Linear(self.c_z, self.c_z),
            ).float()

            self.position_bins = 128
            self.pairwise_positional_embedding = RelativePosition(self.position_bins, self.c_z).float()

        self.trunk = nn.ModuleList()

        for i in range(num_layers):

                self.trunk.append(
                    TriangularSelfAttentionBlock(
                        sequence_state_dim=self.c_s,
                        pairwise_state_dim=self.c_z,
                        sequence_head_width=32,
                        pairwise_head_width=32,
                        dropout=0.0,
                        maxvit_depth=2
                    ).float()
                )
                    
        
        self.recycle_bins = 30
        self.recycle_s_norm = nn.LayerNorm(self.c_s).float()
        self.recycle_z_norm = nn.LayerNorm(self.c_z).float()
        ###self.recycle_s_g_norm = nn.LayerNorm(self.c_s).float()
        ###self.recycle_z_g_norm = nn.LayerNorm(self.c_z).float()
        
        
        #self.recycle_s_emb = nn.Embedding(self.recycle_bins, self.c_s).float()
        #self.recycle_z_emb = nn.Embedding(self.recycle_bins, self.c_z).float()

        self.aa_emb = nn.Embedding(33, self.c_s).float()
        
        #self.pair_top = nn.Conv2d(self.c_z, self.distogram_bins, kernel_size=(1,1),padding='same').float()
        self.pair_top = nn.Linear(self.c_z, self.distogram_bins).float()


        self.cs_mpr = nn.Linear(self.c_s, 384).float()
        self.cz_mpr = nn.Linear(self.c_z, 128).float()
        

        if first_training_pth is not None:
            self.load_state_dict(torch.load(first_training_pth,map_location='cpu')['model'])

        

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward


    def custom_msk(self, module):
        def custom_forward_msk(*inputs):
            inputs = module(inputs[0], mask=inputs[1])
            return inputs
        return custom_forward_msk
    
    
    #repr
                
    # prediction
    def forward(
        self,
        seq_repr,
        pair_repr,
        aa_tokens,
        af2_aa_tokens,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 0,
        recycle_s_in = None,
        recycle_z_in = None,
        first_stage_tgl = False
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

        #residx = torch.arange(self.croplen, device=device)
        #print(seq_repr.dtype)
        #seq_repr = seq_repr.float()
        #pair_repr = pair_repr.float()
        #print(seq_repr.dtype)


        b,crp_len,_ = seq_repr.shape

        if self.first_stage:
            if first_stage_tgl:
                s_s_0 = seq_repr + self.aa_emb(aa_tokens)
                s_z_0 = pair_repr + self.pairwise_positional_embedding( torch.arange(crp_len,dtype=torch.long,device=aa_tokens.device).reshape(1,crp_len).repeat(len(mask),1) , mask=mask)
            else:
                s_s_0 = checkpoint.checkpoint(self.custom(self.esm_s_mlp), seq_repr, use_reentrant=False) + self.aa_emb(aa_tokens)
                s_z_0 = checkpoint.checkpoint(self.custom(self.esm_z_mlp), pair_repr, use_reentrant=False) + self.pairwise_positional_embedding( torch.arange(crp_len,dtype=torch.long,device=aa_tokens.device).reshape(1,crp_len).repeat(len(mask),1) , mask=mask)            
        else:
            s_s_0 = seq_repr
            s_z_0 = pair_repr


        if recycle_s_in is None:
            recycle_s_global = torch.zeros_like(s_s_0).detach()                
        else:
            recycle_s_global = recycle_s_in#.detach()
            
        if recycle_z_in is None:
            recycle_z_global = torch.zeros_like(s_z_0).detach()
        else:
            recycle_z_global = recycle_z_in#.detach()




        recycle_s = torch.zeros_like(s_s_0).detach()
        recycle_z = torch.zeros_like(s_z_0).detach()

        #print(s_s_0.shape, checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False).shape, recycle_s_global.shape)
        #print(s_z_0.shape, checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False).shape, recycle_z_global.shape)
        for i in range(len(self.trunk)):
            if i == 0:                
                s_s , s_z = self.trunk[i](
                        s_s_0 + checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False) + recycle_s_global,
                        s_z_0 + checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False) + recycle_z_global, 
                        mask)
            else:
                s_s , s_z = self.trunk[i](s_s, s_z, mask)



        for rind in range(num_recycles):

            recycle_s = s_s.detach() #+ self.recycle_s_emb(torch.Tensor([rind]).to(torch.long).to(s_s.device))
            recycle_z = s_z.detach() #+ self.recycle_z_emb(torch.Tensor([rind]).to(torch.long).to(s_z.device))

            for i in range(len(self.trunk)):
                if i == 0:
                    s_s , s_z = self.trunk[i](
                        s_s_0 + checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False) + recycle_s_global,
                        s_z_0 + checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False) + recycle_z_global, 
                        mask)
                else:
                    s_s , s_z = self.trunk[i](s_s, s_z, mask)



        #logits = self.seq_top(s_s.permute(0,2,1))        
        ###dist = checkpoint.checkpoint(self.custom(self.pair_top), s_z, use_reentrant=False)

        #logits = F.log_softmax(logits,dim=1)
        ###dist = F.log_softmax(dist,dim=-1)

        
        #dist_full = checkpoint.checkpoint(self.custom(self.full_dist_predict), s_z, use_reentrant=False)
        #dist_full = F.log_softmax(dist_full,dim=-1)

        
        #return logits, dist

        ###cs_feats = checkpoint.checkpoint(self.custom(self.cs_mpr), s_s, use_reentrant=False)
        ###cz_feats = checkpoint.checkpoint(self.custom(self.cz_mpr), s_z, use_reentrant=False)

        ###_,_,struc_out = self.frozen_structure_module(cs_feats,cz_feats,af2_aa_tokens,mask.to(torch.float32).to(cs_feats.device))

        ###return dist, struc_out
        return s_s, s_z

    
    def get_all_repr(    
        self,
        seq_repr,
        pair_repr,
        aa_tokens,
        af2_aa_tokens,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 0,
        recycle_s_in = None,
        recycle_z_in = None,
        first_stage_tgl = False
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

        #residx = torch.arange(self.croplen, device=device)
        #print(seq_repr.dtype)
        #seq_repr = seq_repr.float()
        #pair_repr = pair_repr.float()
        #print(seq_repr.dtype)


        b,crp_len,_ = seq_repr.shape

        s_s_out = []
        s_z_out = []

        if self.first_stage:
            if first_stage_tgl:
                s_s_0 = seq_repr + self.aa_emb(aa_tokens)
                s_z_0 = pair_repr + self.pairwise_positional_embedding( torch.arange(crp_len,dtype=torch.long,device=aa_tokens.device).reshape(1,crp_len).repeat(len(mask),1) , mask=mask)
            else:
                s_s_0 = checkpoint.checkpoint(self.custom(self.esm_s_mlp), seq_repr, use_reentrant=False) + self.aa_emb(aa_tokens)
                s_z_0 = checkpoint.checkpoint(self.custom(self.esm_z_mlp), pair_repr, use_reentrant=False) + self.pairwise_positional_embedding( torch.arange(crp_len,dtype=torch.long,device=aa_tokens.device).reshape(1,crp_len).repeat(len(mask),1) , mask=mask)            
        else:
            s_s_0 = seq_repr
            s_z_0 = pair_repr


        if recycle_s_in is None:
            recycle_s_global = torch.zeros_like(s_s_0).detach()                
        else:
            recycle_s_global = recycle_s_in.detach()
            
        if recycle_z_in is None:
            recycle_z_global = torch.zeros_like(s_z_0).detach()
        else:
            recycle_z_global = recycle_z_in.detach()




        recycle_s = torch.zeros_like(s_s_0).detach()
        recycle_z = torch.zeros_like(s_z_0).detach()

        #print(s_s_0.shape, checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False).shape, recycle_s_global.shape)
        #print(s_z_0.shape, checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False).shape, recycle_z_global.shape)
        for i in range(len(self.trunk)):
            if i == 0:                
                s_s , s_z = self.trunk[i](
                        s_s_0 + checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False) + recycle_s_global,
                        s_z_0 + checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False) + recycle_z_global, 
                        mask)

            else:
                s_s , s_z = self.trunk[i](s_s, s_z, mask)
            s_s_out.append(s_s)
            s_z_out.append(s_z)



        for rind in range(num_recycles):

            recycle_s = s_s.detach() #+ self.recycle_s_emb(torch.Tensor([rind]).to(torch.long).to(s_s.device))
            recycle_z = s_z.detach() #+ self.recycle_z_emb(torch.Tensor([rind]).to(torch.long).to(s_z.device))

            for i in range(len(self.trunk)):
                if i == 0:
                    s_s , s_z = self.trunk[i](
                        s_s_0 + checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False) + recycle_s_global,
                        s_z_0 + checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False) + recycle_z_global, 
                        mask)
                else:
                    s_s , s_z = self.trunk[i](s_s, s_z, mask)

                s_s_out.append(s_s)
                s_z_out.append(s_z)



        #logits = self.seq_top(s_s.permute(0,2,1))        
        ###dist = checkpoint.checkpoint(self.custom(self.pair_top), s_z, use_reentrant=False)

        #logits = F.log_softmax(logits,dim=1)
        ###dist = F.log_softmax(dist,dim=-1)

        
        #dist_full = checkpoint.checkpoint(self.custom(self.full_dist_predict), s_z, use_reentrant=False)
        #dist_full = F.log_softmax(dist_full,dim=-1)

        
        #return logits, dist

        ###cs_feats = checkpoint.checkpoint(self.custom(self.cs_mpr), s_s, use_reentrant=False)
        ###cz_feats = checkpoint.checkpoint(self.custom(self.cz_mpr), s_z, use_reentrant=False)

        ###_,_,struc_out = self.frozen_structure_module(cs_feats,cz_feats,af2_aa_tokens,mask.to(torch.float32).to(cs_feats.device))

        ###return dist, struc_out
        return s_s, s_z, s_s_out, s_z_out
        
    def inout_repr(
        self,
        seq_repr,
        aa_tokens,
        af2_aa_tokens,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 0,
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

        #residx = torch.arange(self.croplen, device=device)
        #print(seq_repr.dtype)
        #seq_repr = seq_repr.float()
        #pair_repr = pair_repr.float()
        #print(seq_repr.dtype)


        with torch.no_grad():
            pair_repr = self.saplm_top(seq_repr, mask)
        

        s_s_0 = self.esm_s_mlp(seq_repr) + self.aa_emb(aa_tokens)
        s_z_0 = self.esm_z_mlp(pair_repr)
        
        for i in range(len(self.trunk)):
            if i == 0:
                s_s , s_z = self.trunk[i](s_s_0, s_z_0, mask)
            else:
                s_s , s_z = self.trunk[i](s_s, s_z, mask)

        for rind in range(num_recycles):

            recycle_s = s_s.detach() #+ self.recycle_s_emb(torch.Tensor([rind]).to(torch.long).to(s_s.device))
            recycle_z = s_z.detach() #+ self.recycle_z_emb(torch.Tensor([rind]).to(torch.long).to(s_z.device))

            for i in range(len(self.trunk)):
                if i == 0:
                    s_s , s_z = self.trunk[i](self.recycle_s_norm(s_s_0 + recycle_s), self.recycle_z_norm(s_z_0 + recycle_z), mask)
                else:
                    s_s , s_z = self.trunk[i](s_s, s_z, mask)


        #logits = self.seq_top(s_s.permute(0,2,1))
        dist = self.pair_top(s_z.permute(0,3,1,2))

        #logits = F.log_softmax(logits,dim=1)
        dist = F.log_softmax(dist,dim=1)

        #logits = logits.permute(0,2,1)
        dist = dist.permute(0,2,3,1)

        #return logits, dist

        cs_feats = self.cs_mpr(s_s)
        cz_feats = self.cz_mpr(s_z)
        

        return s_z_0, s_z

    


    def pred_dist(
        self,
        seq_repr,
        pair_repr,
        aa_tokens,
        af2_aa_tokens,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 1,
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

        #residx = torch.arange(self.croplen, device=device)
        #print(seq_repr.dtype)
        seq_repr = seq_repr.float()
        pair_repr = pair_repr.float()
        #print(seq_repr.dtype)
        

        s_s_0 = self.esm_s_mlp(seq_repr) + self.aa_emb(aa_tokens)
        s_z_0 = pair_repr #self.esm_z_mlp(pair_repr)
        
        for i in range(len(self.trunk)):
            if i == 0:
                s_s , s_z = self.trunk[i](s_s_0, s_z_0, mask)
            else:
                s_s , s_z = self.trunk[i](s_s, s_z, mask)

        for rind in range(1,num_recycles):

            recycle_s = s_s.detach() + self.recycle_s_emb(torch.Tensor([rind]).to(torch.long).to(s_s.device))
            recycle_z = s_z.detach() + self.recycle_z_emb(torch.Tensor([rind]).to(torch.long).to(s_z.device))

            for i in range(len(self.trunk)):
                if i == 0:
                    s_s , s_z = self.trunk[i](self.recycle_s_norm(s_s_0 + recycle_s), self.recycle_z_norm(s_z_0 + recycle_z), mask)
                else:
                    s_s , s_z = self.trunk[i](s_s, s_z, mask)


        #logits = self.seq_top(s_s.permute(0,2,1))
        dist = self.pair_top(s_z.permute(0,3,1,2))

        #logits = F.log_softmax(logits,dim=1)
        dist = F.log_softmax(dist,dim=1)

        #logits = logits.permute(0,2,1)
        dist = dist.permute(0,2,3,1)

        #return logits, dist

        cs_feats = self.cs_mpr(s_s)
        cz_feats = self.cz_mpr(s_z)
        

        return dist

   

class Trunk2(nn.Module):
    def __init__(self, cs,cz, num_layers,distogram_bins,first_training_pth, last_cs=512,last_cz=192):
        super().__init__()

        
        self.esm_feats = 1280 #512
        self.pair_feats = 512

        self.last_cs = last_cs
        self.last_cz = last_cz

        self.c_s = cs
        self.c_z = cz

        
        self.distogram_bins = distogram_bins

        
        
        
        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, self.c_s),
            nn.ReLU(),
            nn.Linear(self.c_s, self.c_s),
        ).float()

        self.dist_s_mlp = nn.Sequential(
            LayerNorm(self.last_cs),
            nn.Linear(self.last_cs, self.c_s),
            nn.ReLU(),
            nn.Linear(self.c_s, self.c_s),
        ).float()

            
        self.esm_z_mlp = nn.Sequential(
            LayerNorm(self.pair_feats),
            nn.Linear(self.pair_feats, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
        ).float()

        self.dist_z_mlp = nn.Sequential(
            LayerNorm(self.last_cz),
            nn.Linear(self.last_cz, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
        ).float()


        self.position_bins = 128
        self.pairwise_positional_embedding = RelativePosition(self.position_bins, self.c_z).float()

        self.trunk = nn.ModuleList()

        for i in range(num_layers):

                self.trunk.append(
                    TriangularSelfAttentionBlock(
                        sequence_state_dim=self.c_s,
                        pairwise_state_dim=self.c_z,
                        sequence_head_width=32,
                        pairwise_head_width=32,
                        dropout=0.0,
                        maxvit_depth=2
                    ).float()
                )
                    
        
        self.recycle_bins = 30
        self.recycle_s_norm = nn.LayerNorm(self.c_s).float()
        self.recycle_z_norm = nn.LayerNorm(self.c_z).float()
        ###self.recycle_s_g_norm = nn.LayerNorm(self.c_s).float()
        ###self.recycle_z_g_norm = nn.LayerNorm(self.c_z).float()
        
        
        #self.recycle_s_emb = nn.Embedding(self.recycle_bins, self.c_s).float()
        #self.recycle_z_emb = nn.Embedding(self.recycle_bins, self.c_z).float()

        self.aa_emb = nn.Embedding(33, self.c_s).float()
        
        #self.pair_top = nn.Conv2d(self.c_z, self.distogram_bins, kernel_size=(1,1),padding='same').float()
        self.pair_top = nn.Linear(self.c_z, self.distogram_bins).float()


        self.cs_mpr = nn.Linear(self.c_s, 384).float()
        self.cz_mpr = nn.Linear(self.c_z, 128).float()
        

        if first_training_pth is not None:
            self.load_state_dict(torch.load(first_training_pth,map_location='cpu')['model'])

        

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward


    def custom_msk(self, module):
        def custom_forward_msk(*inputs):
            inputs = module(inputs[0], mask=inputs[1])
            return inputs
        return custom_forward_msk
    
    
    #repr
                
    # prediction
    def forward(
        self,
        plm_seq_repr,
        plm_pair_repr,
        dist_seq_repr,
        dist_pair_repr,
        aa_tokens,
        af2_aa_tokens,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 0,
        recycle_s_in = None,
        recycle_z_in = None,
        first_stage_tgl = False,
        plm_add = True
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

        #residx = torch.arange(self.croplen, device=device)
        #print(seq_repr.dtype)
        #seq_repr = seq_repr.float()
        #pair_repr = pair_repr.float()
        #print(seq_repr.dtype)


        b,crp_len,_ = plm_seq_repr.shape

        if plm_add:
            s_s_0 = checkpoint.checkpoint(self.custom(self.esm_s_mlp), plm_seq_repr, use_reentrant=False) + checkpoint.checkpoint(self.custom(self.dist_s_mlp), dist_seq_repr, use_reentrant=False) + self.aa_emb(aa_tokens)
            s_z_0 = checkpoint.checkpoint(self.custom(self.esm_z_mlp), plm_pair_repr, use_reentrant=False) + checkpoint.checkpoint(self.custom(self.dist_z_mlp), dist_pair_repr, use_reentrant=False) + self.pairwise_positional_embedding( torch.arange(crp_len,dtype=torch.long,device=aa_tokens.device).reshape(1,crp_len).repeat(len(mask),1) , mask=mask)            
        else:
            s_s_0 = checkpoint.checkpoint(self.custom(self.dist_s_mlp), dist_seq_repr, use_reentrant=False) + self.aa_emb(aa_tokens)
            s_z_0 = checkpoint.checkpoint(self.custom(self.dist_z_mlp), dist_pair_repr, use_reentrant=False) + self.pairwise_positional_embedding( torch.arange(crp_len,dtype=torch.long,device=aa_tokens.device).reshape(1,crp_len).repeat(len(mask),1) , mask=mask)            

        if recycle_s_in is None:
            recycle_s_global = torch.zeros_like(s_s_0).detach()                
        else:
            recycle_s_global = recycle_s_in.detach()
            
        if recycle_z_in is None:
            recycle_z_global = torch.zeros_like(s_z_0).detach()
        else:
            recycle_z_global = recycle_z_in.detach()




        recycle_s = torch.zeros_like(s_s_0).detach()
        recycle_z = torch.zeros_like(s_z_0).detach()

        
        for i in range(len(self.trunk)):
            if i == 0:                
                s_s , s_z = self.trunk[i](
                        s_s_0 + checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False) + recycle_s_global,
                        s_z_0 + checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False) + recycle_z_global, 
                        mask)
            else:
                s_s , s_z = self.trunk[i](s_s, s_z, mask)



        for rind in range(num_recycles):

            recycle_s = s_s.detach() #+ self.recycle_s_emb(torch.Tensor([rind]).to(torch.long).to(s_s.device))
            recycle_z = s_z.detach() #+ self.recycle_z_emb(torch.Tensor([rind]).to(torch.long).to(s_z.device))

            for i in range(len(self.trunk)):
                if i == 0:
                    s_s , s_z = self.trunk[i](
                        s_s_0 + checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False) + recycle_s_global,
                        s_z_0 + checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False) + recycle_z_global, 
                        mask)
                else:
                    s_s , s_z = self.trunk[i](s_s, s_z, mask)



        #logits = self.seq_top(s_s.permute(0,2,1))        
        ###dist = checkpoint.checkpoint(self.custom(self.pair_top), s_z, use_reentrant=False)

        #logits = F.log_softmax(logits,dim=1)
        ###dist = F.log_softmax(dist,dim=-1)

        
        #dist_full = checkpoint.checkpoint(self.custom(self.full_dist_predict), s_z, use_reentrant=False)
        #dist_full = F.log_softmax(dist_full,dim=-1)

        
        #return logits, dist

        ###cs_feats = checkpoint.checkpoint(self.custom(self.cs_mpr), s_s, use_reentrant=False)
        ###cz_feats = checkpoint.checkpoint(self.custom(self.cz_mpr), s_z, use_reentrant=False)

        ###_,_,struc_out = self.frozen_structure_module(cs_feats,cz_feats,af2_aa_tokens,mask.to(torch.float32).to(cs_feats.device))

        ###return dist, struc_out
        return s_s, s_z

    def get_all_repr(
        self,
        plm_seq_repr,
        plm_pair_repr,
        dist_seq_repr,
        dist_pair_repr,
        aa_tokens,
        af2_aa_tokens,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 0,
        recycle_s_in = None,
        recycle_z_in = None,
        first_stage_tgl = False,
        plm_add = True
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

        #residx = torch.arange(self.croplen, device=device)
        #print(seq_repr.dtype)
        #seq_repr = seq_repr.float()
        #pair_repr = pair_repr.float()
        #print(seq_repr.dtype)


        b,crp_len,_ = plm_seq_repr.shape
        s_s_out = [] 
        s_z_out = []

        if plm_add:
            s_s_0 = checkpoint.checkpoint(self.custom(self.esm_s_mlp), plm_seq_repr, use_reentrant=False) + checkpoint.checkpoint(self.custom(self.dist_s_mlp), dist_seq_repr, use_reentrant=False) + self.aa_emb(aa_tokens)
            s_z_0 = checkpoint.checkpoint(self.custom(self.esm_z_mlp), plm_pair_repr, use_reentrant=False) + checkpoint.checkpoint(self.custom(self.dist_z_mlp), dist_pair_repr, use_reentrant=False) + self.pairwise_positional_embedding( torch.arange(crp_len,dtype=torch.long,device=aa_tokens.device).reshape(1,crp_len).repeat(len(mask),1) , mask=mask)            
        else:
            s_s_0 = checkpoint.checkpoint(self.custom(self.dist_s_mlp), dist_seq_repr, use_reentrant=False) + self.aa_emb(aa_tokens)
            s_z_0 = checkpoint.checkpoint(self.custom(self.dist_z_mlp), dist_pair_repr, use_reentrant=False) + self.pairwise_positional_embedding( torch.arange(crp_len,dtype=torch.long,device=aa_tokens.device).reshape(1,crp_len).repeat(len(mask),1) , mask=mask)            

        if recycle_s_in is None:
            recycle_s_global = torch.zeros_like(s_s_0).detach()                
        else:
            recycle_s_global = recycle_s_in.detach()
            
        if recycle_z_in is None:
            recycle_z_global = torch.zeros_like(s_z_0).detach()
        else:
            recycle_z_global = recycle_z_in.detach()




        recycle_s = torch.zeros_like(s_s_0).detach()
        recycle_z = torch.zeros_like(s_z_0).detach()

        
        for i in range(len(self.trunk)):
            if i == 0:                
                s_s , s_z = self.trunk[i](
                        s_s_0 + checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False) + recycle_s_global,
                        s_z_0 + checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False) + recycle_z_global, 
                        mask)
            else:
                s_s , s_z = self.trunk[i](s_s, s_z, mask)

            s_s_out.append(s_s)
            s_z_out.append(s_z)

        for rind in range(num_recycles):

            recycle_s = s_s.detach() #+ self.recycle_s_emb(torch.Tensor([rind]).to(torch.long).to(s_s.device))
            recycle_z = s_z.detach() #+ self.recycle_z_emb(torch.Tensor([rind]).to(torch.long).to(s_z.device))

            for i in range(len(self.trunk)):
                if i == 0:
                    s_s , s_z = self.trunk[i](
                        s_s_0 + checkpoint.checkpoint(self.custom(self.recycle_s_norm), recycle_s, use_reentrant=False) + recycle_s_global,
                        s_z_0 + checkpoint.checkpoint(self.custom(self.recycle_z_norm), recycle_z, use_reentrant=False) + recycle_z_global, 
                        mask)
                else:
                    s_s , s_z = self.trunk[i](s_s, s_z, mask)
            
            s_s_out.append(s_s)
            s_z_out.append(s_z)


        #logits = self.seq_top(s_s.permute(0,2,1))        
        ###dist = checkpoint.checkpoint(self.custom(self.pair_top), s_z, use_reentrant=False)

        #logits = F.log_softmax(logits,dim=1)
        ###dist = F.log_softmax(dist,dim=-1)

        
        #dist_full = checkpoint.checkpoint(self.custom(self.full_dist_predict), s_z, use_reentrant=False)
        #dist_full = F.log_softmax(dist_full,dim=-1)

        
        #return logits, dist

        ###cs_feats = checkpoint.checkpoint(self.custom(self.cs_mpr), s_s, use_reentrant=False)
        ###cz_feats = checkpoint.checkpoint(self.custom(self.cz_mpr), s_z, use_reentrant=False)

        ###_,_,struc_out = self.frozen_structure_module(cs_feats,cz_feats,af2_aa_tokens,mask.to(torch.float32).to(cs_feats.device))

        ###return dist, struc_out
        return s_s, s_z, s_s_out, s_z_out

    
                
        
    def inout_repr(
        self,
        seq_repr,
        aa_tokens,
        af2_aa_tokens,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 0,
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

        #residx = torch.arange(self.croplen, device=device)
        #print(seq_repr.dtype)
        #seq_repr = seq_repr.float()
        #pair_repr = pair_repr.float()
        #print(seq_repr.dtype)


        with torch.no_grad():
            pair_repr = self.saplm_top(seq_repr, mask)
        

        s_s_0 = self.esm_s_mlp(seq_repr) + self.aa_emb(aa_tokens)
        s_z_0 = self.esm_z_mlp(pair_repr)
        
        for i in range(len(self.trunk)):
            if i == 0:
                s_s , s_z = self.trunk[i](s_s_0, s_z_0, mask)
            else:
                s_s , s_z = self.trunk[i](s_s, s_z, mask)

        for rind in range(num_recycles):

            recycle_s = s_s.detach() #+ self.recycle_s_emb(torch.Tensor([rind]).to(torch.long).to(s_s.device))
            recycle_z = s_z.detach() #+ self.recycle_z_emb(torch.Tensor([rind]).to(torch.long).to(s_z.device))

            for i in range(len(self.trunk)):
                if i == 0:
                    s_s , s_z = self.trunk[i](self.recycle_s_norm(s_s_0 + recycle_s), self.recycle_z_norm(s_z_0 + recycle_z), mask)
                else:
                    s_s , s_z = self.trunk[i](s_s, s_z, mask)


        #logits = self.seq_top(s_s.permute(0,2,1))
        dist = self.pair_top(s_z.permute(0,3,1,2))

        #logits = F.log_softmax(logits,dim=1)
        dist = F.log_softmax(dist,dim=1)

        #logits = logits.permute(0,2,1)
        dist = dist.permute(0,2,3,1)

        #return logits, dist

        cs_feats = self.cs_mpr(s_s)
        cz_feats = self.cz_mpr(s_z)
        

        return s_z_0, s_z

    


    def pred_dist(
        self,
        seq_repr,
        pair_repr,
        aa_tokens,
        af2_aa_tokens,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = 1,
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

        #residx = torch.arange(self.croplen, device=device)
        #print(seq_repr.dtype)
        seq_repr = seq_repr.float()
        pair_repr = pair_repr.float()
        #print(seq_repr.dtype)
        

        s_s_0 = self.esm_s_mlp(seq_repr) + self.aa_emb(aa_tokens)
        s_z_0 = pair_repr #self.esm_z_mlp(pair_repr)
        
        for i in range(len(self.trunk)):
            if i == 0:
                s_s , s_z = self.trunk[i](s_s_0, s_z_0, mask)
            else:
                s_s , s_z = self.trunk[i](s_s, s_z, mask)

        for rind in range(1,num_recycles):

            recycle_s = s_s.detach() + self.recycle_s_emb(torch.Tensor([rind]).to(torch.long).to(s_s.device))
            recycle_z = s_z.detach() + self.recycle_z_emb(torch.Tensor([rind]).to(torch.long).to(s_z.device))

            for i in range(len(self.trunk)):
                if i == 0:
                    s_s , s_z = self.trunk[i](self.recycle_s_norm(s_s_0 + recycle_s), self.recycle_z_norm(s_z_0 + recycle_z), mask)
                else:
                    s_s , s_z = self.trunk[i](s_s, s_z, mask)


        #logits = self.seq_top(s_s.permute(0,2,1))
        dist = self.pair_top(s_z.permute(0,3,1,2))

        #logits = F.log_softmax(logits,dim=1)
        dist = F.log_softmax(dist,dim=1)

        #logits = logits.permute(0,2,1)
        dist = dist.permute(0,2,3,1)

        #return logits, dist

        cs_feats = self.cs_mpr(s_s)
        cz_feats = self.cz_mpr(s_z)
        

        return dist

