import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn import LayerNorm as ESM1bLayerNorm
from einops import rearrange
from .esm import MultiheadAttention, ESM2, ESM1bLayerNorm


class AxialAttention(nn.Module):
    """
    Axial attention module for 2D inputs.
    """
    def __init__(self, embed_dim, attention_heads):
        """
        Initialize the AxialAttention module.

        Args:
            embed_dim (int): The dimension of the embedding.
            attention_heads (int): The number of attention heads.
        """
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

    def forward(self, x, pad_msk, chunk_size=None):
        """
        Forward pass for the AxialAttention module.

        Args:
            x (torch.Tensor): The input tensor.
            pad_msk (torch.Tensor): The padding mask.
            chunk_size (int): The size of the chunks to process.

        Returns:
            torch.Tensor: The output tensor.
        """
        
        ## ESM MHA requires (tokens,batch,embed)

        if chunk_size is None:

            b,e,l,w = x.shape

            x_r = rearrange(x, 'b e x y -> x (b y) e')
            x_c = rearrange(x, 'b e x y -> y (b x) e')

            if pad_msk is not None:            
                pad_msk = torch.unsqueeze(pad_msk,dim=1)
                pad_msk = torch.flatten(pad_msk.repeat(1,w,1),end_dim=1)    
            
            x_r = self.row_attn_layer_norm(x_r)
            x_c = self.col_attn_layer_norm(x_c)
            
            x_r,_ = self.row_attn(
                query=x_r,
                key=x_r,
                value=x_r,
                key_padding_mask=pad_msk,
                need_weights=False,
                need_head_weights=False,
                attn_mask=None
            )

            x_c,_ = self.col_attn(
                query=x_c,
                key=x_c,
                value=x_c,
                key_padding_mask=pad_msk,
                need_weights=False,
                need_head_weights=False,
                attn_mask=None
            )

            
            x_r = rearrange(x_r, 'x (b y) e -> b e x y',b=b)
            x_c = rearrange(x_c, 'y (b x) e -> b e x y',b=b)
            
            return x_r + x_c
        
        else:

            b,e,l,w = x.shape

            x_r = rearrange(x, 'b e x y -> x (b y) e')
            x_c = rearrange(x, 'b e x y -> y (b x) e')

            x_r = self.row_attn_layer_norm(x_r)
            x_c = self.col_attn_layer_norm(x_c)
            
            if pad_msk is not None:            
                pad_msk = torch.unsqueeze(pad_msk,dim=1)
                pad_msk = torch.flatten(pad_msk.repeat(1,w,1),end_dim=1)
            
            
            for ii in range(0,x_r.shape[1],chunk_size):

                x_r_ii = x_r[:,ii:min(ii+chunk_size,x_r.shape[1]),:]
                x_c_ii = x_c[:,ii:min(ii+chunk_size,x_c.shape[1]),:]
                
                #if pad_msk is not None:                            
                pad_msk_ii = pad_msk[ii:min(ii+chunk_size,x_r.shape[1]),:]
                
                x_r_ii,_ = self.row_attn(
                    query=x_r_ii,
                    key=x_r_ii,
                    value=x_r_ii,
                    key_padding_mask=pad_msk_ii,
                    need_weights=False,
                    need_head_weights=False,
                    attn_mask=None
                )

                x_c_ii,_ = self.col_attn(
                    query=x_c_ii,
                    key=x_c_ii,
                    value=x_c_ii,
                    key_padding_mask=pad_msk_ii,
                    need_weights=False,
                    need_head_weights=False,
                    attn_mask=None
                )

                if(ii==0):
                    x_r_out = x_r_ii
                    x_c_out = x_c_ii
                else:
                    x_r_out = torch.cat((x_r_out,x_r_ii),dim=1)
                    x_c_out = torch.cat((x_c_out,x_c_ii),dim=1)
            
            x_r = rearrange(x_r_out, 'x (b y) e -> b e x y',b=b)
            x_c = rearrange(x_c_out, 'y (b x) e -> b e x y',b=b)
            
            return x_r + x_c

        

class ProtLAMBDA(nn.Module):
    """
    ProtLAMBDA model.    
    """
    
    def __init__(
        self,
        embed_dim: int = 1280,
        cm_embed_dim = 512,
        frozen_backbone_lyrs: int=22,
        cm_dropout: float = 0.05,        
        ):
        """
        Initialize the ProtLAMBDA model.

        Args:            
            embed_dim (int): The dimension of the embedding.
            cm_embed_dim (int): The dimension of the contact map embedding.
            frozen_backbone_lyrs (int): The number of frozen backbone layers.
            cm_dropout (float): The dropout rate for the contact map.
        """

        super().__init__()

        self.esm2_backbone = ESM2()
        self.frozen_backbone = frozen_backbone_lyrs

        for layer_num in range(self.frozen_backbone):
            for param in self.esm2_backbone.layers[layer_num].parameters():
                param.requires_grad = False


        self.layers = nn.ModuleDict({})        
        self.layers['cm_drpout'] = nn.Dropout(cm_dropout)        
        self.layers['cm_mapper'] = nn.Linear(embed_dim,cm_embed_dim)        
        self.layers['cm_attn'] = AxialAttention(
                                                    embed_dim = cm_embed_dim,       # embedding dimension
                                                    attention_heads = 8,            # number of heads for multi-head attention                                                            
                                                )
        self.layers['cm_layer'] = nn.Conv2d(cm_embed_dim, 1, kernel_size=(1,1),padding='same')
        self.layers['cm_layer_act'] = nn.Sigmoid()
        
        

        

    def forward(self, seqs, masking_ratio=1.00, chunk_size=None):
        """
        Forward pass for the ProtLAMBDA model.

        Args:
            seqs (list): A list of sequences.
            masking_ratio (float): The ratio of tokens to mask.
            chunk_size (int): The size of the chunks to process.
        """
        
        device = next(self.parameters()).device
                
        tokens = self.esm2_backbone.batch_converter(seqs)        
        tokens = tokens.to(device)

        padding_mask = tokens.eq(1) + tokens.eq(2)
        valid_mask_0 =  (tokens>2)*1  #1 - (padding_mask * 1.0)
        rndm_msk = (( (torch.rand(tokens.shape, device=valid_mask_0.device) * valid_mask_0)>masking_ratio ) *1)
        
        tokens = ((1-rndm_msk) * tokens) + (rndm_msk * self.esm2_backbone.mask_idx)


        x = self.esm2_backbone.embed_scale * self.esm2_backbone.embed_tokens(tokens)        
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None


        with torch.no_grad():
            for layer_idx in range(self.frozen_backbone):

                x, _ = self.esm2_backbone.layers[layer_idx](
                            x,
                            self_attn_padding_mask=padding_mask,
                            need_head_weights=False,
                        )


            for layer_idx in range(self.frozen_backbone,len(self.esm2_backbone.layers)):

                x, _ = self.esm2_backbone.layers[layer_idx](
                            x,
                            self_attn_padding_mask=padding_mask,
                            need_head_weights=False,
                        )
            
            dist_feats = x

            x = self.esm2_backbone.emb_layer_norm_after(x)

            x = x.transpose(0, 1) 
            
            x2 = self.layers['cm_mapper'](dist_feats) 
            x2 = rearrange(x2, 't b e -> b e t')
            
                        
            x3 = torch.flatten(x2,end_dim=1)
            x3 = x3.unsqueeze(-1)
            x3 = torch.cdist(x3,x3)
            x3 = x3.reshape(list(x2.size())+[list(x2.size())[-1]])


            cm_in = self.layers['cm_attn'](x3,padding_mask, chunk_size)

            cm_in = self.layers['cm_drpout'](cm_in)

        cm_bin = self.layers['cm_layer'](cm_in)
        
        logits = self.esm2_backbone.lm_head(x)
        logits  = F.log_softmax(logits,dim=-1)
            
        return logits,cm_bin



    def get_repr(self, seqs, masking_ratio=1.00, chunk_size=None):
        """
        Get the sequence and pairwise representations from the ProtLAMBDA model.

        Args:
            seqs (list): A list of sequences.
            masking_ratio (float): The ratio of tokens to mask.
            chunk_size (int): The size of the chunks to process.
        """
        device = next(self.parameters()).device
        
        tokens = self.esm2_backbone.batch_converter(seqs)        
        tokens = tokens.to(device)


        padding_mask = tokens.eq(1) + tokens.eq(2)
        valid_mask_0 =  (tokens>2)*1  #1 - (padding_mask * 1.0)

        rndm_msk = (( (torch.rand(tokens.shape, device=valid_mask_0.device) * valid_mask_0)>masking_ratio ) *1)
        
        tokens = ((1-rndm_msk) * tokens) + (rndm_msk * self.esm2_backbone.mask_idx)

        x = self.esm2_backbone.embed_scale * self.esm2_backbone.embed_tokens(tokens)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None


        with torch.no_grad():
            for layer_idx in range(self.frozen_backbone):
                x, _ = self.esm2_backbone.layers[layer_idx](
                            x,
                            self_attn_padding_mask=padding_mask,
                            need_head_weights=False,
                        )

        for layer_idx in range(self.frozen_backbone,len(self.esm2_backbone.layers)):
            x, _ = self.esm2_backbone.layers[layer_idx](
                        x,
                        self_attn_padding_mask=padding_mask,
                        need_head_weights=False,
                    )
        
        dist_feats = x

    
        x2 = self.layers['cm_mapper'](dist_feats) # (T, B, E)
        x2 = rearrange(x2, 't b e -> b e t')
                
        x3 = torch.flatten(x2,end_dim=1)
        x3 = x3.unsqueeze(-1)
        x3 = torch.cdist(x3,x3)
        x3 = x3.reshape(list(x2.size())+[list(x2.size())[-1]])

        cm_in = self.layers['cm_attn'](x3,padding_mask, chunk_size)

        cm_in = self.layers['cm_drpout'](cm_in)

        seq_repr = dist_feats.permute(1,0,2)       # T,B,E -> B,T,E
        pair_repr = cm_in.permute(0,2,3,1)       # B,E,T,T -> B,T,T,E

        ## NOTE THAT the reprs contain bos and eos tokens, so the user should remove them if needed.
        
        return seq_repr,pair_repr



    def get_seq_repr(self, seqs, masking_ratio=1.00, allow_norm=False):
        """
        Get the sequence representation from the ProtLAMBDA model.

        Args:
            seqs (list): A list of sequences.
            masking_ratio (float): The ratio of tokens to mask.
            allow_norm (bool): Whether to allow normalization of the output representation.
        """
        device = next(self.parameters()).device
                
        tokens = self.esm2_backbone.batch_converter(seqs)        
        tokens = tokens.to(device)
        
        padding_mask = tokens.eq(1) + tokens.eq(2)
        valid_mask_0 =  (tokens>2)*1  #1 - (padding_mask * 1.0)
        
        rndm_msk = (( (torch.rand(tokens.shape, device=valid_mask_0.device) * valid_mask_0)>masking_ratio ) *1)
        
        tokens = ((1-rndm_msk) * tokens) + (rndm_msk * self.esm2_backbone.mask_idx)
        x = self.esm2_backbone.embed_scale * self.esm2_backbone.embed_tokens(tokens)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None


        with torch.no_grad():
            for layer_idx in range(self.frozen_backbone):
                x, _ = self.esm2_backbone.layers[layer_idx](
                            x,
                            self_attn_padding_mask=padding_mask,
                            need_head_weights=False,
                        )

        for layer_idx in range(self.frozen_backbone,len(self.esm2_backbone.layers)):

            x, _ = self.esm2_backbone.layers[layer_idx](
                        x,
                        self_attn_padding_mask=padding_mask,
                        need_head_weights=False,
                    )
        
        dist_feats = x

        if allow_norm:
            dist_feats = self.esm2_backbone.emb_layer_norm_after(dist_feats) 

        seq_repr = dist_feats.permute(1,0,2)       # T,B,E -> B,T,E        
        
        return seq_repr

    def get_all(self, seqs, masking_ratio=1.00, chunk_size=None):
        
        """
        Get all the outputs from the ProtLAMBDA model.

        Args:
            seqs (list): A list of sequences.
            masking_ratio (float): The ratio of tokens to mask.
            chunk_size (int): The size of the chunks to process.
        """
        device = next(self.parameters()).device                
        tokens = self.esm2_backbone.batch_converter(seqs)        
        tokens = tokens.to(device)

        padding_mask = tokens.eq(1) + tokens.eq(2)
        valid_mask_0 =  (tokens>2)*1  #1 - (padding_mask * 1.0)

        
        rndm_msk = (( (torch.rand(tokens.shape, device=valid_mask_0.device) * valid_mask_0)>masking_ratio ) *1)
        
        tokens = ((1-rndm_msk) * tokens) + (rndm_msk * self.esm2_backbone.mask_idx)

        x = self.esm2_backbone.embed_scale * self.esm2_backbone.embed_tokens(tokens)
        
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None


        with torch.no_grad():
            for layer_idx in range(self.frozen_backbone):

                x, _ = self.esm2_backbone.layers[layer_idx](
                            x,
                            self_attn_padding_mask=padding_mask,
                            need_head_weights=False,
                        )


        for layer_idx in range(self.frozen_backbone,len(self.esm2_backbone.layers)):

            x, _ = self.esm2_backbone.layers[layer_idx](
                        x,
                        self_attn_padding_mask=padding_mask,
                        need_head_weights=False,
                    )

        
        dist_feats = x

        x = self.esm2_backbone.emb_layer_norm_after(x)

        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
        
        x2 = self.layers['cm_mapper'](dist_feats) # (T, B, E)
        x2 = rearrange(x2, 't b e -> b e t')
        
        logits = self.esm2_backbone.lm_head(x)
        logits  = F.log_softmax(logits,dim=-1)
        
        x3 = torch.flatten(x2,end_dim=1)
        x3 = x3.unsqueeze(-1)
        x3 = torch.cdist(x3,x3)
        x3 = x3.reshape(list(x2.size())+[list(x2.size())[-1]])


        cm_in = self.layers['cm_attn'](x3,padding_mask, chunk_size)

        cm_in = self.layers['cm_drpout'](cm_in)

        

        cm_bin = self.layers['cm_layer'](cm_in)
        seq_repr = dist_feats.permute(1,0,2)       # T,B,E -> B,T,E
        pair_repr = cm_in.permute(0,2,3,1)       # B,E,T,T -> B,T,T,E

        
        
        return seq_repr,pair_repr,logits,cm_bin
