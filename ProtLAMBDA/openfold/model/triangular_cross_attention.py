from functools import partialmethod
import torch
import torch.nn as nn
from .openfold.primitives import Linear, Attention
from .openfold.tensor_utils import (
    chunk_layer,
    permute_final_dims,
    flatten_final_dims,
)


class TriangleCrossAttention(nn.Module):
    """
    Cross-attention variant of OpenFold's TriangleAttention.

    Q comes from P (predicted pair repr, always complete).
    K and V come from T (template pair repr, may have missing entries).

    Two masks are accepted and combined additively as biases:
        pad_mask   : (*, I, J)  or (*, J)   1=valid, 0=padding
                     Blocks padded positions regardless of source.
        templ_mask : (*, I, J)              1=known in template, 0=missing
                     Blocks unknown template edges from contributing to P.

    Starting node (starting=True):
        Rows are fixed (i), attention is over columns (k).
        templ_mask[i, k] gates which T[i,k] edges are valid.
        No transpose needed.

    Ending node (starting=False):
        Before attention, x and masks are transposed so rows become j.
        After attention, output is transposed back.
        templ_mask[k, j] gates which T[k,j] edges are valid.
        templ_mask is transposed along with x: templ_mask.transpose(-1,-2)
        gives templ_mask[j, k], which then gates correctly in the
        transposed attention space.
    """

    def __init__(self, c_in, c_hidden, no_heads, starting, inf=1e9):
        """
        Args:
            c_in     : input channel dimension (same for P and T)
            c_hidden : overall hidden channel dimension (not per-head)
            no_heads : number of attention heads
            starting : True = starting node (Algorithm 13 style)
                       False = ending node  (Algorithm 14 style)
            inf      : large value used for masking (default 1e9)
        """
        super().__init__()
        self.c_in    = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf      = inf

        # Separate norms: queries from P, keys/values from T
        self.layer_norm_p = nn.LayerNorm(c_in)
        self.layer_norm_t = nn.LayerNorm(c_in)

        # Triangle bias is computed from T (the source of structural evidence)
        self.linear = Linear(c_in, no_heads, bias=False, init="normal")

        # Q from P, K and V from T
        self.mha = Attention(c_in, c_in, c_in, c_hidden, no_heads)

    def forward(
        self,
        p,
        t,
        templ_mask=None,
        pad_mask=None,
        chunk_size=128,
    ):
        """
        Args:
            p          : [*, I, J, c_in]  predicted pair repr   — queries
            t          : [*, I, J, c_in]  template pair repr    — keys/values
            templ_mask : [*, I, J]        1=known in template, 0=missing  [optional]
            pad_mask   : [*, I, J]        1=valid pair, 0=padding         [optional]
                         Pass pad_mask_to_pair(residue_pad_mask) from outside.
            chunk_size : chunk size for memory-efficient attention
        Returns:
            [*, I, J, c_in]  update to add to P
        """
        # Default masks to all-ones if not provided
        if templ_mask is None:
            templ_mask = p.new_ones(p.shape[:-1])                       # [*, I, J]
        if pad_mask is None:
            pad_mask = p.new_ones(p.shape[:-1])                         # [*, I, J]

        # For ending node: transpose I and J so attention runs over the right axis
        if not self.starting:
            p          = p.transpose(-2, -3)                            # [*, J, I, c_in]
            t          = t.transpose(-2, -3)
            templ_mask = templ_mask.transpose(-1, -2)                   # [*, J, I]
            pad_mask   = pad_mask.transpose(-1, -2)

        # Normalise queries (P) and keys/values (T) separately
        p_n = self.layer_norm_p(p)                                      # [*, I, J, c_in]
        t_n = self.layer_norm_t(t)                                      # [*, I, J, c_in]

        # --- Mask biases ---------------------------------------------------
        # templ_mask_bias: block unknown template edges
        # Shape: [*, I, 1, 1, J]  broadcasts over (I, heads, J_q, J_k)
        templ_mask_bias = (self.inf * (templ_mask - 1))[..., :, None, None, :]

        # pad_mask_bias: block padded positions
        # Shape: [*, I, 1, 1, J]  same layout
        pad_mask_bias = (self.inf * (pad_mask - 1))[..., :, None, None, :]

        # --- Triangle bias from T ------------------------------------------
        # [*, H, I, J] -> [*, 1, H, I, J]
        triangle_bias = permute_final_dims(self.linear(t_n), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)

        # --- Cross-attention -----------------------------------------------
        # Q from P, K and V from T
        # All three mask biases passed together — additive in logit space
        mha_inputs = {
            "q_x": p_n,
            "k_x": t_n,
            "v_x": t_n,
            "biases": [templ_mask_bias, pad_mask_bias, triangle_bias],
        }

        if chunk_size is not None:
            out = chunk_layer(
                self.mha,
                mha_inputs,
                chunk_size=chunk_size,
                no_batch_dims=len(p_n.shape[:-2]),
            )
        else:
            out = self.mha(**mha_inputs)

        # Transpose back for ending node
        if not self.starting:
            out = out.transpose(-2, -3)

        return out


class TriangleCrossAttentionStartingNode(TriangleCrossAttention):
    """Cross-attention, starting node (Q=P[i,j], K/V=T[i,k])."""
    __init__ = partialmethod(TriangleCrossAttention.__init__, starting=True)


class TriangleCrossAttentionEndingNode(TriangleCrossAttention):
    """Cross-attention, ending node (Q=P[i,j], K/V=T[k,j])."""
    __init__ = partialmethod(TriangleCrossAttention.__init__, starting=False)