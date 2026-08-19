import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .openfold.model.triangular_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from .openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from .openfold.model.triangular_cross_attention import (
    TriangleCrossAttentionStartingNode,
    TriangleCrossAttentionEndingNode,
)
# =============================================================================
# Head reshape helpers
# =============================================================================

def to_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, L, L, H*D) -> (B, H, L, L, D)"""
    B, L1, L2, HD = x.shape
    D = HD // n_heads
    return x.view(B, L1, L2, n_heads, D).permute(0, 3, 1, 2, 4).contiguous()


def from_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, L, L, D) -> (B, L, L, H*D)"""
    B, H, L1, L2, D = x.shape
    return x.permute(0, 2, 3, 1, 4).contiguous().view(B, L1, L2, H * D)


def pad_mask_to_k_gate(pad_mask: torch.Tensor) -> torch.Tensor:
    """
    Expand residue-level pad_mask to block the k-dimension in attention.
    pad_mask : (B, L)  ->  gate : (B, 1, 1, 1, L)
    Broadcasted over (B, H, i, j, k) — blocks all positions along k that are padding.
    """
    return (1.0 - pad_mask).unsqueeze(1).unsqueeze(2).unsqueeze(3) * -1e9


def pad_mask_to_pair(pad_mask: torch.Tensor) -> torch.Tensor:
    """
    Expand residue-level pad_mask to pair-level.
    A pair (i,j) is valid only if BOTH i and j are valid residues.
    pad_mask : (B, L)  ->  pair_mask : (B, L, L)
    """
    return pad_mask.unsqueeze(2) * pad_mask.unsqueeze(1)


# =============================================================================
# Primitives
# =============================================================================

class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class Linear(nn.Module):
    """Linear with optional initialisation strategy."""
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, init: str = "default"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        if init == "zeros":
            nn.init.zeros_(self.linear.weight)
            if bias:
                nn.init.zeros_(self.linear.bias)
        elif init == "glorot":
            nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# =============================================================================
# Relative Position Encoding
# =============================================================================

class RelativePositionEncoding(nn.Module):
    """
    Sequence-separation encoding for residue pairs.
    Clipped to [-max_rel_pos, +max_rel_pos] and embedded as learned vectors.
    """
    def __init__(self, max_rel_pos: int = 32, d_out: int = 128):
        super().__init__()
        self.max_rel_pos = max_rel_pos
        self.embedding   = nn.Embedding(2 * max_rel_pos + 1, d_out)

    def forward(self, residue_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residue_index : (B, L)
        Returns:
            relpos        : (B, L, L, d_out)
        """
        diff = residue_index[:, :, None] - residue_index[:, None, :]
        diff = diff.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        return self.embedding(diff)


# =============================================================================
# Input Embeddings
# =============================================================================

class PredEmbedding(nn.Module):
    """
    Embeds the predicted distogram into pair representation P.
    P is always complete — no masking required here.
    """
    def __init__(self, n_bins: int, d_pair: int, max_rel_pos: int = 32):
        super().__init__()
        self.relpos      = RelativePositionEncoding(max_rel_pos, d_pair)
        self.mpr = nn.Embedding(n_bins, d_pair)
        self.proj        = nn.Sequential(
            LayerNorm(d_pair),
            Linear(d_pair, d_pair),
            nn.ReLU(),
            Linear(d_pair, d_pair),
        )
        self.relpos_proj = Linear(d_pair, d_pair)
        self.norm        = LayerNorm(d_pair)

    def forward(self, pred_distogram: torch.Tensor,
                residue_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_distogram : (B, L, L, n_bins)
            residue_index  : (B, L)
        Returns:
            P              : (B, L, L, d_pair)
        """
        #print(pred_distogram.shape)
        pred_distogram = self.mpr(pred_distogram)
        #print(pred_distogram.shape)
        P      = self.proj(pred_distogram)
        relpos = self.relpos(residue_index)
        return self.norm(P + self.relpos_proj(relpos))

class TemplEmbedding(nn.Module):
    def __init__(self, n_bins: int, d_pair: int, max_rel_pos: int = 32):
        super().__init__()
        self.mpr = nn.Embedding(n_bins, d_pair)
        self.relpos      = RelativePositionEncoding(max_rel_pos, d_pair)
        self.known_proj  = nn.Sequential(
            LayerNorm(d_pair),
            Linear(d_pair, d_pair),
            nn.ReLU(),
            Linear(d_pair, d_pair),
        )
        self.unknown_emb = nn.Parameter(torch.zeros(d_pair))
        self.relpos_proj = Linear(d_pair, d_pair)
        self.norm        = LayerNorm(d_pair)

    def forward(self, templ_distogram, templ_mask, residue_index):
        relpos      = self.relpos(residue_index)
        templ_distogram = self.mpr(templ_distogram)
        known_emb   = self.known_proj(templ_distogram)
        unknown_emb = self.unknown_emb.view(1, 1, 1, -1).expand_as(known_emb)

        gate = templ_mask.unsqueeze(-1).float()                          # hard binary switch
        T    = gate * known_emb + (1.0 - gate) * unknown_emb

        return self.norm(T + self.relpos_proj(relpos))

# =============================================================================
# Pair Transition MLP
# =============================================================================

class PairTransition(nn.Module):
    """Position-wise MLP, 4x expansion (AF2 default)."""
    def __init__(self, d_pair: int, expansion: int = 4):
        super().__init__()
        self.norm = LayerNorm(d_pair)
        self.ff   = nn.Sequential(
            Linear(d_pair, d_pair * expansion),
            nn.ReLU(),
            Linear(d_pair * expansion, d_pair, init="zeros"),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.ff(self.norm(pair))


# =============================================================================
# Fusion Block
# =============================================================================

class FusionBlock(nn.Module):
    """
    One full fusion block. Routing of the two masks:

    Step  Module                    pad_mask   templ_mask   Rationale
    ────  ──────────────────────    ────────   ──────────   ─────────────────────────────
    1-2   TriangleSelfAttention      YES        NO           P is complete; block padding
    3-4   TriangleCrossAttention     YES        YES          Block padding + unknown T edges
    5-6   TriangleMulUpdate          YES        YES          Block padding + unknown T edges
    7     PairTransition             —          —            No masking needed (MLP)

    Only P is updated. T is a fixed conditioning signal.
    """
    def __init__(self, d_pair: int, d_hidden: int = 128, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            d_pair,
            d_pair,
        ).float()
        self.tri_mul_in = TriangleMultiplicationIncoming(
            d_pair,
            d_pair,
        ).float()
        

        self.self_start = TriangleAttentionStartingNode(
            d_pair,
            d_pair // n_heads,
            n_heads,
            inf=1e9,
        ).float()  # type: ignore
        self.self_end = TriangleAttentionEndingNode(
            d_pair,
            d_pair // n_heads,
            n_heads,
            inf=1e9,
        ).float()  # type: ignore

        self.cross_start = TriangleCrossAttentionStartingNode(
            d_pair,
            d_pair // n_heads,
            n_heads,
            inf=1e9,
        ).float()  # type: ignore
        self.cross_end = TriangleCrossAttentionEndingNode(
            d_pair,
            d_pair // n_heads,
            n_heads,
            inf=1e9,
        ).float()  # type: ignore

        self.transition  = PairTransition(d_pair)
        self.drop        = nn.Dropout(dropout)

    def forward(self, P: torch.Tensor,
                T: torch.Tensor,
                templ_mask: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            P          : (B, L, L, d_pair)  predicted pair repr  [updated]
            T          : (B, L, L, d_pair)  template pair repr   [fixed]
            templ_mask : (B, L, L)          1=known in template
            pad_mask   : (B, L)             1=valid residue       [optional]
        Returns:
            P          : (B, L, L, d_pair)
        """
        # Self-attention: P builds context; block padded positions only

        pad_mask = pad_mask[:,:,None] * pad_mask[:,None,:]
        

        #print(P.shape, T.shape, templ_mask.shape, pad_mask.shape)

        P = P + self.drop(self.self_start(P, mask=pad_mask, chunk_size=None))
        P = P + self.drop(self.self_end(P, mask=pad_mask, chunk_size=None))

        # Cross-attention: P queries T; block padding AND unknown template edges
        P = P + self.drop(self.cross_start(P, T, templ_mask=templ_mask, pad_mask=pad_mask, chunk_size=None))
        P = P + self.drop(self.cross_end(P, T, templ_mask=templ_mask, pad_mask=pad_mask, chunk_size=None))

        # Multiplicative updates: geometric propagation; block padding AND unknown edges
        P = P + self.drop(self.tri_mul_out(P, mask=templ_mask * pad_mask))
        P = P + self.drop(self.tri_mul_in(P, mask=templ_mask * pad_mask))

        # Transition: no masking needed
        P = P + self.drop(self.transition(P))
        return P


# =============================================================================
# Output Head
# =============================================================================

class DistogramOutputHead(nn.Module):
    """
    Projects pair embeddings to distogram bin probabilities.
    Symmetry enforced by averaging (i,j) and (j,i) logits before softmax.
    """
    def __init__(self, d_pair: int, n_bins: int):
        super().__init__()
        self.norm = LayerNorm(d_pair)
        self.proj = Linear(d_pair, n_bins)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair  : (B, L, L, d_pair)
        Returns:
            probs : (B, L, L, n_bins)  symmetric,
        """
        logits = self.proj(self.norm(pair))
        logits = (logits + logits.transpose(1, 2)) * 0.5                # enforce symmetry
        return F.log_softmax(logits, dim=-1)


# =============================================================================
# Full Model
# =============================================================================

class DistogramFusionModel(nn.Module):
    """
    Template-guided distogram fusion using AF2-style triangular modules.

    P  <-  predicted distogram  (complete, lower confidence)
    T  <-  template distogram   (incomplete, high confidence)

    N FusionBlocks refine P via cross-attention to T.
    Both pad_mask (sequence padding) and templ_mask (missing template entries)
    are correctly routed to each sub-module.

    Args:
        n_bins      : distance bin count (AF2 default: 64)
        d_pair      : pair embedding dim  (default: 128)
        d_hidden    : hidden dim for multiplicative updates (default: 128)
        n_heads     : number of attention heads (default: 4)
        n_blocks    : number of FusionBlocks (default: 4)
        max_rel_pos : relative position clip range (default: 32)
        dropout     : dropout rate per block (default: 0.1)
    """
    def __init__(
        self,
        n_bins:      int   = 40,
        d_pair:      int   = 64,
        d_hidden:    int   = 64,
        n_heads:     int   = 4,
        n_blocks:    int   = 2,
        max_rel_pos: int   = 32,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.pred_embed  = PredEmbedding(n_bins, d_pair, max_rel_pos)
        self.templ_embed = TemplEmbedding(n_bins, d_pair, max_rel_pos)
        self.blocks      = nn.ModuleList([
            FusionBlock(d_pair, d_hidden, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.output_head = DistogramOutputHead(d_pair, n_bins)

    def forward(
        self,
        pred_distogram:  torch.Tensor,
        templ_distogram: torch.Tensor,
        templ_mask:      torch.Tensor,
        residue_index:   torch.Tensor,
        pad_mask:        Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_distogram  : (B, L, L, n_bins)  complete predicted distogram
            templ_distogram : (B, L, L, n_bins)  template (0 where missing)
            templ_mask      : (B, L, L)           1=known in template, 0=missing
            residue_index   : (B, L)              integer residue indices
            pad_mask        : (B, L)              1=valid residue, 0=padding  [optional]

        Returns:
            fused_distogram : (B, L, L, n_bins)  complete fused (softmax probs)
        """
        P = self.pred_embed(pred_distogram, residue_index)               # (B,L,L,d_pair)
        T = self.templ_embed(templ_distogram, templ_mask, residue_index) # (B,L,L,d_pair)

        for block in self.blocks:
            P = block(P, T, templ_mask, pad_mask)

        return self.output_head(P)


# =============================================================================
# Loss
# =============================================================================

class DistogramFusionLoss(nn.Module):
    """
    Composite training loss.

    L_impute : cross-entropy at missing positions  (primary)
    L_anchor : cross-entropy at known positions    (don't drift from template)
    L_sym    : symmetry regularisation
    L_tri    : soft triangle inequality penalty    (optional, O(L^3))

    pad_mask is used to exclude padded positions from all loss terms.
    """
    def __init__(
        self,
        lambda_anchor: float = 0.1,
        lambda_sym:    float = 0.01,
        lambda_tri:    float = 0.0,
        bin_edges:     Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lambda_anchor = lambda_anchor
        self.lambda_sym    = lambda_sym
        self.lambda_tri    = lambda_tri
        if bin_edges is not None:
            self.register_buffer("bin_edges", bin_edges)
        else:
            self.bin_edges = None

    def forward(
        self,
        pred:       torch.Tensor,
        target:     torch.Tensor,
        templ_mask: torch.Tensor,
        pad_mask:   Optional[torch.Tensor] = None,
    ):
        """
        Args:
            pred       : (B, L, L, n_bins)  model output (softmax probs)
            target     : (B, L, L, n_bins)  ground truth (softmax probs)
            templ_mask : (B, L, L)          1=known in template, 0=missing
            pad_mask   : (B, L)             1=valid residue               [optional]
        Returns:
            total_loss, dict of component losses
        """
        # Valid pair mask: exclude padding from all loss terms
        if pad_mask is not None:
            valid = pad_mask_to_pair(pad_mask)                           # (B, L, L)
        else:
            valid = torch.ones_like(templ_mask)

        log_pred = torch.log(pred.clamp(min=1e-8))
        ce       = -(target * log_pred).sum(dim=-1)                      # (B, L, L)

        missing = (1.0 - templ_mask) * valid
        known   = templ_mask * valid

        n_miss  = missing.sum().clamp(min=1)
        n_known = known.sum().clamp(min=1)

        loss_impute = (ce * missing).sum() / n_miss
        loss_anchor = (ce * known).sum()   / n_known

        # Symmetry loss over valid pairs only
        sym_err  = (pred - pred.transpose(1, 2)).pow(2).sum(dim=-1)     # (B, L, L)
        loss_sym = (sym_err * valid).sum() / valid.sum().clamp(min=1)

        total = (loss_impute
                 + self.lambda_anchor * loss_anchor
                 + self.lambda_sym    * loss_sym)

        loss_tri = torch.tensor(0.0, device=pred.device)
        if self.lambda_tri > 0 and self.bin_edges is not None:
            d    = (pred * self.bin_edges.to(pred.device)).sum(-1)
            viol = F.relu(d.unsqueeze(3) - d.unsqueeze(2) - d.unsqueeze(1))
            loss_tri = viol.mean()
            total    = total + self.lambda_tri * loss_tri

        return total, {
            "loss_impute": loss_impute.item(),
            "loss_anchor": loss_anchor.item(),
            "loss_sym":    loss_sym.item(),
            "loss_tri":    loss_tri.item(),
        }


# =============================================================================
# Training utilities
# =============================================================================

def make_synthetic_mask(B: int, L: int, missing_frac: float = 0.3,
                        device: str = "cpu") -> torch.Tensor:
    """
    Simulate AF2 template missing patterns:
      - Contiguous blocks (loops, disordered regions, termini)
      - Scattered single residues
    Returns pair mask (B, L, L): 1 iff BOTH residues i and j are known.
    """
    residue_known = torch.ones(B, L, device=device)
    for b in range(B):
        n_blocks = torch.randint(1, 4, (1,)).item()
        for _ in range(n_blocks):
            start  = torch.randint(0, L, (1,)).item()
            length = torch.randint(3, max(4, int(L * missing_frac)), (1,)).item()
            residue_known[b, start:min(start + length, L)] = 0.0
        scatter = torch.rand(L, device=device) < (missing_frac * 0.3)
        residue_known[b] *= (~scatter).float()
    return residue_known.unsqueeze(2) * residue_known.unsqueeze(1)


def make_synthetic_mask(residue_mask: torch.Tensor,
                        missing_frac: float = 0.3) -> torch.Tensor:
    """
    Apply synthetic AF2-style missing patterns to an existing residue mask.

    Takes a 1D boolean residue mask (1=present, 0=absent) and additionally
    drops residues via:
      - Contiguous blocks  (simulate loops, disordered regions, termini)
      - Scattered singles  (simulate sparse no-hit residues)

    Residues already absent in the input mask are preserved as absent.

    Args:
        residue_mask : (L,) or (B, L)  bool or float, 1=present, 0=absent
        missing_frac : fraction of residues to additionally drop (default 0.3)

    Returns:
        pair_mask    : (L, L) or (B, L, L)  float, 1 iff BOTH residues known
    """
    # Handle both (L,) and (B, L) inputs
    unbatched = residue_mask.dim() == 1
    if unbatched:
        residue_mask = residue_mask.unsqueeze(0)                        # (1, L)

    B, L      = residue_mask.shape
    device    = residue_mask.device
    known     = residue_mask.float().clone()                            # (B, L)

    for b in range(B):
        # Contiguous blocks
        n_blocks = torch.randint(1, 5, (1,)).item()
        for _ in range(n_blocks):
            start  = torch.randint(0, L, (1,)).item()
            length = torch.randint(3, max(4, int(L * missing_frac)), (1,)).item()
            known[b, start:min(start + length, L)] = 0.0

        # Scattered singles
        scatter = torch.rand(L, device=device) < (missing_frac * 0.3)
        known[b] *= (~scatter).float()

    # Combine with original mask: a residue is known only if it was
    # present in the input AND not dropped by the synthetic pattern
    known = known * residue_mask.float()

    # Expand to pair mask: (B, L, L) — both residues must be known
    pair_mask = known.unsqueeze(2) * known.unsqueeze(1)

    return pair_mask.squeeze(0) if unbatched else pair_mask


def make_pad_mask(B: int, L: int, lengths: Optional[list] = None,
                  device: str = "cpu") -> torch.Tensor:
    """
    Create a padding mask from sequence lengths.
    If lengths not provided, simulates random padding (last 10-20% of L).

    Returns pad_mask (B, L): 1=valid residue, 0=padding.
    """
    if lengths is None:
        lengths = [torch.randint(int(L * 0.8), L + 1, (1,)).item() for _ in range(B)]
    mask = torch.zeros(B, L, device=device)
    for b, length in enumerate(lengths):
        mask[b, :length] = 1.0
    return mask


def make_bin_edges(n_bins: int = 64, d_min: float = 2.0,
                   d_max: float = 22.0) -> torch.Tensor:
    """AF2-style evenly-spaced bin centres."""
    return torch.linspace(d_min, d_max, n_bins)


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == "__main__":
    B, L, n_bins = 2, 64, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = DistogramFusionModel(
        n_bins=n_bins, d_pair=128, d_hidden=128,
        n_heads=4, n_blocks=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Inputs ────────────────────────────────────────────────────────────────
    pred_distogram   = F.softmax(torch.randn(B, L, L, n_bins, device=device), dim=-1)
    target_distogram = F.softmax(torch.randn(B, L, L, n_bins, device=device), dim=-1)

    templ_mask       = make_synthetic_mask(B, L, missing_frac=0.3, device=str(device))
    templ_distogram  = target_distogram * templ_mask.unsqueeze(-1)

    # Simulate variable-length sequences: seq 0 has length 60, seq 1 has length 55
    pad_mask      = make_pad_mask(B, L, lengths=[60, 55], device=str(device))
    residue_index = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    # ── Forward (with pad_mask) ───────────────────────────────────────────────
    fused = model(pred_distogram, templ_distogram, templ_mask, residue_index, pad_mask)
    print(f"\nWith pad_mask:")
    print(f"  Output shape : {fused.shape}")
    print(f"  Sums to 1    : {fused.sum(-1).mean():.4f}  (expect 1.0)")
    sym_err = (fused - fused.transpose(1, 2)).abs().max().item()
    print(f"  Symmetry err : {sym_err:.6f}  (expect ~0)")

    # ── Forward (without pad_mask) ────────────────────────────────────────────
    fused_no_pad = model(pred_distogram, templ_distogram, templ_mask, residue_index)
    print(f"\nWithout pad_mask:")
    print(f"  Output shape : {fused_no_pad.shape}")
    print(f"  Sums to 1    : {fused_no_pad.sum(-1).mean():.4f}  (expect 1.0)")

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = DistogramFusionLoss(
        lambda_anchor=0.1, lambda_sym=0.01,
        lambda_tri=0.001, bin_edges=make_bin_edges(n_bins).to(device),
    )

    print(f"\nLoss with pad_mask:")
    total_loss, loss_dict = criterion(fused, target_distogram, templ_mask, pad_mask)
    for k, v in loss_dict.items():
        print(f"  {k:15s}: {v:.4f}")
    print(f"  {'total':15s}: {total_loss.item():.4f}")

    print(f"\nLoss without pad_mask:")
    total_loss2, loss_dict2 = criterion(fused_no_pad, target_distogram, templ_mask)
    for k, v in loss_dict2.items():
        print(f"  {k:15s}: {v:.4f}")
    print(f"  {'total':15s}: {total_loss2.item():.4f}")

    # ── Backward ──────────────────────────────────────────────────────────────
    total_loss.backward()
    print("\nBackward pass : OK")