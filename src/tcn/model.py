"""
Transformer pose estimator: 6 tracker poses -> 22 SMPL joint rotations -> joint positions.

A causal RoPE transformer encoder reads a window of tracker features and predicts one
6D rotation per joint and frame. Rotations are made orthonormal (Gram-Schmidt), chained
into global rotations, and passed through forward kinematics with the body's bone
offsets, so positions are a differentiable function of the predicted rotations.
"""

import torch
import torch.nn as nn

from .rotations import sixd_to_rotmat
from .skeleton import SMPL_NEUTRAL_OFFSETS, SMPL_PARENTS, fk_positions


# ─── Lightweight Skeletal FK ──────────────────────────────────────


class SkeletalFK(nn.Module):
    """
    Forward kinematics from global joint rotations (see skeleton.fk_positions).
    Holds the neutral SMPL skeleton as the default body; pass per-sample offsets
    to use a different one.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("bone_offsets", torch.tensor(SMPL_NEUTRAL_OFFSETS, dtype=torch.float32))

    def forward(self, global_rotmats, offsets=None):
        """
        global_rotmats: [..., 22, 3, 3]  (already chained)
        offsets: optional per-sample T-pose bone offsets [B, 22, 3];
                 defaults to the neutral SMPL skeleton.
        Returns: [..., 22, 3] global joint positions
        """
        if offsets is None:
            offsets = self.bone_offsets  # [22, 3]
        return fk_positions(global_rotmats, offsets.to(global_rotmats.dtype))


def local_to_global_rotmat(local_rotmats):
    """
    Chain local rotation matrices → global rotations.
    local_rotmats: [..., 22, 3, 3] → [..., 22, 3, 3]
    """
    global_rots = [local_rotmats[..., 0, :, :]]
    for i in range(1, 22):
        p = SMPL_PARENTS[i]
        g = torch.matmul(
            global_rots[p],
            local_rotmats[..., i, :, :],
        )
        global_rots.append(g)
    return torch.stack(global_rots, dim=-3)


# ─── Rotary Positional Encoding ──────────────────────────────────


class RotaryPositionalEncoding(nn.Module):
    """RoPE with pre-computed cos/sin buffers."""

    def __init__(self, dim, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # Pre-compute full cos/sin tables up to max_len
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len):
        """Slice pre-computed tables to seq_len."""
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply rotary embeddings to Q and K."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (
        q * cos + _rotate_half(q) * sin,
        k * cos + _rotate_half(k) * sin,
    )


# ─── Transformer Layer with RoPE ─────────────────────────────────


class RoPETransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
        B, W, D = x.shape
        H, Dh = self.nhead, self.head_dim

        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).reshape(B, W, H, Dh).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(B, W, H, Dh).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(B, W, H, Dh).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        scale = Dh**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        out = self.out_proj(out.transpose(1, 2).reshape(B, W, D))

        x = x + self.dropout(out)
        x = x + self.ff(self.norm2(x))
        return x


# ─── Main Model ──────────────────────────────────────────────────


class TransformerBodyPose(nn.Module):
    """
    Input:  [B, W, input_dim]  (6 trackers x pos3 + 6D rot + their velocities = 108)
    Output: local_rotmats [B,W,22,3,3], global_rotmats [B,W,22,3,3], fk_pos [B,W,22,3]
    """

    def __init__(
        self,
        input_dim=108,
        embed_dim=256,
        num_layers=3,
        num_heads=8,
        max_seq_len=512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        # Pre-computed RoPE tables (no per-forward allocation)
        self.rope = RotaryPositionalEncoding(
            embed_dim // num_heads, max_len=max_seq_len
        )

        # Pre-computed causal mask (no per-forward allocation)
        causal = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal)

        self.layers = nn.ModuleList(
            [
                RoPETransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_ff=embed_dim * 4,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # Predict 22 joints × 6D = 132
        self.rotation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 22 * 6),
        )

        # Lightweight FK solver (no SMPL!)
        self.fk = SkeletalFK()

    def forward(self, x, do_fk=True, bone_offsets=None):
        """
        x: [B, W, input_dim]
        bone_offsets: optional [B, 22, 3] per-sample T-pose bone offsets for
            FK (default: neutral SMPL skeleton).
        Returns:
            local_rotmats:  [B, W, 22, 3, 3]
            global_rotmats: [B, W, 22, 3, 3]
            fk_pos:         [B, W, 22, 3] or None
        """
        B, W, _ = x.shape

        h = self.input_proj(x)
        h = self.input_norm(h)

        # Slice pre-computed buffers (no GPU allocation)
        cos, sin = self.rope(W)
        mask = self.causal_mask[:W, :W]

        for layer in self.layers:
            h = layer(h, cos, sin, mask)

        h = self.final_norm(h)

        # 6D → rotation matrices (all W frames)
        raw_6d = self.rotation_head(h).reshape(B, W, 22, 6)
        local_rotmats = sixd_to_rotmat(raw_6d)

        # Single kinematic chain computation
        global_rotmats = local_to_global_rotmat(local_rotmats)

        # FK reuses global_rotmats (no redundant chain)
        fk_pos = None
        if do_fk:
            fk_pos = self.fk(global_rotmats, bone_offsets)

        return local_rotmats, global_rotmats, fk_pos
