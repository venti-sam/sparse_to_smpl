"""
Transformer-based full-body pose estimator (v3).

Key changes from v2:
- Lightweight SkeletalFK replaces full SMPL model
  → No axis-angle conversion (eliminates 180° singularity)
  → ~100x less VRAM (pure matmul + add, no mesh skinning)
  → Fully differentiable by construction
- 6D rotation output (continuous, Gram-Schmidt)
- Many-to-many with causal RoPE Transformer
"""

import torch
import torch.nn as nn

from .skeleton import SMPL_PARENTS, fk_positions


# ─── T-pose bone offsets (extracted from SMPL neutral) ────────────
# These are constant — child_joint - parent_joint in T-pose (Y-up).
# The dataset converts to Z-up, but bone offsets are in SMPL's
# native space since FK operates in local frames.

SMPL_BONE_OFFSETS_22 = [
    [-0.001795, -0.223333, 0.028219],  # 0  Pelvis (root)
    [0.069520, -0.091406, -0.006815],  # 1  L_Hip
    [-0.067670, -0.090522, -0.004320],  # 2  R_Hip
    [-0.002533, 0.108963, -0.026696],  # 3  Spine1
    [0.034277, -0.375199, -0.004496],  # 4  L_Knee
    [-0.038290, -0.382569, -0.008850],  # 5  R_Knee
    [0.005487, 0.135180, 0.001092],  # 6  Spine2
    [-0.013596, -0.397960, -0.043693],  # 7  L_Ankle
    [0.015774, -0.398415, -0.042312],  # 8  R_Ankle
    [0.001457, 0.052922, 0.025425],  # 9  Spine3
    [0.026358, -0.055791, 0.119288],  # 10 L_Foot
    [-0.025372, -0.048144, 0.123348],  # 11 R_Foot
    [-0.002778, 0.213870, -0.042857],  # 12 Neck
    [0.078845, 0.121749, -0.034090],  # 13 L_Collar
    [-0.081759, 0.118833, -0.038615],  # 14 R_Collar
    [0.005152, 0.064970, 0.051349],  # 15 Head
    [0.090977, 0.030469, -0.008868],  # 16 L_Shoulder
    [-0.096012, 0.032551, -0.009143],  # 17 R_Shoulder
    [0.259612, -0.012772, -0.027456],  # 18 L_Elbow
    [-0.253742, -0.013329, -0.021401],  # 19 R_Elbow
    [0.249234, 0.008986, -0.001171],  # 20 L_Wrist
    [-0.255298, 0.007772, -0.005559],  # 21 R_Wrist
]


# ─── 6D Rotation Utilities ────────────────────────────────────────


def sixd_to_rotmat(sixd):
    """
    6D rotation → 3×3 rotation matrix via Gram-Schmidt.
    sixd: [..., 6] → [..., 3, 3]
    """
    a1, a2 = sixd[..., :3], sixd[..., 3:]
    b1 = nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = nn.functional.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotmat_to_sixd(rotmat):
    """[...,3,3] → [...,6] first two columns."""
    return torch.cat([rotmat[..., :, 0], rotmat[..., :, 1]], dim=-1)


# ─── Lightweight Skeletal FK ──────────────────────────────────────


class SkeletalFK(nn.Module):
    """
    Pure PyTorch skeletal forward kinematics.

    Computes global joint positions from the kinematic chain.
    Accepts pre-computed global_rotmats to avoid redundant
    chain computation when global rotations are already known.

    Memory: O(22 * B) vs SMPL's O(6890_vertices * B)
    """

    def __init__(self):
        super().__init__()
        # 1. Base offsets in SMPL Y-up space
        offsets_yup = torch.tensor(SMPL_BONE_OFFSETS_22, dtype=torch.float32)

        # 2. Convert to project Z-up convention
        # H maps Y-up to Z-up: X->Y, Y->Z, Z->X
        H = torch.tensor([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=torch.float32)
        offsets_zup = torch.einsum("ij,nj->ni", H, offsets_yup)

        self.register_buffer("bone_offsets", offsets_zup)

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
    Transformer encoder for VR full-body pose estimation (v3).

    Input:  [B, W, input_dim]
    Output: local_rotmats [B,W,22,3,3], global_rotmats, fk_pos

    Uses SkeletalFK instead of SMPL — no axis-angle conversion,
    no mesh skinning, ~100x less VRAM, fully differentiable.
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
