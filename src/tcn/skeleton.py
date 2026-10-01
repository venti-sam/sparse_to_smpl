"""Kinematic tree and functional forward kinematics shared by dataset and model."""

import torch

# SMPL 22-joint kinematic tree (parent indices)
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]

# Left/right bone pairs (child joint indices) that share a length
SYMMETRIC_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)]


def fk_positions(global_rotmats, offsets):
    """
    Global joint positions from global rotations and T-pose bone offsets.

    global_rotmats: [..., 22, 3, 3]
    offsets:        [22, 3] shared, or [B, 22, 3] per sample when
                    global_rotmats is [B, W, 22, 3, 3]
    Returns:        [..., 22, 3], root placed at offsets[..., 0, :]
    """
    batch_shape = global_rotmats.shape[:-3]
    if offsets.dim() == 3:
        offsets = offsets.unsqueeze(-3)  # [B, 1, 22, 3] broadcasts over W

    global_pos = []
    for i, p in enumerate(SMPL_PARENTS):
        if p == -1:
            g_pos = offsets[..., i, :].expand(*batch_shape, 3)
        else:
            step = global_rotmats[..., p, :, :] @ offsets[..., i, :].unsqueeze(-1)
            g_pos = global_pos[p] + step.squeeze(-1)
        global_pos.append(g_pos)
    return torch.stack(global_pos, dim=-2)
