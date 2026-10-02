"""Kinematic tree, neutral skeleton, forward kinematics and mirroring (all in the v2 frame:
x forward, y left, z up)."""

import numpy as np
import torch

# SMPL 22-joint kinematic tree (parent indices)
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]

# Neutral SMPL skeleton: child joint minus parent joint in the T-pose, in SMPL's native Y-up space
# (row 0 is the pelvis joint position). Taken from the SMPL neutral model.
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

# Y-up (SMPL native) -> v2 frame: new_x = old_z, new_y = old_x, new_z = old_y
YUP_TO_V2 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
SMPL_NEUTRAL_OFFSETS = np.asarray(SMPL_BONE_OFFSETS_22) @ YUP_TO_V2.T  # [22, 3] in the v2 frame

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


# Left/right partner of every joint (centre joints map to themselves)
MIRROR_PERM = list(range(22))
for _l, _r in SYMMETRIC_PAIRS:
    MIRROR_PERM[_l], MIRROR_PERM[_r] = _r, _l


def mirror_motion(gr, root_pos, offsets):
    """
    Left-right mirror of a motion in the v2 frame (x forward, y left, z up):
    reflect y and swap each left/right joint pair. A rotation R becomes S R S
    with S = diag(1, -1, 1). Left and right joint frames are both identity in the
    T-pose, so no per-joint frame fix is needed. Measured against BONES-SEED's own
    mirrored clips this reproduces them to 0.4 deg on average.

    gr [..., 22, 3, 3], root_pos [..., 3], offsets [..., 22, 3] -> same shapes.
    """
    flip = gr.new_tensor([1.0, -1.0, 1.0])
    gr_m = gr[..., MIRROR_PERM, :, :] * flip[:, None] * flip[None, :]
    return gr_m, root_pos * flip, offsets[..., MIRROR_PERM, :] * flip
