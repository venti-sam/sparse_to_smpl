"""Rotation helpers shared by the model, the store and the GPU sample builder."""

import torch
import torch.nn as nn


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


def extract_yaw(rotmat, eps=1e-6):
    """
    Extract Z-axis yaw rotation matrix from (..., 3, 3) rotmat.
    Uses projected forward axis on XY plane for robustness.
    Returns (..., 3, 3).
    """
    fwd = rotmat[..., :, 0]          # forward axis = column 0
    fwd_xy = fwd[..., :2]            # drop Z
    n = torch.linalg.norm(fwd_xy, dim=-1, keepdim=True).clamp_min(eps)
    fwd_xy = fwd_xy / n
    yaw = torch.atan2(fwd_xy[..., 1], fwd_xy[..., 0])
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    return torch.stack(
        [
            cos_y,
            -sin_y,
            zeros,
            sin_y,
            cos_y,
            zeros,
            zeros,
            zeros,
            ones,
        ],
        dim=-1,
    ).reshape(*yaw.shape, 3, 3)


def axis_angle_to_rotmat(axis_angle, eps=1e-8):
    """
    Rodrigues formula.
    axis_angle: [..., 3] -> [..., 3, 3]
    """
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(eps)
    axis = axis_angle / theta

    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=-1,
    ).reshape(*axis.shape[:-1], 3, 3)

    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    while eye.dim() < K.dim():
        eye = eye.unsqueeze(0)

    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    return eye + sin_t * K + (1.0 - cos_t) * torch.matmul(K, K)
