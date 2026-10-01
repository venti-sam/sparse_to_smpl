"""
VR Teleop Dataset v3 — reads the v2 on-disk format (convert_dataset_v2.py).

Only gt_rotmat + root_pos + per-actor bone offsets are stored. Tracker inputs
and target positions are derived per window with FK, so body size can be
randomised consistently in inputs and targets.

v2.2 behaviour (unchanged):
Fixes from v2.1:
- Heading canonicalization is per-WINDOW (not per-sequence), so the
  model always sees heading-invariant input regardless of where in
  the sequence the window starts
- Uses extended window (start-1) for velocity so frame 0 of every
  window has a real velocity, not zero
- First-frame velocity padded with vel[1] (constant-velocity extrap)
"""

import os
import glob
import math
import torch
from torch.utils.data import Dataset, DataLoader

from .skeleton import SMPL_PARENTS, SYMMETRIC_PAIRS, fk_positions  # noqa: F401

# Pelvis = index 0; tracker joints in SMPL ordering
TRACKER_JOINTS = [0, 7, 8, 15, 20, 21]  # pelvis, L/R ankle, head, L/R wrist

# Where each Vive tracker rigidly sits: (kind, host joint, anchor joint).
# The tracker follows the host joint's frame and is offset from the anchor
# joint. Ankle trackers are strapped to the shin just above the ankle, so
# they follow the knee joint's frame (the shin), not the foot's.
TRACKER_MOUNTS = [
    ("pelvis", 0, 0),
    ("ankle", 4, 7),
    ("ankle", 5, 8),
    ("head", 15, 15),
    ("hand", 20, 20),
    ("hand", 21, 21),
]

# Reference tracker for root-centering and heading.
# Pelvis = index 0 in our tracker ordering.
REF_TRACKER_IDX = 0


def quat_to_rotmat(quat):
    """Quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = (
        quat[..., 0],
        quat[..., 1],
        quat[..., 2],
        quat[..., 3],
    )
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z

    return torch.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


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


def compute_velocity(tensor, fps=60.0):
    """
    Finite difference velocity with first-frame padding.
    tensor: [T, ...] → [T, ...]
    """
    vel = torch.zeros_like(tensor)
    vel[1:] = (tensor[1:] - tensor[:-1]) * fps
    if tensor.shape[0] > 1:
        vel[0] = vel[1]  # constant-velocity extrapolation
    return vel


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

    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    while I.dim() < K.dim():
        I = I.unsqueeze(0)

    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    return I + sin_t * K + (1.0 - cos_t) * torch.matmul(K, K)


class VRTeleopDataset(Dataset):
    """
    Sliding window dataset v2.2.

    Key design: canonicalization and velocity are computed per-window
    in __getitem__, not pre-computed per-sequence. This ensures:
    - Heading invariance at every window position
    - Proper velocity context via extended window (start-1)
    """

    def __init__(
        self, dataset_dir, window_size=40, stride=1, split="train", train_ratio=0.9,
        folders=None, augmentation=None
    ):
        super().__init__()
        self.window_size = window_size
        self.split = split

        aug_cfg = augmentation or {}
        self.aug_enabled = split == "train" and aug_cfg.get("enabled", False)
        self.aug_pos_jitter_std = float(aug_cfg.get("pos_jitter_std", 0.0))
        self.aug_rot_jitter_deg = float(aug_cfg.get("rot_jitter_deg", 0.0))
        self.aug_pos_bias_std = float(aug_cfg.get("pos_bias_std", 0.0))
        self.aug_rot_bias_deg = float(aug_cfg.get("rot_bias_deg", 0.0))
        self.aug_vel_noise_std = float(aug_cfg.get("vel_noise_std", 0.0))
        self.aug_dropout_prob = float(aug_cfg.get("dropout_prob", 0.0))
        # Body-size randomisation: global scale ~ U(1-r, 1+r), plus per-bone
        # lognormal jitter (left/right bones share a value).
        self.aug_scale_global = float(aug_cfg.get("bone_scale_global", 0.0))
        self.aug_scale_limb_std = float(aug_cfg.get("bone_scale_limb_std", 0.0))
        lo, hi = aug_cfg.get("dropout_frames", [5, 30])
        self.aug_dropout_frames = (int(lo), int(hi))
        # Vive mounting: sampled per window in train, nominal (means) otherwise
        self.mount = self._parse_mount(aug_cfg.get("mount"))

        # 1. Get all seq_*.pt files
        all_pt_files = sorted(glob.glob(os.path.join(dataset_dir, "*.pt")))
        if not all_pt_files:
            raise FileNotFoundError(f"No .pt files in {dataset_dir}")

        # 2. Filter by folders if specified
        if folders is not None:
            pt_files = []
            for f in all_pt_files:
                basename = os.path.basename(f)
                # Check if file matches any folder prefix (e.g. CMU_seq_*)
                if any(basename.startswith(folder + "_") for folder in folders):
                    pt_files.append(f)
                # Fallback for old files without prefix if we want to allow mixed?
                # For now, strict folder matching if folders are provided.
        else:
            # 3. Fallback to ratio-based split on all files
            n = len(all_pt_files)
            split_idx = int(n * train_ratio)
            pt_files = all_pt_files[:split_idx] if split == "train" else all_pt_files[split_idx:]

        print(f"[{split}] Loading {len(pt_files)} sequences (folders={folders})...")

        self.windows = []
        self.sequences = []

        for seq_idx, f in enumerate(pt_files):
            data = torch.load(f, weights_only=False)

            if "bone_offsets" not in data.get("meta", {}):
                raise ValueError(
                    f"{f} is not in the v2 format (no meta['bone_offsets']). "
                    "Run src/convert_dataset_v2.py and point data.dataset_dir at its output."
                )
            gt_rotmat = data["gt_rotmat"].float()

            T = gt_rotmat.shape[0]
            if T < window_size + 1:
                continue

            # Store raw data — FK, canonicalization and velocities per-window
            self.sequences.append(
                {
                    "gt_rotmat": gt_rotmat,
                    "root_pos": data["root_pos"].float(),
                    "bone_offsets": data["meta"]["bone_offsets"].float(),
                }
            )
            
            # Use actual index in self.sequences, NOT seq_idx from enumerate
            curr_idx = len(self.sequences) - 1

            num_windows = max(0, T - window_size) // stride + 1
            for w in range(num_windows):
                self.windows.append((curr_idx, w * stride))

        print(
            f"[{split}] {len(self.sequences)} sequences, "
            f"{len(self.windows)} windows"
        )

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def _parse_mount(cfg):
        """YAML mount block -> tensors per tracker kind (None = exact joint copies)."""
        if not cfg:
            return None
        deg = math.pi / 180.0
        mount = {}
        for kind in {k for k, _, _ in TRACKER_MOUNTS}:
            if kind not in cfg:
                raise KeyError(f"augmentation.mount is missing '{kind}'")
            c = cfg[kind]
            mount[kind] = {
                "pos_mean": torch.tensor(c.get("pos_mean", [0.0] * 3), dtype=torch.float32),
                "pos_std": torch.tensor(c.get("pos_std", [0.0] * 3), dtype=torch.float32),
                "rot_std": torch.tensor(c.get("rot_std_deg", [0.0] * 3), dtype=torch.float32) * deg,
                "along_forearm": float(c.get("along_forearm", 0.0)),
            }
        return mount

    def _mount_trackers(self, gp, gr, offsets):
        """
        Fake Vive trackers from the body: each is rigidly attached to its host
        joint's frame at a mounting offset/rotation (x forward, y left, z up).
        Train: offset and rotation are sampled once per window around the
        configured mean. Otherwise: the mean mount, no randomness.
        gp: [T, 22, 3], gr: [T, 22, 3, 3], offsets: [22, 3] (body used for FK)
        Returns tracker pos [T, 6, 3], rot [T, 6, 3, 3].
        """
        pos, rot = [], []
        for kind, host, anchor in TRACKER_MOUNTS:
            c = self.mount[kind]
            t = c["pos_mean"].clone()
            if kind == "hand":
                # controller held like a stick: along the forearm, toward the fingers
                t = t + torch.nn.functional.normalize(offsets[anchor], dim=0) * c["along_forearm"]
            R_mount = torch.eye(3)
            if self.aug_enabled:
                t = t + torch.randn(3) * c["pos_std"]
                R_mount = axis_angle_to_rotmat(torch.randn(3) * c["rot_std"])
            pos.append(gp[:, anchor] + torch.einsum("tij,j->ti", gr[:, host], t))
            rot.append(gr[:, host] @ R_mount)
        return torch.stack(pos, dim=1), torch.stack(rot, dim=1)

    def _apply_augmentation(self, tp, tr):
        """
        Apply train-time input augmentation:
        1. Persistent Bias (offset per window)
        2. Per-frame Jitter
        3. Sensor Dropout (short freeze of one non-pelvis tracker)
        tp: [T, 6, 3], tr: [T, 6, 3, 3]
        """
        if not self.aug_enabled:
            return tp, tr

        # 1. Persistent Sensor Bias (offset for the whole window)
        if self.aug_pos_bias_std > 0:
            bias = torch.randn(1, tp.shape[1], 3, dtype=tp.dtype, device=tp.device) * self.aug_pos_bias_std
            tp = tp + bias

        if self.aug_rot_bias_deg > 0:
            std_rad = self.aug_rot_bias_deg * (math.pi / 180.0)
            bias_aa = torch.randn(1, tr.shape[1], 3, dtype=tr.dtype, device=tr.device) * std_rad
            bias_R = axis_angle_to_rotmat(bias_aa)
            tr = torch.matmul(tr, bias_R)

        # 2. Per-frame Jitter
        if self.aug_pos_jitter_std > 0:
            tp = tp + torch.randn_like(tp) * self.aug_pos_jitter_std

        if self.aug_rot_jitter_deg > 0:
            std_rad = self.aug_rot_jitter_deg * (math.pi / 180.0)
            delta_aa = torch.randn(
                *tr.shape[:-2], 3, dtype=tr.dtype, device=tr.device
            ) * std_rad
            delta_R = axis_angle_to_rotmat(delta_aa)
            tr = torch.matmul(tr, delta_R)

        # 3. Sensor Dropout: one non-root tracker freezes at its last good
        # value for a short burst, then jumps back to the true pose
        if self.aug_dropout_prob > 0 and torch.rand(1).item() < self.aug_dropout_prob:
            T = tp.shape[0]
            drop_idx = int(torch.randint(low=1, high=tr.shape[1], size=(1,)).item())
            lo, hi = self.aug_dropout_frames
            dur = int(torch.randint(lo, hi + 1, (1,)).item())
            s = int(torch.randint(1, max(2, T - dur + 1), (1,)).item())
            tp[s : s + dur, drop_idx] = tp[s - 1, drop_idx]
            tr[s : s + dur, drop_idx] = tr[s - 1, drop_idx]

        return tp, tr

    def _sample_bone_scale(self):
        """Per-bone multiplicative scale [22, 1]; left/right bones are equal."""
        scale = torch.ones(22, 1)
        if self.aug_scale_limb_std > 0:
            scale = torch.exp(torch.randn(22, 1) * self.aug_scale_limb_std)
            for left, right in SYMMETRIC_PAIRS:
                scale[right] = scale[left]
        if self.aug_scale_global > 0:
            g = 1.0 + (torch.rand(1).item() * 2 - 1) * self.aug_scale_global
            scale = scale * g
        return scale

    def __getitem__(self, idx):
        seq_idx, start = self.windows[idx]
        seq = self.sequences[seq_idx]
        W = self.window_size
        end = start + W

        # ── Extended window for velocity (1 extra leading frame) ──
        ext_start = max(0, start - 1)
        canon_frame = start - ext_start  # 0 if start==0, else 1

        gr = seq["gt_rotmat"][ext_start:end].clone()
        root_pos = seq["root_pos"][ext_start:end]

        # ── Body: per-actor bone offsets, optionally randomised in size ──
        # Positions are derived from rotations, so inputs and targets
        # always describe the same (scaled) body.
        offsets = seq["bone_offsets"]
        scale = torch.ones(22, 1)
        if self.aug_enabled:
            scale = self._sample_bone_scale()
            offsets = offsets * scale
        gp = fk_positions(gr, offsets)
        gp = gp - gp[:, 0:1] + root_pos.unsqueeze(1) * scale[0]
        if self.mount is None:
            tp = gp[:, TRACKER_JOINTS].clone()
            tr = gr[:, TRACKER_JOINTS].clone()
        else:
            tp, tr = self._mount_trackers(gp, gr, offsets)

        # ── Train-time augmentation on INPUT trackers only ──
        tp, tr = self._apply_augmentation(tp, tr)

        # ── Per-window heading canonicalization ──
        # Uses TRACKER pelvis at the window's first frame
        ref_rot = tr[canon_frame, REF_TRACKER_IDX]
        R_yaw_inv = extract_yaw(ref_rot.unsqueeze(0)).transpose(-1, -2)[0]  # [3, 3]

        tp = torch.einsum("ij,tnj->tni", R_yaw_inv, tp)
        gp = torch.einsum("ij,tnj->tni", R_yaw_inv, gp)
        tr = torch.einsum("ij,tnjk->tnik", R_yaw_inv, tr)
        gr = torch.einsum("ij,tnjk->tnik", R_yaw_inv, gr)

        # ── Root-relative (tracker pelvis, not GT) ──
        t_root = tp[:, REF_TRACKER_IDX : REF_TRACKER_IDX + 1]
        tp_rel = tp - t_root

        # GT labels: relative to GT pelvis (correct for supervision)
        gp_rel = gp - gp[:, 0:1, :]

        # ── 6D rotation features ──
        tr_sixd = rotmat_to_sixd(tr)

        # ── Velocity on extended window (proper context) ──
        tp_vel = compute_velocity(tp_rel)
        tr_sixd_vel = compute_velocity(tr_sixd)

        if self.aug_enabled and self.aug_vel_noise_std > 0:
            tp_vel = tp_vel + torch.randn_like(tp_vel) * self.aug_vel_noise_std
            tr_sixd_vel = tr_sixd_vel + torch.randn_like(tr_sixd_vel) * self.aug_vel_noise_std

        # ── Trim to actual window ──
        tp_rel = tp_rel[canon_frame : canon_frame + W]
        tr_sixd = tr_sixd[canon_frame : canon_frame + W]
        tp_vel = tp_vel[canon_frame : canon_frame + W]
        tr_sixd_vel = tr_sixd_vel[canon_frame : canon_frame + W]
        gp_rel = gp_rel[canon_frame : canon_frame + W]
        gr = gr[canon_frame : canon_frame + W]
        t_root = t_root[canon_frame : canon_frame + W]

        # ── Flatten input ──
        # per tracker: pos(3)+6d(6)+vel_pos(3)+vel_6d(6) = 18
        tracker_input = torch.cat(
            [tp_rel, tr_sixd, tp_vel, tr_sixd_vel], dim=-1
        ).reshape(
            W, -1
        )  # [W, 108]

        gt_sixd = rotmat_to_sixd(gr)

        return {
            "input": tracker_input,  # [W, 108]
            "target_pos": gp_rel,  # [W, 22, 3]
            "target_sixd": gt_sixd,  # [W, 22, 6]
            "pelvis_pos": t_root.squeeze(1),  # [W, 3]
            "bone_offsets": offsets,  # [22, 3] body used for FK / targets
        }


def create_dataloaders(cfg):
    aug_cfg = cfg["data"].get("augmentation", {})

    train_ds = VRTeleopDataset(
        dataset_dir=cfg["data"]["dataset_dir"],
        window_size=cfg["data"]["window_size"],
        split="train",
        train_ratio=cfg["data"]["train_split"],
        folders=cfg["data"].get("train_folders"),
        augmentation=aug_cfg,
    )
    val_ds = VRTeleopDataset(
        dataset_dir=cfg["data"]["dataset_dir"],
        window_size=cfg["data"]["window_size"],
        split="val",
        train_ratio=cfg["data"]["train_split"],
        folders=cfg["data"].get("val_folders"),
        augmentation=aug_cfg,  # only the nominal tracker mount applies outside train
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # Optional held-out test set (cross-dataset, never used for model selection)
    test_loader = None
    test_folders = cfg["data"].get("test_folders")
    if test_folders:
        test_ds = VRTeleopDataset(
            dataset_dir=cfg["data"]["dataset_dir"],
            window_size=cfg["data"]["window_size"],
            split="test",
            folders=test_folders,
            augmentation=aug_cfg,  # only the nominal tracker mount applies
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
        )
    return train_loader, val_loader, test_loader
