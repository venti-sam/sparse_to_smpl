"""
Batched, on-GPU construction of training samples from raw motion windows.

Workers only slice the memory-mapped store (rot6d fp16 [E,22,6], root [E,3],
offsets [22,3] with E = window + 1 leading frame). Everything else happens here
for the whole batch: random mirror, body-size scaling, forward kinematics, Vive
tracker mounting, sensor noise and dropouts, heading canonicalization, features
and targets.

Frame 0 is the velocity context and frame 1 is the window's first frame.
Outside training (train=False) the tracker mount is its mean, with no noise.
"""

import torch

from .rotations import axis_angle_to_rotmat, extract_yaw, rotmat_to_sixd, sixd_to_rotmat
from .skeleton import SYMMETRIC_PAIRS, fk_positions, mirror_motion
from .trackers import REF_TRACKER_IDX, TRACKER_JOINTS, TRACKER_MOUNTS, parse_mount

_LEFT = [left for left, _ in SYMMETRIC_PAIRS]
_RIGHT = [right for _, right in SYMMETRIC_PAIRS]


class GPUSampleBuilder:
    def __init__(self, augmentation, device, fps=60.0):
        aug = augmentation or {}
        self.enabled = bool(aug.get("enabled", False))
        self.device = device
        self.fps = fps
        def val(key):
            return float(aug.get(key, 0.0))

        deg = torch.pi / 180
        self.pos_jitter, self.rot_jitter = val("pos_jitter_std"), val("rot_jitter_deg") * deg
        self.pos_bias, self.rot_bias = val("pos_bias_std"), val("rot_bias_deg") * deg
        self.vel_noise, self.dropout_prob = val("vel_noise_std"), val("dropout_prob")
        self.scale_global, self.scale_limb = val("bone_scale_global"), val("bone_scale_limb_std")
        self.mirror_prob = val("mirror_prob")
        lo, hi = aug.get("dropout_frames", [5, 30])
        self.drop_lo, self.drop_hi = int(lo), int(hi)
        mount = parse_mount(aug.get("mount"))
        self.mount = None
        if mount is not None:
            self.mount = {k: {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in c.items()}
                          for k, c in mount.items()}

    # ── helpers ──
    def _velocity(self, x):
        """Finite difference along dim 1, first frame padded with the second."""
        v = torch.zeros_like(x)
        v[:, 1:] = (x[:, 1:] - x[:, :-1]) * self.fps
        v[:, 0] = v[:, 1]
        return v

    def _mount_trackers(self, gp, gr, offsets, train):
        B = gp.shape[0]
        pos, rot = [], []
        for kind, host, anchor in TRACKER_MOUNTS:
            c = self.mount[kind]
            t = c["pos_mean"].expand(B, 3).clone()
            if kind == "hand":  # controller held like a stick, along the forearm
                t = t + torch.nn.functional.normalize(offsets[:, anchor], dim=-1) * c["along_forearm"]
            R_mount = torch.eye(3, device=gp.device).expand(B, 3, 3)
            if train:
                t = t + torch.randn(B, 3, device=gp.device) * c["pos_std"]
                R_mount = axis_angle_to_rotmat(torch.randn(B, 3, device=gp.device) * c["rot_std"])
            pos.append(gp[:, :, anchor] + torch.einsum("beij,bj->bei", gr[:, :, host], t))
            rot.append(gr[:, :, host] @ R_mount[:, None])
        return torch.stack(pos, dim=2), torch.stack(rot, dim=2)

    def _dropout(self, tp, tr):
        """With prob p per window, one non-pelvis tracker freezes at its last good value for a short burst."""
        B, E = tp.shape[:2]
        dev = tp.device
        active = torch.rand(B, device=dev) < self.dropout_prob
        trk = torch.randint(1, 6, (B,), device=dev)
        dur = torch.randint(self.drop_lo, self.drop_hi + 1, (B,), device=dev)
        hi = (E - dur + 1).clamp(min=2)                       # start in [1, E - dur]
        s = 1 + (torch.rand(B, device=dev) * (hi - 1)).long()
        t = torch.arange(E, device=dev)[None]
        frozen = (t >= s[:, None]) & (t < (s + dur)[:, None]) & active[:, None]          # [B,E]
        src = torch.where(frozen, (s - 1)[:, None], t.expand(B, E))                       # [B,E]
        b = torch.arange(B, device=dev)[:, None]
        sel = frozen[:, :, None] & (torch.arange(6, device=dev)[None, None] == trk[:, None, None])  # [B,E,6]
        tp = torch.where(sel[..., None], tp[b, src], tp)
        tr = torch.where(sel[..., None, None], tr[b, src], tr)
        return tp, tr

    # ── main entry ──
    @torch.no_grad()
    def __call__(self, raw, train):
        train = bool(train and self.enabled)
        dev = self.device
        gr = sixd_to_rotmat(raw["rot6d"].to(dev, non_blocking=True).float())     # [B,E,22,3,3]
        root = raw["root"].to(dev, non_blocking=True)
        offsets = raw["offsets"].to(dev, non_blocking=True)
        B, E = gr.shape[:2]
        W = E - 1

        if train and self.mirror_prob > 0:
            m = torch.rand(B, device=dev) < self.mirror_prob
            gr_m, root_m, off_m = mirror_motion(gr, root, offsets)
            gr = torch.where(m[:, None, None, None, None], gr_m, gr)
            root = torch.where(m[:, None, None], root_m, root)
            offsets = torch.where(m[:, None, None], off_m, offsets)

        scale0 = torch.ones(B, device=dev)
        if train and (self.scale_limb > 0 or self.scale_global > 0):
            limb = torch.ones(B, 22, 1, device=dev)
            if self.scale_limb > 0:
                limb = torch.exp(torch.randn(B, 22, 1, device=dev) * self.scale_limb)
                limb[:, _RIGHT] = limb[:, _LEFT]                                  # left/right bones share a value
            if self.scale_global > 0:
                limb = limb * (1.0 + (torch.rand(B, 1, 1, device=dev) * 2 - 1) * self.scale_global)
            offsets = offsets * limb
            scale0 = limb[:, 0, 0]

        gp = fk_positions(gr, offsets)                                            # [B,E,22,3]
        gp = gp - gp[:, :, 0:1] + (root * scale0[:, None, None])[:, :, None]

        if self.mount is None:
            tp, tr = gp[:, :, TRACKER_JOINTS].clone(), gr[:, :, TRACKER_JOINTS].clone()
        else:
            tp, tr = self._mount_trackers(gp, gr, offsets, train)

        if train:
            if self.pos_bias > 0:
                tp = tp + torch.randn(B, 1, 6, 3, device=dev) * self.pos_bias
            if self.rot_bias > 0:
                tr = tr @ axis_angle_to_rotmat(torch.randn(B, 1, 6, 3, device=dev) * self.rot_bias)
            if self.pos_jitter > 0:
                tp = tp + torch.randn_like(tp) * self.pos_jitter
            if self.rot_jitter > 0:
                tr = tr @ axis_angle_to_rotmat(torch.randn(B, E, 6, 3, device=dev) * self.rot_jitter)
            if self.dropout_prob > 0:
                tp, tr = self._dropout(tp, tr)

        # heading canonicalization on the tracker pelvis at the window's first frame (frame 1)
        R_yaw_inv = extract_yaw(tr[:, 1, REF_TRACKER_IDX]).transpose(-1, -2)      # [B,3,3]
        tp = torch.einsum("bij,betj->beti", R_yaw_inv, tp)
        gp = torch.einsum("bij,betj->beti", R_yaw_inv, gp)
        tr = torch.einsum("bij,betjk->betik", R_yaw_inv, tr)
        gr = torch.einsum("bij,betjk->betik", R_yaw_inv, gr)

        t_root = tp[:, :, REF_TRACKER_IDX : REF_TRACKER_IDX + 1]
        tp_rel = tp - t_root
        gp_rel = gp - gp[:, :, 0:1]
        tr_sixd = rotmat_to_sixd(tr)                                              # [B,E,6,6]
        tp_vel, tr_vel = self._velocity(tp_rel), self._velocity(tr_sixd)
        if train and self.vel_noise > 0:
            tp_vel = tp_vel + torch.randn_like(tp_vel) * self.vel_noise
            tr_vel = tr_vel + torch.randn_like(tr_vel) * self.vel_noise

        feats = torch.cat([tp_rel, tr_sixd, tp_vel, tr_vel], dim=-1)[:, 1:]       # [B,W,6,18]
        return {
            "input": feats.reshape(B, W, -1),
            "target_pos": gp_rel[:, 1:],
            "target_sixd": rotmat_to_sixd(gr[:, 1:]),
            "pelvis_pos": t_root[:, 1:, 0],
            "bone_offsets": offsets,
        }


class DeviceLoader:
    """Wraps a loader of raw windows: each batch is built on the GPU before it is yielded."""

    def __init__(self, loader, builder, train):
        self.loader, self.builder, self.train = loader, builder, train
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for raw in self.loader:
            yield self.builder(raw, self.train)
