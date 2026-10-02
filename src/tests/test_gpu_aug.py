import copy
import os
import unittest

import torch
import torch.nn.functional as F
import yaml

from tcn.gpu_aug import GPUSampleBuilder
from tcn.rotations import axis_angle_to_rotmat, sixd_to_rotmat
from tcn.skeleton import fk_positions, mirror_motion
from tcn.trackers import TRACKER_MOUNTS
from tests.synthetic import neutral_offsets, random_motion, raw_batch

with open(os.path.join(os.path.dirname(__file__), "..", "tcn", "config.yaml")) as _f:
    CFG = yaml.safe_load(_f)["data"]["augmentation"]
CPU = torch.device("cpu")
W = 40


def quiet_cfg(**overrides):
    """Augmentation config with every random effect off, then selected ones switched on."""
    cfg = copy.deepcopy(CFG)
    cfg.update(enabled=True, pos_jitter_std=0, rot_jitter_deg=0, pos_bias_std=0, rot_bias_deg=0, vel_noise_std=0,
               dropout_prob=0, bone_scale_global=0, bone_scale_limb_std=0, mirror_prob=0)
    cfg.update(overrides)
    return cfg


class GPUSampleBuilderTest(unittest.TestCase):
    def test_shapes_and_target_consistency(self):
        out = GPUSampleBuilder(CFG, CPU)(raw_batch(random_motion(W + 1)), train=False)
        self.assertEqual(out["input"].shape, (1, W, 108))
        self.assertEqual(out["target_pos"].shape, (1, W, 22, 3))
        self.assertEqual(out["target_sixd"].shape, (1, W, 22, 6))
        pos = fk_positions(sixd_to_rotmat(out["target_sixd"]), out["bone_offsets"])
        self.assertTrue(torch.allclose(pos - pos[:, :, :1], out["target_pos"], atol=1e-4))

    def test_nominal_mount_geometry(self):
        """T-pose, upright: each tracker sits at its mount offset from its joint."""
        off = neutral_offsets()
        eye = torch.eye(3).expand(W + 1, 22, 3, 3)
        out = GPUSampleBuilder(CFG, CPU)(raw_batch(eye, offsets=off), train=False)
        joints = fk_positions(eye[:1], off)[0]
        tp = []
        for kind, _, anchor in TRACKER_MOUNTS:
            c = CFG["mount"][kind]
            t = torch.tensor(c.get("pos_mean", [0.0] * 3))
            if kind == "hand":
                t = t + F.normalize(off[anchor], dim=0) * c.get("along_forearm", 0.0)
            tp.append(joints[anchor] + t)
        tp = torch.stack(tp)
        got = out["input"].reshape(1, W, 6, 18)[0, 0, :, :3]  # position block, relative to the pelvis tracker
        self.assertTrue(torch.allclose(got, tp - tp[0], atol=1e-5))

    def test_heading_invariance(self):
        """Turning the whole motion about the vertical axis must not change the sample."""
        gr = random_motion(W + 1, seed=3)
        root = torch.randn(W + 1, 3)
        Rz = axis_angle_to_rotmat(torch.tensor([0.0, 0.0, 1.1]))
        # float32 windows: separate fp16 rounding of the two motions would show up in the 60x velocity features
        a = GPUSampleBuilder(CFG, CPU)(raw_batch(gr, root, half=False), train=False)
        b = GPUSampleBuilder(CFG, CPU)(raw_batch(Rz @ gr, root @ Rz.T, half=False), train=False)
        for key in ("input", "target_pos", "target_sixd"):
            self.assertTrue(torch.allclose(a[key], b[key], atol=1e-4), key)

    def test_velocity_features_are_finite_differences(self):
        out = GPUSampleBuilder(CFG, CPU)(raw_batch(random_motion(W + 1, seed=5)), train=False)
        f = out["input"].reshape(1, W, 6, 18)
        pos, vel = f[0, :, :, 0:3], f[0, :, :, 9:12]
        self.assertTrue(torch.allclose(vel[1:], (pos[1:] - pos[:-1]) * 60.0, atol=1e-3))

    def test_dropout_freezes_a_tracker(self):
        cfg = quiet_cfg(dropout_prob=1.0)
        batch = {k: v.expand(8, *v.shape[1:]).clone() for k, v in raw_batch(random_motion(W + 1, seed=7)).items()}
        f = GPUSampleBuilder(cfg, CPU)(batch, train=True)["input"].reshape(8, W, 6, 18)
        rot = f[..., 3:9]  # tracker rotations are absolute, so a frozen tracker repeats exactly
        repeated = ((rot[:, 1:] - rot[:, :-1]).abs().amax(-1) == 0)[:, :, 1:].sum(1)  # [B, 5 non-pelvis trackers]
        self.assertTrue(bool((repeated.amax(1) >= cfg["dropout_frames"][0]).all()))
        self.assertTrue(bool((repeated > 0).sum(1).eq(1).all()))  # exactly one tracker per window

    def test_mirror_equals_mirroring_the_input(self):
        gr, root = random_motion(W + 1, seed=9), torch.randn(W + 1, 3)
        off = neutral_offsets() + 0.01 * torch.randn(22, 3)  # asymmetric body, so the swap matters
        a = GPUSampleBuilder(quiet_cfg(mirror_prob=1.0, mount=None), CPU)(raw_batch(gr, root, off), train=True)
        gr_m, root_m, off_m = mirror_motion(gr, root, off)
        b = GPUSampleBuilder(quiet_cfg(mount=None), CPU)(raw_batch(gr_m, root_m, off_m), train=False)
        for key in ("input", "target_pos", "target_sixd", "bone_offsets"):
            self.assertTrue(torch.allclose(a[key], b[key], atol=2e-3), key)

    def test_training_noise_changes_inputs_but_not_targets(self):
        batch = raw_batch(random_motion(W + 1, seed=11))
        builder = GPUSampleBuilder(CFG, CPU)
        clean, noisy = builder(batch, train=False), builder(batch, train=True)
        self.assertFalse(torch.allclose(clean["input"], noisy["input"], atol=1e-3))
        # targets are the (possibly rescaled / mirrored) body, never perturbed by sensor noise
        self.assertTrue(torch.isfinite(noisy["input"]).all())


if __name__ == "__main__":
    unittest.main()
