"""Where the six Vive trackers sit on the body (shared by the GPU sample builder and tests)."""

import math

import torch

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


def parse_mount(cfg):
    """YAML mount block -> tensors per tracker kind (None = trackers are exact joint copies)."""
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
