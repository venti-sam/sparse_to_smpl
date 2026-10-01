"""
Convert the extract_vr_data.py output (v1) into the v2 training format.

Pipeline: extract_vr_data.py  ->  convert_dataset_v2.py  ->  tcn.train

What changes vs v1:
- World frame is fixed. v1 stored AMASS joints (already Z-up) after an extra
  Y-up->Z-up permutation, so the body's up axis was X. v2 applies the inverse
  permutation: up is Z and the pelvis rotation's column 0 is the forward axis,
  which is what the dataset's heading canonicalization assumes.
- Per-actor T-pose bone offsets (from betas) are stored in meta["bone_offsets"],
  so FK reproduces each actor's real body instead of one neutral skeleton.
- Only gt_rotmat + root_pos are stored. Tracker inputs and target positions are
  derived from them at load time (FK), which lets the dataset randomise body
  size and sensor mounting.

Run from src/:  python convert_dataset_v2.py [--limit N]
The v1 directory is never modified.
"""

import argparse
import glob
import os

import torch
import smplx
from tqdm import tqdm

from extract_vr_data import BODY_MODEL_PATH, YUPTOZUP, compute_bone_offsets, DEVICE
from tcn.skeleton import fk_positions

SRC_DIR = "../support_data/vr_teleop_dataset"
DST_DIR = "../support_data/vr_teleop_dataset_v2"

# stored_v1 = H @ amass  ->  amass = H^T @ stored_v1  (H is a proper rotation)
M = torch.tensor(YUPTOZUP).T.contiguous()  # [3, 3]


def convert(src_dir, dst_dir, limit=None):
    os.makedirs(dst_dir, exist_ok=True)
    bm = smplx.SMPL(model_path=BODY_MODEL_PATH, gender="neutral", num_betas=16).to(DEVICE)

    files = sorted(glob.glob(os.path.join(src_dir, "*.pt")))
    if limit:
        files = files[:limit]

    worst = 0.0
    for f in tqdm(files, unit="seq"):
        out_path = os.path.join(dst_dir, os.path.basename(f))
        if os.path.exists(out_path):
            continue

        d = torch.load(f, weights_only=False)
        gt_pos = d["ground_truth"]["gt_pos"].float()
        gt_rot = d["ground_truth"]["gt_rotmat"].float()
        betas = d["meta"]["betas"].float().to(DEVICE)[None]

        offsets = compute_bone_offsets(bm, betas)[0].cpu()  # [22, 3]

        # World fix: rotate both positions and rotations by M
        pos = gt_pos @ M.T
        rot = M @ gt_rot
        root_pos = pos[:, 0].clone()

        # FK with per-actor offsets must reproduce the stored joints (root-relative)
        fk = fk_positions(rot, offsets)
        err = ((fk - fk[:, :1]) - (pos - pos[:, :1])).norm(dim=-1).max().item() * 1000
        worst = max(worst, err)
        if err > 1.0:
            tqdm.write(f"  ! {os.path.basename(f)}: FK mismatch {err:.2f} mm, skipped")
            continue

        meta = dict(d["meta"])
        meta.update(bone_offsets=offsets, coord_system="Z-up (true)", up_axis_fixed=True)
        torch.save(
            {"gt_rotmat": rot, "root_pos": root_pos, "meta": meta},
            out_path,
        )
    print(f"Done. Worst FK reconstruction error: {worst:.3f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DIR)
    ap.add_argument("--dst", default=DST_DIR)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    convert(args.src, args.dst, args.limit)
