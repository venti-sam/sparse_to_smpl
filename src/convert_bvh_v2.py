"""
Convert mocap BVH files into the v2 training format (see convert_dataset_v2.py).

BVH rotations are relative to each skeleton's zero pose, which is arbitrary
(LAFAN1 and SOMA have every bone along +X). To make the rotation labels mean what
they mean in SMPL (all joint frames = identity in a T-pose; x forward, y left,
z up) we take one reference pose in which the actor stands in a T-pose, call it
the zero rotation, and express every frame relative to it:

    R_can(t) = P  G(t) G_ref^-1  P^T

G = BVH global rotation, P = fixed axis change that makes the reference actor
face +x with +y left and +z up. Bone offsets are the T-pose bone vectors of the
clip's own skeleton in that frame.

Body: by default (--body smpl) the saved bone offsets are the SMPL neutral
skeleton, so the motion is that of an SMPL body, the same body structure as
AMASS and as the deployed model's FK. The rotations are unchanged. Measured on
an early v2 checkpoint, LAFAN1 on its own skeleton scored 68 mm MPJPE vs 34 mm
on the SMPL body (AMASS val: 25 mm), because the model's tracker-to-joint
geometry is SMPL's. --body source keeps the file's own skeleton.

Joints that SMPL does not have (SOMA's Neck2) are skipped: the mapped child's
rotation is its own global rotation and its bone is the T-pose vector across the
gap. Joints below a skipped joint are not position-exact; the per-file check
only fails on joints whose whole chain is direct.

Run from src/:
    python3 convert_bvh_v2.py --src /path/to/lafan1 --dst ../support_data/lafan1_v2 \
        --skeleton lafan1 --prefix LAFAN1
    python3 convert_bvh_v2.py --src ../support_data/bones_seed/soma_uniform \
        --dst ../support_data/bones_uniform_v2 --skeleton soma --prefix BONES \
        --ref-file ../support_data/bones_seed/soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh
"""

import argparse
import glob
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm

from bvh_utils import bvh_fk, parse_bvh
from tcn.skeleton import SMPL_NEUTRAL_OFFSETS, SMPL_PARENTS, fk_positions

TARGET_FPS = 60.0

# BVH joint names in SMPL-22 order (pelvis, L/R hip, spine1, L/R knee, spine2,
# L/R ankle, spine3, L/R foot, neck, L/R collar, head, L/R shoulder, L/R elbow,
# L/R wrist). Each mapped joint's SMPL parent must be a BVH ancestor.
SKELETONS = {
    "lafan1": {
        "unit": 0.01,  # file is in cm
        "up": (0.0, 1.0, 0.0),
        "joints": [
            "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
            "Spine1", "LeftFoot", "RightFoot", "Spine2", "LeftToe", "RightToe",
            "Neck", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
            "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
        ],
    },
    # BONES-SEED SOMA (uniform and proportional share names). Neck2 is skipped.
    "soma": {
        "unit": 0.01,  # cm
        "up": (0.0, 1.0, 0.0),
        "joints": [
            "Hips", "LeftLeg", "RightLeg", "Spine1", "LeftShin", "RightShin",
            "Spine2", "LeftFoot", "RightFoot", "Chest", "LeftToeBase", "RightToeBase",
            "Neck1", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
            "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
        ],
    },
}

# SMPL neutral T-pose bone directions in the v2 frame, used to pick / grade the reference pose.
_SMPL_T = SMPL_NEUTRAL_OFFSETS
_SMPL_T_DIR = _SMPL_T[1:] / np.linalg.norm(_SMPL_T[1:], axis=1, keepdims=True)

# Pose-defining bones (thighs, shins, feet, upper arms, forearms), indexed by joint - 1.
# Hip, collar, neck and head bones differ between skeletons by design, not by pose.
_LIMB = np.array([4, 5, 7, 8, 10, 11, 18, 19, 20, 21]) - 1


def load_clip(path, cfg, max_frames=None):
    """BVH -> mapped SMPL-22 global rotations and positions (metres, BVH axes)."""
    d = parse_bvh(path, max_frames=max_frames)
    idx = [d["names"].index(n) for n in cfg["joints"]]
    direct = np.ones(22, dtype=bool)
    exact = np.ones(22, dtype=bool)  # whole chain to the root is direct
    for i in range(1, 22):
        want, j, hops = idx[SMPL_PARENTS[i]], d["parents"][idx[i]], 0
        while j != want:
            if j < 0:
                raise ValueError(f"{cfg['joints'][i]}: {cfg['joints'][SMPL_PARENTS[i]]} is not a BVH ancestor")
            j, hops = d["parents"][j], hops + 1
        direct[i] = hops == 0
        exact[i] = direct[i] and exact[SMPL_PARENTS[i]]
    u = cfg["unit"]
    G, P = bvh_fk(d["parents"], d["offsets"] * u, d["trans"] * u, d["has_trans"], d["rot"])
    return {"G": G[:, idx], "pos": P[:, idx], "raw": d, "idx": idx, "exact": exact, "fps": d["fps"]}


def axis_change(pos, up):
    """P (3x3, rows = forward, left, up) from one pose: left = hip line, forward = left x up."""
    up = np.asarray(up, dtype=np.float64)
    left = pos[1] - pos[2]  # L_Hip - R_Hip
    left = left - (left @ up) * up
    left /= np.linalg.norm(left)
    return np.stack([np.cross(left, up), left, up])


def tpose_error_deg(pos, up):
    """Angle between this pose's bones and the SMPL neutral T-pose bones.
    Returns (mean over limb bones, per-bone angles)."""
    Pm = axis_change(pos, up)
    c = (pos - pos[0]) @ Pm.T
    par = np.asarray(SMPL_PARENTS[1:])
    v = c[1:] - c[par]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((v * _SMPL_T_DIR).sum(1), -1, 1)))
    return ang[_LIMB].mean(), ang


def make_reference(clip, frame, path):
    return {
        "G": clip["G"][frame],  # [22,3,3] mapped global rotations in the reference pose
        "local": clip["raw"]["rot"][frame],  # [J,3,3] all local rotations in the reference pose
        "pos": clip["pos"][frame],
        "names": clip["raw"]["names"],
        "file": os.path.basename(path),
        "frame": frame,
    }


def pick_reference(files, cfg):
    """Frame 0 of the clip whose pose is closest to the SMPL T-pose."""
    best = None
    for f in tqdm(files, desc="reference search", unit="clip"):
        c = load_clip(f, cfg, max_frames=1)
        score, _ = tpose_error_deg(c["pos"][0], cfg["up"])
        if best is None or score < best[0]:
            best = (score, f, c)
    return best[0], make_reference(best[2], 0, best[1])


def to_canonical(clip, ref, Pm, cfg):
    """Rotations, T-pose bone offsets and root path in the v2 frame."""
    d = clip["raw"]
    if d["names"] != ref["names"]:
        raise ValueError("clip skeleton differs from the reference skeleton")
    rel = clip["G"] @ ref["G"].swapaxes(-1, -2)  # rotation relative to the reference pose
    R_can = Pm @ rel @ Pm.T
    # this clip's own skeleton held in the reference pose -> T-pose bone vectors
    J = len(d["names"])
    _, P_ref = bvh_fk(d["parents"], d["offsets"] * cfg["unit"], np.zeros((1, J, 3)),
                      np.zeros(J, dtype=bool), ref["local"][None])
    p_ref = P_ref[0][clip["idx"]]
    par = np.asarray(SMPL_PARENTS[1:])
    off = np.zeros((22, 3))
    off[1:] = (p_ref[1:] - p_ref[par]) @ Pm.T
    root = clip["pos"][:, 0] @ Pm.T
    return R_can, off, root


def resample(R_can, root, fps_src, fps_dst=TARGET_FPS):
    if abs(fps_src - fps_dst) < 0.05:
        return R_can, root
    ratio = fps_src / fps_dst
    if abs(ratio - round(ratio)) < 1e-3 and round(ratio) >= 1:
        s = int(round(ratio))
        return R_can[::s], root[::s]
    T = R_can.shape[0]
    t_src = np.arange(T) / fps_src
    t_dst = np.arange(int(np.floor(t_src[-1] * fps_dst)) + 1) / fps_dst
    out = np.empty((len(t_dst),) + R_can.shape[1:])
    for j in range(R_can.shape[1]):
        out[:, j] = Slerp(t_src, R.from_matrix(R_can[:, j]))(t_dst).as_matrix()
    root_out = np.stack([np.interp(t_dst, t_src, root[:, k]) for k in range(3)], axis=1)
    return out, root_out


def make_context(args, files=None):
    """Everything needed to convert clips of one skeleton: config, T-pose reference, axis change."""
    cfg = SKELETONS[args.skeleton]
    if args.ref_file:
        c = load_clip(args.ref_file, cfg, max_frames=args.ref_frame + 1)
        ref = make_reference(c, args.ref_frame, args.ref_file)
        score, ang = tpose_error_deg(ref["pos"], cfg["up"])
    else:
        score, ref = pick_reference(files, cfg)
        _, ang = tpose_error_deg(ref["pos"], cfg["up"])
    print(f"reference: {ref['file']} frame {ref['frame']}, limb bones vs SMPL T-pose: "
          f"mean {score:.1f} deg, max {ang[_LIMB].max():.1f}")
    return {
        "cfg": cfg, "ref": ref, "Pm": axis_change(ref["pos"], cfg["up"]),
        "dst": args.dst, "prefix": args.prefix, "skeleton": args.skeleton,
        "body": args.body, "body_scale": args.body_scale,
    }


def process_clip(ctx, source, stem, extra_meta=None, out_name=None):
    """
    Convert one BVH (path or bytes) and write <dst>/<prefix>_<stem>.pt atomically.
    Returns {"ok", "msg", "e_exact", "e_other", "toe", "height"}.
    """
    cfg, ref, Pm = ctx["cfg"], ctx["ref"], ctx["Pm"]
    out_path = os.path.join(ctx["dst"], out_name or f"{ctx['prefix']}_{stem}.pt")
    try:
        clip = load_clip(source, cfg)
        R_can, off, root = to_canonical(clip, ref, Pm, cfg)
    except Exception as e:  # truncated / malformed file
        return {"ok": False, "msg": f"{type(e).__name__}: {e}"}

    # exactness: FK of the canonical rotations must reproduce the BVH joints
    fk = fk_positions(torch.from_numpy(R_can), torch.from_numpy(off)).numpy()
    want = clip["pos"] @ Pm.T
    err = np.linalg.norm((fk - fk[:, :1]) - (want - want[:, :1]), axis=-1).max(axis=0) * 1000  # per joint, mm
    e_exact = err[clip["exact"]].max()
    e_other = err[~clip["exact"]].max() if (~clip["exact"]).any() else 0.0
    res = {"ok": True, "msg": "", "e_exact": float(e_exact), "e_other": float(e_other),
           "toe": float(want[:, 10:12, 2].min()),  # toes should reach the floor
           "height": float(want[0, 15, 2] - want[0, 7:9, 2].min())}  # head-to-ankle
    if e_exact > 1.0:
        return {**res, "ok": False, "msg": f"FK mismatch {e_exact:.3f} mm"}

    R_out, root_out = resample(R_can, root, clip["fps"])
    if ctx["body"] == "smpl":
        save_off = _SMPL_T * ctx["body_scale"]
        save_off[0] = 0.0
        root_out = root_out * ctx["body_scale"]
    else:
        save_off = off
    meta = {
        "fps": TARGET_FPS,
        "bone_offsets": torch.from_numpy(save_off).float(),
        "body": ctx["body"],
        "source_file": stem + ".bvh",
        "source_dataset": ctx["prefix"],
        "coord_system": "Z-up (true)",
        "up_axis_fixed": True,
        "skeleton": ctx["skeleton"],
        "reference": f"{ref['file']}:{ref['frame']}",
        **(extra_meta or {}),
    }
    tmp = out_path + ".tmp"
    torch.save({"gt_rotmat": torch.from_numpy(R_out).float(),
                "root_pos": torch.from_numpy(root_out).float(), "meta": meta}, tmp)
    os.replace(tmp, out_path)
    return res


def summarize(results):
    ok = [r for r in results if r["ok"]]
    print(f"converted {len(ok)} clips, skipped {len(results) - len(ok)}")
    if not ok:
        return
    print(f"worst FK error, exact joints: {max(r['e_exact'] for r in ok):.4f} mm | "
          f"joints below a skipped joint: {max(r['e_other'] for r in ok):.1f} mm")
    toe = np.array([r["toe"] for r in ok])
    print(f"lowest toe height per clip (cm): median {100*np.median(toe):.1f}, "
          f"min {100*toe.min():.1f}, max {100*toe.max():.1f}")
    print(f"head-to-ankle height at frame 0 (m): median {np.median([r['height'] for r in ok]):.2f}")


def convert(args):
    files = sorted(glob.glob(os.path.join(args.src, "**", "*.bvh"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .bvh under {args.src}")
    os.makedirs(args.dst, exist_ok=True)
    ctx = make_context(args, files)
    if args.limit:
        files = files[: args.limit]
    results = []
    for f in tqdm(files, desc="convert", unit="clip"):
        stem = os.path.splitext(os.path.basename(f))[0]
        r = process_clip(ctx, f, stem)
        if not r["ok"]:
            tqdm.write(f"  ! {stem}: {r['msg']}")
        results.append(r)
    summarize(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with .bvh files (searched recursively)")
    ap.add_argument("--dst", default="../support_data/lafan1_v2")
    ap.add_argument("--skeleton", default="lafan1", choices=sorted(SKELETONS))
    ap.add_argument("--prefix", default="LAFAN1", help="filename prefix = folder name in tcn config")
    ap.add_argument("--body", default="smpl", choices=["smpl", "source"],
                    help="bone lengths to save: SMPL neutral (default) or the BVH's own skeleton")
    ap.add_argument("--body-scale", type=float, default=1.0, help="scale for --body smpl")
    ap.add_argument("--ref-file", default=None, help="BVH containing the T-pose (default: auto-pick)")
    ap.add_argument("--ref-frame", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    convert(ap.parse_args())
