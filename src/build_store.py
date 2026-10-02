"""
Pack the v2 datasets (AMASS, BONES-SEED, LAFAN1) into one memory-mapped store
(see tcn/store.py), so training can sample windows from ~35M frames without
loading thousands of .pt files into RAM.

Run from src/:
    python3 build_store.py                       # default paths below
    python3 build_store.py --amass DIR --bones DIR --lafan1 DIR --out DIR

Splits: AMASS keeps its dataset-folder subsets (CMU, SFU, ...) and the training
config chooses train/val/test folders. BONES-SEED is split by ACTOR (hash of the
actor id: 90% train / 5% val / 5% test), so no actor appears in two splits.
LAFAN1 is test-only. BONES clips are tagged "bones_low" when their body position
involves sitting, crouching, kneeling, crawling, lying, handstands, ... so
training can oversample the poses AMASS covers least.
"""

import argparse
import glob
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

MIN_LEN = 41  # window 40 + 1 extra leading frame for velocity

LOW_KEYWORDS = ["sitting", "crouch", "croach", "kneel", "all fours", "crawl", "lying", "handstand",
                "plank", "squat", "flipping", "heels", "bent down"]


def actor_split(actor):
    h = int(hashlib.sha1(str(actor).encode()).hexdigest(), 16) % 100
    return "train" if h < 90 else ("val" if h < 95 else "test")


def _load(job):
    path, source, subset, group, split, actor = job
    torch.set_num_threads(1)
    d = torch.load(path, weights_only=False)
    R = d["gt_rotmat"].float()
    if R.shape[0] < MIN_LEN:
        return None
    sixd = torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1).half().numpy()  # [T,22,6]
    return {
        "name": os.path.basename(path)[:-3], "source": source, "subset": subset, "group": group,
        "split": split, "actor": actor, "rot6d": sixd, "root": d["root_pos"].float().numpy(),
        "offsets": d["meta"]["bone_offsets"].float().numpy(),
    }


def make_jobs(args):
    jobs = []
    if args.amass:
        for f in sorted(glob.glob(os.path.join(args.amass, "*.pt"))):
            subset = os.path.basename(f).split("_seq_")[0]
            jobs.append((f, "AMASS", subset, "amass", "", ""))
    if args.bones:
        with open(os.path.join(args.bones, "index.json")) as f:
            index = json.load(f)
        for f in sorted(glob.glob(os.path.join(args.bones, "*.pt"))):
            stem = os.path.basename(f)[len("BONES_"):-3]
            info = index[stem]
            pos = (info["content_body_position"] or "").lower()
            group = "bones_low" if any(k in pos for k in LOW_KEYWORDS) else "bones_std"
            jobs.append((f, "BONES", "BONES", group, actor_split(info["actor_uid"]), info["actor_uid"]))
    if args.lafan1:
        for f in sorted(glob.glob(os.path.join(args.lafan1, "*.pt"))):
            jobs.append((f, "LAFAN1", "LAFAN1", "lafan1", "test", ""))
    return jobs


def build(args):
    jobs = make_jobs(args)
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(jobs)} sequences to pack", flush=True)
    rows = {k: [] for k in ["name", "source", "subset", "group", "split", "actor", "start", "length"]}
    offsets, n_frames, skipped, t0 = [], 0, 0, time.time()
    with open(os.path.join(args.out, "rot6d.bin"), "wb") as frot, open(os.path.join(args.out, "root.bin"), "wb") as froot, \
            Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap(_load, jobs, chunksize=16)):
            if r is None:
                skipped += 1
                continue
            frot.write(r["rot6d"].tobytes())
            froot.write(r["root"].astype(np.float32).tobytes())
            T = len(r["rot6d"])
            for k in ["name", "source", "subset", "group", "split", "actor"]:
                rows[k].append(r[k])
            rows["start"].append(n_frames)
            rows["length"].append(T)
            offsets.append(r["offsets"])
            n_frames += T
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(jobs)} sequences, {n_frames / 1e6:.1f}M frames, {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(args.out, "index.npz"),
             **{k: np.array(v) for k, v in rows.items() if k not in ("start", "length")},
             start=np.array(rows["start"], dtype=np.int64), length=np.array(rows["length"], dtype=np.int32),
             offsets=np.stack(offsets).astype(np.float32))
    meta = {"n_frames": n_frames, "n_seq": len(rows["name"]), "skipped_short": skipped, "min_len": MIN_LEN,
            "format": "rot6d float16 [N,22,6], root float32 [N,3]", "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done: {meta['n_seq']} sequences, {n_frames / 1e6:.1f}M frames ({n_frames / 216000:.0f} h), "
          f"{skipped} too short, {time.time() - t0:.0f}s", flush=True)


def verify(args, n=300):
    """Rebuild rotations from the store and compare FK joint positions to the source files."""
    import random
    from tcn.skeleton import fk_positions
    from tcn.store import MotionStore
    store = MotionStore(args.out)
    random.seed(0)
    errs = []
    for s in random.sample(range(store.n_seq), min(n, store.n_seq)):
        src = {"AMASS": args.amass, "BONES": args.bones, "LAFAN1": args.lafan1}[str(store.idx["source"][s])]
        d = torch.load(os.path.join(src, str(store.idx["name"][s]) + ".pt"), weights_only=False)
        T = int(store.idx["length"][s])
        a = random.randint(0, max(0, T - 200))
        R_store, root, off = store.get(s, a, min(200, T - a))
        R_src = d["gt_rotmat"][a : a + len(R_store)].float()
        p0 = fk_positions(R_src, off)
        p1 = fk_positions(R_store, off)
        errs.append(((p1 - p1[:, :1]) - (p0 - p0[:, :1])).norm(dim=-1).max().item() * 1000)
        assert torch.allclose(off, d["meta"]["bone_offsets"].float())
    print(f"verify: {len(errs)} sequences, fp16 6D round-trip FK error: mean {np.mean(errs):.3f} mm, max {np.max(errs):.3f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass", default="../support_data/vr_teleop_dataset_v2")
    ap.add_argument("--bones", default="../support_data/bones_uniform_v2")
    ap.add_argument("--lafan1", default="../support_data/lafan1_v2")
    ap.add_argument("--out", default="../support_data/store_v2")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    build(a)
    if not a.no_verify:
        verify(a)
