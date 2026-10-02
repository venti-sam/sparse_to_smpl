"""Small synthetic motion + store helpers for the tests (no real datasets needed)."""

import json
import os
from types import SimpleNamespace

import torch

from tcn.model import local_to_global_rotmat
from tcn.rotations import axis_angle_to_rotmat, rotmat_to_sixd
from tcn.skeleton import SMPL_NEUTRAL_OFFSETS


def random_motion(T, seed=0, amplitude=0.3):
    """Global joint rotations [T,22,3,3] of a random but valid pose sequence."""
    g = torch.Generator().manual_seed(seed)
    local = axis_angle_to_rotmat(amplitude * torch.randn(T, 22, 3, generator=g))
    return local_to_global_rotmat(local)


def neutral_offsets():
    return torch.tensor(SMPL_NEUTRAL_OFFSETS, dtype=torch.float32)


def raw_batch(gr, root=None, offsets=None, half=True):
    """Wrap one motion [E,22,3,3] as a batch of one raw window for GPUSampleBuilder."""
    E = gr.shape[0]
    root = torch.zeros(E, 3) if root is None else root
    offsets = neutral_offsets() if offsets is None else offsets
    rot6d = rotmat_to_sixd(gr)[None]
    return {"rot6d": rot6d.half() if half else rot6d, "root": root[None], "offsets": offsets[None]}


def write_v2_sequence(path, T, seed):
    torch.save({"gt_rotmat": random_motion(T, seed), "root_pos": torch.randn(T, 3).cumsum(0) * 0.01,
                "meta": {"bone_offsets": neutral_offsets()}}, path)


def make_store(tmp, n_amass=4, n_bones=6, T=120):
    """Build a tiny store: AMASS-style sequences in two subsets, BONES-style with an index.json."""
    from build_store import build
    amass, bones = os.path.join(tmp, "amass"), os.path.join(tmp, "bones")
    os.makedirs(amass)
    os.makedirs(bones)
    for i in range(n_amass):
        write_v2_sequence(os.path.join(amass, f"{'AAA' if i % 2 else 'BBB'}_seq_{i:04d}.pt"), T + 10 * i, seed=i)
    index = {}
    for i in range(n_bones):
        stem = f"clip{i}"
        write_v2_sequence(os.path.join(bones, f"BONES_{stem}.pt"), T, seed=100 + i)
        index[stem] = {"content_body_position": "sitting on floor" if i % 3 == 0 else "standing", "actor_uid": f"A{i % 4}"}
    with open(os.path.join(bones, "index.json"), "w") as f:
        json.dump(index, f)
    out = os.path.join(tmp, "store")
    build(SimpleNamespace(amass=amass, bones=bones, lafan1=None, out=out, workers=1))
    return out
