"""
Memory-mapped motion store: every sequence of every dataset in two flat arrays
plus a small index (written by src/build_store.py).

    rot6d.bin   float16 [N, 22, 6]   first two columns of each global rotation
    root.bin    float32 [N, 3]       pelvis path (v2 frame)
    index.npz   per sequence: name, source, subset, split, group, actor,
                start, length, offsets[22, 3]
    meta.json   frame count and build info

Opened lazily per process, so DataLoader workers share the OS page cache.
"""

import json
import os

import numpy as np
import torch

from .rotations import sixd_to_rotmat


class MotionStore:
    def __init__(self, store_dir):
        self.dir = store_dir
        with open(os.path.join(store_dir, "meta.json")) as f:
            self.meta = json.load(f)
        with np.load(os.path.join(store_dir, "index.npz")) as idx:
            self.idx = {k: idx[k] for k in idx.files}
        self.n_seq = len(self.idx["length"])
        self._rot = self._root = None

    def _open(self):
        n = self.meta["n_frames"]
        self._rot = np.memmap(os.path.join(self.dir, "rot6d.bin"), dtype=np.float16, mode="r", shape=(n, 22, 6))
        self._root = np.memmap(os.path.join(self.dir, "root.bin"), dtype=np.float32, mode="r", shape=(n, 3))

    def select(self, source=None, subsets=None, split=None, groups=None, min_len=0):
        """Indices of the sequences matching all given filters."""
        m = self.idx["length"] >= min_len
        if source is not None:
            m &= self.idx["source"] == source
        if subsets is not None:
            m &= np.isin(self.idx["subset"], list(subsets))
        if split is not None:
            m &= self.idx["split"] == split
        if groups is not None:
            m &= np.isin(self.idx["group"], list(groups))
        return np.nonzero(m)[0]

    def get_raw(self, seq, start, n):
        """Same frames, undecoded: rot6d fp16 [n,22,6], root [n,3], offsets [22,3]."""
        if self._rot is None:
            self._open()
        s = int(self.idx["start"][seq]) + int(start)
        return (torch.from_numpy(np.array(self._rot[s : s + n])), torch.from_numpy(np.array(self._root[s : s + n])),
                torch.from_numpy(self.idx["offsets"][seq]))

    def get(self, seq, start, n):
        """Frames [start, start+n) of sequence `seq`: rotations [n,22,3,3], root [n,3], offsets [22,3]."""
        if self._rot is None:
            self._open()
        s = int(self.idx["start"][seq]) + int(start)
        rot6d = torch.from_numpy(np.asarray(self._rot[s : s + n], dtype=np.float32))
        root = torch.from_numpy(np.array(self._root[s : s + n]))
        return sixd_to_rotmat(rot6d), root, torch.from_numpy(self.idx["offsets"][seq])
