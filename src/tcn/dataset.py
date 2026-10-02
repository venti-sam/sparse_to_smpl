"""
Training data: random or strided windows from the memory-mapped motion store.

Workers only slice raw frames out of the store (tcn/store.py); the Vive tracker
simulation, augmentation and feature building happen on the GPU (tcn/gpu_aug.py).
Windows start at frame >= 1 so the leading velocity-context frame always exists
(window + 1 frames are read).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .gpu_aug import DeviceLoader, GPUSampleBuilder
from .store import MotionStore


class StoreDataset(Dataset):
    """
    train=True: every call draws a fresh random window. A group is chosen by its
        `mix` weight, then a sequence in it proportionally to its number of windows,
        then a uniform start, so every frame of a group is equally likely. The epoch
        length is `samples_per_epoch`.
    train=False: all windows with the given stride, in order (deterministic).

    Items are raw windows {rot6d [W+1,22,6] fp16, root [W+1,3], offsets [22,3], seq_id}.
    """

    def __init__(self, store, seqs, window_size=40, train=False, mix=None, samples_per_epoch=None,
                 stride=20, label=""):
        self.store = store
        self.window_size = window_size
        self.train = train
        self._rng = None
        seqs = np.asarray(seqs)
        W = window_size
        lengths = store.idx["length"]
        n_starts = np.maximum(lengths[seqs] - W, 0)  # possible starts (1 .. T-W) per sequence

        if train:
            groups = store.idx["group"][seqs]
            self.group_names = [g for g in mix if mix[g] > 0 and (groups == g).any()]
            if not self.group_names:
                raise ValueError(f"mix {mix} matches none of the groups present: {sorted(set(groups))}")
            w = np.array([mix[g] for g in self.group_names], dtype=np.float64)
            self.group_p = w / w.sum()
            self.group_seqs = [seqs[groups == g] for g in self.group_names]
            self.group_cum = [np.cumsum(np.maximum(lengths[s] - W, 0)) for s in self.group_seqs]
            self.samples = int(samples_per_epoch)
            sizes = ", ".join(f"{g}: {len(s)} seq / {c[-1] / 1e6:.1f}M windows / p={p:.2f}"
                              for g, s, c, p in zip(self.group_names, self.group_seqs, self.group_cum, self.group_p))
            print(f"[train] random windows, {self.samples} per epoch | {sizes}")
        else:
            starts = [np.arange(1, n + 1, stride) for n in n_starts]
            self.win_seq = np.concatenate([np.full(len(a), s) for s, a in zip(seqs, starts)])
            self.win_start = np.concatenate(starts)
            print(f"[{label or 'eval'}] {len(seqs)} sequences, {len(self.win_seq)} windows (stride {stride})")

    def __len__(self):
        return self.samples if self.train else len(self.win_seq)

    def _draw(self):
        if self._rng is None:  # per DataLoader worker (and per epoch): torch seeds each differently
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        g = self._rng.choice(len(self.group_names), p=self.group_p)
        cum = self.group_cum[g]
        k = int(np.searchsorted(cum, self._rng.integers(cum[-1]), side="right"))
        n = int(cum[k] - (cum[k - 1] if k else 0))
        return int(self.group_seqs[g][k]), 1 + int(self._rng.integers(n))

    def __getitem__(self, idx):
        seq, start = self._draw() if self.train else (int(self.win_seq[idx]), int(self.win_start[idx]))
        rot6d, root, offsets = self.store.get_raw(seq, start - 1, self.window_size + 1)
        return {"rot6d": rot6d, "root": root, "offsets": offsets, "seq_id": torch.tensor(seq)}


def select_sequences(store, spec, min_len):
    """Sequences matching a config spec such as {source: AMASS, folders: [SFU], split: val}."""
    return store.select(source=spec.get("source"), subsets=spec.get("folders"), split=spec.get("split"),
                        groups=spec.get("groups"), min_len=min_len)


def make_eval_dataset(cfg, spec, stride=None, label="eval"):
    """Strided windows over the sequences of one val/test spec from the config."""
    d = cfg["data"]
    store = MotionStore(d["store_dir"])
    return StoreDataset(store, select_sequences(store, spec, d["window_size"] + 1), d["window_size"],
                        stride=stride or d.get("eval_stride", 20), label=label)


def create_dataloaders(cfg):
    """Returns (train_loader, {name: val_loader}, {name: test_loader}); batches are built on the GPU."""
    d = cfg["data"]
    aug = d.get("augmentation", {})
    W, bs, nw = d["window_size"], d["batch_size"], d["num_workers"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = GPUSampleBuilder(aug, device, fps=d.get("fps", 60.0))
    store = MotionStore(d["store_dir"])
    min_len = W + 1
    print(f"store: {store.n_seq} sequences, {store.meta['n_frames'] / 1e6:.1f}M frames; samples built on {device}")

    def loader(ds, train):
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, drop_last=train,
                        pin_memory=device.type == "cuda", prefetch_factor=2 if nw else None)
        return DeviceLoader(dl, builder, train)

    train_seqs = np.unique(np.concatenate([select_sequences(store, sp, min_len) for sp in d["train"]["sets"]]))
    train_ds = StoreDataset(store, train_seqs, W, train=True, mix=d["train"]["mix"],
                            samples_per_epoch=d["samples_per_epoch"])

    def eval_loaders(specs):
        return {name: loader(StoreDataset(store, select_sequences(store, spec, min_len), W,
                                          stride=d.get("eval_stride", 20), label=name), False)
                for name, spec in (specs or {}).items()}

    return loader(train_ds, True), eval_loaders(d.get("val")), eval_loaders(d.get("test"))
