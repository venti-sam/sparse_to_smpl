import shutil
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_store import actor_split
from tcn.dataset import StoreDataset
from tcn.skeleton import fk_positions
from tcn.store import MotionStore
from tests.synthetic import make_store

W = 40


class StoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.dir = make_store(cls.tmp)
        cls.store = MotionStore(cls.dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_index_contents(self):
        idx = self.store.idx
        self.assertEqual(self.store.n_seq, 10)
        self.assertEqual(set(idx["source"]), {"AMASS", "BONES"})
        self.assertEqual(set(idx["group"]), {"amass", "bones_std", "bones_low"})
        self.assertTrue((idx["group"][idx["source"] == "AMASS"] == "amass").all())
        self.assertEqual(int(idx["length"].sum()), self.store.meta["n_frames"])

    def test_select_filters(self):
        s = self.store
        self.assertEqual(len(s.select(source="AMASS")), 4)
        self.assertEqual(len(s.select(source="AMASS", subsets=["AAA"])), 2)
        self.assertEqual(len(s.select(groups=["bones_low"])), 2)  # clips 0 and 3 are "sitting on floor"
        self.assertEqual(len(s.select(source="AMASS", min_len=10_000)), 0)

    def test_bones_split_is_by_actor(self):
        idx, bones = self.store.idx, self.store.idx["source"] == "BONES"
        by_actor = {}
        for actor, split in zip(idx["actor"][bones], idx["split"][bones]):
            by_actor.setdefault(str(actor), set()).add(str(split))
            self.assertEqual(str(split), actor_split(str(actor)))
        self.assertTrue(all(len(v) == 1 for v in by_actor.values()))

    def test_decoded_frames_match_source(self):
        s = self.store
        seq = int(s.select(source="AMASS")[0])
        gr, root, off = s.get(seq, 5, 60)
        raw_rot, raw_root, _ = s.get_raw(seq, 5, 60)
        self.assertEqual(raw_rot.dtype, torch.float16)
        src = torch.load(f"{self.tmp}/amass/{s.idx['name'][seq]}.pt")
        p0, p1 = fk_positions(src["gt_rotmat"][5:65], off), fk_positions(gr, off)
        self.assertLess(((p1 - p0).norm(dim=-1).max() * 1000).item(), 1.0)  # fp16 6D storage < 1 mm
        self.assertTrue(torch.allclose(raw_root, src["root_pos"][5:65]))

    def test_eval_windows(self):
        seqs = self.store.select(source="AMASS")
        ds = StoreDataset(self.store, seqs, W, stride=7)
        expected = sum(len(range(1, int(self.store.idx["length"][s]) - W + 1, 7)) for s in seqs)
        self.assertEqual(len(ds), expected)
        self.assertTrue(((ds.win_start >= 1) & (ds.win_start + W <= self.store.idx["length"][ds.win_seq])).all())
        item = ds[0]
        self.assertEqual(tuple(item["rot6d"].shape), (W + 1, 22, 6))

    def test_train_sampling_follows_the_mix(self):
        mix = {"amass": 0.5, "bones_std": 0.3, "bones_low": 0.2}
        ds = StoreDataset(self.store, np.arange(self.store.n_seq), W, train=True, mix=mix, samples_per_epoch=100)
        counts = {g: 0 for g in mix}
        n = 6000
        for _ in range(n):
            seq, start = ds._draw()
            counts[str(self.store.idx["group"][seq])] += 1
            self.assertTrue(1 <= start <= self.store.idx["length"][seq] - W)
        for g, p in mix.items():
            self.assertAlmostEqual(counts[g] / n, p, delta=0.03)

    def test_dataloader_batches(self):
        ds = StoreDataset(self.store, self.store.select(source="BONES"), W, stride=10)
        batch = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
        self.assertEqual(tuple(batch["rot6d"].shape), (4, W + 1, 22, 6))
        self.assertEqual(tuple(batch["offsets"].shape), (4, 22, 3))


if __name__ == "__main__":
    unittest.main()
