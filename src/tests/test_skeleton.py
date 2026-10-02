import unittest

import torch

from tcn.skeleton import MIRROR_PERM, SMPL_PARENTS, fk_positions, mirror_motion
from tests.synthetic import neutral_offsets, random_motion


class SkeletonTest(unittest.TestCase):
    def test_fk_preserves_bone_lengths(self):
        gr, off = random_motion(30), neutral_offsets()
        pos = fk_positions(gr, off)
        for i in range(1, 22):
            length = (pos[:, i] - pos[:, SMPL_PARENTS[i]]).norm(dim=-1)
            self.assertTrue(torch.allclose(length, off[i].norm().expand_as(length), atol=1e-5))

    def test_per_sample_offsets_match_shared_offsets(self):
        gr = random_motion(40).reshape(4, 10, 22, 3, 3)
        off = neutral_offsets()
        batched = fk_positions(gr, off[None].expand(4, -1, -1))
        self.assertTrue(torch.allclose(batched, fk_positions(gr, off), atol=1e-6))

    def test_mirror_twice_is_identity(self):
        gr, root, off = random_motion(20), torch.randn(20, 3), neutral_offsets()
        gr2, root2, off2 = mirror_motion(*mirror_motion(gr, root, off))
        self.assertTrue(torch.allclose(gr2, gr, atol=1e-6))
        self.assertTrue(torch.equal(root2, root) and torch.equal(off2, off))

    def test_mirror_reflects_the_body(self):
        gr, root, off = random_motion(20), torch.zeros(20, 3), neutral_offsets()
        pos = fk_positions(gr, off)
        gr_m, _, off_m = mirror_motion(gr, root, off)
        pos_m = fk_positions(gr_m, off_m)
        expected = pos[:, MIRROR_PERM] * torch.tensor([1.0, -1.0, 1.0])  # swap left/right, reflect y
        self.assertTrue(torch.allclose(pos_m - pos_m[:, :1], expected - expected[:, :1], atol=1e-5))
        self.assertTrue(bool((torch.linalg.det(gr_m) > 0.999).all()))


if __name__ == "__main__":
    unittest.main()
