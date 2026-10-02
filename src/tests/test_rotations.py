import math
import unittest

import torch

from tcn.rotations import axis_angle_to_rotmat, extract_yaw, rotmat_to_sixd, sixd_to_rotmat
from tests.synthetic import random_motion


class RotationsTest(unittest.TestCase):
    def test_sixd_round_trip(self):
        R = random_motion(50)
        self.assertTrue(torch.allclose(sixd_to_rotmat(rotmat_to_sixd(R)), R, atol=1e-5))

    def test_sixd_output_is_a_rotation(self):
        R = sixd_to_rotmat(torch.randn(100, 6))
        eye = torch.eye(3).expand_as(R)
        self.assertTrue(torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5))
        self.assertTrue(bool((torch.linalg.det(R) > 0.999).all()))

    def test_axis_angle(self):
        R = axis_angle_to_rotmat(torch.tensor([0.0, 0.0, math.pi / 2]))  # 90 deg about z
        self.assertTrue(torch.allclose(R @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1.0, 0]), atol=1e-6))
        self.assertTrue(torch.allclose(axis_angle_to_rotmat(torch.zeros(3)), torch.eye(3), atol=1e-6))

    def test_extract_yaw_is_the_heading_about_z(self):
        yaw = torch.tensor([0.4, -1.2, 2.5])
        R = axis_angle_to_rotmat(torch.stack([torch.zeros(3), torch.zeros(3), yaw], dim=-1))
        self.assertTrue(torch.allclose(extract_yaw(R), R, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
