"""
Loss functions v4 — with Hierarchical Rotation Loss and MPJPE metric.
"""

import torch
import torch.nn as nn

from .model import rotmat_to_sixd


class BodyPoseLoss(nn.Module):
    """
    Combined loss: L = λ_pos * L_pos + λ_rot * L_rot + λ_vel * L_vel

    L_pos: L1 on root-relative FK positions
    L_rot: Hierarchical L1 on 6D rotation representations
    L_vel: L1 on position velocity (scaled by FPS)
    """

    def __init__(
        self,
        lambda_pos=5.0,  # Increased default weight to prioritize FK positions
        lambda_rot=0.1,  # Decreased default to allow position to dominate
        lambda_vel=0.1,
        fps=60.0,
    ):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_rot = lambda_rot
        self.lambda_vel = lambda_vel
        self.fps = fps
        self.l1 = nn.L1Loss()

        # ── Hierarchical Joint Weights ──
        # Forces the network to prioritize the core kinematic chain
        weights = torch.tensor([
            1.0,  # 0: Pelvis (Root)
            1.0,  # 1: L_Hip
            1.0,  # 2: R_Hip
            1.0,  # 3: Spine1
            0.5,  # 4: L_Knee
            0.5,  # 5: R_Knee
            1.0,  # 6: Spine2
            0.5,  # 7: L_Ankle
            0.5,  # 8: R_Ankle
            1.0,  # 9: Spine3
            0.5,  # 10: L_Foot
            0.5,  # 11: R_Foot
            1.0,  # 12: Neck
            0.5,  # 13: L_Collar
            0.5,  # 14: R_Collar
            0.1,  # 15: Head
            0.5,  # 16: L_Shoulder
            0.5,  # 17: R_Shoulder
            0.5,  # 18: L_Elbow
            0.5,  # 19: R_Elbow
            1.0,  # 20: L_Wrist
            1.0,  # 21: R_Wrist
        ], dtype=torch.float32)
        
        # Register as a buffer so it maps to the correct device automatically
        self.register_buffer("joint_weights", weights)

    def forward(self, pred, target):
        """
        pred:   global_rotmats [B,W,22,3,3], fk_pos [B,W,22,3]
        target: target_pos [B,W,22,3], target_sixd [B,W,22,6]
        Returns: total_loss, loss_dict
        """
        losses = {}

        # ── Hierarchical Rotation loss (Weighted 6D L1) ──
        pred_sixd = rotmat_to_sixd(pred["global_rotmats"])
        target_sixd = target["target_sixd"]
        
        # Calculate raw L1 error: [B, W, 22, 6]
        raw_rot_error = torch.abs(pred_sixd - target_sixd)
        
        # Average the 6D representation components: [B, W, 22]
        mean_6d_error = raw_rot_error.mean(dim=-1)
        
        # Apply the hierarchical kinematic weights: [B, W, 22] * [22]
        weighted_rot_error = mean_6d_error * self.joint_weights
        
        # Final rotation loss is the mean across batch, window, and joints
        rot_loss = weighted_rot_error.mean()
        
        losses["rot_loss"] = rot_loss
        total = self.lambda_rot * rot_loss

        fk_pos = pred.get("fk_pos")
        if fk_pos is not None:
            # ── FK Position loss (root-relative) ──
            fk_rel = fk_pos - fk_pos[:, :, 0:1]
            pos_loss = self.l1(fk_rel, target["target_pos"])
            losses["pos_loss"] = pos_loss
            total = total + self.lambda_pos * pos_loss

            # ── Velocity loss (scaled by FPS) ──
            pred_vel = (fk_rel[:, 1:] - fk_rel[:, :-1]) * self.fps
            gt_vel = (
                target["target_pos"][:, 1:] - target["target_pos"][:, :-1]
            ) * self.fps
            vel_loss = self.l1(pred_vel, gt_vel)
            losses["vel_loss"] = vel_loss
            total = total + self.lambda_vel * vel_loss

        losses["total"] = total
        return total, losses


def compute_mpjpe(pred_pos, gt_pos):
    """
    Mean Per Joint Position Error in millimeters.
    pred_pos, gt_pos: [B, (W,) 22, 3] in meters.
    Returns: scalar (mm)
    """
    err = (pred_pos - gt_pos).norm(dim=-1)  # [B, (W,) 22]
    return err.mean() * 1000.0  # meters → mm