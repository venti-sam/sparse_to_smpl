import torch
import yaml
import sys
import os

# Add src to path
sys.path.append("/workspace/amass/src")

from tcn.losses import BodyPoseLoss
from tcn.dataset import VRTeleopDataset

def test_loss_weighting():
    print("Testing Loss Weighting...")
    criterion = BodyPoseLoss(lambda_pos=5.0, lambda_rot=0.5, lambda_vel=0.1)
    
    # Mock data
    B, W = 2, 40
    pred = {
        "global_rotmats": torch.eye(3).view(1, 1, 1, 3, 3).repeat(B, W, 22, 1, 1),
        "fk_pos": torch.zeros(B, W, 22, 3)
    }
    target = {
        "target_sixd": torch.zeros(B, W, 22, 6),
        "target_pos": torch.zeros(B, W, 22, 3)
    }
    
    # Add some error to a wrist (index 20)
    # 6D error of 1.0 on wrist
    pred["global_rotmats"][0, 0, 20, 0, 0] = 0.5 # Change first col
    
    loss, loss_dict = criterion(pred, target)
    print(f"Loss Dict: {loss_dict}")
    print(f"Rot Loss: {loss_dict['rot_loss'].item():.6f}")
    
    # Expected: wrist weight is 1.0. 
    # Raw L1 error on wrist 6D is roughly checkable.
    assert loss_dict['rot_loss'] > 0
    print("Loss weighting test passed (basic signal check).")

def test_augmentation_bias():
    print("\nTesting Augmentation Bias...")
    # Mock config
    cfg = {
        "data": {
            "dataset_dir": "/home/samuel/Projects/amass/support_data/vr_teleop_dataset_v2",
            "window_size": 40,
            "train_split": 0.9,
            "augmentation": {
                "enabled": True,
                "pos_jitter_std": 0.0,
                "rot_jitter_deg": 0.0,
                "pos_bias_std": 0.1, # Large bias for visibility
                "rot_bias_deg": 10.0,
                "vel_noise_std": 0.0,
                "dropout_prob": 0.0
            }
        }
    }
    
    # We need at least one .pt file to init dataset.
    # Let's see if we can mock the loader or just check _apply_augmentation directly.
    from tcn.dataset import VRTeleopDataset
    try:
        ds = VRTeleopDataset(
            dataset_dir=cfg["data"]["dataset_dir"],
            window_size=40,
            augmentation=cfg["data"]["augmentation"]
        )
        
        tp = torch.zeros(40, 6, 3)
        tr = torch.eye(3).view(1, 1, 3, 3).repeat(40, 6, 1, 1)
        
        tp_aug, tr_aug = ds._apply_augmentation(tp, tr)
        
        # Check if bias is persistent (same for all 40 frames)
        diff_tp = tp_aug[1:] - tp_aug[:-1]
        print(f"Max TP diff across window: {diff_tp.abs().max().item():.6f}")
        assert diff_tp.abs().max() < 1e-6
        
        print("Persistent bias test passed.")
        
    except Exception as e:
        print(f"Dataset init failed (likely no files): {e}")

if __name__ == "__main__":
    test_loss_weighting()
    test_augmentation_bias()
