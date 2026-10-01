"""
Training script v4 for VR full-body pose estimation.

Key upgrades:
- AdamW with weight decay decoupling (safe for LayerNorm)
- Step-based learning rate warmup for large Transformers
- Non-blocking async GPU memory transfers
- Corrected weighted batch averaging for validation metrics
- Full resume logic including GradScaler states
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .dataset import create_dataloaders
from .model import TransformerBodyPose
from .losses import BodyPoseLoss, compute_mpjpe


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── wandb ──
    import wandb

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        config=cfg,
    )

    # ── Data ──
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    print(
        f"Train: {len(train_loader.dataset)} windows, "
        f"Val: {len(val_loader.dataset)} windows, "
        f"Test: {len(test_loader.dataset) if test_loader else 0} windows"
    )

    # ── Model ──
    model = TransformerBodyPose(
        input_dim=cfg["model"]["input_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    wandb.log({"num_params": num_params})

    # ── Optimizer (AdamW + Parameter Grouping) ──
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for biases or 1D tensors (like LayerNorm parameters)
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optim_groups, lr=cfg["training"]["lr"])
    scaler = torch.amp.GradScaler("cuda")

    # ── Schedulers (Step-based Warmup + MultiStep) ──
    steps_per_epoch = len(train_loader)
    warmup_steps = 2000  # Number of batches for linear warmup
    step_milestones = [
        m * steps_per_epoch for m in cfg["training"]["scheduler_milestones"]
    ]

    def lr_lambda(step):
        # 1. Linear Warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # 2. MultiStep Decay
        decay_factor = 1.0
        for milestone in step_milestones:
            if step >= milestone:
                decay_factor *= cfg["training"]["scheduler_gamma"]
        return decay_factor

    scheduler = LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_val_loss = float("inf")

    # ── Resume Logic ──
    if cfg.get("resume"):
        print(f"Resuming from checkpoint: {cfg['resume']}")
        checkpoint = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"  → Resumed at epoch {start_epoch} (best val: {best_val_loss:.4f})")

    # ── Loss ──
    criterion = BodyPoseLoss(
        lambda_pos=cfg["training"]["lambda_pos"],
        lambda_rot=cfg["training"]["lambda_rot"],
        lambda_vel=cfg["training"]["lambda_vel"],
        fps=cfg["data"].get("fps", 60.0),
    ).to(device)

    # ── Directories ──
    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    log_interval = cfg["training"]["log_interval"]
    val_interval = cfg["training"]["val_interval"]

    # ── Training Loop ──
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        model.train()
        epoch_losses = {}
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:3d}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, batch in enumerate(pbar):
            # Using non_blocking=True for async transfers
            inputs = batch["input"].to(device, non_blocking=True)
            target_pos = batch["target_pos"].to(device, non_blocking=True)
            target_sixd = batch["target_sixd"].to(device, non_blocking=True)
            bone_offsets = batch["bone_offsets"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                local_rotmats, global_rotmats, fk_pos = model(
                    inputs, bone_offsets=bone_offsets
                )

                pred = {
                    "global_rotmats": global_rotmats,
                    "fk_pos": fk_pos,
                }
                target = {
                    "target_pos": target_pos,
                    "target_sixd": target_sixd,
                }
                loss, loss_dict = criterion(pred, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if cfg["training"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg["training"]["grad_clip"],
                )

            scaler.step(optimizer)
            scaler.update()

            # Step the scheduler every batch for smooth warmup
            scheduler.step()

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v.item()
            num_batches += 1

            # Live terminal display
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "pos": f"{loss_dict.get('pos_loss', torch.tensor(0)).item():.3f}",
                    "rot": f"{loss_dict.get('rot_loss', torch.tensor(0)).item():.3f}",
                }
            )

            if (batch_idx + 1) % log_interval == 0:
                step = epoch * steps_per_epoch + batch_idx
                log = {f"train/{k}": v / num_batches for k, v in epoch_losses.items()}
                log["lr"] = scheduler.get_last_lr()[0]
                wandb.log(log, step=step)

        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        # ── Validation ──
        if (epoch + 1) % val_interval == 0:
            val_loss, val_mpjpe = validate(model, val_loader, criterion, device)

            log = {"epoch": epoch + 1}
            for k, v in val_loss.items():
                log[f"val/{k}"] = v
            for k, v in epoch_losses.items():
                log[f"train/{k}"] = v
            log["val/mpjpe_mm"] = val_mpjpe
            log["lr"] = scheduler.get_last_lr()[0]
            wandb.log(log)

            print(
                f"Epoch {epoch+1:3d} | "
                f"train={epoch_losses.get('total', 0):.4f} | "
                f"val={val_loss.get('total', 0):.4f} | "
                f"mpjpe={val_mpjpe:.1f}mm | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            if val_loss.get("total", float("inf")) < best_val_loss:
                best_val_loss = val_loss["total"]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "val_loss": best_val_loss,
                        "config": cfg,
                    },
                    os.path.join(save_dir, "best_model.pt"),
                )
                print(f"  \u2713 Best model (val={best_val_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": cfg,
                },
                os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"),
            )

    # ── Held-out test: best-by-val checkpoint, evaluated once ──
    best_path = os.path.join(save_dir, "best_model.pt")
    if test_loader is not None and os.path.exists(best_path):
        best = torch.load(best_path, map_location=device)
        model.load_state_dict(best["model_state_dict"])
        test_loss, test_mpjpe = validate(model, test_loader, criterion, device)
        wandb.log(
            {
                **{f"test/{k}": v for k, v in test_loss.items()},
                "test/mpjpe_mm": test_mpjpe,
                "test/best_epoch": best["epoch"],
            }
        )
        print(
            f"\nTest (best model, epoch {best['epoch']}): "
            f"loss={test_loss.get('total', 0):.4f} | mpjpe={test_mpjpe:.1f}mm"
        )

    wandb.finish()
    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    losses = {}
    mpjpe_sum = 0.0
    total_samples = 0

    for batch in val_loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target_pos = batch["target_pos"].to(device, non_blocking=True)
        target_sixd = batch["target_sixd"].to(device, non_blocking=True)
        bone_offsets = batch["bone_offsets"].to(device, non_blocking=True)
        batch_size = inputs.size(0)

        with torch.amp.autocast("cuda"):
            _, global_rotmats, fk_pos = model(inputs, bone_offsets=bone_offsets)

            pred = {
                "global_rotmats": global_rotmats,
                "fk_pos": fk_pos,
            }
            target = {
                "target_pos": target_pos,
                "target_sixd": target_sixd,
            }
            _, loss_dict = criterion(pred, target)

        for k, v in loss_dict.items():
            # Accumulate sum (mean * batch_size)
            losses[k] = losses.get(k, 0.0) + v.item() * batch_size

        # MPJPE (root-relative)
        if fk_pos is not None:
            fk_rel = fk_pos - fk_pos[:, :, 0:1]
            # Accumulate sum (mean * batch_size)
            mpjpe_sum += compute_mpjpe(fk_rel, target_pos).item() * batch_size

        total_samples += batch_size

    for k in losses:
        losses[k] /= max(total_samples, 1)
    mpjpe_avg = mpjpe_sum / max(total_samples, 1)

    model.train()
    return losses, mpjpe_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume from",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg["resume"] = args.resume

    train(cfg)