# VR Pose Estimation: Training & Refactoring Log

This document tracks the architectural refactoring, optimization, and training stability updates made to the VR full-body pose estimation pipeline.

## 1. Core Architecture Updates

### Skeletal FK vs SMPL Model
- **Issue:** The original pipeline relied on the heavy `smplx` body model for forward kinematics, causing a massive 100x VRAM bottleneck and enforcing axis-angle conversions that created a 180° `acos` singularity (resulting in NaN gradients).
- **Fix:** Implemented a lightweight, pure-PyTorch `SkeletalFK` module that computes joint positions directly from 6D-derived rotation matrices using true T-pose bone offsets. This allows fully differentiable training without quaternion or axis-angle discontinuities.

### Sequence Modeling: TCN to RoPE Transformer
- **Update:** Completely migrated the sequence model from a TCN to a Transformer encoder (`TransformerBodyPose`).
- **Positional Encoding:** Added 2D Rotary Positional Embeddings (RoPE) to provide robust, relative temporal awareness across the sliding window.
- **Continuous Rotations:** The model now outputs continuous 6D rotation representations, mapped to true 3x3 matrices via Gram-Schmidt orthogonalization.

## 2. Performance & VRAM Optimizations

### Automatic Mixed Precision (AMP)
- **Issue:** Multi-head attention and dense MLPs are memory-bandwidth-bound and run comparatively slowly in pure FP32.
- **Fix:** Wrapped the forward pass and loss calculations in `torch.amp.autocast("cuda")` and utilized a `GradScaler`. This halved the VRAM footprint and approximately doubled the training speed.

### Asynchronous GPU Transfers
- **Issue:** Tensors were moved to the GPU synchronously (`.to(device)`), forcing the CPU to momentarily idle.
- **Fix:** Appended `non_blocking=True` to all data transfers to enable overlapping GPU compute and CPU page-locked memory transfers from the data loaders.

### On-the-Fly Allocation Bottlenecks
- **Issue:** RoPE frequencies and causal masks were dynamically allocated on the GPU every single forward pass, slightly slowing the run loop.
- **Fix:** Refactored `RotaryPositionalEncoding` and the Transformer causal mask to pre-compute up to a `max_seq_len` buffer. These buffers are now cleanly sliced during the forward pass, sidestepping runtime allocation overhead.

## 3. Training Stability & Usability Fixes

### Validation Metric Averaging
- **Issue:** The validation loop averaged batch losses by dividing the sum of batch means by `num_batches`. Because `drop_last=False` in validation, the final, smaller batch received a disproportionately heavy mathematical weight.
- **Fix:** Switched to a mathematically sound moving average—multiplying batch means by `batch_size` before accumulating, and dividing the final sums by `total_samples`.

### GradScaler State Checkpointing (Resume Logic)
- **Issue:** While models and optimizers were saved, the AMP `GradScaler` state was missing from checkpoints. Resuming training would abruptly reset the scaler, guaranteeing an immediate gradient overflow (NaNs) and an awkward optimizer recalibration phase.
- **Fix:** Wrote a complete `--resume` logic flow, exposing the CLI argument, updating the starting epoch, and integrating `"scaler_state_dict"` into the `checkpoint_epoch{}.pt` and `best_model.pt` save/load logic.

### Input Embedding Stabilization (LayerNorm)
- **Issue:** The 108D raw input vector concatenated normalized 6D rotations (bounded `[-1, 1]`) with FPS-scaled velocities (often exceeding `50+`). Blasting these wildly disparate scales directly into a standard linear projection destabilized the Transformer's initial embedding layers (std deviations exploded to roughly ~5.17 on initialization).
- **Fix:** Injected an `nn.LayerNorm` directly after the initial `input_proj`, successfully aggressively clamping the embedded standard deviation to a perfectly stable `1.00`.

## 4. Debugging the Loss Plateau (~106mm MPJPE)

### Diagnosing the Barrier
- **Symptom:** During training, validation MPJPE rapidly fell to ~106mm by epoch 11, but completely flatlined for the next 50+ epochs despite learning rate decay.
- **Investigation:** We initially suspected the FPS velocity scaling imbalance (`lambda_vel * L_vel` being mathematically massive on initialization at `lambda_vel=0.1`) was entirely dominating the gradients, and dropped it to `0.01`. However, true `wandb` logs cleanly revealed that the core positional, rotational, and velocity losses naturally found a perfect mathematical equilibrium within 4 epochs (`val/pos_loss` ≈ 0.05, `val/rot_loss` ≈ 0.22, `val/vel_loss` ≈ 0.089). We have since **reverted** `lambda_vel` back up to `0.1`, as the gradient signal is completely healthy and necessary for temporal smoothness.

### The True Bottleneck: Network Capacity (Underfitting)
- **Cause:** The 106mm wall was a structural underfitting limit. The AMASS dataset contains over 2,000 highly complex, diverse motion sequences. Our model, running at `embed_dim=256` and `num_layers=3`, held only **2,497,924** parameters. This tiny capacity was insufficient to memorize and generalize the vast kinematics manifold, causing it to comfortably average out its predictions and plateau at ~10cm.
- **Fix:** In `config.yaml`, we doubled the embedding width and depth:
  - `embed_dim`: 256 → 512
  - `num_layers`: 3 → 6
  - This roughly quadruples the parameter count to **~11M parameters**, granting the Transformer the actual representational capacity needed to push the MPJPE below the 100mm barrier.

## 5. Structural Loss Imbalance & LR Schedule

### The "Lazy Optimizer" and Kinematic Cascades
- **Issue:** The large model (~11M params) aggressively optimized the direct `rot_loss` and `vel_loss`. However, it essentially ignored the `pos_loss` because backpropagating through a kinematic chain of 22 matrix multiplications (`SkeletalFK`) suffers from vanishing gradients. A 1-degree rotation error at the Pelvis is fundamentally worse for position than a 1-degree error at the Hand, but `rot_loss` penalized both equally.
- **Fix:** Switched the configuration multipliers from `lambda_pos: 1.0` & `lambda_rot: 0.5` to **`lambda_pos: 5.0`** & **`lambda_rot: 0.1`**. This forcefully prioritizes the final 3D positions, ensuring the network minimizes actual joint error.

### Learning Rate Decay Timing
- **Issue:** The large model ran significantly slower per-epoch than the small model, meaning it failed to hit the first `MultiStepLR` decay milestone in the same wall-time footprint. It continued to bounce around the local minimum at `1e-4` instead of settling.
- **Fix:** Updated `scheduler_milestones` to `[15, 30, 45, 60, 90, 120]` to introduce learning rate drops much earlier in the run.

### 1. Loss Function Overhaul
* **Hierarchical Rotation Loss:** Replaced standard uniform `L1Loss` with a weighted L1 calculation for 6D rotations.
    * *Reasoning:* The model was aggressively optimizing leaf joint rotations (wrists, ankles) while leaving core joints (pelvis, spine) sloppy, causing massive 3D positional errors down the kinematic chain.
    * *Weights:* Core joints (1.0), Mid joints (0.5), Leaf joints (0.1).
* **Loss Weight Rebalancing:** Updated `config.yaml` to heavily prioritize positional output (`lambda_pos: 5.0`) over pure rotation (`lambda_rot: 0.1`).

### 2. Optimizer & Scheduler Stabilization (Large Transformers)
* **AdamW & Weight Decay Decoupling:** Replaced standard `Adam` with `AdamW`. Separated parameters into two groups to ensure 1D tensors (LayerNorm weights and biases) receive `0.0` weight decay. Shrinking LayerNorm parameters previously contributed to network collapse.
* **Step-Based LR Warmup:** Replaced immediate `1.0e-4` start with a `LambdaLR` linear warmup over the first 2,000 steps. 
    * *Reasoning:* Large transformers suffer from gradient shattering at initialization. Warmup allows the variance to settle before taking full-sized steps.

### 3. Training Loop Efficiency & Bug Fixes
* **Corrected Validation Averaging:** Fixed a statistical bug where the final, smaller validation batch was given disproportionate weight due to mean-of-means averaging. Now accumulates sum and divides by total samples.
* **Async Data Transfers:** Added `non_blocking=True` to `.to(device)` calls to prevent the CPU from halting during GPU memory transfers.
* **GradScaler Checkpointing:** Added `scaler_state_dict` to the `.pt` checkpoint saves. Resuming training without the scaler state previously reset the AMP scale factor to its massive default, causing immediate `NaN` gradients.

## 6. Dataset Optimization & Generalization

### Rotation Representation (Redundancy Removal)
* **Issue:** `extract_vr_data.py` was converting 3x3 rotation matrices to quaternions for storage, and `dataset.py` was converting them back to 3x3 matrices for model input. This added unnecessary CPU overhead and potential precision loss.
* **Fix:** Updated the extraction pipeline to save 3x3 rotation matrices directly (`tracker_rotmat` and `gt_rotmat`). Updated the dataset loader to ingest these matrices natively, streamlining the data loading path.

### Folder-based Splitting (Pre-emptive Leakage Prevention)
* **Issue:** Randomly splitting sequences into Train/Val/Test risks "data leakage," where the model memorizes specific lab-related biases or actor-specific motions rather than learning general human kinematics.
* **Fix:** Implemented a prefix-based filename system during extraction (e.g., `CMU_seq_00001.pt`). Updated `config.yaml` and `dataset.py` to support explicit folder-based splits (e.g., training on CMU and KIT, validating on SFU). This provides a "Gold Standard" test of cross-dataset generalization.
* **Verified Folders:** Confirmed successful processing of newly added datasets: `BMLmovi`, `BMLrub`, `CMU`, `HDM05`, `KIT`, `PosePrior`, `SFU`, `SSM`.

## 7. Data fixes and BONES-SEED (Oct 2026)

The 100 mm and later 24 mm plateaus were partly data problems, not model problems.

- **Up-axis bug.** AMASS joints are already Z-up, but the v1 extraction applied a second Y-up to Z-up permutation, so the stored data had X as its up axis. Heading canonicalization (yaw about Z) then tipped the body over: only 1.3% of validation windows ended up upright, and 28% had a degenerate heading. `convert_dataset_v2.py` undoes it (92.9% upright afterwards). A v1-trained model scored 28.2 mm on its own convention but 33.8 mm on true Z-up input, which is what the live pipeline feeds it.
- **Neutral skeleton.** FK used one neutral skeleton against per-actor ground truth, a 25.7 mm mean error floor (up to 55 mm). Training now uses per-actor bone offsets, with random body-size scaling.
- **Tracker model.** Trackers were exact joint copies. They are now mounted like real Vive trackers (belt buckle, shin strap, HMD, stick-held controllers) with jitter and short dropouts.
- **BONES-SEED.** 71k clips (144 h, 522 actors) are converted to the same form by `convert_bvh_v2.py`, using the SMPL neutral body: LAFAN1 scored 68 mm on its own skeleton but 34 mm on the SMPL body. Mirrored clips are exact reflections (0.4 deg), so they are skipped and mirroring is done on the fly. An AMASS-only model overfit after about 12 passes (val minimum epoch 12) and was 14 mm worse on unseen BONES actors than a 2-epoch AMASS+BONES mix.
- **Training pipeline.** All datasets live in one memory-mapped store; workers slice raw windows and the tracker simulation runs on the GPU (a loader that shipped pre-built batches was 7x slower and overflowed Docker's 64 MB /dev/shm).

AMASS-only baseline (best epoch 12): AMASS val 23.8 mm, SSM test 23.5 mm, unseen BONES 32.2 mm, LAFAN1 32.5 mm. Mixed run, 2 epochs: 24.9 / 24.5 / 18.4 / 29.2 mm.

## 8. Mixed run (80 epochs) and last-frame check (Oct 2 2026)

The model is not a TCN. The `tcn/` package name is a leftover from the first version (see section 1): `model.py` is a 6-layer causal RoPE transformer (512 wide, 8 heads, 19.3M parameters) over a 40-frame window at 60 Hz. The plateaus above were data problems, so an architecture swap (TCN, GRU) is not expected to move the numbers by more than a few mm.

Run `x6d7gh1n` (AMASS 40% / BONES standing 30% / BONES low-pose 30%, 2M windows per epoch, 80 epochs, 07:23 to about 13:50 UTC, about 5 min per epoch). It finished cleanly and the best checkpoint is the last epoch. Validation MPJPE:

| epoch | train loss | val loss | amass_val | bones_val |
|---|---|---|---|---|
| 1 | 0.2011 | 0.1051 | 28.8 mm | 19.8 mm |
| 2 | 0.1191 | 0.0984 | 26.8 mm | 18.1 mm |
| 3 | 0.1122 | 0.0964 | 26.0 mm | 18.0 mm |
| 8 | 0.0999 | 0.0936 | 25.9 mm | 17.2 mm |
| 9 | 0.0930 | 0.0739 | 22.9 mm | 16.6 mm |
| 22 | 0.0705 | 0.0695 | 21.9 mm | 15.5 mm |
| 45 | 0.0644 | 0.0688 | 21.8 mm | 15.3 mm |
| 80 | 0.0599 | 0.0675 | 21.4 mm | 14.9 mm |

The earlier 2-epoch mixed run (`fev6ooak`) had 25.9 / 20.0 mm at epoch 1 and 24.9 / 18.4 mm at epoch 2, so this run is slightly behind on AMASS early on, within epoch-1 noise.

- **Step at epoch 9.** Val loss fell from 0.0936 to 0.0739 in one epoch and train loss dropped with it (0.0999 to 0.0930 to 0.0820). It looks like a phase transition; cause not investigated.
- **Plateau and LR decay.** Val loss was flat at 0.0685-0.0694 from about epoch 28 while train kept falling. Each LR halving (epochs 45, 60, 70) gave a small step; the first was the largest (val 0.0694 at epoch 44 to 0.0679 at epoch 46). The last 35 epochs improved val loss by about 0.001 and AMASS val by 0.4 mm, so the run could have stopped near epoch 50 and saved about 2.5 h. Val never rose, so there is no overfitting; the final train/val gap (0.0599 vs 0.0675) is partly train-only augmentation.
- **Checkpoints.** `checkpoints_mix/` holds `checkpoint_epoch{10..80}.pt` and `best_model.pt` (identical to epoch 80).

Final test scores (best model, MPJPE, root-relative), against the earlier baselines from section 7:

| test set | this run (80 ep) | 2-epoch mix | AMASS-only |
|---|---|---|---|
| amass_test (SSM) | 20.2 mm | 24.5 | 23.5 |
| bones_test (unseen actors) | 15.9 mm | 18.4 | 32.2 |
| lafan1 | 25.5 mm | 29.2 | 32.5 |

Longer training on the mix improved every set by 2.5-4.3 mm over the 2-epoch mix. Per-joint error at the last frame (checkpoint epoch 80, 8192 random windows per set): elbows are the worst joints on AMASS and BONES (30-43 mm), then feet (25-32 mm) and wrists (22-35 mm); on LAFAN1 the feet are the worst (48-50 mm). Elbows are not tracked and their swivel is underdetermined by hand and head poses, which is the likely reason. The LAFAN1 foot error probably comes from the BVH-to-SMPL foot geometry (section 7), but that is a guess.

**Train/deploy frame check.** Training and validation average the loss and MPJPE over all 40 output frames, but the live estimator (`htc_vive_pro2_socket/.../estimator.py`) uses only the last one (`fk_pos[0, -1]`). The suspicion was that early frames, which have little causal context, make the reported mm optimistic. Measured on the epoch-3 best checkpoint (4096 random val windows per set, MPJPE by frame position):

| set | all-frame mean | last frame | frames >= 20 | frame 0 |
|---|---|---|---|---|
| amass_val | 26.0 mm | 26.0 | 25.8 | 28.3 |
| bones_val | 18.0 mm | 17.8 | 17.8 | 19.9 |
| lafan1 | 28.6 mm | 29.6 | 28.8 | 31.4 |

- The suspicion was mostly wrong. Error is nearly flat across the window; only frames 0-2 are 2-3 mm worse. Velocity features and window-start heading canonicalization give even early frames enough to go on, so the all-frame mean tracks the deployed number to within 0.2 mm on AMASS and BONES (1.0 mm on LAFAN1). No loss reweighting is needed.
- At epoch 3, LAFAN1 was the exception: its error grew with context (28.0 mm at frame 9, 29.6 mm at frame 39). In the final model this is nearly gone (25.4 mm at frame 9, 25.8 mm at frame 39), so it was an early-training effect, not a standing problem.
- Live pipeline settings match training (`window_size: 40`, `fps: 60.0`).

Re-measured on the final checkpoint (epoch 80, 8192 random windows per set, 199 for amass_test); the conclusion holds, error is flat across the window and the last frame is as good as or better than the mean:

| set | all-frame mean | last frame | frame 0 |
|---|---|---|---|
| amass_val | 21.4 mm | 21.2 | 21.9 |
| bones_val | 14.9 mm | 14.6 | 15.6 |
| amass_test | 20.2 mm | 19.6 | 21.1 |
| bones_test | 15.8 mm | 15.5 | 16.2 |
| lafan1 | 25.6 mm | 25.8 | 25.5 |

Ideas not yet tried: a shorter schedule (the run was flat after about epoch 50; e.g. 50 epochs with the first LR drop around epoch 30); log a `last_mm` metric in `validate()` so the selection number is the deployment number; a mild loss ramp (weight 0.5 on frames 0-9, 1.0 after), which the final curve shows is unnecessary; a longer window (90-120 frames, within the 512-frame RoPE limit) for ambiguous poses such as sitting versus standing, judged on last-frame error.

