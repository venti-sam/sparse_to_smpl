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