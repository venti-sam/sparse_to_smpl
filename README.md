# sparse_to_smpl

Sparse-to-dense body pose for VR teleoperation: a causal transformer that predicts a full 22-joint SMPL body from 6 VR trackers (pelvis, both ankles, head, both hands). It is trained on motion capture (AMASS and BONES-SEED) with simulated Vive trackers, and feeds the live pipeline in `htc_vive_pro2_socket`, which retargets the body to the G1.

Everything below runs **inside the Docker container** from `/workspace/amass/src` unless noted. (The image and container keep their original names `amass_env` / `amass_container`, and the repo is mounted at `/workspace/amass` inside it.)

## 0. Prerequisites

- Docker with NVIDIA GPU support
- AMASS `.npz` files in `support_data/amass_npz/<dataset>/` (BMLmovi, BMLrub, CMU, HDM05, KIT, PosePrior, SFU, SSM)
- SMPL-H body model in `support_data/body_models/smplh/neutral/`

## 1. Container

```bash
./docker_scripts/start.sh     # build image + start container (host)
./docker_scripts/join.sh      # shell into it (host)
./docker_scripts/stop.sh      # stop it (host)
```

The container restarts with the machine; the repo is mounted at `/workspace/amass`.

## 2. Build the data (once)

```bash
cd /workspace/amass/src

# AMASS
python3 convert_pkl_to_npz.py       # only if support_data/body_models/smplh/neutral/model_clean.pkl is missing
python3 extract_vr_data.py          # AMASS -> support_data/vr_teleop_dataset (v1), ~2 min
python3 convert_dataset_v2.py       # v1 -> support_data/vr_teleop_dataset_v2, ~2 min

# BONES-SEED (optional, 71k extra clips; research-use licence, never commit it)
# Accept the licence at https://huggingface.co/datasets/bones-studio/seed and save a read token to
# ~/.cache/huggingface/token (or export HF_TOKEN). The archive is 45 GB and takes ~70 min.
python3 bones_pipeline.py download  # resumable + sha256-checked
python3 bones_pipeline.py convert   # skips mirrored clips, ~15 min -> support_data/bones_uniform_v2 (24 GB)
python3 bones_pipeline.py index    # writes index.json used for sampling (needs pandas + pyarrow)

# optional extra test set: python3 convert_bvh_v2.py --src <lafan1 folder> --dst ../support_data/lafan1_v2

# pack everything into one memory-mapped store (13 s)
python3 build_store.py              # -> support_data/store_v2
```

Training reads the **store** (`data.store_dir` in the config). For AMASS-only training, remove the BONES entry and the `bones_*` groups from `data.train` in the config.

## 3. Train

```bash
cd /workspace/amass/src
wandb login                         # one-time
python3 -m tcn.train                # uses tcn/config.yaml
```

```bash
python3 -m tcn.train --config tcn/config.yaml --resume ../checkpoints_mix/checkpoint_epoch20.pt
WANDB_MODE=disabled python3 -m tcn.train      # run without wandb
```

- Settings: `src/tcn/config.yaml` (`data.train.mix` = share of AMASS / BONES-standing / BONES-low-pose windows, `training.epochs`, augmentation, tracker mount).
- An "epoch" is `data.samples_per_epoch` random windows (2M), about 5 min. Mirroring, body-size scaling and the Vive tracker simulation run on the GPU.
- Output in `checkpoints_mix/`: `best_model.pt` (best mean validation loss over AMASS and BONES) and `checkpoint_epochN.pt` every 10 epochs.
- At the end the best checkpoint is scored on the held-out test sets (AMASS SSM, unseen BONES actors, LAFAN1).
- Tests (synthetic data, a few seconds, no GPU needed): `python3 -m unittest discover -s tests -t .`

## 4. Visualize a checkpoint (RViz2)

```bash
# terminal 1
rviz2     # Fixed Frame: map, add MarkerArray on /vr_pose/skeleton_markers

# terminal 2
cd /workspace/amass/src
python3 infer_rviz.py --checkpoint ../checkpoints_mix/best_model.pt [--test-set bones_test]
```

Green = prediction, red = ground truth (offset 1 m in X). `--test-set` is a key of `data.test` in the config (default `amass_test`).

## 5. Deploy to the live pipeline

```bash
# host
cp checkpoints_mix/best_model.pt ~/Projects/htc_vive_pro2_socket/src/skeletal_dense/checkpoints/<name>.pt
```

Then launch `skeletal_dense` with `checkpoint:=<that file>` (see the `htc_vive_pro2_socket` README). A v2-trained model already includes the tracker mounting, so set `pelvis_offset`, `ankle_offset` and `ankle_drop` to `0` in `skeletal_dense.launch.py`.

## Notes

- Frame: **Z-up, X forward**. Checkpoints trained on v1 data (including `all_data_ckpt/best_model.pt`) used X-up and are **not** compatible with v2 data or `infer_rviz.py`; they stay usable only in the old live pipeline.
- `all_data_ckpt/best_model.pt` is the currently deployed (v1) model.
- Training history and past debugging: `TRAINING_LOG.md`.

## Layout

```
Dockerfile, docker_scripts/       container
src/extract_vr_data.py            AMASS -> v1 .pt (needs smplx)
src/convert_dataset_v2.py         v1 -> v2 (Z-up fix, per-actor bone offsets)
src/convert_bvh_v2.py             BVH mocap -> v2 (LAFAN1, BONES-SEED)
src/bones_pipeline.py             BONES-SEED download / convert / index
src/build_store.py                v2 datasets -> one memory-mapped store
src/infer_rviz.py                 RViz playback of a checkpoint
src/tcn/                          the model and training code
  config.yaml                       data mix, augmentation, tracker mount, schedule
  train.py, model.py, losses.py     training loop, transformer, loss + MPJPE
  dataset.py, store.py              random/strided windows from the store
  gpu_aug.py                        on-GPU tracker simulation and augmentation
  skeleton.py, rotations.py, trackers.py   joint tree + FK + mirror, rotation helpers, tracker mounts
src/tests/                        unit tests (python3 -m unittest discover -s tests -t .)
support_data/                     amass_npz, body_models, vr_teleop_dataset_v2, bones_*, store_v2 (not in git)
checkpoints_mix/                  new checkpoints (created by training, not in git)
all_data_ckpt/                    deployed v1 model (not in git)
```
