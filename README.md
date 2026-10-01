# AMASS → VR Full-Body Pose Estimation

Trains a Transformer that predicts a 22-joint SMPL body from 6 VR trackers (pelvis, both ankles, head, both hands). Used by the live pipeline in `htc_vive_pro2_socket` to feed GMR.

Everything below runs **inside the Docker container** from `/workspace/amass/src` unless noted.

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

## 2. Build the dataset (once)

```bash
cd /workspace/amass/src

# (only if support_data/body_models/smplh/neutral/model_clean.pkl is missing)
python3 convert_pkl_to_npz.py

python3 extract_vr_data.py          # AMASS -> support_data/vr_teleop_dataset (v1), ~2 min
python3 convert_dataset_v2.py       # v1 -> support_data/vr_teleop_dataset_v2, ~2 min
```

Training reads **v2** only. The v1 folder is left untouched and is not used by `tcn.train`.

## 3. Train

```bash
cd /workspace/amass/src
wandb login                         # one-time
python3 -m tcn.train                # uses tcn/config.yaml
```

```bash
python3 -m tcn.train --config tcn/config.yaml --resume ../checkpoints_v2/checkpoint_epoch20.pt
WANDB_MODE=disabled python3 -m tcn.train      # run without wandb
```

- Settings: `src/tcn/config.yaml` (`training.epochs`, train/val/test folders, `data.augmentation.mount`).
- Output in `checkpoints_v2/`: `best_model.pt` (best val loss) and `checkpoint_epochN.pt` every 10 epochs.
- At the end of training the best checkpoint is scored once on the test folders (`test/mpjpe_mm` in wandb, printed in the log).
- Quick sanity check of the loss/augmentation code: `python3 -m tcn.verify_env`

## 4. Visualize a checkpoint (RViz2)

```bash
# terminal 1
rviz2     # Fixed Frame: map, add MarkerArray on /vr_pose/skeleton_markers

# terminal 2
cd /workspace/amass/src
python3 infer_rviz.py --checkpoint ../checkpoints_v2/best_model.pt
```

Green = prediction, red = ground truth (offset 1 m in X).

## 5. Deploy to the live pipeline

```bash
# host
cp checkpoints_v2/best_model.pt ~/Projects/htc_vive_pro2_socket/src/skeletal_dense/checkpoints/<name>.pt
```

Then launch `skeletal_dense` with `checkpoint:=<that file>` (see the `htc_vive_pro2_socket` README). A v2-trained model already includes the tracker mounting, so set `pelvis_offset`, `ankle_offset` and `ankle_drop` to `0` in `skeletal_dense.launch.py`.

## Notes

- Frame: **Z-up, X forward**. Checkpoints trained on v1 data (including `all_data_ckpt/best_model.pt`) used X-up and are **not** compatible with v2 data or `infer_rviz.py`; they stay usable only in the old live pipeline.
- `all_data_ckpt/best_model.pt` is the currently deployed (v1) model.
- Training history and past debugging: `TRAINING_LOG.md`.

## Layout

```
Dockerfile, docker_scripts/       container
src/extract_vr_data.py            AMASS -> v1 .pt
src/convert_dataset_v2.py         v1 -> v2 (Z-up fix, per-actor bone offsets)
src/infer_rviz.py                 RViz playback of a checkpoint
src/tcn/                          model.py, dataset.py, skeleton.py, losses.py, train.py, config.yaml
support_data/                     amass_npz, body_models, vr_teleop_dataset(_v2)
checkpoints_v2/                   new checkpoints (created by training)
all_data_ckpt/                    deployed v1 model
```
