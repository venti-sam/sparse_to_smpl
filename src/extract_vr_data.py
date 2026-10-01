import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import smplx
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------
# 1. CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BODY_MODEL_PATH = "../support_data/body_models/smplh/neutral/model_clean.pkl"
AMASS_DIR = "../support_data/amass_npz"
OUTPUT_DIR = "../support_data/vr_teleop_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standard 22-joint SMPL kinematic tree (Parents array)
SMPL_PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    7,
    8,
    9,
    10,
    #  0  1  2  3  4  5  6  7  8  9  10
    # pel Lh Rh sp Lk Rk sp2 La Ra sp3 nk
    # Note: indices 7,8 are L/R ankle, 9 is spine3, 10 is neck
]

# Full 22-joint SMPL parent indices
SMPL_PARENTS = [
    -1,  # 0  Pelvis
    0,  # 1  L_Hip
    0,  # 2  R_Hip
    0,  # 3  Spine1
    1,  # 4  L_Knee
    2,  # 5  R_Knee
    3,  # 6  Spine2
    4,  # 7  L_Ankle
    5,  # 8  R_Ankle
    6,  # 9  Spine3
    7,  # 10 L_Foot
    8,  # 11 R_Foot
    9,  # 12 Neck
    9,  # 13 L_Collar
    9,  # 14 R_Collar
    12,  # 15 Head
    13,  # 16 L_Shoulder
    14,  # 17 R_Shoulder
    16,  # 18 L_Elbow
    17,  # 19 R_Elbow
    18,  # 20 L_Wrist
    19,  # 21 R_Wrist
]

# 6 VR Tracker joints mapped to SMPL joint indices
# pelvis, L_Ankle, R_Ankle, Head, L_Wrist, R_Wrist
TRACKER_IDX = [0, 7, 8, 15, 20, 21]

# Body part names (matching GMR human_data keys)
TRACKER_NAMES = [
    "pelvis",
    "left_ankle",
    "right_ankle",
    "head",
    "left_hand",
    "right_hand",
]

JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
]

TARGET_FPS = 60.0


# ---------------------------------------------------------
# 2. COORDINATE CONVERSION: SMPL Y-up → Z-up
# ---------------------------------------------------------
# SMPL/AMASS uses Y-up:  X=right, Y=up,   Z=forward
# MuJoCo/GMR uses Z-up:  X=right, Y=front, Z=up
# Axis permutation: X→Y, Y→Z, Z→X  (same as vive_config.yaml)
#
# NOTE: AMASS joints from smplx are ALREADY Z-up (the up axis lives in
# global_orient), so applying this permutation to them leaves the saved
# dataset with X as the up axis. convert_dataset_v2.py undoes it. The
# permutation is left as-is here so extraction output stays reproducible.
YUPTOZUP = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float32,
)


def positions_y_to_z(pos):
    """Convert positions from Y-up to Z-up. pos: [..., 3]"""
    return pos @ YUPTOZUP.T


def rotmats_y_to_z(rotmats):
    """Convert 3x3 rotation matrices from Y-up to Z-up. rotmats: [..., 3, 3]"""
    H = torch.tensor(YUPTOZUP, device=rotmats.device, dtype=rotmats.dtype)
    # R_zup = H @ R_yup @ H^T
    return H @ rotmats @ H.T


# Note: rotmats_to_quat_wxyz is no longer used for saving, but kept for reference if needed
def rotmats_to_quat_wxyz(rotmats_np):
    """Convert rotation matrices [..., 3, 3] to quaternions [..., 4] in [w,x,y,z]."""
    orig_shape = rotmats_np.shape[:-2]
    flat = rotmats_np.reshape(-1, 3, 3)
    quats = R.from_matrix(flat).as_quat()  # scipy returns [x,y,z,w]
    # Reorder to [w,x,y,z] (scalar-first, matching GMR convention)
    quats = quats[:, [3, 0, 1, 2]]
    return quats.reshape(orig_shape + (4,))


# ---------------------------------------------------------
# 3. HELPER: AXIS-ANGLE → ROTATION MATRIX (Rodrigues)
# ---------------------------------------------------------
def aa2matrot(aa):
    """
    Convert axis-angle vectors to rotation matrices.
    Input:  [N, 3] axis-angle
    Output: [N, 3, 3] rotation matrices
    """
    angle = torch.norm(aa, dim=1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle

    K = torch.zeros(aa.shape[0], 3, 3, device=aa.device, dtype=aa.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0)
    sin_a = torch.sin(angle).unsqueeze(2)
    cos_a = torch.cos(angle).unsqueeze(2)

    return eye + sin_a * K + (1 - cos_a) * (K @ K)


# ---------------------------------------------------------
# 4. HELPER: COMPUTE GLOBAL ROTATIONS
# ---------------------------------------------------------
def get_global_rotations(local_rot_mats):
    """
    Accumulates local rotation matrices along the kinematic tree
    to get global rotation matrices w.r.t the world origin.
    Input:  [T, 22, 3, 3]
    Output: [T, 22, 3, 3]
    """
    global_rot_mats = torch.zeros_like(local_rot_mats)
    global_rot_mats[:, 0] = local_rot_mats[:, 0]

    for i in range(1, 22):
        parent = SMPL_PARENTS[i]
        global_rot_mats[:, i] = torch.matmul(
            global_rot_mats[:, parent], local_rot_mats[:, i]
        )

    return global_rot_mats


# ---------------------------------------------------------
# 4b. HELPER: PER-ACTOR BONE OFFSETS (T-pose)
# ---------------------------------------------------------
@torch.no_grad()
def compute_bone_offsets(bm, betas):
    """
    Rest-pose bone offsets for a batch of body shapes, in Z-up.
    betas: [N, 16] -> [N, 22, 3], offsets[i] = joint[i] - joint[parent[i]].
    The root row is the pelvis joint itself (unused by root-relative FK).
    """
    n = betas.shape[0]
    zeros = lambda *s: torch.zeros(*s, device=betas.device, dtype=betas.dtype)
    body = bm(
        betas=betas,
        body_pose=zeros(n, 69),
        global_orient=zeros(n, 3),
        transl=zeros(n, 3),
    )
    joints = body.joints[:, :22, :]  # [N, 22, 3] Y-up
    parents = torch.tensor(SMPL_PARENTS[1:], device=betas.device)
    offsets = joints.clone()
    offsets[:, 1:] = joints[:, 1:] - joints[:, parents]
    H = torch.tensor(YUPTOZUP, device=betas.device, dtype=betas.dtype)
    return offsets @ H.T


# ---------------------------------------------------------
# 5. MAIN EXTRACTION ROUTINE
# ---------------------------------------------------------
def extract_dataset():
    print("Loading SMPL Body Model via smplx...")
    bm = smplx.SMPL(
        model_path=BODY_MODEL_PATH,
        gender="neutral",
        num_betas=16,
    ).to(DEVICE)
    print(f"Model loaded on {DEVICE}")

    npz_files = glob.glob(os.path.join(AMASS_DIR, "**/*_poses.npz"), recursive=True)
    print(f"Found {len(npz_files)} motion sequences.\n")

    saved, skipped, errored = 0, 0, 0
    pbar = tqdm(enumerate(npz_files), total=len(npz_files), unit="seq")

    for file_idx, npz_path in pbar:
        try:
            bdata = np.load(npz_path)

            # --- Handle Framerate Downsampling ---
            mocap_fps = bdata["mocap_framerate"].item()
            stride = max(1, int(round(mocap_fps / TARGET_FPS)))

            # Extract data and apply stride
            poses = torch.tensor(bdata["poses"][::stride], dtype=torch.float32).to(
                DEVICE
            )
            trans = torch.tensor(bdata["trans"][::stride], dtype=torch.float32).to(
                DEVICE
            )
            betas = (
                torch.tensor(bdata["betas"][:16], dtype=torch.float32)
                .unsqueeze(0)
                .repeat(poses.shape[0], 1)
                .to(DEVICE)
            )

            T = poses.shape[0]
            if T < 30:
                skipped += 1
                pbar.set_postfix(saved=saved, skip=skipped, err=errored)
                continue

            # =============================================
            # A. FORWARD KINEMATICS (in SMPL Y-up space)
            # =============================================
            with torch.no_grad():
                body = bm(
                    body_pose=poses[:, 3:72],
                    global_orient=poses[:, :3],
                    transl=trans,
                    betas=betas,
                )
            # Joint positions in Y-up world space
            joint_pos_yup = body.joints[:, :22, :]  # [T, 22, 3]

            # Local rotations from axis-angle
            local_aa = poses[:, :66].reshape(-1, 22, 3)
            local_rot_mats = aa2matrot(local_aa.reshape(-1, 3)).reshape(
                -1, 22, 3, 3
            )  # [T, 22, 3, 3]

            # Global rotations via FK chain (in Y-up)
            global_rot_mats = get_global_rotations(local_rot_mats)

            # =============================================
            # B. CONVERT TO Z-UP (matching GMR / MuJoCo)
            # =============================================
            # Convert positions: Y-up → Z-up
            joint_pos_zup = positions_y_to_z(joint_pos_yup.cpu().numpy())  # [T, 22, 3]

            # Convert global rotations: Y-up → Z-up
            global_rot_zup = rotmats_y_to_z(global_rot_mats).cpu().numpy() # [T, 22, 3, 3]

            # =============================================
            # C. EXTRACT TRACKING DATA
            # =============================================
            tracker_pos = joint_pos_zup[:, TRACKER_IDX, :]  # [T, 6, 3]
            tracker_rotmat = global_rot_zup[:, TRACKER_IDX]  # [T, 6, 3, 3]
            
            # Ground truth in pure Z-up convention for training/inference
            global_rot_zup_gt = global_rot_zup # [T, 22, 3, 3]

            # =============================================
            # D. SAVE SEQUENCE
            # =============================================
            sequence_data = {
                "inputs": {
                    "tracker_pos": torch.tensor(tracker_pos),
                    "tracker_rotmat": torch.tensor(tracker_rotmat),
                },
                "ground_truth": {
                    "gt_pos": torch.tensor(joint_pos_zup),
                    "gt_rotmat": torch.tensor(global_rot_zup_gt),
                },
                "meta": {
                    "fps": TARGET_FPS,
                    "betas": betas[0].cpu(),
                    "source_file": os.path.basename(npz_path),
                    "source_dataset": os.path.relpath(npz_path, AMASS_DIR).split(os.sep)[0],
                    "tracker_names": TRACKER_NAMES,
                    "joint_names": JOINT_NAMES,
                    "coord_system": "Z-up",
                    "rotation_rep": "rotmat",
                },
            }

            dataset_name = os.path.relpath(npz_path, AMASS_DIR).split(os.sep)[0]
            save_name = f"{dataset_name}_seq_{file_idx:05d}.pt"
            torch.save(sequence_data, os.path.join(OUTPUT_DIR, save_name))
            saved += 1

            # Free GPU memory
            del body, joint_pos_yup, local_rot_mats, global_rot_mats
            del joint_pos_zup, global_rot_zup
            del tracker_pos, tracker_rotmat, poses, trans, betas
            torch.cuda.empty_cache()

        except Exception as e:
            errored += 1
            tqdm.write(f"  \u2717 {npz_path}: {e}")
            torch.cuda.empty_cache()

        pbar.set_postfix(saved=saved, skip=skipped, err=errored)

    print(f"\n{'='*50}")
    print(f"  Saved:   {saved}")
    print(f"  Skipped: {skipped} (too short)")
    print(f"  Errors:  {errored}")
    print(f"  Total:   {len(npz_files)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    extract_dataset()
