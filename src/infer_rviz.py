import os
import argparse
import torch
import time

try:
    import rclpy
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
except ImportError:
    print("Error: rclpy or visualization_msgs not found.")
    print(
        "This script requires a ROS 2 environment (e.g., Humble) to publish to RViz2."
    )
    print("Ensure you have sourced '/opt/ros/humble/setup.bash'.")
    exit(1)

from tcn.dataset import make_eval_dataset
from tcn.gpu_aug import GPUSampleBuilder
from tcn.model import TransformerBodyPose
from tcn.rotations import sixd_to_rotmat
from tcn.skeleton import SMPL_PARENTS
from tcn.train import load_config


def create_skeleton_marker(
    node, positions, parents, color_rgb, ns="skeleton", m_id=0, offset_x=0.0
):
    """
    Creates a ROS 2 visualization_msgs/Marker (LINE_LIST) for the 22-joint SMPL skeleton.
    positions: [22, 3] tensor or numpy array
    """
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.ns = ns
    marker.id = m_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.02  # 2cm thick lines
    marker.color.r = float(color_rgb[0])
    marker.color.g = float(color_rgb[1])
    marker.color.b = float(color_rgb[2])
    marker.color.a = 1.0
    marker.pose.orientation.w = 1.0

    for i in range(1, len(parents)):
        p = parents[i]
        if p == -1:
            continue

        p1 = Point()
        p1.x = float(positions[i, 0]) + offset_x
        p1.y = float(positions[i, 1])
        p1.z = float(positions[i, 2])

        p2 = Point()
        p2.x = float(positions[p, 0]) + offset_x
        p2.y = float(positions[p, 1])
        p2.z = float(positions[p, 2])

        marker.points.append(p1)
        marker.points.append(p2)

    return marker


def create_axis_markers(
    node, position, rotation, ns="axis", m_id_start=0, scale=0.1, offset_x=0.0
):
    """
    Creates 3 Marker.ARROW markers representing the XYZ axes.
    Convention: Red=X (Forward), Cyan=Y (Left), Blue=Z (Up)
    position: [3]
    rotation: [3, 3] rotation matrix
    """
    markers = []
    # X=Forward (Red), Y=Left (Cyan), Z=Up (Blue)
    colors = [(1, 0, 0), (0, 1, 1), (0, 0, 1)]
    axes = rotation.T  # Columns are the axis vectors in world space

    for i in range(3):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = m_id_start + i
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Start and End points for the arrow
        p_start = Point(x=float(position[0]) + offset_x, y=float(position[1]), z=float(position[2]))
        vec = axes[i] * scale
        p_end = Point(
            x=p_start.x + float(vec[0]), 
            y=p_start.y + float(vec[1]), 
            z=p_start.z + float(vec[2])
        )
        
        marker.points = [p_start, p_end]
        
        marker.scale.x = 0.01 # shaft diameter
        marker.scale.y = 0.02 # head diameter
        marker.scale.z = 0.04 # head length
        
        marker.color.r = float(colors[i][0])
        marker.color.g = float(colors[i][1])
        marker.color.b = float(colors[i][2])
        marker.color.a = 0.8
        
        markers.append(marker)
        
    return markers


@torch.no_grad()
def main(args):
    rclpy.init()
    node = rclpy.create_node("vr_pose_inference")
    pub = node.create_publisher(MarkerArray, "/vr_pose/skeleton_markers", 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Prioritize local config for data/folders, use checkpoint for model architecture
    local_cfg = load_config(args.config)
    ckpt_cfg = ckpt.get("config", {})
    
    # Merge: Model from checkpoint (must match weights), Data from local (flexible paths)
    cfg = local_cfg.copy()
    if "model" in ckpt_cfg:
        cfg["model"] = ckpt_cfg["model"]
    
    print(f"Using window size: {cfg['data']['window_size']}")

    # ── Model ──
    model = TransformerBodyPose(
        input_dim=cfg["model"]["input_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")

    # ── Test windows: consecutive, non-overlapping, from one test set of the config ──
    W = cfg["data"]["window_size"]
    dataset = make_eval_dataset(cfg, cfg["data"]["test"][args.test_set], stride=W, label=args.test_set)
    builder = GPUSampleBuilder(cfg["data"].get("augmentation"), device)  # nominal tracker mount, no noise

    # Frame rate delay
    fps = cfg["data"].get("fps", 60.0)
    delay = 1.0 / fps

    print("\nStarting RViz2 playback...")
    print("In RViz2, set your Fixed Frame to 'map'.")
    print("Add a MarkerArray display listening to '/vr_pose/skeleton_markers'.")
    print("  -> GREEN Skeleton: Predicted Pose")
    print("  -> RED Skeleton:   Ground Truth Pose (offset by 1.0m on X axis)")

    for idx in range(len(dataset)):
        if not rclpy.ok():
            break

        raw = {k: v[None] for k, v in dataset[idx].items() if k != "seq_id"}
        batch = builder(raw, train=False)  # batch of one, built on the GPU
        inputs, gt_pos = batch["input"], batch["target_pos"]

        with torch.amp.autocast("cuda"):
            _, global_rotmats, fk_pos = model(inputs, bone_offsets=batch["bone_offsets"])

        # `fk_pos` and `global_rotmats` are root-relative/absolute
        # Center the prediction properly
        root_pos = fk_pos[:, :, 0:1]
        pred_rel = fk_pos - root_pos

        # Extract [W, 22, 3] and [W, 22, 3, 3] shapes
        pred_seq = pred_rel[0].cpu().numpy()
        pred_rot_seq = global_rotmats[0].cpu().numpy()
        gt_seq = gt_pos[0].cpu().numpy()

        # Tracker poses for the axis arrows: per tracker the 18 input features are
        # pos(3), 6D rotation(6), pos velocity(3), rot velocity(6)
        in_np = inputs[0].cpu()
        tracker_viz_pos = []
        tracker_viz_rot = []
        for i in range(6):
            start_off = i * 18
            pos = in_np[:, start_off : start_off + 3]
            sixd = in_np[:, start_off + 3 : start_off + 9]
            rot = sixd_to_rotmat(sixd).numpy()
            tracker_viz_pos.append(pos.numpy())
            tracker_viz_rot.append(rot)
            
        # Playback the window frame-by-frame
        for t in range(W):
            if not rclpy.ok():
                break

            start_time = time.time()

            markers = MarkerArray()

            # Green for Prediction (Centered at X=0)
            m_pred = create_skeleton_marker(
                node,
                pred_seq[t],
                SMPL_PARENTS,
                color_rgb=(0, 1, 0),
                ns="pred",
                m_id=0,
                offset_x=0.0,
            )

            # Red for Ground Truth (Offset at X=1.0 for side-by-side viewing)
            m_gt = create_skeleton_marker(
                node,
                gt_seq[t],
                SMPL_PARENTS,
                color_rgb=(1, 0, 0),
                ns="gt",
                m_id=1,
                offset_x=1.0,
            )

            # Add Axis Arrows for Predicted joints (sampling a few key joints to avoid clutter)
            # Pelvis, L/R Ankle, Head, L/R Wrist
            viz_joints = [0, 7, 8, 15, 20, 21]
            for i, j_idx in enumerate(viz_joints):
                m_axes = create_axis_markers(
                    node,
                    pred_seq[t, j_idx],
                    pred_rot_seq[t, j_idx],
                    ns=f"pred_axis_{j_idx}",
                    m_id_start=100 + i*3,
                    scale=0.15,
                    offset_x=0.0
                )
                markers.markers.extend(m_axes)

            # Add Axis Arrows for Input Trackers (offset slightly to avoid overlap)
            # Yellow-ish for trackers
            for i in range(6):
                m_track_axes = create_axis_markers(
                    node,
                    tracker_viz_pos[i][t],
                    tracker_viz_rot[i][t],
                    ns=f"input_tracker_{i}",
                    m_id_start=200 + i*3,
                    scale=0.1,
                    offset_x=-0.5 # Offset left to see inputs separately
                )
                markers.markers.extend(m_track_axes)

            markers.markers.extend([m_pred, m_gt])
            pub.publish(markers)

            # Enforce playback FPS
            elapsed = time.time() - start_time
            sleep_time = delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RViz Inference Playback")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained .pt checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "tcn", "config.yaml"),
    )
    parser.add_argument("--test-set", default="amass_test", help="a key of data.test in the config")
    args = parser.parse_args()
    main(args)
