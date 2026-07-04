#!/usr/bin/env python3.8
"""
Pose Tracking V2: Multi-view mask-based object pose trajectory recovery.

For each sequence, for each frame with valid masks in >=2 cameras:
1. Compute mask centroid in each camera view
2. Triangulate 3D position using DLT with known camera matrices
3. Estimate orientation from mask principal axes (top-view)
4. Output trajectory as sequence of (timestamp, T_4x4) entries

Mask definition: visible-region mask centroid -> 3D position
"""
import os, sys, json, argparse
import numpy as np
import cv2

DATA_ROOT = '/mnt/workspace/Hackthon/data/human_demo'
MASK_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mask_pose')

OBJECT_SEQUENCES = {
    'bread': ['weigh_bread__2026_0701_0044_30', 'weigh_bread__left__2026_0701_0046_02'],
    'pipette': [
        'grasp_pipette_stand__2026_0701_0019_19',
        'grasp_pipette_rotate__2026_0701_0025_42',
        'grasp_pipette_press__2026_0701_0028_11',
    ],
    'drink_ad': ['weigh_drink_ad__2026_0701_0047_56', 'weigh_drink_ad__left__2026_0701_0049_04'],
    'drink_yykx': [
        'weigh_drink_yykx__2026_0701_0051_12',
        'weigh_drink_yykx__left__2026_0701_0052_53',
        'grasp_drink_yykx__2026_0701_0054_45',
    ],
}

CAMERAS = ['camera_top', 'camera_side_1', 'camera_side_2']


def load_camera_calib(seq_name, cam_name):
    calib_path = os.path.join(DATA_ROOT, seq_name, 'camera_calib', cam_name, 'calib.json')
    if not os.path.exists(calib_path):
        return None
    with open(calib_path) as f:
        calib = json.load(f)
    K = np.array(calib['K'])
    E = np.array(calib['E'])  # world-to-camera extrinsics
    # Camera matrix: P = K @ [R|t] = K @ E[:3, :]
    P = K @ E[:3, :4]
    return {'K': K, 'E': E, 'P': P}


def triangulate_point(points_2d, cameras):
    """DLT triangulation from multiple 2D observations with known camera matrices."""
    A = []
    for (u, v), cam in zip(points_2d, cameras):
        P = cam['P']
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]


def get_mask_centroid(mask):
    """Compute centroid of binary mask."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def get_mask_orientation_2d(mask):
    """Estimate 2D orientation from mask PCA (in top view)."""
    ys, xs = np.where(mask)
    if len(ys) < 5:
        return 0.0, None
    points = np.column_stack([xs, ys])
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal_axis = eigenvectors[:, -1]  # largest eigenvalue
    angle = np.arctan2(principal_axis[1], principal_axis[0])
    return angle, mean


def estimate_pose_from_masks(masks_dict, cameras_dict, frame_idx):
    """Estimate 3D pose from multi-view masks."""
    points_2d = []
    valid_cameras = []

    for cam_name in CAMERAS:
        if cam_name not in masks_dict:
            continue
        mask = masks_dict.get(cam_name)
        if mask is None or mask.sum() == 0:
            continue
        centroid = get_mask_centroid(mask)
        if centroid is None:
            continue
        if cam_name not in cameras_dict or cameras_dict[cam_name] is None:
            continue
        points_2d.append(centroid)
        valid_cameras.append(cameras_dict[cam_name])

    if len(points_2d) < 2:
        return None

    # Triangulate 3D position
    try:
        pos_3d = triangulate_point(points_2d, valid_cameras)
    except np.linalg.LinAlgError:
        return None

    # Estimate rotation from top-view mask orientation
    angle = 0.0
    if 'camera_top' in masks_dict and masks_dict['camera_top'] is not None:
        mask_top = masks_dict['camera_top']
        if mask_top.sum() > 100:
            angle, _ = get_mask_orientation_2d(mask_top)

    # Build 4x4 transform (rotation around Z axis from top-view orientation)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos_3d
    return T


def track_sequence(obj_name, seq_name):
    """Track object pose across all frames."""
    mask_dir = os.path.join(MASK_ROOT, obj_name, seq_name, 'masks')
    meta_file = os.path.join(MASK_ROOT, obj_name, seq_name, 'mask_meta.json')

    if not os.path.exists(mask_dir):
        print(f"  WARNING: No masks found for {seq_name}")
        return None

    # Load mask metadata to get frame indices
    with open(meta_file) as f:
        meta = json.load(f)

    # Load camera calibrations
    cameras_dict = {}
    for cam_name in CAMERAS:
        cameras_dict[cam_name] = load_camera_calib(seq_name, cam_name)

    # Group masks by frame index across cameras
    frame_groups = {}
    for cam_name, cam_data in meta['cameras'].items():
        for mask_key, mask_info in cam_data['masks'].items():
            fidx = mask_info['frame']
            if fidx not in frame_groups:
                frame_groups[fidx] = {}
            frame_groups[fidx][cam_name] = mask_key

    # Track each frame
    trajectory = []
    frame_indices = sorted(frame_groups.keys())

    for fidx in frame_indices:
        group = frame_groups[fidx]

        # Load all masks for this frame
        masks_dict = {}
        for cam_name, mask_key in group.items():
            mask_path = os.path.join(mask_dir, f'{mask_key}.png')
            if os.path.exists(mask_path):
                masks_dict[cam_name] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                masks_dict[cam_name] = None

        # Estimate pose
        T = estimate_pose_from_masks(masks_dict, cameras_dict, fidx)
        if T is not None:
            trajectory.append({
                'frame': fidx,
                'timestamp': fidx / 15.0,  # 15 fps
                'transform': T.tolist(),
                'position': T[:3, 3].tolist(),
            })

    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=['bread', 'pipette', 'drink_ad', 'drink_yykx'])
    args = parser.parse_args()

    for obj_name in args.objects:
        sequences = OBJECT_SEQUENCES.get(obj_name, [])
        if not sequences:
            print(f"Unknown object: {obj_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Pose Tracking: {obj_name}")
        print(f"{'='*60}")

        for seq_name in sequences:
            print(f"  Processing: {seq_name}")
            trajectory = track_sequence(obj_name, seq_name)

            if trajectory is None or len(trajectory) == 0:
                print(f"    WARNING: No valid trajectory")
                continue

            # Save trajectory
            out_dir = os.path.join(MASK_ROOT, obj_name, seq_name)
            os.makedirs(out_dir, exist_ok=True)

            traj_path = os.path.join(out_dir, 'object_trajectory.json')
            with open(traj_path, 'w') as f:
                json.dump({
                    'object': obj_name,
                    'sequence': seq_name,
                    'num_frames': len(trajectory),
                    'fps': 15.0,
                    'method': 'multi-view mask centroid triangulation',
                    'trajectory': trajectory,
                }, f, indent=2)

            # Summarize
            positions = np.array([t['position'] for t in trajectory])
            extent_min = positions.min(axis=0)
            extent_max = positions.max(axis=0)
            extent_range = extent_max - extent_min
            print(f"    {len(trajectory)} frames tracked")
            print(f"    Position range (m): X [{extent_min[0]:.3f}, {extent_max[0]:.3f}]")
            print(f"                        Y [{extent_min[1]:.3f}, {extent_max[1]:.3f}]")
            print(f"                        Z [{extent_min[2]:.3f}, {extent_max[2]:.3f}]")
            print(f"    Travel dist (m):    X {extent_range[0]:.3f}, Y {extent_range[1]:.3f}, Z {extent_range[2]:.3f}")

    print(f"\nDone. Trajectories saved to: {MASK_ROOT}")


if __name__ == '__main__':
    main()
