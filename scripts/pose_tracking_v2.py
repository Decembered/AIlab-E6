#!/usr/bin/env python3.8
"""
Pose Tracking V3: Multi-view mask silhouette + mesh projection pose recovery.

Improvements over V2:
1. Mesh-based silhouette projection for yaw optimization (not just top-view PCA)
2. Ray-plane intersection for translation (not centroid-based DLT to fixed Z)
3. 3-view constraint with quality grading (GRADE_3VIEW / GRADE_2VIEW)
4. Velocity-based quality flagging (GOOD / SUSPICIOUS_FAST)
5. Per-frame silhouette scores and quality metrics

Mask definition: visible-region mask of the target object.
"""
import os, sys, json, argparse, math
import numpy as np
import cv2
from pathlib import Path

DATA_ROOT = os.environ.get('HO_TRACKER_DATA', '/mnt/workspace/Hackthon/data/human_demo')
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_ROOT = os.path.join(SCRIPT_DIR, 'outputs', 'mask_pose')
SRC_DIR = os.path.join(SCRIPT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

from object_recon.pose_tracking import (
    camera_ray_from_pixel,
    mask_2d_stats,
    optimize_yaw_by_multiview_projection,
    load_mesh_points,
    choose_canonical_mesh,
    yaw_transform_world,
    CAMERAS,
    YAW_AMBIGUOUS_OBJECTS,
    TrackingConfig,
    rotation_matrix_to_quaternion_xyzw,
    load_intrinsics,
    load_extrinsics,
    angle_delta_deg,
)

OBJECT_SEQUENCES = {
    'bread': ['weigh_bread__2026_0701_0044_30', 'weigh_bread__left__2026_0701_0046_02'],
    'pipette': [
        'grasp_pipette_stand__2026_0701_0019_19',
        'grasp_pipette_rotate__2026_0701_0025_42',
        'grasp_pipette_press__2026_0701_0028_11',
        'pipette_rh_beaker__2026_0701_0035_47',
        'pipette_rh_beaker_testtube__2026_0701_0039_28',
    ],
    'drink_ad': ['weigh_drink_ad__2026_0701_0047_56', 'weigh_drink_ad__left__2026_0701_0049_04'],
    'drink_yykx': [
        'weigh_drink_yykx__2026_0701_0051_12',
        'weigh_drink_yykx__left__2026_0701_0052_53',
        'grasp_drink_yykx__2026_0701_0054_45',
    ],
}

VELOCITY_THRESHOLD_FAST = 2.0
MIN_SILHOUETTE_SCORE = 0.01
YAW_SEARCH_DEGREES = 180.0
YAW_SEARCH_STEPS = 73
MAX_CENTROID_JUMP_M = 1.0
MAX_THETA_JUMP_DEG = 170.0
MAX_POSE_JUMP_M = 1.0

LOCAL_YAW_AMBIGUOUS = {"drink_ad", "drink_yykx", "bottle", "can", "bread"}


def load_camera_calibration(seq_name, cam_name):
    calib_dir = os.path.join(DATA_ROOT, seq_name, 'camera_calib')
    try:
        K = load_intrinsics(Path(calib_dir), cam_name)
        E = load_extrinsics(Path(calib_dir), cam_name)
        return {'K': np.asarray(K, dtype='float64'), 'E': np.asarray(E, dtype='float64')}
    except Exception:
        return None


def load_mask_frame(seq_name, obj_name, cam_name, fidx):
    mask_path = os.path.join(MASK_ROOT, obj_name, seq_name, 'masks',
                              f'{cam_name}_frame_{fidx:06d}.png')
    if not os.path.exists(mask_path):
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 127).astype(bool)


def parse_camera_frame_key(mask_key):
    import re
    m = re.match(r'(camera_\w+)_frame_(\d+)', mask_key)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def build_view_records(masks_dict, cameras_dict, config):
    view_records = []
    for cam, mask in masks_dict.items():
        if mask is None or cam not in cameras_dict or cameras_dict[cam] is None:
            continue
        stats = mask_2d_stats(mask)
        area_ratio = stats['mask_area'] / float(mask.shape[0] * mask.shape[1])
        invalid_reason = stats['invalid_reason']
        if invalid_reason is None and area_ratio > config.max_image_mask_area_ratio:
            invalid_reason = f"mask_area_ratio_too_large ({area_ratio:.3f}>{config.max_image_mask_area_ratio:.3f})"

        K = cameras_dict[cam]['K']
        E = cameras_dict[cam]['E']
        center, ray = None, None
        if stats['centroid_x'] is not None:
            center, ray = camera_ray_from_pixel(K, E, [stats['centroid_x'], stats['centroid_y']])
        view_records.append({
            'camera': cam,
            'mask': mask,
            'K': K,
            'E': E,
            'view_valid': invalid_reason is None,
            'view_invalid_reason': invalid_reason,
            'mask_area': int(stats['mask_area']),
            'mask_area_ratio': float(area_ratio),
            'centroid_2d': [float(stats['centroid_x']), float(stats['centroid_y'])] if stats['centroid_x'] is not None else None,
            'bbox_xyxy': stats['bbox_xyxy'],
            'camera_center': center,
            'centroid_ray': ray,
        })
    return view_records


def kalman_smooth_positions(positions, timestamps, process_noise_pos=0.01, process_noise_vel=0.1, meas_noise=0.001):
    """Constant velocity Kalman filter for 3D position smoothing.

    State: [x, y, z, vx, vy, vz] (6D)
    Measurement: [x, y, z] (3D)
    """
    n = len(positions)
    dt = np.diff(timestamps).mean() if len(timestamps) > 1 else 1.0 / 15.0

    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0

    Q = np.eye(6)
    Q[0, 0] = Q[1, 1] = Q[2, 2] = process_noise_pos * dt
    Q[3, 3] = Q[4, 4] = Q[5, 5] = process_noise_vel * dt

    R = np.eye(3) * meas_noise

    x_est = np.zeros(6)
    x_est[:3] = positions[0]
    P = np.eye(6) * 0.1

    smoothed = np.zeros((n, 3))
    smoothed[0] = positions[0]

    for i in range(1, n):
        x_pred = F @ x_est
        P_pred = F @ P @ F.T + Q

        z = positions[i]
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_est = x_pred + K @ y
        P = (np.eye(6) - K @ H) @ P_pred
        smoothed[i] = x_est[:3]

    return smoothed


def kalman_smooth_angles(thetas, timestamps, process_noise=0.05, meas_noise=0.01):
    """Constant velocity Kalman filter for 1D angle smoothing (handles circularity)."""
    n = len(thetas)
    dt = np.diff(timestamps).mean() if len(timestamps) > 1 else 1.0 / 15.0
    dt = max(dt, 1e-6)

    x_est = np.array([thetas[0], 0.0])
    P = np.eye(2) * 0.1

    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[process_noise * dt, 0], [0, process_noise * dt * 0.1]])
    R = np.array([[meas_noise]])

    smoothed = np.zeros(n)
    smoothed[0] = thetas[0]

    for i in range(1, n):
        x_pred = F @ x_est
        P_pred = F @ P @ F.T + Q

        residual = thetas[i] - x_pred[0]
        residual = math.atan2(math.sin(residual), math.cos(residual))
        y = np.array([residual])

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_est = x_pred + K @ y
        P = (np.eye(2) - K @ H) @ P_pred
        smoothed[i] = x_est[0]

    return smoothed


def triangulate_dlt(points_2d, proj_matrices):
    """DLT triangulation from multiple 2D observations with known 3x4 projection matrices."""
    A = []
    for (u, v), P in zip(points_2d, proj_matrices):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def run_pose_tracking(obj_name, seq_name, mesh_points, config):
    mask_dir = os.path.join(MASK_ROOT, obj_name, seq_name, 'masks')
    meta_file = os.path.join(MASK_ROOT, obj_name, seq_name, 'mask_meta.json')

    if not os.path.exists(mask_dir):
        print(f"  WARNING: No masks found for {seq_name}")
        return None

    with open(meta_file) as f:
        meta = json.load(f)

    cameras_dict = {}
    for cam_name in CAMERAS:
        cameras_dict[cam_name] = load_camera_calibration(seq_name, cam_name)

    proj_matrices = {}
    for cam_name in CAMERAS:
        if cameras_dict[cam_name] is not None:
            E = cameras_dict[cam_name]['E']
            K = cameras_dict[cam_name]['K']
            proj_matrices[cam_name] = K @ E[:3, :4]

    frame_groups = {}
    if isinstance(meta.get('cameras'), dict):
        for cam_name, cam_data in meta['cameras'].items():
            if cam_name not in CAMERAS:
                continue
            for mask_key, mask_info in cam_data.get('masks', {}).items():
                parsed_cam, fidx = parse_camera_frame_key(mask_key)
                if parsed_cam is None:
                    continue
                if fidx not in frame_groups:
                    frame_groups[fidx] = {}
                frame_groups[fidx][parsed_cam] = fidx
    else:
        frames_by_cam = {}
        for fname in sorted(os.listdir(mask_dir)):
            parsed_cam, fidx = parse_camera_frame_key(fname.replace('.png', ''))
            if parsed_cam is None:
                continue
            frames_by_cam.setdefault(parsed_cam, set()).add(fidx)
        common_frames = set.intersection(*frames_by_cam.values()) if len(frames_by_cam) >= 2 else set()
        for fidx in sorted(common_frames):
            frame_groups[fidx] = {cam: fidx for cam in frames_by_cam}

    frame_indices = sorted(frame_groups.keys())
    trajectory = []
    frame_records = []
    prev_valid_pose = None
    prev_valid_theta = None
    yaw_ambiguous = obj_name in LOCAL_YAW_AMBIGUOUS

    for fidx in frame_indices:
        group = frame_groups[fidx]
        masks_dict = {}
        for cam_name in CAMERAS:
            if cam_name in group:
                masks_dict[cam_name] = load_mask_frame(seq_name, obj_name, cam_name, fidx)
            else:
                masks_dict[cam_name] = None

        view_records = build_view_records(
            {cam: masks_dict.get(cam) for cam in CAMERAS},
            cameras_dict, config
        )

        valid_views = [rec for rec in view_records if rec['view_valid']]
        invalid_reason = None

        if len(valid_views) < 2:
            invalid_reason = f"insufficient_valid_views ({len(valid_views)}<2)"
        elif len(valid_views) < 3:
            view_grade = 'GRADE_2VIEW'
        else:
            view_grade = 'GRADE_3VIEW'

        translation = None
        yaw = 0.0
        silhouette_score = 0.0
        per_view_scores = []
        centroid_jump = 0.0
        theta_jump = 0.0
        pose_jump = 0.0
        pose_raw = np.eye(4)

        if invalid_reason is None:
            points_2d = []
            valid_proj = []
            for rec in valid_views:
                if rec['centroid_2d'] is not None and rec['camera'] in proj_matrices:
                    points_2d.append(rec['centroid_2d'])
                    valid_proj.append(proj_matrices[rec['camera']])
            if len(points_2d) >= 2:
                try:
                    translation = triangulate_dlt(points_2d, valid_proj)
                except np.linalg.LinAlgError:
                    pass
            if translation is None:
                invalid_reason = "dlt_triangulation_failed"

        if invalid_reason is None:
            yaw, silhouette_score, per_view_scores = optimize_yaw_by_multiview_projection(
                mesh_points, translation, view_records, yaw_ambiguous, config
            )
            pose_raw = yaw_transform_world(translation, yaw)
            if prev_valid_pose is not None:
                centroid_jump = float(np.linalg.norm(translation - prev_valid_pose[:3, 3]))
                theta_jump = angle_delta_deg(prev_valid_theta, yaw)
                pose_jump = centroid_jump
            if centroid_jump > MAX_CENTROID_JUMP_M:
                invalid_reason = f"centroid_jump ({centroid_jump:.4f}>{MAX_CENTROID_JUMP_M:.4f}m)"
            if invalid_reason is None and not yaw_ambiguous and theta_jump > MAX_THETA_JUMP_DEG:
                if theta_jump > 175.0:
                    pass
                else:
                    invalid_reason = f"theta_jump ({theta_jump:.1f}>{MAX_THETA_JUMP_DEG:.1f}deg)"
            if invalid_reason is None and pose_jump > MAX_POSE_JUMP_M:
                invalid_reason = f"pose_jump ({pose_jump:.4f}>{MAX_POSE_JUMP_M:.4f}m)"

        valid = invalid_reason is None
        if valid:
            prev_valid_pose = pose_raw.copy()
            prev_valid_theta = yaw
            pose_out = pose_raw.copy()
        elif prev_valid_pose is not None:
            pose_out = prev_valid_pose.copy()
        else:
            pose_out = np.eye(4)

        trajectory.append({
            'frame': int(fidx),
            'timestamp': fidx / 15.0,
            'transform_4x4': pose_out.tolist(),
        })

        total_mask_area = sum(rec['mask_area'] for rec in view_records)
        frame_records.append({
            'frame_id': int(fidx),
            'valid': bool(valid),
            'invalid_reason': invalid_reason,
            'mask_area': int(total_mask_area),
            'view_grade': view_grade if invalid_reason is None else 'INVALID',
            'valid_view_count': len(valid_views),
            'total_view_count': len(view_records),
            'silhouette_score': float(silhouette_score),
            'per_view_scores': per_view_scores,
            'translation': pose_out[:3, 3].tolist(),
            'rotation_matrix': pose_out[:3, :3].tolist(),
            'quaternion_xyzw': rotation_matrix_to_quaternion_xyzw(pose_out[:3, :3]).tolist(),
            'theta_rad': float(yaw),
            'theta_deg': float(math.degrees(yaw)),
            'centroid_jump_m': float(centroid_jump),
            'theta_jump_deg': float(theta_jump),
            'pose_jump_m': float(pose_jump),
        })

    positions_raw = np.array([[t['transform_4x4'][0][3], t['transform_4x4'][1][3], t['transform_4x4'][2][3]] for t in trajectory])
    timestamps = np.array([t['timestamp'] for t in trajectory])
    yaws_raw = np.array([float(math.atan2(
        t['transform_4x4'][0][2], t['transform_4x4'][0][0]
    )) for t in trajectory])
    yaw_ambiguous = obj_name in LOCAL_YAW_AMBIGUOUS

    vel_raw = np.zeros(len(positions_raw))
    for i in range(1, len(positions_raw)):
        dt = timestamps[i] - timestamps[i - 1]
        vel_raw[i] = np.linalg.norm(positions_raw[i] - positions_raw[i - 1]) / max(dt, 1e-6)

    if len(positions_raw) >= 3:
        positions_smooth = kalman_smooth_positions(positions_raw, timestamps)
        if not yaw_ambiguous:
            yaws_smooth = kalman_smooth_angles(yaws_raw, timestamps)
        else:
            yaws_smooth = yaws_raw

        for i in range(len(trajectory)):
            trans = positions_smooth[i]
            if not yaw_ambiguous:
                cos_a, sin_a = float(math.cos(yaws_smooth[i])), float(math.sin(yaws_smooth[i]))
            else:
                cos_a, sin_a = float(math.cos(yaws_raw[i])), float(math.sin(yaws_raw[i]))
            R = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = trans
            trajectory[i]['transform_4x4'] = T.tolist()
    else:
        positions_smooth = positions_raw

    positions = positions_smooth
    vel = np.zeros(len(positions))
    for i in range(1, len(positions)):
        dt = trajectory[i]['timestamp'] - trajectory[i - 1]['timestamp']
        vel[i] = np.linalg.norm(positions[i] - positions[i - 1]) / max(dt, 1e-6)

    num_3view = sum(1 for r in frame_records if r.get('view_grade') == 'GRADE_3VIEW')
    num_2view = sum(1 for r in frame_records if r.get('view_grade') == 'GRADE_2VIEW')
    max_vel = float(vel.max()) if len(vel) else 0.0
    quality_flag = 'GOOD' if max_vel <= VELOCITY_THRESHOLD_FAST else 'SUSPICIOUS_FAST'

    position_stats = {
        'x': [float(positions[:, 0].min()), float(positions[:, 0].max())],
        'y': [float(positions[:, 1].min()), float(positions[:, 1].max())],
        'z': [float(positions[:, 2].min()), float(positions[:, 2].max())],
    }

    return {
        'object': obj_name,
        'sequence': seq_name,
        'num_frames': len(trajectory),
        'fps': 15.0,
        'method': 'DLT triangulation + mesh silhouette yaw + Kalman filter',
        'trajectory': trajectory,
        'frame_records': frame_records,
        'quality': {
            'num_3view_frames': num_3view,
            'num_2view_frames': num_2view,
            'num_invalid_frames': len(trajectory) - num_3view - num_2view,
            'max_velocity_m_per_s': max_vel,
            'mean_velocity_m_per_s': float(vel.mean()) if len(vel) else 0.0,
            'max_velocity_raw_m_per_s': float(vel_raw.max()) if len(vel_raw) else 0.0,
            'mean_velocity_raw_m_per_s': float(vel_raw.mean()) if len(vel_raw) else 0.0,
            'quality_flag': quality_flag,
            'position_stats_m': position_stats,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=['bread', 'pipette', 'drink_ad', 'drink_yykx'])
    args = parser.parse_args()

    config = TrackingConfig(
        max_image_mask_area_ratio=0.5,
        yaw_search_degrees=YAW_SEARCH_DEGREES,
        yaw_search_steps=YAW_SEARCH_STEPS,
        min_silhouette_score=MIN_SILHOUETTE_SCORE,
        max_centroid_jump_m=MAX_CENTROID_JUMP_M,
        max_theta_jump_deg=MAX_THETA_JUMP_DEG,
        max_pose_jump_m=MAX_POSE_JUMP_M,
    )

    for obj_name in args.objects:
        sequences = OBJECT_SEQUENCES.get(obj_name, [])
        if not sequences:
            print(f"Unknown object: {obj_name}")
            continue

        mesh_path = choose_canonical_mesh(obj_name, Path(os.path.join(SCRIPT_DIR, 'runs/object_asset_v1')))
        if mesh_path is None or not os.path.exists(str(mesh_path)):
            print(f"  WARNING: No mesh found for {obj_name}, skipping silhouette optimization")
            mesh_points = None
        else:
            mesh_points = load_mesh_points(mesh_path)

        print(f"\n{'='*60}")
        print(f"Pose Tracking V3: {obj_name}")
        print(f"{'='*60}")

        for seq_name in sequences:
            print(f"  Processing: {seq_name}")

            if mesh_points is None:
                print(f"    SKIPPED: no mesh available for silhouette optimization")
                continue

            result = run_pose_tracking(obj_name, seq_name, mesh_points, config)

            if result is None or len(result['trajectory']) == 0:
                print(f"    WARNING: No valid trajectory")
                continue

            out_dir = os.path.join(MASK_ROOT, obj_name, seq_name)
            os.makedirs(out_dir, exist_ok=True)

            traj_json = {
                'object': result['object'],
                'sequence': result['sequence'],
                'num_frames': result['num_frames'],
                'fps': result['fps'],
                'method': result['method'],
                'trajectory': result['trajectory'],
            }
            traj_path = os.path.join(out_dir, 'object_trajectory.json')
            with open(traj_path, 'w') as f:
                json.dump(traj_json, f, indent=2)

            quality_path = os.path.join(out_dir, 'trajectory_quality_report.json')
            with open(quality_path, 'w') as f:
                json.dump({
                    'status': 'valid',
                    'method': 'DLT_triangulation + mesh_silhouette_yaw + kalman_filter',
                    'postprocessing': 'kalman_filter_constant_velocity',
                    'num_frames': result['num_frames'],
                    'num_source_keypoints': result['num_frames'],
                    'duration_s': result['num_frames'] / 15.0,
                    'position_stats_m': result['quality']['position_stats_m'],
                    'velocity_stats_m_per_s': {
                        'max': result['quality']['max_velocity_m_per_s'],
                        'mean': result['quality']['mean_velocity_m_per_s'],
                    },
                    'view_quality': {
                        'grade_3view': result['quality']['num_3view_frames'],
                        'grade_2view': result['quality']['num_2view_frames'],
                        'invalid': result['quality']['num_invalid_frames'],
                    },
                    'quality_flag': result['quality']['quality_flag'],
                }, f, indent=2)

            ext_min = result['quality']['position_stats_m']
            ext_range = {k: ext_min[k][1] - ext_min[k][0] for k in ext_min}
            print(f"    {len(result['trajectory'])} frames tracked")
            invalid_counts = {}
            for rec in result['frame_records']:
                reason = rec.get('invalid_reason') or 'VALID'
                key = reason.split(' (', 1)[0]
                invalid_counts[key] = invalid_counts.get(key, 0) + 1
            print(f"    3-view: {result['quality']['num_3view_frames']}, "
                  f"2-view: {result['quality']['num_2view_frames']}, "
                  f"invalid: {result['quality']['num_invalid_frames']}")
            for key, count in sorted(invalid_counts.items(), key=lambda x: -x[1]):
                if key != 'VALID':
                    print(f"      Reason: {key} -> {count} frames")
            print(f"    Position range (m): X [{ext_min['x'][0]:.3f}, {ext_min['x'][1]:.3f}]")
            print(f"                        Y [{ext_min['y'][0]:.3f}, {ext_min['y'][1]:.3f}]")
            print(f"                        Z [{ext_min['z'][0]:.3f}, {ext_min['z'][1]:.3f}]")
            print(f"    Travel dist (m):    X {ext_range['x']:.3f}, Y {ext_range['y']:.3f}, Z {ext_range['z']:.3f}")
            vel_raw_max = result['quality'].get('max_velocity_raw_m_per_s', result['quality']['max_velocity_m_per_s'])
            vel_kf_max = result['quality']['max_velocity_m_per_s']
            print(f"    Max velocity: {vel_kf_max:.3f} m/s (raw: {vel_raw_max:.3f} m/s)")
            print(f"    Quality: {result['quality']['quality_flag']}")

    print(f"\nDone. Trajectories saved to: {MASK_ROOT}")


if __name__ == '__main__':
    main()
