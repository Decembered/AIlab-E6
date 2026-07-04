"""Phase 5: Object pose tracking from video using mask-based tracking."""
import os, pickle, json
import numpy as np
import cv2

FRAMES_DIR = 'experiments/2026-07-04_obj_recon_bread/frames'
MASKS_DIR = 'experiments/2026-07-04_obj_recon_bread/masks'
OUT_DIR = 'experiments/2026-07-04_obj_recon_bread'
SEQ = 'data/human_demo/weigh_bread__2026_0701_0044_30'

# Load camera params (top camera)
K = pickle.load(open(f'{SEQ}/camera_calib/camera_top/cam_intr.pkl', 'rb'))
E = pickle.load(open(f'{SEQ}/camera_calib/camera_top/cam_extr.pkl', 'rb'))
R, t = E[:3, :3], E[:3, 3]
C = -R.T @ t
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

print("Phase 5: Object Pose Tracking")
print("=" * 60)

frame_files = sorted([f for f in os.listdir(f'{FRAMES_DIR}/camera_top') if f.endswith('.jpg')])
print(f"Processing {len(frame_files)} frames from top camera...")

poses = []
timestamps = []
track_data = []

for i, fname in enumerate(frame_files):
    frame_path = f'{FRAMES_DIR}/camera_top/{fname}'
    image = cv2.imread(frame_path)
    h, w = image.shape[:2]

    # GrabCut mask
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    ch, cw = int(h * 0.4), int(w * 0.4)
    c_x, c_y = (w - cw) // 2, (h - ch) // 2
    rect = (c_x, c_y, cw, ch)

    mask, _, _ = cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = ((mask == 1) | (mask == 3)).astype(np.uint8) * 255

    # Find centroid
    ys, xs = np.where(fg_mask > 128)
    if len(ys) < 50:
        if poses:
            poses.append(poses[-1].copy())
        else:
            poses.append(np.eye(4))
        timestamps.append(i)
        track_data.append({"frame": fname, "fg_pixels": len(ys), "status": "skip"})
        continue

    mx, my = xs.mean(), ys.mean()

    # Orientation from moments
    moments = cv2.moments(fg_mask)
    if moments['mu20'] + moments['mu02'] > 0:
        theta = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
    else:
        theta = 0.0

    # Project centroid to 3D (ground plane)
    xn = (mx - cx) / fx
    yn = (my - cy) / fy
    d_cam = np.array([xn, yn, 1.0])
    d_cam = d_cam / np.linalg.norm(d_cam)
    d_world = R.T @ d_cam
    table_y = 0.05
    alpha = (table_y - C[1]) / d_world[1] if abs(d_world[1]) > 1e-6 else 3.0
    pos_3d = C + alpha * d_world

    # Build 4x4 pose
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    pose = np.eye(4)
    pose[:3, 3] = pos_3d
    pose[0, 0] = cos_t
    pose[0, 2] = sin_t
    pose[2, 0] = -sin_t
    pose[2, 2] = cos_t

    poses.append(pose)
    timestamps.append(i)
    track_data.append({
        "frame": fname, "fg_pixels": len(ys),
        "centroid_2d": [float(mx), float(my)],
        "centroid_3d": pos_3d.tolist(),
        "orientation_rad": float(theta), "status": "ok",
    })

    if i % 10 == 0:
        print(f"  [{i+1}/{len(frame_files)}] {fname}: pos=({pos_3d[0]:.3f},{pos_3d[1]:.3f},{pos_3d[2]:.3f}), "
              f"theta={np.degrees(theta):.1f}deg, fg={len(ys)}px")

poses = np.array(poses)
timestamps = np.array(timestamps)
print(f"\n  Total: {len(poses)} poses")

# Smooth trajectory
window = 3
poses_smooth = poses.copy()
for i in range(len(poses)):
    start = max(0, i - window // 2)
    end = min(len(poses), i + window // 2 + 1)
    poses_smooth[i, :3, 3] = poses[start:end, :3, 3].mean(axis=0)

# Save
traj_path = f'{OUT_DIR}/object_trajectory.npz'
np.savez(traj_path, poses=poses_smooth, timestamps=timestamps, raw_poses=poses)
print(f"  Trajectory saved: {traj_path}")

traj_json = f'{OUT_DIR}/object_trajectory.json'
with open(traj_json, 'w') as f:
    json.dump({
        "sequence": "weigh_bread__2026_0701_0044_30",
        "method": "GrabCut mask centroid tracking + ground-plane back-projection",
        "num_frames": len(poses),
        "first_pose": poses_smooth[0].tolist(),
        "last_pose": poses_smooth[-1].tolist(),
        "position_range": {
            "x": [float(poses_smooth[:,0,3].min()), float(poses_smooth[:,0,3].max())],
            "y": [float(poses_smooth[:,1,3].min()), float(poses_smooth[:,1,3].max())],
            "z": [float(poses_smooth[:,2,3].min()), float(poses_smooth[:,2,3].max())],
        },
        "track_data": track_data,
    }, f, indent=2)

print(f"  Trajectory JSON saved: {traj_json}")

pos_diff = np.linalg.norm(poses_smooth[-1, :3, 3] - poses_smooth[0, :3, 3])
print(f"\n  Total displacement: {pos_diff*100:.1f} cm over {len(poses)} frames")
print(f"  Position range X: [{poses_smooth[:,0,3].min():.3f}, {poses_smooth[:,0,3].max():.3f}] m")
print(f"  Position range Z: [{poses_smooth[:,2,3].min():.3f}, {poses_smooth[:,2,3].max():.3f}] m")
print(f"\nDone! Phase 5 complete.")
