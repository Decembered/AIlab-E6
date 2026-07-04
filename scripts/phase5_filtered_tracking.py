"""Phase 5 v2: Pose tracking with quality gating — filter SAM prompt failures.

Gates:
1. Mask area: valid in [4000, 14000] px
2. Theta jump: <20° from previous valid frame
3. Centroid jump: <30 px from previous valid frame

Anomaly handling: hold previous valid pose, mark frame invalid.
"""
import os, pickle, json
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEQ = 'data/human_demo/weigh_bread__2026_0701_0044_30'
FRAMES_DIR = 'experiments/2026-07-04_obj_recon_bread/frames'
OUT_DIR = 'experiments/2026-07-04_obj_recon_bread'
os.makedirs(OUT_DIR, exist_ok=True)

# --- Config ---
VALID_AREA_MIN = 4000
VALID_AREA_MAX = 14000
MAX_THETA_JUMP_DEG = 20.0
MAX_CENTROID_JUMP_PX = 30.0

CKPT = os.path.expanduser('~/.cache/sam/sam_vit_b_01ec64.pth')

# v4.1 prompts
POS_PTS = np.array([[555,300],[590,305],[625,300],[640,325],[600,345],[595,360]])
NEG_PTS = np.array([[500,350],[500,390],[535,405],[560,285],[585,320],[680,340]])
BOX = np.array([515, 250, 670, 395])

# Camera
K = pickle.load(open(f'{SEQ}/camera_calib/camera_top/cam_intr.pkl', 'rb'))
E = pickle.load(open(f'{SEQ}/camera_calib/camera_top/cam_extr.pkl', 'rb'))
R, t = E[:3, :3], E[:3, 3]
C = -R.T @ t
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
table_y = 0.05

print("=" * 60)
print("Phase 5 v2: Filtered Pose Tracking")
print(f"  Area gate: [{VALID_AREA_MIN}, {VALID_AREA_MAX}]")
print(f"  Theta gate: {MAX_THETA_JUMP_DEG} deg")
print(f"  Centroid gate: {MAX_CENTROID_JUMP_PX} px")
print("=" * 60)

# --- Load SAM ---
print("Loading SAM...")
sam = sam_model_registry['vit_b'](checkpoint=CKPT)
sam.to(device='cpu')
predictor = SamPredictor(sam)

def run_sam_v41(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    all_pts = np.vstack([POS_PTS, NEG_PTS])
    all_lbl = np.concatenate([np.ones(len(POS_PTS)), np.zeros(len(NEG_PTS))])
    masks, scores, _ = predictor.predict(
        point_coords=all_pts, point_labels=all_lbl,
        box=BOX[None, :], multimask_output=True,
    )
    best = np.argmax(scores)
    m = masks[best]

    # Post-process: keep main component
    n_labels, labels, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    pos_c = POS_PTS.mean(axis=0)
    best_comp, best_dist = 0, float('inf')
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] > 1000:
            d = np.sqrt((cents[i][0]-pos_c[0])**2 + (cents[i][1]-pos_c[1])**2)
            if d < best_dist and d < 300:
                best_dist = d
                best_comp = i
    mask = (labels == best_comp).astype(np.uint8)*255 if best_comp > 0 else np.zeros_like(m, dtype=np.uint8)
    return mask, float(scores[best])

# --- Process all frames ---
frame_files = sorted([f for f in os.listdir(f'{FRAMES_DIR}/camera_top') if f.endswith('.jpg')])
print(f"Processing {len(frame_files)} frames...")

poses = []
track_data = []
prev_valid_pose = None
prev_valid_theta = None
prev_valid_centroid = None
prev_valid_mask = None  # for visualization

stats = {"total": 0, "valid": 0, "area_gate": 0, "theta_gate": 0, "centroid_gate": 0, "low_px": 0}

for i, fname in enumerate(frame_files):
    image = cv2.imread(f'{FRAMES_DIR}/camera_top/{fname}')
    mask, sam_score = run_sam_v41(image)

    ys, xs = np.where(mask > 128)
    fg_pixels = len(ys)
    stats["total"] += 1

    # Extract features
    mx, my = xs.mean(), ys.mean() if len(ys) > 0 else (-1, -1)
    moments = cv2.moments(mask)
    theta = 0.5*np.arctan2(2*moments['mu11'], moments['mu20']-moments['mu02']) if moments['mu20']+moments['mu02']>0 else 0.0

    # --- Quality gates ---
    invalid_reason = None

    # Gate 1: Area
    if fg_pixels < 50:
        invalid_reason = "empty_mask"
        stats["low_px"] += 1
    elif fg_pixels < VALID_AREA_MIN:
        invalid_reason = f"area_too_small ({fg_pixels}<{VALID_AREA_MIN})"
        stats["area_gate"] += 1
    elif fg_pixels > VALID_AREA_MAX:
        invalid_reason = f"area_too_large ({fg_pixels}>{VALID_AREA_MAX})"
        stats["area_gate"] += 1

    # Gate 2: Theta jump (only if area is OK and we have a previous valid frame)
    if invalid_reason is None and prev_valid_theta is not None:
        theta_deg = abs(np.degrees(theta))
        prev_theta_deg = abs(np.degrees(prev_valid_theta))
        jump = abs(theta_deg - prev_theta_deg)
        if jump > MAX_THETA_JUMP_DEG:
            invalid_reason = f"theta_jump ({jump:.1f}deg>{MAX_THETA_JUMP_DEG}deg)"

    # Gate 3: Centroid jump
    if invalid_reason is None and prev_valid_centroid is not None:
        jump = np.sqrt((mx-prev_valid_centroid[0])**2 + (my-prev_valid_centroid[1])**2)
        if jump > MAX_CENTROID_JUMP_PX:
            invalid_reason = f"centroid_jump ({jump:.1f}px>{MAX_CENTROID_JUMP_PX}px)"

    # --- Pose estimation ---
    if invalid_reason is None:
        # Valid frame: compute pose
        xn_i = (mx - cx) / fx
        yn_i = (my - cy) / fy
        d_cam_i = np.array([xn_i, yn_i, 1.0])
        d_cam_i = d_cam_i / np.linalg.norm(d_cam_i)
        d_world_i = R.T @ d_cam_i
        alpha_i = (table_y - C[1]) / d_world_i[1] if abs(d_world_i[1]) > 1e-6 else 3.0
        pos = C + alpha_i * d_world_i

        pose = np.eye(4)
        pose[:3, 3] = pos
        pose[0, 0] = np.cos(theta); pose[0, 2] = np.sin(theta)
        pose[2, 0] = -np.sin(theta); pose[2, 2] = np.cos(theta)

        prev_valid_pose = pose.copy()
        prev_valid_theta = theta
        prev_valid_centroid = (mx, my)
        prev_valid_mask = mask.copy()
        stats["valid"] += 1
        valid = True
    else:
        # Anomaly: hold previous valid pose
        if prev_valid_pose is not None:
            pose = prev_valid_pose.copy()
        else:
            pose = np.eye(4)
        valid = False
        if "theta_jump" in invalid_reason:
            stats["theta_gate"] += 1
        if "centroid_jump" in invalid_reason:
            stats["centroid_gate"] += 1

    poses.append(pose)
    track_data.append({
        "frame_id": i,
        "frame_name": fname,
        "fg_pixels": int(fg_pixels),
        "centroid_2d": [float(mx), float(my)],
        "centroid_3d": pose[:3, 3].tolist(),
        "theta_rad": float(theta),
        "theta_deg": float(np.degrees(theta)),
        "sam_score": sam_score,
        "valid": valid,
        "invalid_reason": invalid_reason,
    })

    if i % 5 == 0 or not valid:
        tag = "OK" if valid else f"REJ({invalid_reason})"
        print(f"  [{i:3d}] {fname}: fg={fg_pixels:5d} theta={np.degrees(theta):5.1f}deg "
              f"centroid=({mx:5.0f},{my:5.0f}) {tag}")

poses = np.array(poses)
print(f"\nGate stats: {json.dumps(stats, indent=2)}")
print(f"Valid: {stats['valid']}/{stats['total']} ({100*stats['valid']/stats['total']:.0f}%)")

# --- Smooth valid trajectory ---
poses_smooth = poses.copy()
window = 3
for i in range(len(poses)):
    s, e = max(0, i-window//2), min(len(poses), i+window//2+1)
    poses_smooth[i, :3, 3] = poses[s:e, :3, 3].mean(axis=0)

# --- Save trajectory ---
np.savez(f'{OUT_DIR}/object_trajectory_v41_filtered.npz',
         poses=poses_smooth, timestamps=np.arange(len(poses)), raw_poses=poses)

traj_json = f'{OUT_DIR}/object_trajectory_v41_filtered.json'
with open(traj_json, 'w') as f:
    json.dump({
        "version": "v4.1_filtered",
        "sequence": "weigh_bread__2026_0701_0044_30",
        "method": "SAM v4.1 + quality gates (area/theta/centroid)",
        "gates": {"area_min": VALID_AREA_MIN, "area_max": VALID_AREA_MAX,
                   "theta_max_deg": MAX_THETA_JUMP_DEG, "centroid_max_px": MAX_CENTROID_JUMP_PX},
        "num_frames": len(poses),
        "num_valid": stats["valid"],
        "rejection_stats": {k: v for k, v in stats.items() if k != "valid"},
        "track_data": track_data,
        "pos_range_x": [float(poses_smooth[:,0,3].min()), float(poses_smooth[:,0,3].max())],
        "pos_range_z": [float(poses_smooth[:,2,3].min()), float(poses_smooth[:,2,3].max())],
    }, f, indent=2)

print(f"Trajectory: {traj_json}")

# --- Quality report ---
report = {
    "version": "v4.1_filtered",
    "gating_config": {
        "area_range": [VALID_AREA_MIN, VALID_AREA_MAX],
        "theta_jump_threshold_deg": MAX_THETA_JUMP_DEG,
        "centroid_jump_threshold_px": MAX_CENTROID_JUMP_PX,
    },
    "statistics": stats,
    "anomaly_frames": [d for d in track_data if not d["valid"]],
    "displacement_cm": float(np.linalg.norm(poses_smooth[-1,:3,3] - poses_smooth[0,:3,3]) * 100),
}
with open(f'{OUT_DIR}/trajectory_quality_report_v41.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"Report: {OUT_DIR}/trajectory_quality_report_v41.json")

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: XY trajectory (top-down)
ax = axes[0]
valid_mask = np.array([d["valid"] for d in track_data])
invalid_mask = ~valid_mask

pos_x = poses_smooth[:, 0, 3] * 100  # cm
pos_z = poses_smooth[:, 2, 3] * 100  # cm

ax.scatter(pos_z[valid_mask], pos_x[valid_mask], c='green', s=30, label=f'Valid ({valid_mask.sum()})', zorder=3)
ax.scatter(pos_z[invalid_mask], pos_x[invalid_mask], c='red', marker='x', s=80, label=f'Rejected ({invalid_mask.sum()})', zorder=4)
ax.plot(pos_z, pos_x, 'gray', alpha=0.3, linewidth=1)

# Annotate rejected frames
for d in track_data:
    if not d["valid"]:
        ax.annotate(str(d["frame_id"]), (d["centroid_3d"][2]*100, d["centroid_3d"][0]*100),
                    fontsize=7, color='red', alpha=0.8)

ax.set_xlabel('Z (cm)')
ax.set_ylabel('X (cm)')
ax.set_title('Bread Position (Top-Down)')
ax.legend(fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Plot 2: FG pixels over time
ax = axes[1]
fg_vals = [d["fg_pixels"] for d in track_data]
frame_ids = [d["frame_id"] for d in track_data]
colors_fg = ['green' if d["valid"] else 'red' for d in track_data]
ax.bar(frame_ids, fg_vals, color=colors_fg, alpha=0.7, width=1)
ax.axhline(y=VALID_AREA_MIN, color='orange', linestyle='--', alpha=0.5, label=f'Min ({VALID_AREA_MIN})')
ax.axhline(y=VALID_AREA_MAX, color='orange', linestyle='--', alpha=0.5, label=f'Max ({VALID_AREA_MAX})')
ax.set_xlabel('Frame')
ax.set_ylabel('FG Pixels')
ax.set_title('Mask Area per Frame')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Plot 3: Theta over time
ax = axes[2]
theta_vals = [abs(np.degrees(d["theta_rad"])) for d in track_data]
colors_theta = ['green' if d["valid"] else 'red' for d in track_data]
ax.bar(frame_ids, theta_vals, color=colors_theta, alpha=0.7, width=1)
ax.set_xlabel('Frame')
ax.set_ylabel('|Theta| (deg)')
ax.set_title('Orientation per Frame')
ax.grid(True, alpha=0.3)

plt.suptitle('Phase 5 v2: Filtered Pose Tracking (SAM v4.1 + Quality Gates)', fontsize=13)
plt.tight_layout()
plot_path = f'{OUT_DIR}/trajectory_plot_v41_filtered.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot: {plot_path}")

# --- Summary ---
print("\n" + "=" * 60)
print("Phase 5 v2 Complete")
print("=" * 60)
print(f"  Valid frames: {stats['valid']}/{stats['total']}")
print(f"  Rejected: area_gate={stats['area_gate']}, theta_gate={stats['theta_gate']}, "
      f"centroid_gate={stats['centroid_gate']}, low_px={stats['low_px']}")
print(f"  Displacement: {report['displacement_cm']:.1f} cm")
print(f"  Output: {traj_json}")
print(f"  Plot: {plot_path}")
print("Done!")
