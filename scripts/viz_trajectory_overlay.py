#!/usr/bin/env python3.8
"""
Generate trajectory replay overlay on top-camera video.
For each sequence, project the 3D trajectory back onto video frames
to verify the trajectory is consistent with the observed motion.
"""
import os, json, argparse
import numpy as np
import cv2

DATA_ROOT = os.environ.get('HO_TRACKER_DATA', '/mnt/workspace/Hackthon/data/human_demo')
TRAJ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mask_pose')
VIZ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'trajectory_viz')

OBJECT_SEQUENCES = {
    'bread':      ['weigh_bread__2026_0701_0044_30', 'weigh_bread__left__2026_0701_0046_02'],
    'pipette':    ['grasp_pipette_stand__2026_0701_0019_19', 'grasp_pipette_rotate__2026_0701_0025_42', 'grasp_pipette_press__2026_0701_0028_11'],
    'drink_ad':   ['weigh_drink_ad__2026_0701_0047_56', 'weigh_drink_ad__left__2026_0701_0049_04'],
    'drink_yykx': ['weigh_drink_yykx__2026_0701_0051_12', 'weigh_drink_yykx__left__2026_0701_0052_53', 'grasp_drink_yykx__2026_0701_0054_45'],
}


def load_camera(seq, cam):
    with open(os.path.join(DATA_ROOT, seq, 'camera_calib', cam, 'calib.json')) as f:
        d = json.load(f)
    K = np.array(d['K']); E = np.array(d['E'])
    R, t = E[:3, :3], E[:3, 3]
    P = K @ np.hstack([R, t.reshape(3, 1)])
    return {'K': K, 'P': P}


def project_3d_to_2d(point_3d, P):
    """Project 3D world point to 2D image point using camera matrix P."""
    p = np.append(point_3d, 1.0)
    uv = P @ p
    if uv[2] <= 0:
        return None
    u, v = uv[0] / uv[2], uv[1] / uv[2]
    return (int(u), int(v))


def generate_overlay(obj_name, seq_name):
    traj_path = os.path.join(TRAJ_ROOT, obj_name, seq_name, 'object_trajectory.json')
    if not os.path.exists(traj_path):
        print(f"  No trajectory for {seq_name}")
        return

    with open(traj_path) as f:
        traj = json.load(f)

    video_path = os.path.join(DATA_ROOT, seq_name, 'video', 'camera_top.mkv')
    if not os.path.exists(video_path):
        print(f"  No top video for {seq_name}")
        return

    cam = load_camera(seq_name, 'camera_top')

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = os.path.join(VIZ_ROOT, obj_name, seq_name)
    os.makedirs(out_dir, exist_ok=True)

    # Extract trajectory positions
    positions = []
    for t in traj['trajectory']:
        positions.append((t['frame'], t['position']))
    positions.sort()

    # Build frame-to-position map
    pos_map = {p[0]: p[1] for p in positions}

    # Sample key frames for static overlay (every 15 frames)
    sample_indices = list(range(0, total_frames, 15))

    for fidx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        overlay = frame.copy()

        # Draw all trajectory points up to this frame
        for p_frame, p_pos in positions:
            if p_frame > fidx:
                break
            uv = project_3d_to_2d(p_pos, cam['P'])
            if uv is None:
                continue
            u, v = uv
            if 0 <= u < 1280 and 0 <= v < 720:
                # Older points: smaller, dimmer
                alpha = 0.3 + 0.7 * (p_frame / max(1, fidx))
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.circle(overlay, (u, v), 3, color, -1)

        # Draw current position larger
        if fidx in pos_map:
            uv = project_3d_to_2d(pos_map[fidx], cam['P'])
            if uv is not None and 0 <= uv[0] < 1280 and 0 <= uv[1] < 720:
                cv2.circle(overlay, uv, 8, (0, 0, 255), -1)
                cv2.circle(overlay, uv, 10, (0, 0, 255), 2)

        # Draw trajectory path
        traj_pts = []
        for p_frame, p_pos in positions:
            if p_frame > fidx:
                break
            uv = project_3d_to_2d(p_pos, cam['P'])
            if uv is not None and 0 <= uv[0] < 1280 and 0 <= uv[1] < 720:
                traj_pts.append(uv)
        for i in range(1, len(traj_pts)):
            cv2.line(overlay, traj_pts[i-1], traj_pts[i], (0, 255, 255), 1)

        # Legend
        cv2.putText(overlay, f'{obj_name} / {seq_name[:30]}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f'Frame {fidx}/{total_frames}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f'Trajectory frames: {len(traj_pts)}', (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        out_path = os.path.join(out_dir, f'frame_{fidx:06d}.jpg')
        cv2.imwrite(out_path, overlay)

    cap.release()
    print(f"  Saved {len(sample_indices)} overlay frames to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=list(OBJECT_SEQUENCES.keys()))
    args = parser.parse_args()

    for obj_name in args.objects:
        print(f"\n{'='*60}")
        print(f"Trajectory Overlay: {obj_name}")
        print(f"{'='*60}")
        for seq_name in OBJECT_SEQUENCES[obj_name]:
            print(f"  {seq_name}")
            generate_overlay(obj_name, seq_name)

    # Generate summary
    total = 0
    for obj_name in args.objects:
        for seq_name in OBJECT_SEQUENCES[obj_name]:
            out_dir = os.path.join(VIZ_ROOT, obj_name, seq_name)
            if os.path.isdir(out_dir):
                n = len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
                total += n
    print(f"\nTotal overlay frames: {total}")
    print(f"Output: {VIZ_ROOT}")


if __name__ == '__main__':
    main()
