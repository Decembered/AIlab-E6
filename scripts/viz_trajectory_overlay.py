#!/usr/bin/env python3.8
"""
Generate trajectory replay overlay frames on top-camera video (V3.1).
Projects 3D trajectory back onto video to verify motion consistency.
"""
import os, sys, json, argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from object_recon.pose_tracking import load_intrinsics, load_extrinsics

DATA_ROOT = os.environ.get('HO_TRACKER_DATA', '/mnt/workspace/Hackthon/data/human_demo')
TRAJ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mask_pose')
VIZ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'trajectory_viz')

OBJECT_SEQUENCES = {
    'bread':      ['weigh_bread__2026_0701_0044_30', 'weigh_bread__left__2026_0701_0046_02'],
    'pipette':    [
        'grasp_pipette_stand__2026_0701_0019_19', 'grasp_pipette_rotate__2026_0701_0025_42',
        'grasp_pipette_press__2026_0701_0028_11', 'pipette_rh_beaker__2026_0701_0035_47',
        'pipette_rh_beaker_testtube__2026_0701_0039_28',
    ],
    'drink_ad':   ['weigh_drink_ad__2026_0701_0047_56', 'weigh_drink_ad__left__2026_0701_0049_04'],
    'drink_yykx': ['weigh_drink_yykx__2026_0701_0051_12', 'weigh_drink_yykx__left__2026_0701_0052_53',
                   'grasp_drink_yykx__2026_0701_0054_45'],
}


def load_projection(seq, cam):
    K = np.asarray(load_intrinsics(Path(DATA_ROOT) / seq / 'camera_calib', cam), dtype='float64')
    E = np.asarray(load_extrinsics(Path(DATA_ROOT) / seq / 'camera_calib', cam), dtype='float64')
    P = K @ E[:3, :4]
    return P


def project_3d(p, P):
    p_h = np.append(p, 1.0)
    uv = P @ p_h
    if uv[2] <= 0:
        return None
    return int(uv[0] / uv[2]), int(uv[1] / uv[2])


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

    try:
        P = load_projection(seq_name, 'camera_top')
    except Exception as e:
        print(f"  Camera error: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir = os.path.join(VIZ_ROOT, obj_name, seq_name)
    os.makedirs(out_dir, exist_ok=True)

    positions = []
    for t in traj.get('trajectory', []):
        tf = t.get('transform_4x4')
        if tf:
            positions.append((t['frame'], [tf[0][3], tf[1][3], tf[2][3]]))
        elif 'position' in t:
            positions.append((t['frame'], t['position']))
    positions.sort()
    pos_map = {p[0]: np.array(p[1]) for p in positions}

    sample_frames = list(range(0, total_frames, 15))
    if not sample_frames:
        sample_frames = [0]

    for fidx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        overlay = frame.copy()
        traj_pts = []
        for p_frame, p_pos in positions:
            if p_frame > fidx:
                break
            uv = project_3d(p_pos, P)
            if uv is None:
                continue
            u, v = uv
            if 0 <= u < 1280 and 0 <= v < 720:
                alpha = 0.3 + 0.7 * (p_frame / max(1, fidx))
                cv2.circle(overlay, (u, v), 3, (0, int(255 * alpha), int(255 * (1 - alpha))), -1)
                traj_pts.append((u, v))

        if fidx in pos_map:
            uv = project_3d(pos_map[fidx], P)
            if uv is not None and 0 <= uv[0] < 1280 and 0 <= uv[1] < 720:
                cv2.circle(overlay, uv, 8, (0, 0, 255), -1)
                cv2.circle(overlay, uv, 10, (0, 0, 255), 2)

        for i in range(1, len(traj_pts)):
            cv2.line(overlay, traj_pts[i - 1], traj_pts[i], (0, 255, 255), 1)

        cv2.putText(overlay, f'{obj_name} / {seq_name[:35]}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f'Frame {fidx}/{total_frames}  Trail: {len(traj_pts)} pts', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, f'frame_{fidx:06d}.jpg'), overlay)

    cap.release()
    print(f"  Saved to {out_dir} ({len(sample_frames)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=list(OBJECT_SEQUENCES.keys()))
    args = parser.parse_args()

    total = 0
    for obj_name in args.objects:
        print(f"\nTrajectory Overlay: {obj_name}")
        for seq_name in OBJECT_SEQUENCES[obj_name]:
            print(f"  {seq_name}")
            generate_overlay(obj_name, seq_name)
            out_dir = os.path.join(VIZ_ROOT, obj_name, seq_name)
            if os.path.isdir(out_dir):
                total += len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
    print(f"\nTotal overlay frames: {total}")
    print(f"Output: {VIZ_ROOT}")


if __name__ == '__main__':
    main()
