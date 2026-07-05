#!/usr/bin/env python3.8
"""
Generate geometry consistency evidence: overlay 3D model on video frames.
Projects reconstructed model onto camera view using tracked pose.
"""
import os, sys, json, argparse
import numpy as np
import cv2
import trimesh
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from object_recon.pose_tracking import load_intrinsics, load_extrinsics

DATA_ROOT = os.environ.get('HO_TRACKER_DATA', '/mnt/workspace/Hackthon/data/human_demo')
TRAJ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mask_pose')
MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'object_asset_v1')
VIZ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'geometry_viz')

OBJECTS = {
    'bread':      'weigh_bread__2026_0701_0044_30',
    'pipette':    'grasp_pipette_stand__2026_0701_0019_19',
    'drink_ad':   'weigh_drink_ad__2026_0701_0047_56',
    'drink_yykx': 'weigh_drink_yykx__2026_0701_0051_12',
}


def load_projection(seq, cam):
    K = np.asarray(load_intrinsics(Path(DATA_ROOT) / seq / 'camera_calib', cam), dtype='float64')
    E = np.asarray(load_extrinsics(Path(DATA_ROOT) / seq / 'camera_calib', cam), dtype='float64')
    return K @ E[:3, :4]


def project_points(pts_3d, P):
    pts_h = np.column_stack([pts_3d, np.ones(len(pts_3d))])
    uv_h = (P @ pts_h.T).T
    in_front = uv_h[:, 2] > 0
    u = uv_h[in_front, 0] / uv_h[in_front, 2]
    v = uv_h[in_front, 1] / uv_h[in_front, 2]
    valid = (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
    return np.column_stack([u, v])[valid]


def draw_mesh_projection(frame, mesh_verts, P, color=(255, 165, 0), alpha=0.4):
    pts_2d = project_points(mesh_verts, P)
    if len(pts_2d) < 3:
        return
    pts_int = pts_2d.astype(np.int32)
    hull = cv2.convexHull(pts_int)
    if hull is not None and len(hull) >= 3:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [hull], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [hull], True, color, 2)


def generate_overlay(obj_name):
    seq = OBJECTS[obj_name]
    model_path = os.path.join(MODEL_ROOT, obj_name, 'mesh', 'visual_mesh.obj')
    if not os.path.exists(model_path):
        print(f"  No model: {model_path}")
        return
    mesh = trimesh.load(model_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.geometry[list(mesh.geometry.keys())[0]]

    traj_path = os.path.join(TRAJ_ROOT, obj_name, seq, 'object_trajectory.json')
    if not os.path.exists(traj_path):
        print(f"  No trajectory")
        return
    with open(traj_path) as f:
        traj = json.load(f)

    pos_map = {}
    for t in traj.get('trajectory', []):
        tf = t.get('transform_4x4')
        if tf:
            pos_map[t['frame']] = np.array([tf[0][3], tf[1][3], tf[2][3]])
        elif 'position' in t:
            pos_map[t['frame']] = np.array(t['position'])

    video_path = os.path.join(DATA_ROOT, seq, 'video', 'camera_top.mkv')
    if not os.path.exists(video_path):
        print(f"  No video")
        return

    try:
        P = load_projection(seq, 'camera_top')
    except Exception as e:
        print(f"  Camera error: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir = os.path.join(VIZ_ROOT, obj_name, seq)
    os.makedirs(out_dir, exist_ok=True)

    sample_frames = list(range(0, total_frames, 15))
    if not sample_frames:
        sample_frames = [0]
    generated = 0

    for fidx in sample_frames:
        if fidx not in pos_map:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        pos_3d = pos_map[fidx]
        model_verts = mesh.vertices.copy() + pos_3d
        draw_mesh_projection(frame, model_verts, P)

        uv_pts = project_points(pos_3d.reshape(1, 3), P)
        if len(uv_pts) > 0:
            u, v = uv_pts[0].astype(int)
            cv2.circle(frame, (u, v), 6, (0, 0, 255), -1)
            cv2.circle(frame, (u, v), 8, (0, 0, 255), 2)

        cv2.putText(frame, f'{obj_name} | frame {fidx}/{total_frames}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'3D: ({pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f})m', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, f'geom_frame_{fidx:06d}.jpg'), frame)
        generated += 1

    cap.release()
    print(f"  {obj_name}: {generated} frames → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=list(OBJECTS.keys()))
    args = parser.parse_args()
    for obj_name in args.objects:
        generate_overlay(obj_name)
    print(f"\nOutput: {VIZ_ROOT}")


if __name__ == '__main__':
    main()
