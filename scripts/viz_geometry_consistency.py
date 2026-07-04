#!/usr/bin/env python3.8
"""
Generate geometry consistency evidence: overlay 3D model on video frames.
Shows that the reconstructed 3D model matches the real object in the video
by projecting the model onto the camera view at the tracked 3D position.
"""
import os, json, argparse
import numpy as np
import cv2
import trimesh

DATA_ROOT = '/mnt/workspace/Hackthon/data/human_demo'
TRAJ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mask_pose')
MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'object_asset_v3')
VIZ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'geometry_viz')

OBJECTS = {
    'bread':      {'seq': 'weigh_bread__2026_0701_0044_30'},
    'pipette':    {'seq': 'grasp_pipette_stand__2026_0701_0019_19'},
    'drink_ad':   {'seq': 'weigh_drink_ad__2026_0701_0047_56'},
    'drink_yykx': {'seq': 'weigh_drink_yykx__2026_0701_0051_12'},
}


def load_camera(seq, cam):
    with open(os.path.join(DATA_ROOT, seq, 'camera_calib', cam, 'calib.json')) as f:
        d = json.load(f)
    K = np.array(d['K']); E = np.array(d['E'])
    R, t = E[:3, :3], E[:3, 3]
    P = K @ np.hstack([R, t.reshape(3, 1)])
    return {'K': K, 'R': R, 't': t, 'P': P, 'C': -R.T @ t}


def project_points(points_3d, P):
    """Project 3D points to 2D image coordinates."""
    pts_h = np.column_stack([points_3d, np.ones(len(points_3d))])
    uv_h = (P @ pts_h.T).T
    in_front = uv_h[:, 2] > 0
    u = uv_h[in_front, 0] / uv_h[in_front, 2]
    v = uv_h[in_front, 1] / uv_h[in_front, 2]
    valid = (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)
    return np.column_stack([u, v])[valid], valid


def draw_mesh_projection(frame, mesh_verts, P, color=(255, 165, 0), alpha=0.4):
    """Draw projected mesh vertices as semi-transparent points on frame."""
    pts_2d, valid = project_points(mesh_verts, P)
    if len(pts_2d) < 3:
        return

    # Draw convex hull outline
    pts_int = pts_2d.astype(np.int32)
    hull = cv2.convexHull(pts_int)
    if hull is not None and len(hull) >= 3:
        # Draw filled hull with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [hull], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw hull outline
        cv2.polylines(frame, [hull], True, color, 2)


def generate_geometry_overlay(obj_name):
    cfg = OBJECTS[obj_name]
    seq = cfg['seq']

    # Load model
    model_path = os.path.join(MODEL_ROOT, obj_name, f'{obj_name}_visual.obj')
    if not os.path.exists(model_path):
        print(f"  No model for {obj_name}")
        return
    mesh = trimesh.load(model_path)

    # Load trajectory
    traj_path = os.path.join(TRAJ_ROOT, obj_name, seq, 'object_trajectory.json')
    if not os.path.exists(traj_path):
        print(f"  No trajectory")
        return
    with open(traj_path) as f:
        traj = json.load(f)

    # Build a map from frame index to 3D position
    pos_map = {}
    for t in traj['trajectory']:
        pos_map[t['frame']] = np.array(t['position'])

    # Open video
    video_path = os.path.join(DATA_ROOT, seq, 'video', 'camera_top.mkv')
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load camera
    cam = load_camera(seq, 'camera_top')

    out_dir = os.path.join(VIZ_ROOT, obj_name, seq)
    os.makedirs(out_dir, exist_ok=True)

    # Generate overlay at key frames (every 15 frames)
    sample_frames = list(range(0, total_frames, 15))
    generated = 0

    for fidx in sample_frames:
        if fidx not in pos_map:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        pos_3d = pos_map[fidx]

        # Place model at tracked 3D position
        model_verts = mesh.vertices.copy()
        # Model is centered at origin; translate to tracked position
        # Model Z center = 0; object sits on table at Z≈0
        # Align model bottom to Z=0 (the model's centroid is at origin after centering)
        model_verts += pos_3d

        # Draw model projection
        draw_mesh_projection(frame, model_verts, cam['P'])

        # Draw tracked position marker
        uv = project_points(pos_3d.reshape(1, 3), cam['P'])[0]
        if len(uv) > 0:
            uv = uv[0].astype(int)
            cv2.circle(frame, tuple(uv), 6, (0, 0, 255), -1)
            cv2.circle(frame, tuple(uv), 8, (0, 0, 255), 2)

        # Info overlay
        cv2.putText(frame, f'{obj_name} | frame {fidx}/{total_frames}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f'3D pos: ({pos_3d[0]:.3f}, {pos_3d[1]:.3f}, {pos_3d[2]:.3f})m', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f'Model extents: {mesh.extents[0]:.3f}x{mesh.extents[1]:.3f}x{mesh.extents[2]:.3f}m', (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        out_path = os.path.join(out_dir, f'geom_frame_{fidx:06d}.jpg')
        cv2.imwrite(out_path, frame)
        generated += 1

    cap.release()
    print(f"  {obj_name}: {generated} geometry overlay frames → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', nargs='+', default=list(OBJECTS.keys()))
    args = parser.parse_args()

    for obj_name in args.objects:
        print(f"  {obj_name}")
        generate_geometry_overlay(obj_name)

    print(f"\nOutput: {VIZ_ROOT}")


if __name__ == '__main__':
    main()
