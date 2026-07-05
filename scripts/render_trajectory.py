#!/usr/bin/env python3.8
"""
IsaacGym Trajectory Renderer: replay object pose trajectory and render video.

Loads object URDF + tracked trajectory, replays in headless IsaacGym,
captures multi-view renders with text overlays, outputs MP4 videos.

Usage:
  python3.8 scripts/render_trajectory.py --object bread
  python3.8 scripts/render_trajectory.py --all
"""
import os, sys, json, argparse, math
import numpy as np
import cv2
from pathlib import Path
from isaacgym import gymapi

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_ROOT = os.path.join(SCRIPT_DIR, 'outputs', 'mask_pose')
ASSET_ROOT = os.path.join(SCRIPT_DIR, 'runs', 'object_asset_v1')
RENDER_ROOT = os.path.join(SCRIPT_DIR, 'outputs', 'isaacgym_trajectory_render')

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

SEQUENCE_LABELS = {
    'bread':      {'weigh': 'Bread — Weigh bread on scale'},
    'pipette':    {'grasp_pipette_stand': 'Pipette — Grasp & stand',
                   'grasp_pipette_rotate': 'Pipette — Grasp & rotate',
                   'grasp_pipette_press': 'Pipette — Grasp & press',
                   'pipette_rh_beaker': 'Pipette — Transfer to beaker',
                   'pipette_rh_beaker_testtube': 'Pipette — Transfer to test tube'},
    'drink_ad':   {'weigh_drink_ad': 'Drink AD — Weigh bottle on scale'},
    'drink_yykx': {'weigh_drink_yykx': 'Drink YYKX — Weigh bottle on scale',
                   'grasp_drink_yykx': 'Drink YYKX — Grasp bottle'},
}

CAM_WIDTH = 1280
CAM_HEIGHT = 720
FPS = 15

CAMERAS = [
    {'name': 'persp',  'offset': (0.25, 0.18, 0.25), 'label': 'Perspective'},
    {'name': 'top',    'offset': (0.01, 0.35, 0.0),   'label': 'Top View'},
    {'name': 'front',  'offset': (0.0, 0.08, 0.40),   'label': 'Front View'},
    {'name': 'side',   'offset': (0.40, 0.08, 0.0),   'label': 'Side View'},
]


def get_sequence_label(obj_name, seq_name):
    task = seq_name.split('__2026_')[0]
    for key, label in SEQUENCE_LABELS.get(obj_name, {}).items():
        if task.startswith(key):
            return label
    return f'{obj_name} / {seq_name[:25]}'


def load_trajectory(obj_name, seq_name):
    traj_path = os.path.join(MASK_ROOT, obj_name, seq_name, 'object_trajectory.json')
    if not os.path.exists(traj_path):
        return None
    with open(traj_path) as f:
        data = json.load(f)
    frames = []
    for t in data.get('trajectory', []):
        tf = t.get('transform_4x4')
        if tf:
            pos = np.array([tf[0][3], tf[1][3], tf[2][3]])
            R = np.array([tf[i][:3] for i in range(3)])
            quat = rotation_matrix_to_quat_xyzw(R)
            frames.append({'frame': t['frame'], 'timestamp': t['timestamp'],
                          'pos': pos, 'quat': quat})
        elif 'position' in t:
            frames.append({'frame': t['frame'], 'timestamp': t['timestamp'],
                          'pos': np.array(t['position']), 'quat': np.array([0, 0, 0, 1])})
    return frames


def rotation_matrix_to_quat_xyzw(R):
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
        qw = 0.25 * s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype='float64')
    return q / np.linalg.norm(q)


def draw_overlay(frame, obj_name, seq_name, cam_name, cam_label, frame_idx, timestamp, n_total, pos):
    h, w = frame.shape[:2]

    # Semi-transparent title bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Object + sequence label
    label = get_sequence_label(obj_name, seq_name)
    cv2.putText(frame, label, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Camera view label (right-aligned)
    cv2.putText(frame, cam_label, (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1, cv2.LINE_AA)

    # Frame info at bottom
    info = f'Frame: {frame_idx:>4}  |  Time: {timestamp:.2f}s  |  Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m'
    cv2.rectangle(overlay, (0, h - 36), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, info, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Progress bar at very bottom
    progress = min(1.0, max(0.0, frame_idx / max(1, n_total - 1)))
    bar_y, bar_h = h - 6, 4
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (0, bar_y), (int(w * progress), bar_y + bar_h), (0, 180, 255), -1)


def render_sequence(obj_name, seq_name):
    frames = load_trajectory(obj_name, seq_name)
    if frames is None or len(frames) == 0:
        print(f"  No trajectory for {seq_name}")
        return

    urdf_dir = os.path.join(ASSET_ROOT, obj_name, 'asset')
    if not os.path.exists(os.path.join(urdf_dir, 'object.urdf')):
        print(f"  URDF not found: {urdf_dir}")
        return

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / FPS
    sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)
    sim_params.up_axis = gymapi.UP_AXIS_Y

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print(f"  FAIL: create_sim")
        return

    gym.add_ground(sim, gymapi.PlaneParams())

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = True
    asset_options.armature = 0.01

    try:
        asset = gym.load_asset(sim, urdf_dir, 'object.urdf', asset_options)
    except Exception as e:
        print(f"  FAIL: load_asset: {e}")
        gym.destroy_sim(sim)
        return
    if asset is None:
        print(f"  FAIL: load_asset returned None")
        gym.destroy_sim(sim)
        return

    env_spacing = 3.0
    env = gym.create_env(sim, gymapi.Vec3(-env_spacing, 0, -env_spacing),
                         gymapi.Vec3(env_spacing, env_spacing, env_spacing), 1)

    f0 = frames[0]
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(f0['pos'][0], f0['pos'][1], f0['pos'][2])
    start_pose.r = gymapi.Quat(f0['quat'][0], f0['quat'][1], f0['quat'][2], f0['quat'][3])
    actor = gym.create_actor(env, asset, start_pose, obj_name, 0, 1)

    cameras = []
    for cam_cfg in CAMERAS:
        cam_props = gymapi.CameraProperties()
        cam_props.width = CAM_WIDTH
        cam_props.height = CAM_HEIGHT
        cam_props.enable_tensors = False
        ch = gym.create_camera_sensor(env, cam_props)
        cameras.append((cam_cfg['name'], cam_cfg['offset'], cam_cfg['label'], ch))

    out_dir = os.path.join(RENDER_ROOT, obj_name, seq_name)
    os.makedirs(out_dir, exist_ok=True)
    writers = {}
    for cam_name, _, _, _ in cameras:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers[cam_name] = cv2.VideoWriter(
            os.path.join(out_dir, f'{cam_name}.mp4'), fourcc, FPS, (CAM_WIDTH, CAM_HEIGHT))

    gym.prepare_sim(sim)

    n_steps = len(frames)
    cam_buffers = {c[0]: None for c in cameras}

    for i in range(n_steps):
        f = frames[i]
        body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        body_states['pose']['p'][0] = (float(f['pos'][0]), float(f['pos'][1]), float(f['pos'][2]))
        body_states['pose']['r'][0] = (float(f['quat'][0]), float(f['quat'][1]),
                                       float(f['quat'][2]), float(f['quat'][3]))
        gym.set_actor_rigid_body_states(env, actor, body_states, gymapi.STATE_ALL)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)

        for cam_name, cam_offset, cam_label, cam_handle in cameras:
            cam_pos = (f['pos'][0] + cam_offset[0],
                       f['pos'][1] + cam_offset[1],
                       f['pos'][2] + cam_offset[2])
            target = (f['pos'][0] + 0.005, f['pos'][1], f['pos'][2])
            gym.set_camera_location(cam_handle, env,
                gymapi.Vec3(cam_pos[0], cam_pos[1], cam_pos[2]),
                gymapi.Vec3(target[0], target[1], target[2]))

        gym.render_all_camera_sensors(sim)

        for cam_name, cam_offset, cam_label, cam_handle in cameras:
            img = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
            if img.size > 0 and img.any():
                rgb = img.reshape(CAM_HEIGHT, CAM_WIDTH, 4)[:, :, :3].copy()
                bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                draw_overlay(bgr, obj_name, seq_name, cam_name, cam_label,
                            f['frame'], f['timestamp'], n_steps, f['pos'])
                writers[cam_name].write(bgr)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"    frame {i + 1}/{n_steps}")

    for w in writers.values():
        w.release()
    gym.destroy_sim(sim)
    del gym

    total_mb = sum(os.path.getsize(os.path.join(out_dir, f'{n}.mp4'))
                   for n in writers if os.path.exists(os.path.join(out_dir, f'{n}.mp4')))
    print(f"  \u2192 {len(cameras)} views ({total_mb/1024/1024:.1f}MB) {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', default=None)
    parser.add_argument('--seq', default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.object and args.seq:
        render_sequence(args.object, args.seq)
    elif args.object:
        for seq_name in OBJECT_SEQUENCES.get(args.object, []):
            print(f"\n{args.object}: {seq_name}")
            render_sequence(args.object, seq_name)
    elif args.all:
        for obj_name in OBJECT_SEQUENCES:
            for seq_name in OBJECT_SEQUENCES[obj_name]:
                print(f"\n{obj_name}: {seq_name}")
                render_sequence(obj_name, seq_name)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
