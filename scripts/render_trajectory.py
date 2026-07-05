#!/usr/bin/env python3.8
"""
IsaacGym Trajectory Renderer: replay object pose trajectory and render video.

Loads object URDF + tracked trajectory, replays in headless IsaacGym,
captures multi-view renders, and outputs MP4 video files.

Usage:
  python3.8 scripts/render_trajectory.py --object bread
  python3.8 scripts/render_trajectory.py --object pipette
  python3.8 scripts/render_trajectory.py --all
"""
import os, sys, json, argparse, math, tempfile
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

CAM_WIDTH = 1280
CAM_HEIGHT = 720
FPS = 15

CAMERAS = [
    {'name': 'persp',  'offset': (0.2, 0.25, 0.2)},
    {'name': 'top',    'offset': (0.0, 0.35, 0.0)},
    {'name': 'front',  'offset': (0.0, 0.05, 0.3)},
    {'name': 'side',   'offset': (0.3, 0.05, 0.0)},
]


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
            frames.append((t['frame'], pos, quat))
        elif 'position' in t:
            frames.append((t['frame'], np.array(t['position']), np.array([0, 0, 0, 1])))
    return frames


def rotation_matrix_to_quat_xyzw(R):
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
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


def create_camera(gym, env, name, width, height):
    cam_props = gymapi.CameraProperties()
    cam_props.width = width
    cam_props.height = height
    cam_props.enable_tensors = False
    return gym.create_camera_sensor(env, cam_props)


def update_camera(gym, env, cam_handle, pos, target):
    gym.set_camera_location(cam_handle, env,
        gymapi.Vec3(pos[0], pos[1], pos[2]),
        gymapi.Vec3(target[0], target[1], target[2]))


def render_sequence(obj_name, seq_name):
    frames = load_trajectory(obj_name, seq_name)
    if frames is None or len(frames) == 0:
        print(f"  No trajectory for {seq_name}")
        return

    urdf_dir = os.path.join(ASSET_ROOT, obj_name, 'asset')
    urdf_path = os.path.join(urdf_dir, 'object.urdf')
    if not os.path.exists(urdf_path):
        print(f"  URDF not found: {urdf_path}")
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

    env_spacing = 2.0
    env_lower = gymapi.Vec3(-env_spacing, 0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(frames[0][1][0], frames[0][1][1], frames[0][1][2])
    start_pose.r = gymapi.Quat(frames[0][2][0], frames[0][2][1], frames[0][2][2], frames[0][2][3])
    actor = gym.create_actor(env, asset, start_pose, obj_name, 0, 1)

    cameras = []
    for cam_cfg in CAMERAS:
        ch = create_camera(gym, env, cam_cfg['name'], CAM_WIDTH, CAM_HEIGHT)
        cameras.append((cam_cfg['name'], cam_cfg['offset'], ch))

    out_dir = os.path.join(RENDER_ROOT, obj_name, seq_name)
    os.makedirs(out_dir, exist_ok=True)
    writers = {}
    frame_buffers = {}
    for cam_name, _, _ in cameras:
        out_path = os.path.join(out_dir, f'{cam_name}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers[cam_name] = cv2.VideoWriter(out_path, fourcc, FPS, (CAM_WIDTH, CAM_HEIGHT))
        frame_buffers[cam_name] = []

    gym.prepare_sim(sim)

    n_steps = len(frames)
    for i in range(n_steps):
        fidx, pos, quat = frames[i]
        body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        body_states['pose']['p'][0] = (float(pos[0]), float(pos[1]), float(pos[2]))
        body_states['pose']['r'][0] = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        gym.set_actor_rigid_body_states(env, actor, body_states, gymapi.STATE_ALL)

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)

        for cam_name, cam_offset, cam_handle in cameras:
            cam_pos = (pos[0] + cam_offset[0], pos[1] + cam_offset[1], pos[2] + cam_offset[2])
            update_camera(gym, env, cam_handle, cam_pos, pos)
            gym.render_all_camera_sensors(sim)

        for cam_name, cam_offset, cam_handle in cameras:
            img = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
            if img.size > 0:
                rgb = img.reshape(CAM_HEIGHT, CAM_WIDTH, 4)[:, :, :3]
                bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                writers[cam_name].write(bgr)

        if i % 50 == 0:
            print(f"    frame {i}/{n_steps}")

    for w in writers.values():
        w.release()

    gym.destroy_sim(sim)
    del gym

    total_size = sum(os.path.getsize(os.path.join(out_dir, f'{n}.mp4')) for n in writers if os.path.exists(os.path.join(out_dir, f'{n}.mp4')))
    print(f"  Saved {len(cameras)} views ({total_size/1024/1024:.1f}MB) → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', default=None, help='Single object name')
    parser.add_argument('--seq', default=None, help='Single sequence name')
    parser.add_argument('--all', action='store_true', help='Render all sequences')
    args = parser.parse_args()

    if args.object and args.seq:
        render_sequence(args.object, args.seq)
        return

    if args.object:
        for seq_name in OBJECT_SEQUENCES.get(args.object, []):
            print(f"\n{args.object}: {seq_name}")
            render_sequence(args.object, seq_name)
        return

    if args.all:
        for obj_name in OBJECT_SEQUENCES:
            for seq_name in OBJECT_SEQUENCES[obj_name]:
                print(f"\n{obj_name}: {seq_name}")
                render_sequence(obj_name, seq_name)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
