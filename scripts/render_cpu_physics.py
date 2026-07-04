#!/usr/bin/env python3
"""Render IsaacGym asset using CPU physics + GPU graphics.

IsaacGym Preview 4 GPU PhysX kernels don't support SM 8.9 (Ada Lovelace),
but graphics rendering (camera sensor, step_graphics, get_camera_image)
uses CUDA graphics interop which IS compatible with Ada GPUs.

Strategy: physics on CPU, rendering on GPU.
"""
import argparse, sys, os
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True, help='Path to URDF file')
    parser.add_argument('--out', default='render_output', help='Output directory')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    args = parser.parse_args()

    urdf_path = Path(args.urdf).resolve()
    asset_dir = str(urdf_path.parent)
    asset_file = urdf_path.name
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading IsaacGym...")
    from isaacgym import gymapi, gymutil
    import numpy as np

    gym = gymapi.acquire_gym()

    # --- KEY: CPU physics, no GPU pipeline ---
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)
    sim_params.up_axis = gymapi.UP_AXIS_Y
    sim_params.use_gpu_pipeline = False  # CPU PhysX (compatible with Ada)

    # Graphics device: use GPU 0 (even with CPU physics, graphics can use GPU)
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("FAIL: create_sim returned None")
        sys.exit(1)

    print(f"Sim created (CPU physics)")

    # Ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 1, 0)
    plane_params.distance = 0.0
    gym.add_ground(sim, plane_params)

    # Load asset
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False
    asset_options.flip_visual_attachments = False

    asset = gym.load_asset(sim, asset_dir, asset_file, asset_options)
    if asset is None:
        print("FAIL: load_asset returned None")
        sys.exit(1)

    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_shapes = gym.get_asset_rigid_shape_count(asset)
    print(f"Asset loaded: {num_bodies} body(s), {num_shapes} shape(s)")

    # Create environment
    env_spacing = 3.0
    env = gym.create_env(sim, gymapi.Vec3(-env_spacing, 0, -env_spacing),
                         gymapi.Vec3(env_spacing, env_spacing, env_spacing), 1)

    # Spawn actor above ground
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.10, 0.0)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    actor = gym.create_actor(env, asset, pose, "object", 0, 1)
    print(f"Actor spawned")

    # --- Try GPU camera rendering ---
    cam_props = gymapi.CameraProperties()
    cam_props.width = args.width
    cam_props.height = args.height
    cam_props.use_collision_geometry = False

    try:
        cam_handle = gym.create_camera_sensor(env, cam_props)
        print(f"Camera sensor created OK")
    except Exception as e:
        print(f"Camera sensor FAILED: {e}")
        gym.destroy_sim(sim)
        sys.exit(1)

    # Camera views
    views = [
        ("front",   gymapi.Vec3(0.15, 0.10, 0.20), gymapi.Vec3(0, 0.02, 0)),
        ("side",    gymapi.Vec3(0.20, 0.06, 0.0),  gymapi.Vec3(0, 0.02, 0)),
        ("top",     gymapi.Vec3(0.0, 0.18, 0.001), gymapi.Vec3(0, 0.02, 0)),
        ("angle",   gymapi.Vec3(0.12, 0.12, 0.15), gymapi.Vec3(0, 0.02, 0)),
    ]

    print(f"Physics settling ({args.steps} steps)...")
    # Step physics (CPU) first to settle
    for i in range(60):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if i % 30 == 0:
            print(f"  step {i} done")

    print(f"Rendering from {len(views)} views...")
    from PIL import Image

    for view_name, cam_pos, cam_target in views:
        gym.set_camera_location(cam_handle, env, cam_pos, cam_target)

        # Simulate a few steps for the render to update
        for _ in range(5):
            gym.simulate(sim)
            gym.fetch_results(sim, True)

        # Render
        try:
            gym.step_graphics(sim)
            img = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)

            if img is not None and img.size > 0:
                # Reshape if needed
                if img.ndim == 3 and img.shape[2] == 4:
                    pil_img = Image.fromarray(img[:, :, :3])
                else:
                    pil_img = Image.fromarray(
                        img[:args.width * args.height * 4].reshape(args.height, args.width, 4)[:, :, :3]
                    )
                path = out_dir / f"{view_name}.png"
                pil_img.save(str(path))
                print(f"  ✅ {view_name}.png saved")
            else:
                print(f"  ❌ {view_name}: empty image — GPU graphics may not work with CPU physics")
        except Exception as e:
            print(f"  ❌ {view_name}: render error: {e}")

    gym.destroy_sim(sim)
    print("Done!")


if __name__ == '__main__':
    main()
