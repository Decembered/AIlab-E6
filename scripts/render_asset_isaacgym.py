#!/usr/bin/env python3
"""Render IsaacGym asset to image — saves screenshots from multiple angles."""
import argparse, sys, os
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--out', default='render_output', help='Output directory')
    parser.add_argument('--steps', type=int, default=120)
    args = parser.parse_args()

    urdf_path = Path(args.urdf).resolve()
    asset_dir = str(urdf_path.parent)
    asset_file = urdf_path.name
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from isaacgym import gymapi, gymutil
    import numpy as np

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)
    sim_params.up_axis = gymapi.UP_AXIS_Y
    sim_params.use_gpu_pipeline = True  # Need GPU for rendering

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("FAIL: create_sim")
        sys.exit(1)

    # Ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 1, 0)
    plane_params.distance = 0.0
    gym.add_ground(sim, plane_params)

    # Load asset
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False
    asset = gym.load_asset(sim, asset_dir, asset_file, asset_options)
    print(f"Asset: {gym.get_asset_rigid_body_count(asset)} body(s), "
          f"{gym.get_asset_rigid_shape_count(asset)} shape(s)")

    # Create env
    env_spacing = 3.0
    env = gym.create_env(sim, gymapi.Vec3(-env_spacing, 0, -env_spacing),
                         gymapi.Vec3(env_spacing, env_spacing, env_spacing), 1)

    # Spawn actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.05, 0.0)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    actor = gym.create_actor(env, asset, pose, "object", 0, 1)

    # Camera setup
    cam_props = gymapi.CameraProperties()
    cam_props.width = 800
    cam_props.height = 600
    cam_props.use_collision_geometry = False

    # Create camera sensor
    cam_handle = gym.create_camera_sensor(env, cam_props)

    # Camera positions for multiple views
    views = [
        ("front", gymapi.Vec3(0.3, 0.2, 0.3), gymapi.Vec3(0, 0.03, 0)),
        ("side", gymapi.Vec3(0.3, 0.15, 0.0), gymapi.Vec3(0, 0.03, 0)),
        ("top", gymapi.Vec3(0.0, 0.3, 0.001), gymapi.Vec3(0, 0.03, 0)),
        ("angle", gymapi.Vec3(0.2, 0.25, 0.2), gymapi.Vec3(0, 0.03, 0)),
    ]

    print(f"Simulating {args.steps} steps for settling, then rendering...")

    # Settle
    for _ in range(60):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    # Render from each view
    for view_name, cam_pos, cam_target in views:
        gym.set_camera_location(cam_handle, env, cam_pos, cam_target)

        # Step to update rendering
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)

        # Grab image
        img = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
        if img is not None and img.size > 0:
            # img is RGBA uint8 (height, width, 4)
            from PIL import Image
            if img.ndim == 3 and img.shape[2] == 4:
                pil_img = Image.fromarray(img[:, :, :3])  # drop alpha
                path = out_dir / f"{view_name}.png"
                pil_img.save(str(path))
                print(f"  Saved: {path}")
            else:
                # Raw bytes — reshape
                w, h = cam_props.width, cam_props.height
                img_rgb = img[:w*h*4].reshape(h, w, 4)[:, :, :3]
                pil_img = Image.fromarray(img_rgb)
                path = out_dir / f"{view_name}.png"
                pil_img.save(str(path))
                print(f"  Saved: {path} (reshaped)")
        else:
            print(f"  {view_name}: no image data (GPU pipeline issue?)")

    gym.destroy_sim(sim)
    print("Done!")


if __name__ == '__main__':
    main()
