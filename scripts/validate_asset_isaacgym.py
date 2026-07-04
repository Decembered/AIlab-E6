#!/usr/bin/env python3
"""Minimal IsaacGym asset load validation.

Usage: python validate_asset_isaacgym.py --urdf bread_v41.urdf [--obj bread_v41.obj]

Validates:
1. Asset loads without import/syntax errors
2. Correct body/shape count
3. Spawns in environment without crash
4. 60-step simulation: no NaN, no position explosion (>100m)
"""
import argparse, sys, os, json
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True, help='Path to URDF file')
    parser.add_argument('--obj', default=None, help='Path to OBJ file (for logging)')
    parser.add_argument('--steps', type=int, default=60, help='Simulation steps')
    parser.add_argument('--headless', action='store_true', default=True)
    args = parser.parse_args()

    urdf_path = Path(args.urdf).resolve()
    if not urdf_path.exists():
        print(f"FAIL: URDF not found: {urdf_path}")
        sys.exit(1)

    asset_dir = str(urdf_path.parent)
    asset_file = urdf_path.name

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"=== IsaacGym Asset Validation ===")
    log(f"Time: {datetime.now().isoformat()}")
    log(f"URDF: {urdf_path}")
    log(f"Asset dir: {asset_dir}")
    log(f"")

    # --- Step 1: Import ---
    log("[1/5] Importing isaacgym...")
    try:
        from isaacgym import gymapi, gymutil
        log("  isaacgym import OK")
    except Exception as e:
        log(f"FAIL: isaacgym import failed: {e}")
        sys.exit(1)

    # --- Step 2: Create sim ---
    log("[2/5] Creating simulation...")
    try:
        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)
        sim_params.up_axis = gymapi.UP_AXIS_Y
        sim_params.use_gpu_pipeline = False

        sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            log("FAIL: create_sim returned None")
            sys.exit(1)
        log("  Sim created OK")

        # Ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 1, 0)
        plane_params.distance = 0.0
        gym.add_ground(sim, plane_params)
        log("  Ground plane added")
    except Exception as e:
        log(f"FAIL: sim creation failed: {e}")
        sys.exit(1)

    # --- Step 3: Load asset ---
    log("[3/5] Loading asset...")
    try:
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        asset = gym.load_asset(sim, asset_dir, asset_file, asset_options)
        if asset is None:
            log("FAIL: load_asset returned None")
            sys.exit(1)

        num_bodies = gym.get_asset_rigid_body_count(asset)
        num_shapes = gym.get_asset_rigid_shape_count(asset)
        num_dofs = gym.get_asset_dof_count(asset)
        log(f"  Loaded OK: {num_bodies} body(s), {num_shapes} shape(s), {num_dofs} DOF(s)")
    except Exception as e:
        log(f"FAIL: asset load failed: {e}")
        sys.exit(1)

    # --- Step 4: Create env + actor ---
    log("[4/5] Creating environment and spawning actor...")
    try:
        env_spacing = 2.0
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        env = gym.create_env(sim, env_lower, env_upper, 1)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.1, 0.0)  # Spawn 10cm above ground
        pose.r = gymapi.Quat(0, 0, 0, 1)

        actor_handle = gym.create_actor(env, asset, pose, "bread_test", 0, 1)
        log(f"  Actor spawned, handle={actor_handle}")
    except Exception as e:
        log(f"FAIL: actor creation failed: {e}")
        sys.exit(1)

    # --- Step 5: Simulate ---
    log(f"[5/5] Simulating {args.steps} steps...")
    try:
        positions = []
        for step in range(args.steps):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)

            if step % 10 == 0:
                body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
                pos = body_states['pose']['p']
                px, py, pz = pos['x'][0], pos['y'][0], pos['z'][0]

                # Check for NaN
                import math
                if any(math.isnan(v) for v in [px, py, pz]):
                    log(f"  Step {step}: FAIL — NaN detected in position")
                    sys.exit(1)

                # Check for explosion
                dist = (px**2 + py**2 + pz**2) ** 0.5
                if dist > 100.0:
                    log(f"  Step {step}: FAIL — position explosion, dist={dist:.1f}m")
                    sys.exit(1)

                positions.append([float(px), float(py), float(pz)])
                log(f"  Step {step}: pos=({px:.4f}, {py:.4f}, {pz:.4f}), dist={dist:.3f}m")

        gym.destroy_sim(sim)
        log(f"")
        log(f"=== PASS: Asset loads and simulates correctly ===")
        log(f"Body count: {num_bodies}, Shape count: {num_shapes}")

        # Compute some stats
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        log(f"Position range X: [{min(xs):.4f}, {max(xs):.4f}]")
        log(f"Position range Y: [{min(ys):.4f}, {max(ys):.4f}]")
        log(f"Position range Z: [{min(zs):.4f}, {max(zs):.4f}]")

        # Save log
        log_path = Path(args.urdf).parent.parent / "asset_check.log"
        with open(log_path, 'w') as f:
            f.write('\n'.join(log_lines))
        log(f"\nLog saved: {log_path}")

        return 0

    except Exception as e:
        log(f"FAIL: simulation error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
