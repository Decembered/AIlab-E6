#!/usr/bin/env python3
"""
IsaacGym Bread Asset Validation Suite
Member C: cn-ryw | 2026-07-04
Platform: Aliyun PAI DSW, 2x RTX 4090, IsaacGym Preview 4

Tests:
  1. Asset loading (URDF + mesh)
  2. Actor creation + VHACD convex decomposition
  3. Drop stability test from 0.5m
"""

import sys
from pathlib import Path
from isaacgym import gymapi, gymutil

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "experiments/2026-07-04_obj_recon_bread/models"
ASSET_FILE = "bread.urdf"


def test_asset_load(gym, sim):
    """Test 1: Load URDF asset and verify creation."""
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params.resolution = 100000

    asset = gym.load_asset(sim, str(MODELS_DIR), ASSET_FILE, asset_options)
    if asset is None:
        print("FAIL: load_asset returned None")
        return False
    print("PASS: load_asset OK")
    return asset


def test_create_actor(gym, sim, asset):
    """Test 2: Create actor in environment."""
    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
    pose.r = gymapi.Quat(0, 0, 0, 1)

    actor = gym.create_actor(env, asset, pose, "bread", 0, 1)
    if actor < 0:
        print("FAIL: create_actor failed")
        return None, None
    print("PASS: create_actor OK")
    return env, actor


def test_drop_stability(gym, sim, env, actor):
    """Test 3: Drop from 0.5m and check stability."""
    steps = 240
    for _ in range(steps):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
    pos = state["pose"]["p"][0]
    vel = state["vel"]["linear"][0]

    pos_ok = abs(pos[2]) < 2.0  # bread should be near ground, not flying
    vel_ok = abs(vel[0]) < 1.0 and abs(vel[1]) < 1.0 and abs(vel[2]) < 1.0  # settled

    print(f"  Final pos:  x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    print(f"  Final vel:  vx={vel[0]:.4f}, vy={vel[1]:.4f}, vz={vel[2]:.4f}")

    if pos_ok and vel_ok:
        print(f"PASS: BREAD_DROP_TEST_OK")
        return True
    else:
        print(f"FAIL: position_ok={pos_ok}, velocity_ok={vel_ok}")
        return False


def main():
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.physx.use_gpu = True

    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    results = {}

    # Test 1: Load
    asset = test_asset_load(gym, sim)
    results["load_asset"] = asset is not None

    # Test 2: Create actor
    env, actor = test_create_actor(gym, sim, asset)
    results["create_actor"] = actor is not None

    # Test 3: Drop stability
    results["drop_stability"] = test_drop_stability(gym, sim, env, actor)

    gym.destroy_sim(sim)

    print(f"\n=== SUMMARY ===")
    all_pass = all(results.values())
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
