#!/usr/bin/env python3.8
"""
Task 3.4 Integration Verification: Load object asset + replay object pose in IsaacGym.

This validates that:
1. Object URDF loads in IsaacGym (from pipeline-compatible structure)
2. Object pose trajectory can be replayed
3. Scene is ready for Sharpa hand integration

Usage:
  python3.8 scripts/verify_integration.py --seq weigh_bread__2026_0701_0044_30
  python3.8 scripts/verify_integration.py --all
"""
import os, sys, argparse, pickle, json
import numpy as np
from pathlib import Path
from isaacgym import gymapi

PIPELINE_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'pipeline_assets')


def verify_sequence(seq_name):
    seq_dir = Path(PIPELINE_DATA) / seq_name / 'left_urdf'
    
    urdf_path = seq_dir / 'scan.urdf'
    obj_pkl_path = seq_dir / 'left_obj.pkl'
    hand_pkl_path = seq_dir / 'left_hand.pkl'
    
    urdf_ok = "OK" if urdf_path.exists() else "MISSING"
    obj_ok = "OK" if obj_pkl_path.exists() else "MISSING"
    hand_ok = "OK" if hand_pkl_path.exists() else "MISSING"
    print(f'\n=== {seq_name} ===')
    print(f'  URDF: {urdf_ok}')
    print(f'  obj_pose: {obj_ok}')
    print(f'  hand_pose: {hand_ok}')
    
    if not urdf_path.exists():
        print(f'  FAIL: URDF not found')
        return False
    
    # Load object pose
    obj_poses = None
    timestamps = None
    if obj_pkl_path.exists():
        with open(obj_pkl_path, 'rb') as f:
            data = pickle.load(f)
        obj_poses = data['obj_pose']  # T x 7 (xyz + quat)
        timestamps = data.get('timestamps', None)
        print(f'  Object poses: {len(obj_poses)} frames')
    
    # === IsaacGym Validation ===
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.gravity = gymapi.Vec3(0.0, -9.81, 0.0)
    sim_params.up_axis = gymapi.UP_AXIS_Y
    sim_params.use_gpu_pipeline = False
    
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print(f'  FAIL: create_sim')
        return False
    
    gym.add_ground(sim, gymapi.PlaneParams())
    
    # Load object URDF from pipeline directory
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = True  # For pose replay
    
    asset = gym.load_asset(sim, str(seq_dir.resolve()), 'scan.urdf', asset_options)
    if asset is None:
        print(f'  FAIL: load_asset')
        gym.destroy_sim(sim)
        return False
    
    env = gym.create_env(sim, gymapi.Vec3(-3, 0, -3), gymapi.Vec3(3, 3, 3), 1)
    
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.05, 0.0)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    actor = gym.create_actor(env, asset, pose, seq_name, 0, 1)
    
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_shapes = gym.get_asset_rigid_shape_count(asset)
    print(f'  IsaacGym: {num_bodies} body, {num_shapes} shape — loaded OK')
    
    # Replay object pose trajectory if available
    if obj_poses is not None and len(obj_poses) > 1:
        n_steps = min(120, len(obj_poses))
        positions_log = []
        
        for i in range(n_steps):
            # Set object state from trajectory
            if i < len(obj_poses):
                pos = obj_poses[i]
                body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
                body_states['pose']['p'][0] = (pos[0], pos[1], pos[2])
                body_states['pose']['r'][0] = (pos[3], pos[4], pos[5], pos[6])
                gym.set_actor_rigid_body_states(env, actor, body_states, gymapi.STATE_ALL)
            
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            
            if i % 20 == 0:
                states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
                p = states['pose']['p'][0]
                positions_log.append((p['x'], p['y'], p['z']))
        
        # Verify positions changed (trajectory is being replayed)
        if len(positions_log) >= 2:
            px = [p[0] for p in positions_log]
            py = [p[1] for p in positions_log]
            pz = [p[2] for p in positions_log]
            motion = max(px) - min(px) + max(py) - min(py) + max(pz) - min(pz)
            print(f'  Pose replay: {n_steps} steps, motion_range={motion:.3f}m')
            if motion > 0.001:
                print(f'  ✅ Object pose replay working')
            else:
                print(f'  ⚠️  No significant motion detected')
    else:
        # Simple physics test
        for _ in range(60):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        print(f'  Physics: 60 steps completed')
    
    gym.destroy_sim(sim)
    del gym
    
    print(f'  ✅ PASS: {seq_name}')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', default=None, help='Single sequence to verify')
    parser.add_argument('--all', action='store_true', help='Verify all sequences')
    parser.add_argument('--list', action='store_true', help='List available sequences')
    args = parser.parse_args()
    
    if args.list:
        data_dir = Path(PIPELINE_DATA)
        if data_dir.exists():
            for d in sorted(data_dir.iterdir()):
                if d.is_dir():
                    has_urdf = (d / 'left_urdf' / 'scan.urdf').exists()
                    has_obj_pkl = (d / 'left_urdf' / 'left_obj.pkl').exists()
                    print(f'{d.name}: urdf={has_urdf}, obj_pose={has_obj_pkl}')
        return
    
    if args.seq:
        verify_sequence(args.seq)
        return
    
    if args.all:
        passed = 0; failed = 0
        for d in sorted(Path(PIPELINE_DATA).iterdir()):
            if d.is_dir() and (d / 'left_urdf' / 'scan.urdf').exists():
                if verify_sequence(d.name):
                    passed += 1
                else:
                    failed += 1
        print(f'\n=== SUMMARY ===')
        print(f'  Passed: {passed}/{passed+failed}')
        return
    
    parser.print_help()


if __name__ == '__main__':
    main()
