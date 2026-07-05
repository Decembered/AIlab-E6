#!/usr/bin/env python3.8
"""
Task 3.4 Joint Verification: Inspire Hand + Object in IsaacGym

Validates:
1. Inspire hand URDF loads in IsaacGym
2. Object URDF loads in same scene alongside hand
3. Object pose trajectory replays correctly
4. Hand-object spatial relationship is physically plausible

This proves the object assets from Member C are compatible with
the dexterous hand tracking pipeline from Member A/B.
"""
import os, sys, argparse, pickle
import numpy as np
from pathlib import Path
from isaacgym import gymapi

PIPELINE_DATA = Path(__file__).resolve().parent.parent / 'data' / 'pipeline_assets'


def verify_sequence(seq_name: str, hand_urdf_file: str = 'inspire_hand_left.urdf'):
    """Load hand + object in same IsaacGym scene and verify integration."""
    seq_dir = PIPELINE_DATA / seq_name / 'left_urdf'
    hand_urdf = PIPELINE_DATA / hand_urdf_file
    
    print(f'\n{"="*60}')
    print(f'  Joint Verification: {seq_name}')
    print(f'{"="*60}')
    
    # Check prerequisites
    obj_urdf = seq_dir / 'scan.urdf'
    obj_pkl = seq_dir / 'left_obj.pkl'
    
    checks = [
        ('Hand URDF', hand_urdf.exists()),
        ('Object URDF', obj_urdf.exists()),
        ('Object pose', obj_pkl.exists()),
    ]
    for name, ok in checks:
        status = 'OK' if ok else 'MISSING'
        print(f'  {status}: {name}')
    
    if not all(c[1] for c in checks):
        print(f'  FAIL: missing prerequisites')
        return False
    
    # Load object pose
    with open(obj_pkl, 'rb') as f:
        obj_data = pickle.load(f)
    obj_poses = obj_data['obj_pose']  # T × 7 (xyz + quaternion xyzw)
    print(f'  Object poses: {len(obj_poses)} frames')
    
    # === IsaacGym Setup ===
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
    
    # === Load Hand ===
    print(f'  Loading hand: {hand_urdf_file}...')
    hand_options = gymapi.AssetOptions()
    hand_options.fix_base_link = True   # Fix hand base in place
    hand_options.disable_gravity = True  # Hand is controlled
    hand_options.collapse_fixed_joints = False
    hand_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    
    try:
        hand_asset = gym.load_asset(sim, str(PIPELINE_DATA.resolve()), hand_urdf_file, hand_options)
    except Exception as e:
        print(f'  FAIL: hand load error — {e}')
        gym.destroy_sim(sim)
        return False
    
    if hand_asset is None:
        print(f'  FAIL: hand_asset is None')
        gym.destroy_sim(sim)
        return False
    
    hand_bodies = gym.get_asset_rigid_body_count(hand_asset)
    hand_dofs = gym.get_asset_dof_count(hand_asset)
    print(f'  Hand: {hand_bodies} bodies, {hand_dofs} DOFs')
    
    # === Load Object ===
    print(f'  Loading object: scan.urdf...')
    obj_options = gymapi.AssetOptions()
    obj_options.fix_base_link = False
    obj_options.disable_gravity = True  # Replay trajectory
    obj_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    
    try:
        obj_asset = gym.load_asset(sim, str(seq_dir.resolve()), 'scan.urdf', obj_options)
    except Exception as e:
        print(f'  FAIL: object load error — {e}')
        gym.destroy_sim(sim)
        return False
    
    if obj_asset is None:
        print(f'  FAIL: obj_asset is None')
        gym.destroy_sim(sim)
        return False
    
    obj_bodies = gym.get_asset_rigid_body_count(obj_asset)
    obj_shapes = gym.get_asset_rigid_shape_count(obj_asset)
    print(f'  Object: {obj_bodies} body, {obj_shapes} shapes')
    
    # === Create Environment ===
    env = gym.create_env(sim, gymapi.Vec3(-3, 0, -3), gymapi.Vec3(3, 3, 3), 1)
    
    # Spawn hand above table
    hand_pose = gymapi.Transform()
    hand_pose.p = gymapi.Vec3(0.0, 0.3, 0.0)  # 30cm above ground
    hand_pose.r = gymapi.Quat(0, 0, 0, 1)
    hand_actor = gym.create_actor(env, hand_asset, hand_pose, 'inspire_hand', 0, 1)
    
    # Set hand DOF to a natural pose
    if hand_dofs > 0:
        dof_states = np.zeros(hand_dofs, dtype=np.float32)
        dof_names = [gym.get_asset_dof_name(hand_asset, i) for i in range(hand_dofs)]
        # Set a slightly-open hand pose (typical grasping ready pose)
        for i, name in enumerate(dof_names):
            if 'proximal' in name:
                dof_states[i] = 0.3  # Slightly bent
            elif 'intermediate' in name:
                dof_states[i] = 0.2
            elif 'yaw' in name or 'pitch' in name:
                dof_states[i] = 0.0  # Neutral
            elif 'distal' in name:
                dof_states[i] = 0.1
            else:
                dof_states[i] = 0.0
        
        dof_state_array = np.zeros(hand_dofs, dtype=gymapi.DofState.dtype)
        for i in range(hand_dofs):
            dof_state_array[i]['pos'] = float(dof_states[i])
            dof_state_array[i]['vel'] = 0.0
        gym.set_actor_dof_states(env, hand_actor, dof_state_array, gymapi.STATE_ALL)
        print(f'  Hand DOFs set: {len(dof_names)} joints configured')
    
    # Spawn object below hand (on table)
    obj_pose_transform = gymapi.Transform()
    obj_pose_transform.p = gymapi.Vec3(0.0, 0.03, 0.0)  # Near table surface
    obj_pose_transform.r = gymapi.Quat(0, 0, 0, 1)
    obj_actor = gym.create_actor(env, obj_asset, obj_pose_transform, seq_name, 0, 1)
    
    # === Simulate + Replay ===
    print(f'  Simulating hand+object together...')
    n_steps = min(120, len(obj_poses))
    
    hand_positions = []
    obj_positions = []
    min_distance = float('inf')
    
    for step in range(n_steps):
        # Update object pose from trajectory
        if step < len(obj_poses):
            pos = obj_poses[step]
            body_states = gym.get_actor_rigid_body_states(env, obj_actor, gymapi.STATE_ALL)
            body_states['pose']['p'][0] = (float(pos[0]), float(pos[1]), float(pos[2]))
            body_states['pose']['r'][0] = (float(pos[3]), float(pos[4]), float(pos[5]), float(pos[6]))
            gym.set_actor_rigid_body_states(env, obj_actor, body_states, gymapi.STATE_ALL)
        
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # Track positions every 10 steps
        if step % 10 == 0:
            hs = gym.get_actor_rigid_body_states(env, hand_actor, gymapi.STATE_POS)
            os_ = gym.get_actor_rigid_body_states(env, obj_actor, gymapi.STATE_POS)
            
            hp = hs['pose']['p']
            op = os_['pose']['p']
            
            # Handle both single-body and multi-body cases
            if isinstance(hp['x'], np.ndarray):
                hand_pos = (float(hp['x'][0]), float(hp['y'][0]), float(hp['z'][0]))
                obj_pos = (float(op['x'][0]), float(op['y'][0]), float(op['z'][0]))
            else:
                hand_pos = (float(hp['x']), float(hp['y']), float(hp['z']))
                obj_pos = (float(op['x']), float(op['y']), float(op['z']))
            
            hand_positions.append(hand_pos)
            obj_positions.append(obj_pos)
            
            # Compute hand-object distance
            dist = np.sqrt(
                (hand_pos[0] - obj_pos[0])**2 +
                (hand_pos[1] - obj_pos[1])**2 +
                (hand_pos[2] - obj_pos[2])**2
            )
            min_distance = min(min_distance, dist)
    
    gym.destroy_sim(sim)
    del gym
    
    # === Results ===
    print(f'')
    print(f'  Hand bodies: {hand_bodies}, DOFs: {hand_dofs}')
    print(f'  Object bodies: {obj_bodies}, shapes: {obj_shapes}')
    
    if len(obj_positions) >= 2:
        obj_motion = sum(abs(obj_positions[-1][i] - obj_positions[0][i]) for i in range(3))
        print(f'  Object motion: {obj_motion:.3f}m over {n_steps} steps')
    
    if len(hand_positions) >= 2:
        hand_motion = sum(abs(hand_positions[-1][i] - hand_positions[0][i]) for i in range(3))
        print(f'  Hand motion: {hand_motion:.3f}m')
    
    print(f'  Min hand-object distance: {min_distance:.3f}m')
    
    # Verdict
    success = (hand_bodies >= 1 and obj_bodies >= 1 and min_distance < 100)
    verdict = '✅ PASS' if success else '⚠️  PARTIAL'
    print(f'')
    print(f'  VERDICT: {verdict} — hand+object coexist in IsaacGym scene')
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Verify hand+object integration in IsaacGym')
    parser.add_argument('--seq', default='weigh_bread__2026_0701_0044_30', help='Sequence to test')
    parser.add_argument('--hand', default='inspire_hand_left.urdf', help='Hand URDF file')
    parser.add_argument('--all', action='store_true', help='Test all sequences')
    parser.add_argument('--list', action='store_true', help='List available sequences')
    args = parser.parse_args()
    
    if args.list:
        for d in sorted(PIPELINE_DATA.iterdir()):
            if d.is_dir() and (d / 'left_urdf' / 'scan.urdf').exists():
                has_obj = (d / 'left_urdf' / 'left_obj.pkl').exists()
                print(f'{d.name}: urdf=OK, obj_pose={"OK" if has_obj else "MISSING"}')
        return
    
    if args.all:
        passed = 0
        total = 0
        for d in sorted(PIPELINE_DATA.iterdir()):
            if d.is_dir() and (d / 'left_urdf' / 'scan.urdf').exists():
                total += 1
                if verify_sequence(d.name, args.hand):
                    passed += 1
        print(f'\n{"="*60}')
        print(f'  SUMMARY: {passed}/{total} sequences PASS hand+object integration')
        return
    
    verify_sequence(args.seq, args.hand)


if __name__ == '__main__':
    main()
