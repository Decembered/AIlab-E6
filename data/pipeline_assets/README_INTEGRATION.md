# Task 3.4 Integration: Object Assets for Hand-Object Tracking

> Member C → Members A & B handoff

## What This Provides

For each of the 12 HO-Tracker sequences, this directory provides:

```
data/pipeline_assets/{sequence_name}/left_urdf/
├── scan.obj          # 3D object mesh (from Task 3.3 reconstruction)
├── scan.urdf         # URDF with visual + collision + inertial
├── left_obj.pkl      # Object 6-DoF pose trajectory (T × 7: xyz + quat)
└── left_hand.pkl     # PLACEHOLDER — replace with Member B's MANO→Sharpa data
```

## Integration Points

### For Task 3.1 (Sharpa Tracking)

1. **Object URDFs** follow the same convention as `assets/obj_urdf_example.urdf`
2. Copy a sequence directory into the pipeline's expected data path:
   ```bash
   cp -r data/pipeline_assets/{sequence_name} data/HO-Tracker/data/test_sample/h1o1/
   ```
3. The URDF loads in IsaacGym alongside Sharpa hand:
   ```python
   gym.load_asset(sim, "data/pipeline_assets/{seq}/left_urdf/", "scan.urdf", options)
   ```

### For Task 3.2 (Hand Reconstruction)

1. **hand_pose placeholder**: Replace `left_hand.pkl` with actual MediaPipe → MANO → Sharpa retargeting output
2. Expected format: `{'hand_pose': array(T, 51), 'timestamps': array(T), ...}`
3. 51 = 17 MANO joints × 3 DOF each

### For Task 3.4 (Comprehensive)

Combined scene setup:
```python
# Load hand (from Sharpa pipeline)
hand_asset = gym.load_asset(sim, hand_urdf_root, hand_urdf_file, hand_options)
hand_actor = gym.create_actor(env, hand_asset, pose, "sharpa_hand", 0, 1)

# Load object (from this package)
obj_asset = gym.load_asset(sim, obj_urdf_root, "scan.urdf", obj_options)
obj_actor = gym.create_actor(env, obj_asset, obj_pose, "object", 0, 1)

# Replay trajectories
for frame in range(num_frames):
    set_hand_dofs(hand_actor, hand_traj[frame])
    set_object_pose(obj_actor, obj_traj[frame])
    gym.simulate(sim)
```

## Verified Sequences (12/12)

| Sequence | Object | Pose Frames | Motion Range | IsaacGym |
|----------|--------|-------------|-------------|----------|
| weigh_bread__2026_0701_0044_30 | Bread | 154 | 0.505m | ✅ PASS |
| weigh_bread__left__2026_0701_0046_02 | Bread | 144 | 0.507m | ✅ PASS |
| grasp_pipette_stand__2026_0701_0019_19 | Pipette | 171 | 0.519m | ✅ PASS |
| grasp_pipette_rotate__2026_0701_0025_42 | Pipette | 234 | 0.598m | ✅ PASS |
| grasp_pipette_press__2026_0701_0028_11 | Pipette | 207 | 0.593m | ✅ PASS |
| pipette_rh_beaker__2026_0701_0035_47 | Pipette | 471 | 0.069m | ✅ PASS |
| pipette_rh_beaker_testtube__2026_0701_0039_28 | Pipette | 351 | 0.122m | ✅ PASS |
| weigh_drink_ad__2026_0701_0047_56 | Drink AD | 161 | 0.449m | ✅ PASS |
| weigh_drink_ad__left__2026_0701_0049_04 | Drink AD | 137 | 0.473m | ✅ PASS |
| weigh_drink_yykx__2026_0701_0051_12 | Drink YYKX | 161 | 0.500m | ✅ PASS |
| weigh_drink_yykx__left__2026_0701_0052_53 | Drink YYKX | 121 | 0.469m | ✅ PASS |
| grasp_drink_yykx__2026_0701_0054_45 | Drink YYKX | 174 | 0.388m | ✅ PASS |

## Verification Command

```bash
# Verify single sequence
python3.8 scripts/verify_integration.py --seq weigh_bread__2026_0701_0044_30

# Verify all sequences
python3.8 scripts/verify_integration.py --all

# List available sequences
python3.8 scripts/verify_integration.py --list
```

## Next Steps (for Members A & B)

1. **Member B**: Generate actual `left_hand.pkl` from MediaPipe → MANO → Sharpa retargeting
2. **Member A**: Copy sequence data into `data/HO-Tracker/data/test_sample/h1o1/` and integrate with Sharpa training pipeline
3. **All**: Run combined hand+object tracking and evaluate rollout success
