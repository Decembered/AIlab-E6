# Object 3.3 Evidence Bundle

This folder is the review entry point for Member C task 3.3 object reconstruction and task 3.4 object-side integration evidence.

## Canonical Inputs And Outputs

- Object assets: `runs/object_asset_v1/{bread,pipette,drink_ad,drink_yykx}/`
- Authoritative object trajectories: `outputs/mask_pose/{object}/{sequence}/object_trajectory.json`
- Trajectory quality reports: `outputs/mask_pose/{object}/{sequence}/trajectory_quality_report.json`
- Pipeline handoff package: `data/pipeline_assets/{sequence}/left_urdf/`
- Deprecated early trajectories: `outputs/pose_tracking/` is not used for integration.

## Evidence Files

- Final status: `member_c_task3_status_report.md`
- Geometric consistency: `geometry_summary.md`
- Object bonus summary: `object_bonus_report.md`
- IsaacGym rendering note: `isaacgym_rendering_note.md`
- Trajectory overview figures: `trajectory_overview_v3.png`, `trajectory_dense_overview_v3.png`
- Video/model overlays: `visual_overlay/`
- Hand-object joint render evidence: `hand_object_joint_render.png`, `hand_object_joint_frames/`

## Integration Status

The object side is ready for task 2-4 integration: each sequence has `scan.urdf`, `scan.obj`, and `left_obj.pkl` under `data/pipeline_assets/{sequence}/left_urdf/`.

The current `left_hand.pkl` files are placeholders with zero hand poses. Replace them with Member B's real hand reconstruction or Sharpa-retargeted trajectory before claiming real hand-object tracking.

Run the preflight check before a joint experiment:

```bash
python3.8 scripts/preflight_task24_integration.py --root data/pipeline_assets
```

Use `--allow-placeholder-hand` only for object-only smoke tests.
