# Member C Object Task Handoff

This is the handoff guide for another teammate, reviewer, or AI agent taking over
the object reconstruction and pose-tracking work for Video2Motion2Action task 3.3.

## Start Here

Current canonical evidence bundle:

- Task status report: `outputs/object3_submission/member_c_task3_status_report.md`
- Geometry summary: `outputs/object3_submission/geometry_summary.md`
- Trajectory overview: `outputs/object3_submission/trajectory_overview_v3.png`
- Dense trajectory overview: `outputs/object3_submission/trajectory_dense_overview_v3.png`
- Object-side preflight result: `outputs/object3_submission/task24_preflight_object_only.json`
- Canonical object assets: `runs/object_asset_v1/`
- Task 2-4 handoff package: `data/pipeline_assets/`

Current canonical scripts:

- Evidence bundle generator: `scripts/prepare_object3_submission_evidence.py`
- Prompt-free pose tracking entrypoint: `scripts/phase5_mask_depth_pose.py`
- Pose tracking implementation: `src/object_recon/pose_tracking.py`
- Synthetic checks: `scripts/test_mask_depth_pose.py`
- Task 2-4 integration preflight: `scripts/preflight_task24_integration.py`

Recommended Python for local object work:

```bash
python3.8
```

## What Is Done

Four simulation-first object assets exist under `runs/object_asset_v1/`:

- `bread`
- `pipette`
- `drink_ad`
- `drink_yykx`

Each object has:

- `mesh/visual_mesh.obj`
- `mesh/collision_mesh.obj`
- `asset/object.urdf`
- `asset/object_meta.json`
- `report/geometry_check_visual.txt`
- `report/geometry_check_collision.txt`
- `renders/front.png`, `side.png`, `top.png`, `angle.png`

The assets are low-face-count and watertight. They are intended to be usable in
IsaacGym rather than high-detail visual reconstructions.

## Data Reality

Important dataset findings:

- `data/human_demo` has RGB videos, camera calibration, and `pose_3d.hdf5`.
- Primary-object GT pose masks are all false for all 12 target sequences.
- A secondary test tube/beaker GT exists in one pipette sequence, but it must not
  be used as the primary pipette pose.
- No raw depth frames are currently discoverable.

Therefore:

- Do not claim GT object pose tracking for the primary objects.
- Do not claim RGB-D ICP on the current raw data.
- Use `multi_view_mask_pose` or `image_plane_pose` until depth or per-frame
  reconstructed point clouds are available.

## Current Pose Outputs

The latest prompt-free output for bread is:

```text
outputs/mask_pose/bread/weigh_bread__2026_0701_0044_30/
```

Most relevant files:

- `object_trajectory_multiview_pose.json`
- `object_trajectory_multiview_pose.npz`
- `trajectory_quality_report.json`
- `trajectory_plot.png`
- `debug_overlay/frame_000115.png`

Important: `object_trajectory.json` is the canonical integration filename. Older files named `object_trajectory_mask_pose.*` are legacy/fallback outputs. Some sequences also keep `object_trajectory_multiview_pose.*` as provenance, but downstream code should consume `object_trajectory.json`.

## Reproduce The Current Evidence Bundle

```bash
python3.8 \
  scripts/prepare_object3_submission_evidence.py \
  --python python3.8
```

This regenerates:

- `outputs/object3_submission/`
- `runs/object_asset_v1/README.md`
- per-object render images
- per-object local IsaacGym probe logs

Local IsaacGym probe can fail if the environment does not have the legacy
`isaacgym` Python module. That is recorded in:

```text
runs/object_asset_v1/<object>/report/asset_check_local.log
```

Cluster IsaacGym Python 3.8 validation has been recorded in `runs/object_asset_v1/isaacgym_validation_summary.json` and per-object `asset_check.log` files: 4/4 object assets load and run 60-step CPU physics successfully.

## Reproduce Dataset Inventory And Pose Tracking

Inventory:

```bash
python3.8 \
  scripts/phase5_mask_depth_pose.py \
  --inventory-only \
  --data-root data/human_demo \
  --output-root outputs
```

Bread current smoke test:

```bash
python3.8 \
  scripts/phase5_mask_depth_pose.py \
  --sequence weigh_bread__2026_0701_0044_30 \
  --data-root data/human_demo \
  --output-root outputs \
  --camera camera_top
```

Synthetic checks:

```bash
python3.8 scripts/test_mask_depth_pose.py
```

## What Is Not Done

Object-side task 3.3 is complete enough for handoff. The remaining work is integration-side:

- Replace placeholder `left_hand.pkl` files in `data/pipeline_assets/{sequence}/left_urdf/` with Member B's real hand reconstruction or Sharpa-retargeted trajectories.
- Run real hand-object replay/tracking with moving hand DOFs; current object-side verification replays object trajectories and uses a fixed hand pose for compatibility checks.
- Replace Inspire hand in `scripts/verify_hand_object_joint.py` with Sharpa hand once the Sharpa URDF and DOF mapping are finalized by Members A/B.
- Re-run the task 2-4 preflight check after hand trajectories are dropped in.

Why:

- Object GT poses are unavailable for primary objects, so the object trajectory is video-derived.
- The current `left_hand.pkl` files are intentionally zero placeholders so the package shape matches HO-Tracker conventions before Member B's data is available.
- Full task 3.4 success requires Members A/B outputs: Sharpa asset support, hand reconstruction, retargeting, and rollout evaluation.

## Next Best Actions

1. Ask Member B for real `left_hand.pkl` trajectories with documented shape, DOF order, and timestamps.
2. Run `python3.8 scripts/preflight_task24_integration.py --root data/pipeline_assets`.
3. Start with one representative sequence such as `weigh_bread__2026_0701_0044_30` or `grasp_pipette_press__2026_0701_0028_11`.
4. Verify real hand-object replay before moving to Sharpa tracking rollout.
5. Keep `outputs/pose_tracking/` deprecated; use `outputs/mask_pose/` for all object trajectories.

## Git Hygiene

Large generated files are intentionally ignored:

- `experiments/*/frames/`
- `experiments/*/pose_tracking/mask_sequence/`
- `experiments/*/pose_tracking/mask_sequence_debug/`
- `runs/*.tar.gz`

Do not commit raw extracted frames, dense mask `.npy` sequences, or large tarballs
unless the team explicitly decides to use Git LFS or an external artifact store.
