# Member C Object Task Handoff

This is the handoff guide for another teammate, reviewer, or AI agent taking over
the object reconstruction and pose-tracking work for Video2Motion2Action task 3.3.

## Start Here

Current canonical evidence bundle:

- Task status report: `outputs/object3_submission/member_c_task3_status_report.md`
- Dataset inventory: `outputs/dataset_inventory.csv`
- Geometry summary: `outputs/object3_submission/geometry_summary.csv`
- Mask audit: `outputs/object3_submission/mask_audit.csv`
- Render contact sheet: `outputs/object3_submission/render_contact_sheet.png`
- Canonical object assets: `runs/object_asset_v1/`

Current canonical scripts:

- Evidence bundle generator: `scripts/prepare_object3_submission_evidence.py`
- Prompt-free pose tracking entrypoint: `scripts/phase5_mask_depth_pose.py`
- Pose tracking implementation: `src/object_recon/pose_tracking.py`
- Synthetic checks: `scripts/test_mask_depth_pose.py`

Recommended Python for local object work:

```bash
/home/ruan/miniconda3/envs/objasset/bin/python
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

Important: older files named `object_trajectory_mask_pose.*` are legacy/fallback
outputs. For the current Phase 5 story, prefer `object_trajectory_multiview_pose.*`.

## Reproduce The Current Evidence Bundle

```bash
/home/ruan/miniconda3/envs/objasset/bin/python \
  scripts/prepare_object3_submission_evidence.py \
  --python /home/ruan/miniconda3/envs/objasset/bin/python
```

This regenerates:

- `outputs/object3_submission/`
- `runs/object_asset_v1/README.md`
- per-object render images
- per-object local IsaacGym probe logs

Local IsaacGym probe currently fails because this environment does not have the
legacy `isaacgym` Python module. That is recorded in:

```text
runs/object_asset_v1/<object>/report/asset_check_local.log
```

Run the same validation on the cluster IsaacGym Python 3.8 environment before
claiming IsaacGym asset load success.

## Reproduce Dataset Inventory And Pose Tracking

Inventory:

```bash
/home/ruan/miniconda3/envs/objasset/bin/python \
  scripts/phase5_mask_depth_pose.py \
  --inventory-only \
  --data-root data/human_demo \
  --output-root outputs
```

Bread current smoke test:

```bash
/home/ruan/miniconda3/envs/objasset/bin/python \
  scripts/phase5_mask_depth_pose.py \
  --sequence weigh_bread__2026_0701_0044_30 \
  --data-root data/human_demo \
  --output-root outputs \
  --camera camera_top
```

Synthetic checks:

```bash
/home/ruan/miniconda3/envs/objasset/bin/python scripts/test_mask_depth_pose.py
```

## What Is Not Done

Still incomplete:

- Stable continuous object masks for pipette, drink_ad, and drink_yykx.
- Full-sequence object pose trajectories for all four objects.
- Cluster IsaacGym validation logs for all four URDFs.
- Strong video-overlay/scale evidence for pipette and drinks.
- Object Bonus articulated modeling, especially a pipette plunger/button joint.

Why:

- Fixed-coordinate SAM prompts fail on long sequences.
- Current masks for pipette and drinks are too large and often include hand or
  background.
- The current dataset has no depth frames.
- Primary GT object poses are intentionally invalidated by all-false masks.

## Next Best Actions

1. Run URDF load validation on the cluster IsaacGym Python 3.8 environment.
2. Build dynamic mask propagation for one sequence per object.
3. Re-run `multi_view_mask_pose` once at least two calibrated views have usable
   masks for the same frames.
4. Add mesh projection/video overlay evidence for pipette and drinks.
5. Add pipette articulated URDF only after the base 3.3 deliverables are stable.

## Git Hygiene

Large generated files are intentionally ignored:

- `experiments/*/frames/`
- `experiments/*/pose_tracking/mask_sequence/`
- `experiments/*/pose_tracking/mask_sequence_debug/`
- `runs/*.tar.gz`

Do not commit raw extracted frames, dense mask `.npy` sequences, or large tarballs
unless the team explicitly decides to use Git LFS or an external artifact store.
