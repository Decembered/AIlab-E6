# DLT/Kalman JSON Backup

This directory preserves the small, information-bearing subset of local changes that were not part of the final curated submission bundle.

## What Is Included

- 26 JSON files copied from `outputs/mask_pose/`.
- Files include `object_trajectory.json`, `trajectory_quality_report.json`, and two `mask_meta.json` files.
- These JSONs record a later local pose-tracking variant using `DLT triangulation + mesh silhouette yaw + Kalman filter`.

## Why This Exists

Before destroying the compute cluster, the local worktree still had 771 modified generated files relative to `origin/task3-object-reconstruction`:

- 26 JSON files, about 0.5 MB, containing useful trajectory/quality metadata.
- 745 PNG/JPG frame-level mask/overlay/trajectory/geometry images, about 132 MB.

The frame-level images are reproducible intermediate visualizations and were not preserved here because the curated presentation artifacts are already tracked under `figure/` and `outputs/object3_submission/`.

## Important Boundary

The authoritative object-side handoff for task2-4/task3.4 remains:

- `data/pipeline_assets/*/left_urdf/left_obj.pkl`
- `data/pipeline_assets/*/left_urdf/scan.urdf`
- `data/pipeline_assets/*/left_urdf/scan.obj`
- `outputs/object3_submission/task24_preflight_*.json`

This backup is retained for traceability and future comparison only. It should not be treated as the primary task3.4 hand-object tracking result.
