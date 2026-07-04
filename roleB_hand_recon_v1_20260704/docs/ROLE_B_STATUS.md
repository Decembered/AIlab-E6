# Role B Status - Hand Reconstruction / Trajectory Capture

Updated: 2026-07-04T10:15:00Z

## Current Pipeline

- Workspace: `/data/autovla/ho_tracker_challenge/roleB_hand_recon`
- Main script: `scripts/roleb_pipeline.py`
- Log: `ROLE_B_LOG.md`
- Metrics table: `SESSION_METRICS.csv`
- Outputs root: `outputs/<session>/`

Pipeline steps:

1. Download selected `human_demo` sessions from HF mirror direct resolve URLs into roleB data only.
2. Run MediaPipe Hands on `camera_side_1`, `camera_side_2`, and `camera_top`.
3. Save per-view raw 21-point `image_joints` and MediaPipe `world_joints` to `*_hand_traj_raw.npz`.
4. Interpolate and smooth per-view trajectories to `*_hand_traj_smooth.npz`.
5. Compute view quality metrics: valid ratio, wrist jitter, bounding-box area, bone-length CV, and an aggregate `view_score`.
6. Select best view per session for downstream single-view baseline use.
7. Generate overlay videos, thumbnails, and proxy hand masks for QA.
8. Triangulate multi-view 2D landmarks with calibration; choose extrinsic convention by lower reprojection error.
9. Save multi-view output as `multiview_hand_traj.npz`.

## Sessions Processed

| session | task | side | object | best view | valid frames by view | triangulation reproj median |
|---|---|---|---|---|---|---|
| `weigh_bread__2026_0701_0044_30` | weigh_bread | rh | Bread #1 | camera_top | side_1 235/235, side_2 233/235, top 235/235 | 2.31 px |
| `weigh_bread__left__2026_0701_0046_02` | weigh_bread__left | lh | Bread #1 | camera_side_2 | side_1 201/217, side_2 217/217, top 217/217 | 2.12 px |
| `weigh_drink_yykx__2026_0701_0051_12` | weigh_drink_yykx | rh | Drink YYKX | camera_side_1 | side_1 235/235, side_2 55/235, top 235/235 | 1.72 px |
| `grasp_pipette_press__2026_0701_0028_11` | grasp_pipette_press | rh | Pipette #1 | camera_top | side_1 235/235, side_2 205/235, top 235/235 | 3.67 px |

## What Improved Since The First Baseline

- Expanded from 1 sequence to 4 sequences across bread, drink, and pipette categories.
- Added automatic view scoring and best-view selection; this caught weak side views such as `weigh_drink_yykx` camera_side_2.
- Added multi-view triangulation using provided calibration, with median reprojection errors in the ~1.7-3.7 px range.
- Added per-session `metrics.json` and global `SESSION_METRICS.csv` for reproducible comparison.
- Kept all role B outputs isolated from the main HO-Tracker repo and A-line runs/logs.

## Limitations

- MediaPipe 21-point landmarks are still a baseline, not final MANO/mesh reconstruction.
- Proxy masks are landmark convex hulls and can overlap manipulated objects.
- Multi-view triangulation depends on MediaPipe 2D quality; bad detections still need filtering or robust fitting.
- No Sharpa retargeting has been run yet; the next training-facing step is MANO fitting or mano2dexhand/Sharpa retargeting.

## Recommended Next Step

Use the triangulated 21-point trajectories as initialization for MANO/HaMeR fitting, then produce `mano_pose`, `mano_betas`, `mano_tsl`, `wrist_pos`, and `wrist_rot` in HO-Tracker-compatible format. After that, connect to Sharpa mano2dexhand retargeting for Hand Bonus and A-line tracking.
