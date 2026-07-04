# Role B Hand Reconstruction V1

This folder contains the isolated Role B trajectory-capture baseline for the
Video2Motion2Action task. It is intentionally packaged without videos, raw
datasets, or generated MP4 artifacts.

## Scope

- Processes `human_demo` sessions from `kelvin34501/HO-Tracker-Challenge`.
- Runs MediaPipe Hands on `camera_side_1`, `camera_side_2`, and `camera_top`.
- Saves per-view 21-point hand landmarks.
- Smooths per-view trajectories.
- Scores each view and chooses a best view per session.
- Triangulates multi-view 2D landmarks with provided camera calibration.
- Records metrics and QA notes for later MANO fitting / Sharpa retargeting.

## Main Script

```bash
scripts/roleb_pipeline.py
```

Example:

```bash
cd /data/autovla/ho_tracker_challenge/roleB_hand_recon
ionice -c2 -n7 nice -n 10 \
  /data/autovla/aILAB_hand_motion/.venv/bin/python scripts/roleb_pipeline.py \
  --download \
  --max-frames 235 \
  --resize-width 640 \
  --sessions \
  weigh_bread__2026_0701_0044_30 \
  weigh_bread__left__2026_0701_0046_02 \
  weigh_drink_yykx__2026_0701_0051_12 \
  grasp_pipette_press__2026_0701_0028_11
```

## Packaged Results

- `docs/ROLE_B_STATUS.md`: compact status and current conclusions.
- `docs/ROLE_B_LOG.md`: chronological exploration log.
- `docs/SESSION_METRICS.csv`: per-session metrics table.

The full local output tree on `xmu75` includes videos, thumbnails, proxy masks,
and `multiview_hand_traj.npz` files under:

```bash
/data/autovla/ho_tracker_challenge/roleB_hand_recon/outputs/<session>/
```

Those large/generated files are deliberately excluded from this code upload.

## Processed Sessions

Current V1 coverage:

- `weigh_bread__2026_0701_0044_30`
- `weigh_bread__left__2026_0701_0046_02`
- `weigh_drink_yykx__2026_0701_0051_12`
- `grasp_pipette_press__2026_0701_0028_11`

The full dataset metadata contains 12 `human_demo` operation trajectories, each
with three camera videos.

## Next Step

Use the triangulated 21-point trajectories as initialization for MANO/HaMeR
fitting, then export HO-Tracker-compatible hand data:

- `mano_pose`
- `mano_betas`
- `mano_tsl`
- `wrist_pos`
- `wrist_rot`

That output is the useful bridge to mano2dexhand / Sharpa retargeting.
