# Recommended Figures For Presentation

This file marks the most important Member C / task3 visuals for PPT, defense, brief report, and GitHub pages. The same files are copied into `figure/recommended/` with numeric prefixes for easy insertion into slides.

## Must Show

| Priority | File | Purpose | Suggested Caption |
|---|---|---|---|
| 1 | `recommended/01_asset_overview_ppt.png` | Object reconstruction and simulation assets | Four required objects reconstructed as watertight visual/collision meshes for IsaacGym. |
| 2 | `recommended/02_geometry_overlay_ppt.png` | Geometry consistency | Reconstructed 3D object models projected back to multi-view videos to verify scale and shape alignment. |
| 3 | `recommended/03_pose_tracking_ppt.png` | Object pose tracking | Video-derived object trajectories for 12 sequences using multi-view triangulation, silhouette yaw search, and Kalman smoothing. |
| 4 | `recommended/04_isaacgym_validation_ppt.png` | IsaacGym asset validation | URDF assets load and pass 60-step physics validation; black render screenshots are intentionally excluded. |
| 5 | `recommended/05_task34_integration_ppt.png` | Task 3.4 object-side integration | Object assets and poses are packed for hand-object replay and Sharpa tracking handoff. |
| 6 | `recommended/06_hand_object_joint_frames.mp4` | Demo video | Short hand-object visualization video compiled from valid frames. |

## Optional Quantitative Slides

| Priority | File | Purpose | Use Notes |
|---|---|---|---|
| 7 | `recommended/07_asset_quality_radar.png` | Multi-metric asset quality comparison | Good for showing normalized quality across mesh detail, collision simplicity, scale match, watertightness, IsaacGym pass, and mask stability. |
| 8 | `recommended/08_trajectory_speed_violin.png` | Trajectory speed distribution | Good for showing motion stability and object-specific velocity distribution. |
| 9 | `recommended/09_trajectory_speed_band_stability_not_gt_error.png` | Trajectory stability band | Use only with the note: this is not GT error, because primary-object GT poses are unavailable. |

## Report Files To Reference

- `06_reports/WORK_SUMMARY.md`: detailed Member C work summary and timeline.
- `06_reports/member_c_task3_status_report.md`: concise task 3.3 status report.
- `06_reports/geometry_summary.md`: geometric consistency summary.
- `06_reports/asset_summary.json`: mesh counts, extents, mass, and validation metadata.
- `06_reports/task24_preflight_object_only.json`: object-side task 2-4 handoff structure check.
- `06_reports/task24_preflight_strict.json`: strict preflight result showing current placeholder hand trajectory blocker.

## Important Presentation Notes

- Do not use black or failed IsaacGym render screenshots; they have been removed from this figure pack.
- For IsaacGym, show `recommended/04_isaacgym_validation_ppt.png` and quote the validation logs instead.
- Use `recommended/09_trajectory_speed_band_stability_not_gt_error.png` as a stability/variation band only, not as a tracking error band.
- Current full 2-4/3.4 integration still depends on Member B replacing placeholder `left_hand.pkl` with real hand trajectories.
