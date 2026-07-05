# Member C Task 3 Figure Pack

This folder is a curated presentation bundle for Member C's object reconstruction work in Video2Motion2Action task 3.3 and the object-side contribution to task 3.4. It is intended for PPT roadshow, brief report, and GitHub README pages.

Start here for the marked high-priority slide assets:

- `RECOMMENDED_FIGURES.md`
- `recommended/`

## Recommended PPT Order

1. **Task Scope And Deliverables**
   - Use: `06_reports/member_c_task3_status_report.md`
   - Message: Member C reconstructed 4 required objects, generated IsaacGym assets, recovered object pose trajectories, and prepared task 2-4/3.4 integration handoff files.

2. **Object Assets Overview**
   - Use: `01_asset_renders/asset_overview_ppt.png`
   - Message: Bread, pipette, Drink AD, and Drink YYKX all have low-face-count visual/collision meshes suitable for simulation.

3. **Geometry Consistency With Videos**
   - Use: `02_video_overlays/geometry_overlay_ppt.png`
   - Message: Reconstructed object geometry is projected back to original multi-view videos to show scale and shape consistency.

4. **Object Pose Tracking**
   - Use: `03_pose_tracking/pose_tracking_ppt.png`
   - Message: Object trajectories are recovered for all 12 sequences. The authoritative trajectories are under `outputs/mask_pose/{object}/{sequence}/object_trajectory.json`.

5. **IsaacGym Asset Validation**
   - Use: `04_isaacgym_validation/isaacgym_validation_ppt.png`
   - Use: `04_isaacgym_validation/isaacgym_validation_summary.json`
   - Message: Four URDF assets load and pass 60-step physics validation on the cluster IsaacGym Preview 4 Python 3.8 environment.

6. **Task 3.4 Object-Side Integration**
   - Use: `05_task24_integration/task34_integration_ppt.png`
   - Use: `05_task24_integration/hand_object_joint_frames.mp4`
   - Message: Object assets and object poses have been packed into `data/pipeline_assets/{sequence}/left_urdf/` for hand-object tracking pipeline integration.

7. **Current Integration Blocker**
   - Use: `06_reports/task24_preflight_strict.json`
   - Use: `06_reports/task24_preflight_object_only.json`
   - Message: Object-side inputs pass structure checks. Full task 2-4 integration still requires Member B's real hand trajectory because current `left_hand.pkl` files are placeholders.

8. **Optional Quantitative Analysis Charts**
   - Use: `07_analysis_charts/asset_quality_radar.png`
   - Use: `07_analysis_charts/trajectory_speed_violin.png`
   - Use with caution: `07_analysis_charts/trajectory_speed_band.png`
   - Message: Radar/spider chart summarizes multi-metric asset quality; violin plot summarizes object trajectory speed distributions. The band plot shows trajectory variation/stability, not ground-truth error.

## Scoring Mapping

| Requirement | Evidence |
|---|---|
| Object 2D mask | Mask overlays referenced in `06_reports/member_c_task3_status_report.md`; selected geometry overlays in `02_video_overlays/` |
| Object 3D model and visualization | `01_asset_renders/asset_overview_ppt.png`; per-object renders in `01_asset_renders/` |
| Geometric quality | `06_reports/asset_summary.json`; per-object logs in `04_isaacgym_validation/*_asset_check.log` |
| Geometric consistency | `02_video_overlays/geometry_overlay_ppt.png`; `06_reports/geometry_summary.md` |
| IsaacGym asset | `04_isaacgym_validation/isaacgym_validation_ppt.png`; `04_isaacgym_validation/isaacgym_validation_summary.json` |
| Object pose tracking | `03_pose_tracking/pose_tracking_ppt.png`; `03_pose_tracking/trajectory_overview_v3.png`; `03_pose_tracking/trajectory_dense_overview_v3.png` |
| Task 3.4 object-side handoff | `05_task24_integration/task34_integration_ppt.png`; `05_task24_integration/hand_object_joint_frames.mp4`; `06_reports/task24_preflight_object_only.json` |
| Failure/limitation analysis | `06_reports/isaacgym_rendering_note.md`; `06_reports/task24_preflight_strict.json` |
| Optional quantitative charts | `07_analysis_charts/asset_quality_radar.png`; `07_analysis_charts/trajectory_speed_violin.png`; `07_analysis_charts/trajectory_speed_band.png` |

## Directory Contents

- `01_asset_renders/`: per-object mesh render images and PPT-ready asset overview.
- `02_video_overlays/`: reconstructed object projection overlays on original videos and PPT-ready overlay grid.
- `03_pose_tracking/`: trajectory overview figures and PPT-ready tracking summary.
- `04_isaacgym_validation/`: validation summary card and asset check logs. Black/invalid IsaacGym render screenshots were removed.
- `05_task24_integration/`: object-side hand-object integration frames, PPT-ready summary, and MP4 video.
- `06_reports/`: compact markdown and JSON reports copied from canonical outputs.
- `07_analysis_charts/`: optional radar/spider, violin, and band charts generated from task3 metadata and object pose trajectories.
- `recommended/`: duplicated high-priority figures and demo video with numeric prefixes for PPT/GitHub insertion.
- `manifest.json`: source-to-figure mapping for traceability.

## Important Notes

- These figures are curated copies. Canonical generated outputs remain in `outputs/`, `runs/`, and `data/pipeline_assets/`.
- The latest authoritative object trajectories are in `outputs/mask_pose/`, not `outputs/pose_tracking/`.
- `hand_object_joint_frames.mp4` is a visualization compiled from existing hand-object frames. It is useful for presentation but does not replace full Sharpa tracking rollout.
- Black or non-informative IsaacGym render images are intentionally excluded. IsaacGym evidence is represented by validation logs and the `isaacgym_validation_ppt.png` status card.
- `trajectory_speed_band.png` is not a true error-band plot because primary-object GT poses are unavailable. Present it as a trajectory stability/variation band only.
- Full task 2-4/3.4 tracking still needs real hand trajectories from Member B; current placeholder hand files are intentionally flagged by `scripts/preflight_task24_integration.py`.
