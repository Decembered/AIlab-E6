# Multiview Hand Triangulation Report

- sequence_id: `weigh_drink_yykx_left_2026_0701_0052_53`
- output: `task2/outputs/by_sequence/weigh_drink_yykx_left_2026_0701_0052_53/trajectories/hand_traj_multiview.npz`
- calibration: `task2/outputs/by_sequence/weigh_drink_yykx_left_2026_0701_0052_53/camera_calib.json`
- cameras: camera_side_1, camera_side_2, camera_top
- primary_camera: `camera_side_2`
- frames: 182
- fps: 15.000
- selected_extrinsic_mode: `world_to_camera`
- median_reprojection_px: 2.0435
- valid_frames: 182/182 (100.00%)
- mean_support_per_joint: 2.637

## Mode Comparison

- `world_to_camera`: median_reprojection_px=2.0435
- `camera_to_world`: median_reprojection_px=909.5961

## View Valid Frames

- `camera_side_1`: 116/182 (63.74%)
- `camera_side_2`: 182/182 (100.00%)
- `camera_top`: 182/182 (100.00%)

## Notes

- This file is a multiview enhancement and does not overwrite the existing single-view baseline.
- `world_alignment_valid=False` means the trajectory is not yet aligned to the IsaacGym task/world frame.
