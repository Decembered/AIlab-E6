# Multiview Hand Triangulation Report

- sequence_id: `pipette_rh_beaker_testtube_2026_0701_0039_28`
- output: `task2/outputs/by_sequence/pipette_rh_beaker_testtube_2026_0701_0039_28/trajectories/hand_traj_multiview.npz`
- calibration: `task2/outputs/by_sequence/pipette_rh_beaker_testtube_2026_0701_0039_28/camera_calib.json`
- cameras: camera_side_1, camera_side_2, camera_top
- primary_camera: `camera_side_2`
- frames: 534
- fps: 15.000
- selected_extrinsic_mode: `world_to_camera`
- median_reprojection_px: 4.4753
- valid_frames: 534/534 (100.00%)
- mean_support_per_joint: 3.000

## Mode Comparison

- `world_to_camera`: median_reprojection_px=4.4753
- `camera_to_world`: median_reprojection_px=881.4807

## View Valid Frames

- `camera_side_1`: 534/534 (100.00%)
- `camera_side_2`: 534/534 (100.00%)
- `camera_top`: 534/534 (100.00%)

## Notes

- This file is a multiview enhancement and does not overwrite the existing single-view baseline.
- `world_alignment_valid=False` means the trajectory is not yet aligned to the IsaacGym task/world frame.
