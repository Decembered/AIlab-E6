# hand_traj.npz 校验报告

- 文件：`task2/outputs/by_sequence/pipette_rh_beaker_testtube_2026_0701_0039_28/trajectories/hand_traj_multiview.npz`
- T：534
- fps：15.000
- valid_ratio：100.00%
- errors：0
- warnings：3

## 字段清单

- `K`: shape=(3, 3, 3), dtype=float32
- `T_world_palm`: shape=(534, 4, 4), dtype=float32
- `T_world_wrist`: shape=(534, 4, 4), dtype=float32
- `active_fingers`: shape=(534, 5), dtype=bool
- `bone_length_error`: shape=(534,), dtype=float32
- `camera_calib_valid`: shape=(), dtype=bool
- `camera_extrinsics_raw`: shape=(3, 4, 4), dtype=float32
- `camera_ids`: shape=(3,), dtype=<U32
- `confidence`: shape=(534, 21), dtype=float32
- `contact_likelihood`: shape=(534, 5), dtype=float32
- `contact_valid`: shape=(), dtype=bool
- `coord_frame`: shape=(), dtype=<U28
- `dataset_name`: shape=(), dtype=<U20
- `failure_reason`: shape=(534,), dtype=<U32
- `fingertips3d`: shape=(534, 5, 3), dtype=float32
- `fingertips3d_palm`: shape=(534, 5, 3), dtype=float32
- `fingertips_score`: shape=(534, 5), dtype=float32
- `fingertips_vel`: shape=(534, 5, 3), dtype=float32
- `fps`: shape=(), dtype=float32
- `frame_ids`: shape=(534,), dtype=int64
- `frame_manifest`: shape=(), dtype=<U1
- `hand_bbox2d`: shape=(534, 3, 4), dtype=float32
- `hand_landmarks_2d`: shape=(534, 21, 2), dtype=float32
- `hand_landmarks_3d`: shape=(534, 21, 3), dtype=float32
- `handedness`: shape=(534,), dtype=<U16
- `handedness_by_view`: shape=(534, 3), dtype=<U16
- `handedness_score`: shape=(534,), dtype=float32
- `handedness_score_by_view`: shape=(534, 3), dtype=float32
- `image_size`: shape=(2,), dtype=int64
- `interpolated_flag`: shape=(534,), dtype=bool
- `keypoint_convention`: shape=(), dtype=<U12
- `keypoints2d`: shape=(534, 21, 2), dtype=float32
- `keypoints2d_score`: shape=(534, 21), dtype=float32
- `keypoints3d`: shape=(534, 21, 3), dtype=float32
- `keypoints3d_raw`: shape=(534, 21, 3), dtype=float32
- `keypoints3d_score`: shape=(534, 21), dtype=float32
- `keypoints3d_smooth`: shape=(534, 21, 3), dtype=float32
- `keypoints3d_vel`: shape=(534, 21, 3), dtype=float32
- `metric_3d_valid`: shape=(), dtype=bool
- `multi_view_image_size`: shape=(3, 2), dtype=int64
- `multi_view_keypoints2d`: shape=(534, 3, 21, 2), dtype=float32
- `multi_view_keypoints2d_score`: shape=(534, 3, 21), dtype=float32
- `notes`: shape=(), dtype=<U272
- `palm_pos`: shape=(534, 3), dtype=float32
- `palm_rot`: shape=(534, 4), dtype=float32
- `palm_rot_valid`: shape=(), dtype=bool
- `phase`: shape=(534,), dtype=<U16
- `phase_valid`: shape=(), dtype=bool
- `primary_camera_id`: shape=(), dtype=<U13
- `projection_matrices`: shape=(3, 3, 4), dtype=float32
- `quality_score`: shape=(534,), dtype=float32
- `quat_order`: shape=(), dtype=<U4
- `reprojection_error`: shape=(534, 3, 21), dtype=float32
- `reprojection_error_frame_median`: shape=(534,), dtype=float32
- `retarget_keypoints3d`: shape=(534, 11, 3), dtype=float32
- `retarget_keypoints3d_palm`: shape=(534, 11, 3), dtype=float32
- `retarget_landmark_names`: shape=(11,), dtype=<U32
- `retarget_weights`: shape=(11,), dtype=float32
- `schema_version`: shape=(), dtype=<U18
- `sequence_id`: shape=(), dtype=<U44
- `source`: shape=(), dtype=<U33
- `source_video`: shape=(), dtype=<U112
- `temporal_jump_score`: shape=(534,), dtype=float32
- `timestamps`: shape=(534,), dtype=float32
- `triangulation_median_reprojection_px`: shape=(), dtype=float32
- `triangulation_mode`: shape=(), dtype=<U15
- `triangulation_support`: shape=(534, 21), dtype=int16
- `units`: shape=(), dtype=<U5
- `valid`: shape=(534,), dtype=bool
- `view_valid`: shape=(534, 3), dtype=bool
- `world_alignment_valid`: shape=(), dtype=bool
- `world_landmarks`: shape=(534, 21, 3), dtype=float32
- `wrist_pos`: shape=(534, 3), dtype=float32
- `wrist_pos_smooth`: shape=(534, 3), dtype=float32
- `wrist_rot`: shape=(534, 4), dtype=float32
- `wrist_rot_valid`: shape=(), dtype=bool
- `wrist_vel`: shape=(534, 3), dtype=float32

## Errors

- 无

## Warnings

- world_alignment_valid=False
- contact_valid=False
- phase_valid=False
