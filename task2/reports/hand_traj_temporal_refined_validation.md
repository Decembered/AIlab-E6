# hand_traj.npz 校验报告

- 文件：`task2/outputs/trajectories/hand_traj_temporal_refined.npz`
- T：182
- fps：15.000
- valid_ratio：100.00%
- errors：0
- warnings：6

## 字段清单

- `T_world_palm`: shape=(182, 4, 4), dtype=float32
- `T_world_wrist`: shape=(182, 4, 4), dtype=float32
- `active_fingers`: shape=(182, 5), dtype=bool
- `bbox_area`: shape=(182,), dtype=float32
- `bone_length_error`: shape=(182,), dtype=float32
- `camera_calib_valid`: shape=(), dtype=bool
- `camera_ids`: shape=(1,), dtype=<U32
- `confidence`: shape=(182, 21), dtype=float32
- `contact_likelihood`: shape=(182, 5), dtype=float32
- `contact_valid`: shape=(), dtype=bool
- `coord_frame`: shape=(), dtype=<U15
- `dataset_name`: shape=(), dtype=<U20
- `failure_reason`: shape=(182,), dtype=<U32
- `fingertip_step`: shape=(182, 5), dtype=float32
- `fingertip_step_max`: shape=(182,), dtype=float32
- `fingertips3d`: shape=(182, 5, 3), dtype=float32
- `fingertips3d_palm`: shape=(182, 5, 3), dtype=float32
- `fingertips_score`: shape=(182, 5), dtype=float32
- `fingertips_vel`: shape=(182, 5, 3), dtype=float32
- `fps`: shape=(), dtype=float32
- `frame_ids`: shape=(182,), dtype=int64
- `frame_manifest`: shape=(), dtype=<U91
- `hand_bbox2d`: shape=(182, 4), dtype=float32
- `hand_landmarks_2d`: shape=(182, 21, 2), dtype=float32
- `hand_landmarks_3d`: shape=(182, 21, 3), dtype=float32
- `handedness`: shape=(182,), dtype=<U16
- `handedness_score`: shape=(182,), dtype=float32
- `image_size`: shape=(2,), dtype=int64
- `interpolated_flag`: shape=(182,), dtype=bool
- `keypoint_convention`: shape=(), dtype=<U12
- `keypoints2d`: shape=(182, 21, 2), dtype=float32
- `keypoints2d_score`: shape=(182, 21), dtype=float32
- `keypoints3d`: shape=(182, 21, 3), dtype=float32
- `keypoints3d_raw`: shape=(182, 21, 3), dtype=float32
- `keypoints3d_score`: shape=(182, 21), dtype=float32
- `keypoints3d_smooth`: shape=(182, 21, 3), dtype=float32
- `keypoints3d_vel`: shape=(182, 21, 3), dtype=float32
- `mask_area`: shape=(182,), dtype=float32
- `mask_bbox_iou`: shape=(182,), dtype=float32
- `mask_bbox_ratio`: shape=(182,), dtype=float32
- `mask_inside_bbox`: shape=(182,), dtype=float32
- `metric_3d_valid`: shape=(), dtype=bool
- `notes`: shape=(), dtype=<U320
- `palm_pos`: shape=(182, 3), dtype=float32
- `palm_rot`: shape=(182, 4), dtype=float32
- `palm_rot_valid`: shape=(), dtype=bool
- `phase`: shape=(182,), dtype=<U16
- `phase_valid`: shape=(), dtype=bool
- `primary_camera_id`: shape=(), dtype=<U13
- `quality_flag_names`: shape=(12,), dtype=<U32
- `quality_flags`: shape=(182, 12), dtype=bool
- `quality_score`: shape=(182,), dtype=float32
- `quat_order`: shape=(), dtype=<U4
- `retarget_keypoints3d`: shape=(182, 11, 3), dtype=float32
- `retarget_keypoints3d_palm`: shape=(182, 11, 3), dtype=float32
- `retarget_landmark_names`: shape=(11,), dtype=<U32
- `retarget_weights`: shape=(11,), dtype=float32
- `schema_version`: shape=(), dtype=<U18
- `sequence_id`: shape=(), dtype=<U53
- `source`: shape=(), dtype=<U18
- `source_video`: shape=(), dtype=<U108
- `temporal_jump_score`: shape=(182,), dtype=float32
- `temporal_refiner_blend_alpha`: shape=(182,), dtype=float32
- `temporal_refiner_keypoints3d`: shape=(182, 21, 3), dtype=float32
- `temporal_refiner_note`: shape=(), dtype=<U131
- `temporal_refiner_quality`: shape=(182,), dtype=float32
- `temporal_refiner_raw_prediction3d`: shape=(182, 21, 3), dtype=float32
- `temporal_refiner_source`: shape=(), dtype=<U76
- `timestamps`: shape=(182,), dtype=float32
- `units`: shape=(), dtype=<U26
- `valid`: shape=(182,), dtype=bool
- `world_alignment_valid`: shape=(), dtype=bool
- `world_landmarks`: shape=(182, 21, 3), dtype=float32
- `wrist_pos`: shape=(182, 3), dtype=float32
- `wrist_pos_smooth`: shape=(182, 3), dtype=float32
- `wrist_rot`: shape=(182, 4), dtype=float32
- `wrist_rot_valid`: shape=(), dtype=bool
- `wrist_step`: shape=(182,), dtype=float32
- `wrist_vel`: shape=(182, 3), dtype=float32

## Errors

- 无

## Warnings

- units=non_metric_mediapipe_world，不能直接当作 IsaacGym/world metric 坐标
- metric_3d_valid=False
- world_alignment_valid=False
- camera_calib_valid=False
- contact_valid=False
- phase_valid=False
