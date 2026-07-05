# 任务2 Baseline 报告

## 输入与输出

- MediaPipe JSON：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/trajectories/mediapipe_landmarks.json`，存在：True
- hand_traj.npz：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/trajectories/hand_traj.npz`，存在：True
- 关键点 overlay 视频：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/videos/mediapipe_overlay.mp4`，存在：True
- hand mask overlay 视频：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/overlays/hand_mask_overlay.mp4`，存在：True
- hand mask 目录：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/masks`，存在：True
- 3D skeleton 视频：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/videos/hand_3d_skeleton.mp4`，存在：True
- hand_traj 校验报告：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/hand_traj_validation.md`，存在：True
- 帧级质量审计：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/frame_quality/frame_quality_audit.md`，存在：True
- 帧级指标 CSV：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/frame_quality/frame_metrics.csv`，存在：True
- 评分审查视频：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/videos/task2_scoring_review.mp4`，存在：True
- temporal refiner 应用报告：`None`，存在：False

## 检测统计

- 输入帧目录：`task2/data/frames/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top`
- 输入帧数：534
- 检测成功帧数：534
- 检测成功帧比例：100.00%
- 缺失帧列表：[]

## 轨迹跳变统计

- 轨迹帧数：534
- 有效帧数：534
- 有效帧比例：100.00%
- 平滑前缺失帧列表：[]
- wrist 平均相邻位移：0.003173
- wrist 最大相邻位移：0.014599
- fingertips 平均相邻位移：0.004683
- fingertips 最大相邻位移：0.023414

## Mask 统计

- mask 文件数：534
- 非空 mask：534
- 空 mask：0
- 平均 mask 面积：11815.88 px
- 最大 mask 面积：14091.00 px

## hand_traj.npz 字段

- `T_world_palm`
- `T_world_wrist`
- `active_fingers`
- `bbox_area`
- `bone_length_error`
- `camera_calib_valid`
- `camera_ids`
- `confidence`
- `contact_likelihood`
- `contact_valid`
- `coord_frame`
- `dataset_name`
- `failure_reason`
- `fingertip_step`
- `fingertip_step_max`
- `fingertips3d`
- `fingertips3d_palm`
- `fingertips_score`
- `fingertips_vel`
- `fps`
- `frame_ids`
- `frame_manifest`
- `hand_bbox2d`
- `hand_landmarks_2d`
- `hand_landmarks_3d`
- `handedness`
- `handedness_score`
- `image_size`
- `interpolated_flag`
- `keypoint_convention`
- `keypoints2d`
- `keypoints2d_score`
- `keypoints3d`
- `keypoints3d_raw`
- `keypoints3d_score`
- `keypoints3d_smooth`
- `keypoints3d_vel`
- `mask_area`
- `mask_bbox_iou`
- `mask_bbox_ratio`
- `mask_inside_bbox`
- `metric_3d_valid`
- `notes`
- `palm_pos`
- `palm_rot`
- `palm_rot_valid`
- `phase`
- `phase_valid`
- `primary_camera_id`
- `quality_flag_names`
- `quality_flags`
- `quality_score`
- `quat_order`
- `retarget_keypoints3d`
- `retarget_keypoints3d_palm`
- `retarget_landmark_names`
- `retarget_weights`
- `schema_version`
- `sequence_id`
- `source`
- `source_video`
- `temporal_jump_score`
- `timestamps`
- `units`
- `valid`
- `world_alignment_valid`
- `world_landmarks`
- `wrist_pos`
- `wrist_pos_smooth`
- `wrist_rot`
- `wrist_rot_valid`
- `wrist_step`
- `wrist_vel`

## 可视化文件路径

- MediaPipe overlay：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/videos/mediapipe_overlay.mp4`
- hand mask overlay：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/overlays/hand_mask_overlay.mp4`，当前为 coarse visible hand mask baseline，后续可替换为 SAM2。
- hand mask 目录：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/masks`
- 3D skeleton 可视化：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/videos/hand_3d_skeleton.mp4`，当前为 MediaPipe world landmarks skeleton，非 MANO mesh。
- hand_traj 校验报告：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/hand_traj_validation.md`
- 帧级质量审计：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/frame_quality/frame_quality_audit.md`
- 帧级指标 CSV：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/reports/frame_quality/frame_metrics.csv`
- 评分审查视频：`task2/outputs/by_view/pipette_rh_beaker_testtube_2026_0701_0039_28_camera_top/videos/task2_scoring_review.mp4`
- temporal refiner 应用报告：未传入

## 当前局限性

- MediaPipe world landmarks 不是严格 metric 3D，不能直接等同真实手部尺度。
- 遮挡、手物接触和运动模糊时可能漏检。
- 当前 baseline 没有真实 MANO mesh，3D 重建得分需要后续接入 HaMeR/MANO 提升。
- 当前 mask 为 bbox/关键点凸包生成的 coarse visible hand mask baseline，不是高质量分割。
- `T_world_wrist` / `T_world_palm` 当前不是 IsaacGym/world metric transform；`world_alignment_valid=False`。
- `contact_likelihood`、`active_fingers`、`phase` 当前是占位字段，不能作为真实手物交互标签。
- temporal refiner 使用 MediaPipe pseudo-label 训练，只作为辅助去噪/质量建模对照，不是 GT 或 metric 3D。

## 下一步改进建议

- 接入 SAM2，用 MediaPipe bbox 或关键点作为 prompt 生成 hand mask。
- 接入 HaMeR 和 MANO，导出 mesh、MANO 参数和更稳定的 3D joints。
- 做左右手 ID 维护，避免多手场景下 handedness 切换。
- 加入骨长约束、OneEuro filter 或鲁棒优化，降低跳变。
- 若有相机参数，补充坐标系说明和重投影误差统计。
