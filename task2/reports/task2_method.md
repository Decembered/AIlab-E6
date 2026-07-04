# 任务2方法说明：MediaPipe Baseline

## 当前真实序列

- 数据根目录：`task1/data/HO-Tracker-Challenge/`
- 处理序列：`weigh_drink_yykx__left__2026_0701_0052_53`
- 推荐视角：`camera_side_2`
- 视频路径：`task1/data/HO-Tracker-Challenge/human_demo/weigh_drink_yykx__left__2026_0701_0052_53/video/camera_side_2.mkv`
- 序列 ID：`weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2`
- FPS：15
- 总帧数：182
- 分辨率：1280x720

## 方法流程

- 视频抽帧：`01_extract_frames.py` 使用 OpenCV 导出连续图片。
- 手部关键点：`02_run_mediapipe_hands.py` 使用 MediaPipe HandLandmarker 输出 21 点 2D landmarks、world landmarks、handedness 和 bbox。
- 模型文件：当前环境使用 MediaPipe Tasks API，模型为 `task2/models/mediapipe/hand_landmarker.task`。
- 手部 mask：`07_generate_coarse_hand_masks.py` 使用关键点凸包或 bbox 生成 coarse visible hand mask。
- 轨迹平滑：`04_smooth_hand_traj.py` 做缺失帧插值和移动平均滤波。
- 统一导出：`05_export_task2_result.py` 导出标准 `hand_traj.npz`，包含 retarget 相关字段和质量字段。
- 校验：`08_validate_hand_traj.py` 校验 required 字段、shape、时间轴和数值范围。
- 3D 可视化：`10_visualize_3d_skeleton.py` 将 MediaPipe world landmarks 渲染为 3D skeleton 视频。
- 帧级审计：`13_frame_quality_audit.py` 输出质量 CSV、曲线、风险帧和可追加到 NPZ 的 quality flags。
- 训练迭代：`31_train_temporal_refiner.py` 用全量真实视频 MediaPipe pseudo-label 训练 temporal refiner；该训练不使用人工 GT，不输出 MANO/mesh。

## 输出

- MediaPipe JSON：`task2/outputs/trajectories/mediapipe_landmarks.json`
- 平滑轨迹：`task2/outputs/trajectories/hand_traj_smooth.npz`
- 统一轨迹：`task2/outputs/trajectories/hand_traj.npz`
- 关键点 overlay：`task2/outputs/videos/mediapipe_overlay.mp4`
- coarse visible hand mask：`task2/outputs/masks/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/%06d.png`
- mask overlay：`task2/outputs/overlays/hand_mask_overlay.mp4`
- 3D skeleton 视频：`task2/outputs/videos/hand_3d_skeleton.mp4`
- 代表截图：`task2/outputs/figures/report_cases/`
- 校验报告：`task2/reports/hand_traj_validation.md`
- baseline 报告：`task2/reports/task2_baseline_report.md`
- 多视角对比：`task2/reports/view_comparison.md`
- 全量主视角汇总：`task2/reports/all_human_demo_side2_summary.md`
- Temporal refiner 训练报告：`task2/outputs/training/temporal_refiner_side2_12seq/training_report.md`
- Temporal refiner 应用报告：`task2/reports/temporal_refiner_apply_report.md`

## 当前结果摘要

- `camera_side_1`：116 / 182 帧检测成功，成功率 63.74%。
- `camera_side_2`：182 / 182 帧检测成功，成功率 100.00%。
- `camera_top`：182 / 182 帧检测成功，成功率 100.00%。
- 当前标准输出采用 `camera_side_2`，因为它与 `camera_top` 同为 100% 成功率且平均质量分略高。
- 已批量处理 12 条 human_demo 的 `camera_side_2` 主视角，结果见 `task2/reports/all_human_demo_side2_summary.md`。其中 8 条主视角检测成功率 >= 90%，2 条饮料称重右手序列主视角低于 40%，需要后续依赖其他视角或多视角融合。

## 坐标系说明

- `keypoints2d` / `hand_landmarks_2d` 为像素坐标 `[x, y]`。
- `keypoints3d` / `hand_landmarks_3d` 来自 MediaPipe world landmarks。
- 当前 `coord_frame=mediapipe_world`。
- 当前 `units=non_metric_mediapipe_world`，不能直接当作 IsaacGym/world metric 坐标。
- `wrist_rot` 和 `palm_rot` 当前由 wrist、index MCP、middle MCP、pinky MCP 的几何关系近似估计，`quat_order=xyzw`。
- `T_world_wrist` 和 `T_world_palm` 当前基于 MediaPipe non-metric 坐标与近似 palm frame 构造，主要服务接口调试；`world_alignment_valid=False`。

## hand_traj.npz 接口

当前 `hand_traj.npz` 已通过 `08_validate_hand_traj.py` 校验，`errors=0`。主要字段包括：

- 时间轴：`sequence_id`、`fps`、`frame_ids`、`timestamps`。
- 2D/3D：`keypoints2d`、`keypoints2d_score`、`keypoints3d`、`keypoints3d_score`、`world_landmarks`。
- 手腕/手掌：`wrist_pos`、`wrist_rot`、`palm_pos`、`palm_rot`、`T_world_wrist`、`T_world_palm`。
- 指尖：`fingertips3d`、`fingertips_score`、`fingertips3d_palm`。
- retarget：`retarget_landmark_names`、`retarget_keypoints3d`、`retarget_keypoints3d_palm`、`retarget_weights`。
- 质量：`valid`、`interpolated_flag`、`quality_score`、`temporal_jump_score`、`bone_length_error`、`failure_reason`。
- 交互占位：`phase`、`contact_likelihood`、`active_fingers`。

## 局限性

- 当前 3D 不是真实 metric 3D，也不是 MANO mesh。
- 当前 mask 是 coarse visible hand mask，不是 SAM2 精细分割。
- 当前 hand rotation 是 21 点几何近似，retarget 前不能直接作为真实机器人 wrist/palm 姿态使用。
- 相机标定已按序列导出到 `task2/outputs/by_sequence/*/camera_calib.json` 和 `.npz`，但外参方向尚未验证，尚未用于三角化或真实重投影误差计算。
- HaMeR/MANO 和 SAM2 当前未接入可运行 pipeline。
- Temporal refiner 的训练目标是 MediaPipe pseudo-label 去噪和质量建模，不能把其输出解释为人工 GT、metric 3D 或 MANO 结果。
