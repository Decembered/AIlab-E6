# 真实数据盘点

## 数据位置

当前已发现 HO-Tracker-Challenge 数据位于：

`task1/data/HO-Tracker-Challenge/`

任务2不复制大数据，直接通过命令行参数读取该目录下的视频，并将任务2产物输出到 `task2/`。

## human_demo 序列

`task1/data/HO-Tracker-Challenge/human_demo/` 下共有 12 条真实演示序列：

- `grasp_drink_yykx__2026_0701_0054_45`
- `grasp_pipette_press__2026_0701_0028_11`
- `grasp_pipette_rotate__2026_0701_0025_42`
- `grasp_pipette_stand__2026_0701_0019_19`
- `pipette_rh_beaker__2026_0701_0035_47`
- `pipette_rh_beaker_testtube__2026_0701_0039_28`
- `weigh_bread__2026_0701_0044_30`
- `weigh_bread__left__2026_0701_0046_02`
- `weigh_drink_ad__2026_0701_0047_56`
- `weigh_drink_ad__left__2026_0701_0049_04`
- `weigh_drink_yykx__2026_0701_0051_12`
- `weigh_drink_yykx__left__2026_0701_0052_53`

每条序列包含三视角视频：

- `video/camera_side_1.mkv`
- `video/camera_side_2.mkv`
- `video/camera_top.mkv`

每条序列包含三视角标定：

- `camera_calib/<camera>/cam_intr.pkl`，`3x3` 相机内参。
- `camera_calib/<camera>/cam_extr.pkl`，`4x4` 相机外参。

每条序列还包含 `pose_3d.hdf5`，可用于后续物体 pose、时间戳和多视角重投影验证。

## 当前任务2标准序列

已用以下真实序列跑通三视角任务2 baseline：

`task1/data/HO-Tracker-Challenge/human_demo/weigh_drink_yykx__left__2026_0701_0052_53/`

视频信息：

- fps：15
- 帧数：182
- 分辨率：1280x720

三视角结果：

| run_id | frames | valid | valid_ratio | 用途 |
|---|---:|---:|---:|---|
| `weigh_drink_yykx_left_2026_0701_0052_53_camera_side_1` | 182 | 116 | 63.74% | 失败案例 |
| `weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2` | 182 | 182 | 100.00% | 标准输出 |
| `weigh_drink_yykx_left_2026_0701_0052_53_camera_top` | 182 | 182 | 100.00% | 备用视角 |

标准输出采用：

`task1/data/HO-Tracker-Challenge/human_demo/weigh_drink_yykx__left__2026_0701_0052_53/video/camera_side_2.mkv`

标准输出位置：

- 抽帧：`task2/data/frames/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/`
- frame manifest：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/frame_manifest.json`
- MediaPipe JSON：`task2/outputs/trajectories/mediapipe_landmarks.json`
- 平滑轨迹：`task2/outputs/trajectories/hand_traj_smooth.npz`
- 统一接口：`task2/outputs/trajectories/hand_traj.npz`
- 关键点 overlay：`task2/outputs/videos/mediapipe_overlay.mp4`
- 3D skeleton 视频：`task2/outputs/videos/hand_3d_skeleton.mp4`
- 粗 hand mask：`task2/outputs/masks/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/`
- hand mask overlay：`task2/outputs/overlays/hand_mask_overlay.mp4`
- 质量报告：`task2/reports/trajectory_quality.md`
- baseline 报告：`task2/reports/task2_baseline_report.md`
- 接口校验报告：`task2/reports/hand_traj_validation.md`
- 多视角对比：`task2/reports/view_comparison.md`
- 按视角归档：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/`

当前检测结果：

- 总帧数：182
- MediaPipe 成功帧数：182
- 成功率：100.00%

## 后续建议

- 将当前单序列三视角流程扩展到其他 11 条 human_demo 序列，并生成全局批量索引。
- 读取 `cam_intr.pkl` 和 `cam_extr.pkl`，新增相机标定读取脚本与 3D 重投影验证。
- 使用 `pose_3d.hdf5` 的 timestamp 对齐视频帧，避免多视角/物体 pose 时间错位。
- 不要把真实大数据复制到 `task2/data/raw/`，除非明确需要子集样例；优先保留原始数据在 `task1/data/HO-Tracker-Challenge/`。
