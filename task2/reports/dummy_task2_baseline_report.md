# Dummy 自检报告，非任务2正式结果

本文件仅用于脚本自检。正式任务2结果见 `task2/reports/task2_baseline_report.md`。

## 输入与输出

- MediaPipe JSON：`task2/outputs/trajectories/dummy_mediapipe_landmarks.json`，存在：True
- hand_traj.npz：`task2/outputs/trajectories/dummy_hand_traj.npz`，存在：True
- 关键点 overlay 视频：`task2/outputs/videos/dummy_mediapipe_overlay.mp4`，存在：True

## 检测统计

- 输入帧目录：`task2/data/samples/dummy_frames`
- 输入帧数：60
- 检测成功帧数：60
- 检测成功帧比例：100.00%
- 缺失帧列表：[]

## 轨迹跳变统计

- 轨迹帧数：60
- 有效帧数：60
- 有效帧比例：100.00%
- 平滑前缺失帧列表：[]
- wrist 平均相邻位移：0.000000
- wrist 最大相邻位移：0.000000
- fingertips 平均相邻位移：0.000000
- fingertips 最大相邻位移：0.000000

## 可视化文件路径

- MediaPipe overlay：`task2/outputs/videos/dummy_mediapipe_overlay.mp4`
- hand mask overlay：`task2/outputs/overlays/dummy_hand_mask_overlay.mp4`
- hand mask 目录：`task2/outputs/masks/dummy/`
- 3D skeleton/mesh 可视化：dummy 自检未作为正式 3D 结果。

## 当前局限性

- MediaPipe world landmarks 不是严格 metric 3D，不能直接等同真实手部尺度。
- 遮挡、手物接触和运动模糊时可能漏检。
- 当前 baseline 没有真实 MANO mesh，3D 重建得分需要后续接入 HaMeR/MANO 提升。
- dummy mask 已由 coarse mask 脚本生成，但不具备真实分割评估意义。

## 下一步改进建议

- 接入 SAM2，用 MediaPipe bbox 或关键点作为 prompt 生成 hand mask。
- 接入 HaMeR 和 MANO，导出 mesh、MANO 参数和更稳定的 3D joints。
- 做左右手 ID 维护，避免多手场景下 handedness 切换。
- 加入骨长约束、OneEuro filter 或鲁棒优化，降低跳变。
- 若有相机参数，补充坐标系说明和重投影误差统计。
