# 多视角 baseline 对比

- 序列前缀：`weigh_drink_yykx_left_2026_0701_0052_53`

- 推荐优先视角：`weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2`，检测成功率 100.00%

## 结果

| run_id | frames | valid | valid_ratio | quality_mean |
|---|---:|---:|---:|---:|
| weigh_drink_yykx_left_2026_0701_0052_53_camera_side_1 | 182 | 116 | 63.74% | 0.5469 |
| weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2 | 182 | 182 | 100.00% | 0.9765 |
| weigh_drink_yykx_left_2026_0701_0052_53_camera_top | 182 | 182 | 100.00% | 0.9762 |

## 输出路径

### weigh_drink_yykx_left_2026_0701_0052_53_camera_side_1
- hand_traj：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_1/trajectories/hand_traj.npz`
- keypoint overlay：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_1/videos/mediapipe_overlay.mp4`
- mask overlay：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_1/overlays/hand_mask_overlay.mp4`
- 缺失帧：[0, 1, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141]

### weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2
- hand_traj：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/trajectories/hand_traj.npz`
- keypoint overlay：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/videos/mediapipe_overlay.mp4`
- mask overlay：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/overlays/hand_mask_overlay.mp4`
- 缺失帧：[]

### weigh_drink_yykx_left_2026_0701_0052_53_camera_top
- hand_traj：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_top/trajectories/hand_traj.npz`
- keypoint overlay：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_top/videos/mediapipe_overlay.mp4`
- mask overlay：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_top/overlays/hand_mask_overlay.mp4`
- 缺失帧：[]

