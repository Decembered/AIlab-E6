# 任务2失败分析

## 当前标准视角

- 序列：`weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2`
- 检测成功帧：182 / 182
- 检测成功率：100.00%
- 当前标准视角没有 MediaPipe 漏检帧。

## 多视角对比中的失败案例

`camera_side_1` 视角存在明显失败，适合作为报告中的失败案例：

- 序列：`weigh_drink_yykx_left_2026_0701_0052_53_camera_side_1`
- 检测成功帧：116 / 182
- 检测成功率：63.74%
- 缺失帧：`[0, 1]` 和 `[78, 141]`
- 最长连续缺失区间：78 到 141，共 64 帧，约 4.27 秒。
- wrist 跳变帧：`[39]`
- handedness 短暂翻转候选帧：`[41, 42]`

相关证据：

- `task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_1/videos/mediapipe_overlay.mp4`
- `task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_1/overlays/hand_mask_overlay.mp4`
- `task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_1/reports/task2_baseline_report.md`

## 失败类型

### 遮挡或视角不佳

- 现象：`camera_side_1` 在 78 到 141 帧连续漏检。
- 影响：该视角会产生大段插值轨迹，不能直接作为高可信 retarget 输入。
- 处理：优先选择 `camera_side_2` 或 `camera_top`；多视角融合时对 `camera_side_1` 的该区间降权。

### 轨迹跳变

- 现象：`camera_side_1` 的第 39 帧被标记为 wrist 跳变帧。
- 排查：查看 `camera_side_1` 的 keypoint overlay 和 3D skeleton。
- 处理：后续加入 OneEuro filter、异常帧重检测或跨视角约束。

### 左右手或 handedness 不稳定

- 现象：`camera_side_1` 的第 41、42 帧出现短暂 Right，而该序列名称包含 left，疑似 handedness 不稳定。
- 处理：后续根据序列 `side` 元信息、轨迹连续性和多视角一致性约束 handedness。

### 粗 mask 局限

- 现象：当前 mask 由关键点凸包/bbox 生成，检测失败帧 mask 为空，手物接触处边界不精细。
- 处理：后续接入 SAM2 或其他视频分割模型，以 MediaPipe bbox/keypoints 作为 prompt。

## 当前结论

当前任务2正式 baseline 应优先使用 `camera_side_2` 视角的结果作为单视角标准输出；`camera_side_1` 保留为失败案例和多视角鲁棒性分析材料。
