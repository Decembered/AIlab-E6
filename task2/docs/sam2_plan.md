# SAM2 扩展计划

## SAM2 用来做什么

SAM2 用于视频手部分割，目标是生成任务2评分需要的 hand 2D mask。它可以用 MediaPipe 或 HaMeR 的手部 bbox、关键点作为 prompt，得到比简单 bbox/凸包更好的可见手部 mask。

当前第一阶段不强制安装 SAM2 或下载 checkpoint。若 SAM2 不可用，先使用 MediaPipe bbox 或关键点凸包生成粗 mask，保证有可交付物。

## 权重和代码放置位置

建议目录：

```text
task2/
  third_party/sam2/
  models/sam2/
    sam2_checkpoint.pt
    sam2_config.yaml
```

具体 checkpoint 文件名以官方 SAM2 发布说明为准。不要把 checkpoint 放到根目录。

## 如何使用 bbox 或关键点作为 prompt

### prompt 来源

- MediaPipe bbox：读取 `task2/outputs/trajectories/mediapipe_landmarks.json` 中每帧 `bbox_xyxy`。
- MediaPipe keypoints：使用 21 个关键点中的 wrist、finger joints 或 convex hull 作为 point/box prompt。
- HaMeR bbox：若后续 HaMeR 接入，可使用 HaMeR detector 或 mesh projection 得到更稳定的 bbox。

### 推荐流程

1. 从 `mediapipe_landmarks.json` 读取每帧 primary hand bbox。
2. 对 bbox 做适当 padding，例如 10% 到 25%，避免漏掉手指边缘。
3. 在 SAM2 中用第一帧或关键帧 bbox 初始化视频分割。
4. 对后续帧传播 mask。
5. 检查失败帧，必要时补充新的 bbox prompt。

## mask 输出如何保存

逐帧 mask 输出：

```text
task2/outputs/masks/sequence_name/%06d.png
```

保存约定：

- 单通道 PNG。
- 手部区域像素值 255，背景 0。
- mask 定义默认为 visible hand mask，即图像中可见的手部区域。
- 若使用 amodal hand mask，必须在报告中明确说明。

## 如何生成 hand mask overlay 视频

建议新增脚本：

```bash
python task2/scripts/08_visualize_hand_mask_overlay.py \
  --frames_dir task2/data/frames/demo \
  --masks_dir task2/outputs/masks/demo \
  --out_video task2/outputs/overlays/hand_mask_overlay.mp4
```

overlay 建议：

- 原图保持 60% 到 80% 权重。
- mask 区域叠加绿色或蓝色半透明颜色。
- 在左上角显示 frame id 和 mask area。

## SAM2 权重不可用时的保底方案

如果 SAM2 checkpoint 不可用、安装失败或显存不足：

- 使用 MediaPipe bbox 生成矩形粗 mask。
- 或对 MediaPipe 21 个 2D keypoints 做 convex hull，膨胀后生成粗 hand mask。
- 输出仍保存到 `task2/outputs/masks/sequence_name/%06d.png`。
- 报告中明确标注：这是 coarse visible hand mask baseline，不是 SAM2 精细分割。

保底方案虽然质量有限，但可以覆盖任务2“有 2D mask、定义清楚、可视化可检查”的基础要求。
