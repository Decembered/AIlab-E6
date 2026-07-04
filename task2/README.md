# task2：手部重建与轨迹恢复

## 目标

任务2：从人类操作演示视频中恢复手部 2D mask、3D 手部表示和连续手部运动轨迹，并输出可检查的可视化结果和轨迹质量分析。

当前已经在真实 HO-Tracker-Challenge human_demo 序列上跑通 MediaPipe baseline：视频抽帧、手部关键点检测、轨迹平滑、coarse visible hand mask、overlay 可视化、3D skeleton 可视化、`hand_traj.npz` 标准接口导出和自动报告。HaMeR/MANO 与 SAM2 作为后续扩展，不在当前 baseline 中强制安装或下载。

## 目录说明

- `env/`：任务2环境配置文件。
- `data/raw/`：原始视频或相机文件。
- `data/frames/`：抽帧结果。
- `data/processed/`：中间处理结果。
- `data/samples/`：小样例数据。
- `models/mano/`：MANO 模型文件放置位置。
- `models/hamer/`：HaMeR 权重放置位置。
- `models/sam2/`：SAM2 checkpoint 放置位置。
- `third_party/`：后续 clone HaMeR、SAM2 等外部代码。
- `src/`：可复用工具模块。
- `scripts/`：可直接运行的 pipeline 脚本。
- `outputs/`：mask、mesh、轨迹、overlay、视频、图表输出。
- `logs/`：环境检查和运行日志。
- `reports/`：质量报告和方法说明。
- `docs/`：HaMeR/SAM2扩展计划。

## 流水线

### 1. 视频抽帧

输入：原始视频。

输出：`task2/data/frames/sequence_name/%06d.jpg`。

脚本：

```bash
python task2/scripts/01_extract_frames.py --video path/to/video.mp4 --out_dir task2/data/frames/demo
```

### 2. 手部检测与关键点 baseline

首选保底工具：MediaPipe Hand Landmarker / MediaPipe Hands。

输出：

- 2D hand landmarks。
- 3D/world hand landmarks。
- handedness 左右手信息。
- 每帧 bbox。

保存到：`task2/outputs/trajectories/mediapipe_landmarks.json`。

脚本：

```bash
python task2/scripts/02_run_mediapipe_hands.py \
  --frames_dir task2/data/frames/demo \
  --out_json task2/outputs/trajectories/mediapipe_landmarks.json \
  --vis_dir task2/outputs/overlays/mediapipe_frames
```

### 3. 手部 mask baseline

推荐方案：

- 用 MediaPipe 或 HaMeR 得到 hand bbox。
- 用 SAM2 或简单 bbox/关键点凸包 prompt 得到 hand mask。

输出：

- `task2/outputs/masks/sequence_name/%06d.png`。
- `task2/outputs/overlays/hand_mask_overlay.mp4`。

当前已提供 `07_generate_coarse_hand_masks.py`，可用 MediaPipe bbox 或关键点凸包生成 coarse visible hand mask 保底。后续 SAM2 可替换该输出。

### 4. 3D 手部重建 baseline

推荐方案：

- 保底：MediaPipe 21 点 3D skeleton。
- 冲分：HaMeR 输出 MANO/mesh。

输出：

- `task2/outputs/meshes/sequence_name/`。
- `task2/outputs/trajectories/hand_traj.npz`。

当前 baseline 中 `hand_landmarks_3d` 来自 MediaPipe world landmarks，不是严格 metric 3D。

### 5. 轨迹平滑与质量检查

处理：

- 缺失帧插值。
- 关节点时间滤波。
- 检查跳变、左右手切换、骨长异常、手指翻折。

输出：

- `task2/outputs/trajectories/hand_traj_smooth.npz`。
- `task2/reports/trajectory_quality.md` 或 `task2/reports/task2_baseline_report.md`。

### 6. 重投影与可视化

输出：

- hand keypoints overlay video。
- hand mask overlay video。
- 3D hand skeleton/mesh visualization。
- success/failure case screenshots。

当前 baseline 输出 `task2/outputs/videos/mediapipe_overlay.mp4` 和 `task2/outputs/videos/hand_3d_skeleton.mp4`。

## 快速开始

```bash
cd /mnt/workspace/hackthon
bash task2/scripts/00_check_env.sh

bash task2/scripts/20_run_single_view_baseline.sh \
  --video task1/data/HO-Tracker-Challenge/human_demo/weigh_drink_yykx__left__2026_0701_0052_53/video/camera_side_2.mkv \
  --sequence_id weigh_drink_yykx_left_0052_53 \
  --view_id camera_side_2 \
  --fps 15
```

多视角对比：

```bash
python task2/scripts/12_compare_views.py --by_view_dir task2/outputs/by_view --sequence_prefix weigh_drink_yykx_left_0052_53 --out_report task2/reports/view_comparison.md
```

## Dummy 自检，可选

当前已有真实数据，dummy 样例只用于验证脚本链路，不作为正式任务2结果：

```bash
python task2/scripts/98_make_dummy_sample.py --frames_dir task2/data/samples/dummy_frames --out_json task2/outputs/trajectories/dummy_mediapipe_landmarks.json
python task2/scripts/03_visualize_mediapipe_overlay.py --frames_dir task2/data/samples/dummy_frames --landmarks_json task2/outputs/trajectories/dummy_mediapipe_landmarks.json --out_video task2/outputs/videos/dummy_mediapipe_overlay.mp4
python task2/scripts/04_smooth_hand_traj.py --in_json task2/outputs/trajectories/dummy_mediapipe_landmarks.json --out_npz task2/outputs/trajectories/dummy_hand_traj_smooth.npz
python task2/scripts/05_export_task2_result.py --in_npz task2/outputs/trajectories/dummy_hand_traj_smooth.npz --out_npz task2/outputs/trajectories/dummy_hand_traj.npz
python task2/scripts/07_generate_coarse_hand_masks.py --frames_dir task2/data/samples/dummy_frames --landmarks_json task2/outputs/trajectories/dummy_mediapipe_landmarks.json --out_masks_dir task2/outputs/masks/dummy --out_overlay_video task2/outputs/overlays/dummy_hand_mask_overlay.mp4
python task2/scripts/06_make_task2_report.py --landmarks_json task2/outputs/trajectories/dummy_mediapipe_landmarks.json --traj_npz task2/outputs/trajectories/dummy_hand_traj.npz --out_report task2/reports/dummy_task2_baseline_report.md --overlay_video task2/outputs/videos/dummy_mediapipe_overlay.mp4
```

## 环境策略

当前机器的 `/usr/local/bin/python` 已能运行 MediaPipe baseline。不要在 base 环境里乱装包；为了跨机器复现，保留 conda 环境配置。当前系统检查显示驱动 CUDA 12.8、RTX 4090 可用，因此 `environment_task2.yml` 默认使用 `pytorch-cuda=12.8`。如果包源暂不支持该版本，可按 PyTorch 官方安装页选择兼容的 CUDA 12.x 版本。

```bash
conda env create -f task2/env/environment_task2.yml
conda activate v2m2a-task2
bash task2/scripts/check_task2_env.sh
```

如需自动执行环境创建，可查看并运行：

```bash
bash task2/scripts/setup_task2_env.sh
```

## hand_traj.npz 字段

当前导出的 `hand_traj.npz` 已通过 `08_validate_hand_traj.py` 校验，主要包含：

- `schema_version`: 接口版本。
- `sequence_id`: 序列和视角 ID。
- `fps`: 帧率。
- `frame_ids`: `[T]`。
- `timestamps`: `[T]`。
- `hand_landmarks_2d`: `[T, 21, 2]`。
- `hand_landmarks_3d`: `[T, 21, 3]`。
- `keypoints2d` / `keypoints3d` 及对应 score。
- `confidence`: `[T, 21]`。
- `handedness`: `[T]`。
- `wrist_pos`: `[T, 3]`，无真实尺度时使用 MediaPipe world landmarks 的 wrist。
- `wrist_rot`, `palm_pos`, `palm_rot`, `T_world_wrist`, `T_world_palm`。
- `fingertips3d`, `fingertips3d_palm`。
- `retarget_landmark_names`, `retarget_keypoints3d`, `retarget_keypoints3d_palm`, `retarget_weights`。
- `valid`, `quality_score`, `interpolated_flag`, `failure_reason`。
- `coord_frame=mediapipe_world`。
- `units=non_metric_mediapipe_world`。
- `source`: 字符串，`mediapipe baseline`。
- `notes`: 字符串，说明这是 baseline，后续可替换为 HaMeR/MANO。

## HaMeR 安装计划

HaMeR 用于单目 3D hand mesh / MANO 恢复。当前 `task2/third_party/hamer/` 已存在源码目录，但 ViTPose/third-party 依赖、MANO 文件和 HaMeR checkpoint 不完整，不能视为已接入能力。第一阶段不强制安装，计划和当前状态见 `task2/docs/hamer_plan.md`。

## SAM2 安装计划

SAM2 用于视频手部分割 mask。它需要 checkpoint，可用手部 bbox 或关键点作为 prompt。第一阶段不强制安装，计划见 `task2/docs/sam2_plan.md`。

## 当前局限性

- MediaPipe world landmarks 不等价于真实尺度 3D。
- 遮挡、运动模糊和强手物接触会导致漏检或跳变。
- 当前 mask 已有粗凸包/bbox baseline，完整 SAM2 接入需手动准备 checkpoint。
- 左右手多手跟踪当前以第一只检测手为 primary baseline，复杂交互需要后续做 ID 维护。
- 当前 `wrist_rot` 和 `palm_rot` 是由 21 点几何估计的近似 palm frame，不能直接用于精确机器人 retarget。
- 当前标准输出采用 `camera_side_2`；`camera_side_1` 保留为失败案例分析。
