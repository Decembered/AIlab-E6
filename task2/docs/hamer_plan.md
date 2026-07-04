# HaMeR 扩展计划

## HaMeR 用来做什么

HaMeR 用于从单目 RGB 图像中恢复 3D hand mesh、MANO 参数和 3D joints。它对应任务2评分中的“手部 3D 重建与可视化”加分项，可替换当前 MediaPipe world landmarks baseline。

## 当前本机状态

截至当前检查，`task2/third_party/hamer/` 已存在 HaMeR 源码，但不是完整可运行状态：

- 未发现 `.git/` 元信息，无法确认完整 clone 历史。
- ViTPose submodule / third-party 依赖不完整。
- `task2/models/mano/` 中尚无 MANO 官方模型文件。
- `task2/models/hamer/` 中尚无可用 HaMeR checkpoint。

因此当前不能把 HaMeR 视为已接入能力。任务2正式 baseline 仍以 MediaPipe + coarse mask + `hand_traj.npz` 为主；HaMeR 只作为后续冲分扩展。

当前 baseline 的 3D 表示是 MediaPipe 21 点 skeleton，不保证真实尺度。HaMeR 接入后，目标是输出：

- 每帧 MANO pose / shape。
- 每帧 hand mesh，通常为 MANO 778 vertices。
- 每帧 3D joints。
- mesh 或 skeleton 可视化视频。
- 与当前 `hand_traj.npz` 兼容的 `hand_landmarks_3d`、`wrist_pos`、`fingertips3d` 字段。

## 需要的权重和 MANO 文件

HaMeR 通常需要以下资源，具体文件名以官方仓库说明为准：

- HaMeR 模型权重，建议放到 `task2/models/hamer/`。
- MANO 模型文件，建议放到 `task2/models/mano/`。
- 常见 MANO 文件包括 `MANO_RIGHT.pkl`、`MANO_LEFT.pkl` 或官方要求的 MANO model directory。

MANO 需要从官方渠道申请或下载，不能自动伪造。当前已在 `task2/models/mano/README.md` 写明放置要求。

## 安装步骤建议

第一阶段不强制安装 HaMeR，避免破坏可运行 baseline。后续建议按以下步骤单独接入：

1. 检查或重新同步 `task2/third_party/hamer`，确认官方代码、submodule 和依赖完整。
2. 阅读官方 README，确认 Python、PyTorch、CUDA、pytorch3d、detectron2、MANO 等依赖版本。
3. 若依赖和 `v2m2a-task2` 冲突，创建独立环境，例如 `v2m2a-hamer`，不要污染 baseline 环境。
4. 将 HaMeR checkpoint 放入 `task2/models/hamer/`。
5. 将 MANO 文件放入 `task2/models/mano/`。
6. 先对 `task2/data/frames/demo` 中 5 到 20 帧做 smoke test。
7. 通过后再批量处理完整视频。

示例目录：

```text
task2/
  third_party/hamer/
  models/hamer/
    hamer_checkpoint.ckpt
  models/mano/
    MANO_RIGHT.pkl
    MANO_LEFT.pkl
```

## 输入输出如何接入当前 hand_traj.npz

### 输入

- RGB frame：`task2/data/frames/sequence_name/%06d.jpg`。
- hand bbox：优先复用 `task2/outputs/trajectories/mediapipe_landmarks.json` 中的 `bbox_xyxy`。
- handedness：复用 MediaPipe handedness，或使用 HaMeR 自带左右手逻辑。

### 输出

- mesh 文件：`task2/outputs/meshes/sequence_name/%06d.obj` 或 `%06d.ply`。
- MANO 参数：可保存到 `task2/outputs/trajectories/hamer_mano_params.npz`。
- 统一轨迹：将 HaMeR joints 写入 `task2/outputs/trajectories/hand_traj.npz` 的 `hand_landmarks_3d`、`wrist_pos`、`fingertips3d`。
- 可视化：`task2/outputs/videos/hamer_mesh_overlay.mp4` 或 `task2/outputs/videos/hand_3d_mesh.mp4`。

建议新增转换脚本：

```bash
python task2/scripts/07_run_hamer.py \
  --frames_dir task2/data/frames/demo \
  --mediapipe_json task2/outputs/trajectories/mediapipe_landmarks.json \
  --out_dir task2/outputs/meshes/demo \
  --out_npz task2/outputs/trajectories/hamer_hand_traj.npz
```

然后可用 `05_export_task2_result.py` 或新增 adapter 将 HaMeR 结果导出为统一接口。

## 安装失败时如何退回 MediaPipe baseline

如果 HaMeR 安装失败、权重缺失或 MANO 文件不可用：

- 不阻塞任务2交付。
- 继续使用 `02_run_mediapipe_hands.py`、`04_smooth_hand_traj.py`、`05_export_task2_result.py`。
- 在报告中说明当前 3D 为 MediaPipe baseline，不是 MANO/mesh。
- 保留 HaMeR 失败日志到 `task2/logs/hamer_install_or_run.log`。
- 优先保证 `task2/outputs/trajectories/hand_traj.npz`、overlay 视频和 baseline 报告存在。
