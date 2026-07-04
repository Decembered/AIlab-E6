# Task2 交付索引

## 正式结果

- 数据集：HO-Tracker-Challenge human_demo
- 序列：`weigh_drink_yykx__left__2026_0701_0052_53`
- 标准视角：`camera_side_2`
- 标准 run id：`weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2`
- 帧数：182
- FPS：15
- 检测成功率：182 / 182，100.00%
- 当前交付等级：MediaPipe non-metric baseline，不是 MANO/mesh 或 metric multi-view 3D。
- 已完成全量主视角批处理：12 条 human_demo 的 `camera_side_2`，见 `task2/reports/all_human_demo_side2_summary.md`。
- 已完成一次 GPU 训练迭代：temporal refiner 使用 12 条 `camera_side_2` MediaPipe pseudo-label 训练，checkpoint 位于 `task2/outputs/training/temporal_refiner_side2_12seq/temporal_refiner_best.pt`。

## 评分对照

| 评分子项 | 分值 | 当前证据 | 当前指标 | 局限性 |
|---|---:|---|---|---|
| 手部 2D mask | 3 | mask 目录、mask overlay、mask 面积曲线 | 182/182 非空，平均 mask 面积 12789.48 px | coarse visible mask，非 SAM2 |
| 手部 3D 重建与可视化 | 8 | 3D skeleton 视频、逐帧 skeleton 图、评分审查视频 | 182 帧 MediaPipe world skeleton | 非 MANO/mesh，非 metric 3D |
| 手部轨迹质量 | 8 | `hand_traj.npz`、帧级 CSV、质量曲线、overlay 视频 | valid 100%，interpolated 0，wrist_step max 0.005666 | step 为 non-metric 相对量 |
| 人体运动约束 | 3 | bone_length_error、temporal_jump_score、quality_flags | bone_length_error mean 0.002762，已标记风险帧 | 未做物理骨骼优化 |
| 方法说明与失败分析 | 3 | method、failure_cases、view_comparison、frame_quality_audit | side1 作为失败案例，side2 风险帧已审计 | 仍需人工视觉复核关键帧 |

## 核心接口

- 标准轨迹：`task2/outputs/trajectories/hand_traj.npz`
- Temporal refiner 对照轨迹：`task2/outputs/trajectories/hand_traj_temporal_refined.npz`
- 平滑轨迹：`task2/outputs/trajectories/hand_traj_smooth.npz`
- 原始检测 JSON：`task2/outputs/trajectories/mediapipe_landmarks.json`
- 接口校验报告：`task2/reports/hand_traj_validation.md`
- 按视角归档轨迹：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/trajectories/hand_traj.npz`

`hand_traj.npz` 已包含：

- 时间轴：`sequence_id`、`fps`、`frame_ids`、`timestamps`。
- 来源：`dataset_name`、`source_video`、`frame_manifest`、`primary_camera_id`、`camera_ids`。
- 2D/3D：`keypoints2d`、`keypoints2d_score`、`keypoints3d`、`keypoints3d_score`、`world_landmarks`。
- 手腕/手掌：`wrist_pos`、`wrist_rot`、`palm_pos`、`palm_rot`、`T_world_wrist`、`T_world_palm`。
- 指尖/retarget：`fingertips3d`、`fingertips3d_palm`、`retarget_landmark_names`、`retarget_keypoints3d`、`retarget_keypoints3d_palm`、`retarget_weights`。
- 质量字段：`valid`、`interpolated_flag`、`quality_score`、`temporal_jump_score`、`bone_length_error`、`failure_reason`。
- 有效性标记：`metric_3d_valid=False`、`world_alignment_valid=False`、`camera_calib_valid=False`、`contact_valid=False`、`phase_valid=False`。

## 可视化

- 关键点 overlay：`task2/outputs/videos/mediapipe_overlay.mp4`
- mask overlay：`task2/outputs/overlays/hand_mask_overlay.mp4`
- 3D skeleton 视频：`task2/outputs/videos/hand_3d_skeleton.mp4`
- 评分审查视频：`task2/outputs/videos/task2_scoring_review.mp4`
- 代表帧截图：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/figures/report_cases/`
- 逐帧 3D skeleton 图：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/figures/hand_3d_skeleton/`

## Mask

- 标准 mask 目录：`task2/outputs/masks/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/`
- 按视角归档 mask：`task2/outputs/by_view/weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2/masks/`
- 文件数：182
- 非空 mask：182
- 类型：coarse visible hand mask，由关键点凸包/bbox 生成，不是 SAM2 精细分割。

## 报告

- 方法说明：`task2/reports/task2_method.md`
- Baseline 报告：`task2/reports/task2_baseline_report.md`
- 轨迹质量：`task2/reports/trajectory_quality.md`
- 失败分析：`task2/reports/failure_cases.md`
- 多视角对比：`task2/reports/view_comparison.md`
- 接口校验：`task2/reports/hand_traj_validation.md`
- 帧级质量审计：`task2/reports/frame_quality/frame_quality_audit.md`
- 帧级指标 CSV：`task2/reports/frame_quality/frame_metrics.csv`
- 帧级质量图表：`task2/reports/frame_quality/figures/`
- 数据盘点：`task2/reports/real_data_inventory.md`
- 全量主视角汇总：`task2/reports/all_human_demo_side2_summary.md`
- 全量主视角 CSV：`task2/reports/all_human_demo_side2_metrics.csv`
- Temporal refiner 训练报告：`task2/outputs/training/temporal_refiner_side2_12seq/training_report.md`
- Temporal refiner 应用报告：`task2/reports/temporal_refiner_apply_report.md`
- Temporal refined 接口校验：`task2/reports/hand_traj_temporal_refined_validation.md`

## 帧级质量审计

- 质量曲线：`task2/reports/frame_quality/figures/quality_scores.png`
- 时序运动曲线：`task2/reports/frame_quality/figures/temporal_motion.png`
- 骨长误差曲线：`task2/reports/frame_quality/figures/bone_length_error.png`
- mask 面积曲线：`task2/reports/frame_quality/figures/mask_area.png`
- mask / bbox 对齐曲线：`task2/reports/frame_quality/figures/mask_alignment.png`
- 关键帧拼图：`task2/reports/frame_quality/figures/quality_keyframe_panel.jpg`

当前主要风险帧：

- fingertip step 风险：32、34、35、127、149。
- wrist / temporal jump 风险：122、124、127。
- low confidence 风险：66、67、73、74、75、78、103、105。
- bone length 风险：68、71、74。

## 多视角结果

| run_id | frames | valid | valid_ratio | quality_mean | 用途 |
|---|---:|---:|---:|---:|---|
| `weigh_drink_yykx_left_2026_0701_0052_53_camera_side_1` | 182 | 116 | 63.74% | 0.5469 | 失败案例 |
| `weigh_drink_yykx_left_2026_0701_0052_53_camera_side_2` | 182 | 182 | 100.00% | 0.9765 | 标准输出 |
| `weigh_drink_yykx_left_2026_0701_0052_53_camera_top` | 182 | 182 | 100.00% | 0.9762 | 备用视角 |

按视角归档根目录：`task2/outputs/by_view/`

## 全量主视角批处理

- 入口脚本：`task2/scripts/30_run_all_human_demo.py`
- 汇总报告：`task2/reports/all_human_demo_side2_summary.md`
- 汇总 CSV：`task2/reports/all_human_demo_side2_metrics.csv`
- 每序列相机标定：`task2/outputs/by_sequence/*/camera_calib.json` 和 `.npz`
- 每序列视角对比：`task2/outputs/by_sequence/*/view_comparison.md`

主视角结果摘要：

- 12 / 12 个 `camera_side_2` 视角跑通 pipeline。
- 8 / 12 个主视角检测成功率 >= 90%。
- 2 / 12 个主视角检测成功率低于 40%，需要后续采用其他视角或多视角融合。
- 全量结果仍为 MediaPipe pseudo-label baseline，不能视作手部 GT。

## 训练迭代

- 训练脚本：`task2/scripts/31_train_temporal_refiner.py`
- 应用脚本：`task2/scripts/32_apply_temporal_refiner.py`
- checkpoint：`task2/outputs/training/temporal_refiner_side2_12seq/temporal_refiner_best.pt`
- loss 曲线：`task2/outputs/training/temporal_refiner_side2_12seq/loss_curve.png`
- 训练报告：`task2/outputs/training/temporal_refiner_side2_12seq/training_report.md`
- 应用报告：`task2/reports/temporal_refiner_apply_report.md`
- 对照 NPZ：`task2/outputs/trajectories/hand_traj_temporal_refined.npz`

训练性质：

- 使用真实 HO-Tracker human_demo 视频跑出的 MediaPipe pseudo-label。
- 不使用人工 hand GT，不使用 MANO/SAM2/HaMeR 权重。
- 输出字段 `temporal_refiner_keypoints3d` 是辅助训练结果；核心 `keypoints3d` 保留原 baseline，避免把模型输出误作为 GT。

## 复现命令

标准单视角：

```bash
cd /mnt/workspace/hackthon
bash task2/scripts/20_run_single_view_baseline.sh \
  --video task1/data/HO-Tracker-Challenge/human_demo/weigh_drink_yykx__left__2026_0701_0052_53/video/camera_side_2.mkv \
  --sequence_id weigh_drink_yykx_left_2026_0701_0052_53 \
  --view_id camera_side_2 \
  --fps 15
```

三视角对比：

```bash
bash task2/scripts/20_run_single_view_baseline.sh --video task1/data/HO-Tracker-Challenge/human_demo/weigh_drink_yykx__left__2026_0701_0052_53/video/camera_side_1.mkv --sequence_id weigh_drink_yykx_left_2026_0701_0052_53 --view_id camera_side_1 --fps 15
bash task2/scripts/20_run_single_view_baseline.sh --video task1/data/HO-Tracker-Challenge/human_demo/weigh_drink_yykx__left__2026_0701_0052_53/video/camera_top.mkv --sequence_id weigh_drink_yykx_left_2026_0701_0052_53 --view_id camera_top --fps 15
bash task2/scripts/20_run_single_view_baseline.sh --video task1/data/HO-Tracker-Challenge/human_demo/weigh_drink_yykx__left__2026_0701_0052_53/video/camera_side_2.mkv --sequence_id weigh_drink_yykx_left_2026_0701_0052_53 --view_id camera_side_2 --fps 15
python task2/scripts/12_compare_views.py --by_view_dir task2/outputs/by_view --sequence_prefix weigh_drink_yykx_left_2026_0701_0052_53 --out_report task2/reports/view_comparison.md
```

全量主视角批处理：

```bash
python task2/scripts/30_run_all_human_demo.py \
  --human_demo_dir task1/data/HO-Tracker-Challenge/human_demo \
  --views camera_side_2 \
  --skip_existing \
  --summary_csv task2/reports/all_human_demo_side2_metrics.csv \
  --summary_md task2/reports/all_human_demo_side2_summary.md \
  --fps 15
```

Temporal refiner 训练与应用：

```bash
python task2/scripts/31_train_temporal_refiner.py \
  --by_view_dir task2/outputs/by_view \
  --views camera_side_2 \
  --include_run_regex '.*_2026_0701_[0-9]{4}_[0-9]{2}_camera_side_2' \
  --out_dir task2/outputs/training/temporal_refiner_side2_12seq \
  --epochs 80 \
  --batch_size 32 \
  --window 32 \
  --stride 8 \
  --hidden 256 \
  --device cuda

python task2/scripts/32_apply_temporal_refiner.py \
  --checkpoint task2/outputs/training/temporal_refiner_side2_12seq/temporal_refiner_best.pt \
  --in_npz task2/outputs/trajectories/hand_traj.npz \
  --out_npz task2/outputs/trajectories/hand_traj_temporal_refined.npz \
  --out_report task2/reports/temporal_refiner_apply_report.md \
  --device cuda
```

## 明确限制

- 当前 3D 是 MediaPipe world landmarks，不是 metric world 3D。
- 当前没有 MANO 参数、MANO mesh 或 HaMeR mesh。
- 当前没有使用相机标定做三角化或重投影误差。
- 当前 `T_world_wrist` / `T_world_palm` 服务接口调试，`world_alignment_valid=False`。
- 当前 `contact_likelihood`、`active_fingers`、`phase` 是占位/未知，不能作为真实接触标签。
- SAM2、HaMeR、MANO 仍是后续扩展，不是当前可运行 baseline 的一部分。
