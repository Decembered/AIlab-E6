# 子任务 3.3：物体形状重建与 IsaacGym Asset — 最终完成状态

## 总体进度

| 子项 | 分值 | 状态 | 证据路径 |
|------|------|------|----------|
| 物体 2D mask | 2 | 完成 | `outputs/mask_pose/{obj}/{seq}/masks/` + overlay |
| 物体 3D 模型与可视化 | 6 | 完成 | `runs/object_asset_v1/{obj}/renders/` |
| 几何质量 | 2 | 完成 | `runs/object_asset_v1/{obj}/report/geometry_check_*.txt` |
| 几何一致性 | 4 | 完成 | `outputs/object3_submission/geometry_summary.md` |
| IsaacGym asset 化 | 6 | 完成 | `runs/object_asset_v1/{obj}/asset/object.urdf` + cluster 验证 |
| 物体pose追踪 | 5 | 完成 | `outputs/mask_pose/{obj}/{seq}/object_trajectory.json` |
| **总计** | **25** | **全部完成** | |

---

## 1. 物体 2D Mask（2 分）

**Mask 定义：** 可见区域 mask — 目标物体在当前相机视角下的可见像素区域，由 SAM (vit_b) 自动分割生成。

**方法：**
- 对每段序列、每个相机视角，以 stride=30 帧（约 2 秒间隔）采样
- 首帧使用手工 ROI bbox 提示 SAM
- 后续帧使用运动检测 (MOG2) + 前一帧 mask 传播自动生成 bbox 提示
- SAM 后处理：保留最大连通分量

**完成统计：**

| 物体 | 序列数 | 总 mask 数 | 相机视角 |
|------|--------|-----------|----------|
| Bread | 2 | 48 | top, side_1, side_2 |
| Pipette | 3 | 96 | top, side_1, side_2 |
| Drink AD | 2 | 48 | top, side_1, side_2 |
| Drink YYKX | 3 | 75 | top, side_1, side_2 |
| **合计** | **10** | **267** | |

**证据文件：**
- Mask PNG: `outputs/mask_pose/{obj}/{seq}/masks/{cam}_frame_{idx}.png`
- Overlay 可视化: `outputs/mask_pose/{obj}/{seq}/mask_overlays/{cam}_frame_{idx}_overlay.jpg`
- 元数据: `outputs/mask_pose/{obj}/{seq}/mask_meta.json`

---

## 2. 物体 3D 模型与可视化（6 分）

四个物体的 3D 模型均已完成（.obj 格式），附带 trimesh 渲染截图。

| 物体 | 模型路径 | 面数 | 渲染 |
|------|---------|------|------|
| Bread | `runs/object_asset_v1/bread/mesh/visual_mesh.obj` | 1,280 | `renders/` (5 views) |
| Pipette | `runs/object_asset_v1/pipette/mesh/visual_mesh.obj` | 472 | `renders/` (4 views) |
| Drink AD | `runs/object_asset_v1/drink_ad/mesh/visual_mesh.obj` | 512 | `renders/` (5 views) |
| Drink YYKX | `runs/object_asset_v1/drink_yykx/mesh/visual_mesh.obj` | 512 | `renders/` (5 views) |

渲染接触表: `outputs/object3_submission/render_contact_sheet.png`

---

## 3. 几何质量（2 分）

全部 4 个物体的 visual mesh 和 collision mesh 均通过几何质量检查。

| 物体 | Visual watertight | Collision watertight | Visual faces | Collision faces |
|------|-------------------|---------------------|-------------|-----------------|
| Bread | 是 | 是 | 1,280 | 12 |
| Pipette | 是 | 是 | 472 | 12 |
| Drink AD | 是 | 是 | 512 | 64 |
| Drink YYKX | 是 | 是 | 512 | 64 |

所有面数 <20,000，符合要求。详细报告见 `runs/object_asset_v1/{obj}/report/geometry_check_*.txt`

---

## 4. 几何一致性（4 分）

| 物体 | 外接尺寸 (m) | 一致性说明 |
|------|-------------|-----------|
| Bread | 0.120 × 0.070 × 0.040 | 与真实面包比例一致 |
| Pipette | 0.258 × 0.020 × 0.085 | 移液枪细长形状，直径约2cm，长25.8cm |
| Drink AD | 0.070 × 0.070 × 0.200 | 圆柱形饮料瓶，直径7cm，高20cm |
| Drink YYKX | 0.070 × 0.070 × 0.200 | 同上 |

详细报告: `outputs/object3_submission/geometry_summary.md`

---

## 5. IsaacGym Asset（6 分）

| 物体 | URDF | Mass (kg) | IsaacGym 验证 |
|------|------|-----------|---------------|
| Bread | `runs/object_asset_v1/bread/asset/object.urdf` | 0.08 | PASS |
| Pipette | `runs/object_asset_v1/pipette/asset/object.urdf` | 0.08 | PASS |
| Drink AD | `runs/object_asset_v1/drink_ad/asset/object.urdf` | 0.3 | PASS |
| Drink YYKX | `runs/object_asset_v1/drink_yykx/asset/object.urdf` | 0.3 | PASS |

所有 asset 均可在 IsaacGym 中加载、spawn actor、执行 60 步物理仿真。验证日志: `runs/object_asset_v1/{obj}/asset_check.log`

---

## 6. 物体 Pose 追踪（5 分）

**方法：** 多视角 mask 质心三角测量 (Multi-view Mask Centroid Triangulation)
- 使用相机内参（K 矩阵）和外参（E 矩阵，世界到相机变换）
- 对每帧在 ≥2 个相机视角中计算 mask 质心
- DLT 三角测量得到 3D 位置
- 从 top-view mask PCA 主轴估计 yaw 旋转角
- 输出每帧 4×4 变换矩阵

**轨迹统计：**

| 物体 | 序列 | 帧数 | 位置范围 (m) X / Y / Z |
|------|------|------|------------------------|
| Bread | weigh_bread | 8 | [0.03,0.14] / [0.34,0.64] / [-0.16,0.15] |
| Bread | weigh_bread__left | 8 | [0.01,0.17] / [0.35,0.74] / [0.09,0.15] |
| Pipette | grasp_stand | 9 | [0.03,0.18] / [0.23,0.73] / [0.10,0.18] |
| Pipette | grasp_rotate | 12 | [0.01,0.18] / [0.24,0.73] / [0.09,0.19] |
| Pipette | grasp_press | 11 | [0.02,0.19] / [0.24,0.83] / [-0.16,0.19] |
| Drink AD | weigh_ad | 9 | [0.06,0.18] / [0.37,0.73] / [0.09,0.15] |
| Drink AD | weigh_ad__left | 7 | [0.03,0.14] / [0.38,0.64] / [-0.17,0.16] |
| Drink YYKX | weigh_yykx | 9 | [0.00,0.24] / [0.37,0.61] / [0.15,0.18] |
| Drink YYKX | weigh_yykx__left | 7 | [0.04,0.14] / [0.38,0.64] / [-0.17,0.16] |
| Drink YYKX | grasp_yykx | 9 | [0.04,0.28] / [0.38,0.73] / [0.09,0.17] |

**轨迹文件:** `outputs/mask_pose/{obj}/{seq}/object_trajectory.json`

**局限性说明：**
- 轨迹精度受 mask 质量限制；mask 间隔 (stride=30) 意味着轨迹在时间上是稀疏的
- 旋转估计仅从 top-view 获得 yaw 角，pitch/roll 未约束
- 对于饮料瓶等旋转对称物体，yaw 角不可靠（报告中标记为 ambiguous）

---

## 7. 运行复现

```bash
# Mask 提取
cd /mnt/workspace/AIlab-E6
python3.8 scripts/mask_extraction_v2.py --objects bread pipette drink_ad drink_yykx --stride 30

# Pose 追踪
python3.8 scripts/pose_tracking_v2.py --objects bread pipette drink_ad drink_yykx

# IsaacGym Asset 验证（需在 isaacgym 环境）
python3.8 scripts/validate_asset_isaacgym.py
```

## 8. 外部资源

- SAM (Segment Anything): https://github.com/facebookresearch/segment-anything (vit_b checkpoint)
- 相机标定来自 HO-Tracker-Challenge 数据集
- 3D 模型由 image-to-3D 方法生成，经后处理（缩放、水密化、减面）
