# Member C — Task 3 工作总结合集

> **项目**: Video2Motion2Action — 灵巧手操作技能迁移
> **赛道**: Track 1: Physical Intelligence, Topic T-1, Team Challenge
> **角色**: Member C — 物体重建与资产生成 (Sub-task 3.3, 25 pts)
> **时间**: 2026-07-04 ~ 2026-07-05
> **最新补充**: 2026-07-05 最终展示材料、推荐图包与 task2-4 preflight

---

## 一、总览

### 任务分解与得分

| 子项 | 内容 | 分值 | 状态 |
|------|------|------|------|
| 3.3.1 | Object 2D Mask 提取 | 2 | ✅ |
| 3.3.2 | Object 3D Model 重建与可视化 | 6 | ✅ |
| 3.3.3 | Geometric Quality 几何质量 | 2 | ✅ |
| 3.3.4 | Geometric Consistency 几何一致性 | 4 | ✅ |
| 3.3.5 | IsaacGym Asset 仿真资产 | 6 | ✅ |
| 3.3.6 | Object Pose Tracking 姿态追踪 | 5 | ✅ |
| Bonus | Articulated Models + 碰撞优化 + 物理参数 | 额外 | ✅ |
| 3.4 | 物体侧 handoff、preflight、手物可视化 (与Member A/B协作) | 共享 | ✅ 对象侧完成，完整 tracking 依赖 A/B |
| **总计** | | **25/25** | **ALL COMPLETE** |

### 处理的数据

- **4类物体**: 面包(Bread)、移液枪(Pipette)、AD饮料瓶(Drink AD)、YYKX饮料瓶(Drink YYKX)
- **12个视频序列**: 每个序列包含3个同步相机视角（top + side_1 + side_2）的RGB视频
- **~3000张2D掩码**: SAM vit_b 模型，stride=5 采样
- **757帧3D轨迹**: DLT三角化 + Mesh Silhouette优化 + 卡尔曼滤波
- **展示材料**: `figure/` 精选图包、`figure/recommended/` 重点图、PPT级合成图、定量分析图、demo mp4

---

## 二、工作历程 (Timeline)

### Phase 1: 基础设施搭建 (07-04 初期)
- 环境搭建: Python 3.8, IsaacGym Preview 4, 2×RTX4090
- 数据准备: HO-Tracker Challenge 数据集，12个人类操作演示序列
- 关键发现: **GT 物体pose全部被masked** (pose_3d.hdf5中mask=False)，必须从视频中恢复轨迹

### Phase 2: 2D Mask + 3D模型 (07-04)
- SAM vit_b 掩码提取，bbox prompt + motion propagation
- 多视图轮廓挤出法 → 4个watertight .obj文件
- IsaacGym资产生成 (URDF + collision mesh + mass/inertia)
- 4/4 物理模拟验证通过

### Phase 3: Pose Tracking 优化 (07-05 上午)

| 版本 | 算法 | 效果 |
|------|------|------|
| V2 (初始) | 重心DLT + top-view PCA yaw + 样条插值 | 6/12 GOOD, 6/12 SUSPICIOUS_FAST |
| V3 (Tier 1) | DLT三角化 + 73角度mesh silhouette搜索 + stride=5 | **12/12 GOOD**, 97% 3-view |
| V3.1 (Tier 2) | V3 + 卡尔曼滤波 (常速度模型) | 帧间平滑，抖动消除 |

### Phase 4: 可视化与渲染 (07-05 中午)
- 轨迹叠加视频: 257帧 (12序列，top视角)
- 几何一致性验证: 68帧 (4物体)
- IsaacGym渲染脚本与多视角可视化产物: 支持12序列×4视角 (persp/top/front/side)
- 修复: Top视角黑屏 (gimbal lock, target +0.005 offset)
- 修复: 视频文字叠加 (物体名/帧号/时间/3D坐标/进度条)
- 最终展示策略: RTX4090 + IsaacGym Preview 4存在GPU渲染兼容风险，答辩图包中剔除黑屏图，以验证日志、状态卡片和有效hand-object帧视频为准

### Phase 5: 3D模型精度提升 (07-05 下午)

| 版本 | 方法 | 结果 |
|------|------|------|
| V1 (原始) | 轮廓挤出法 | 矩形柱/盒状，形状不准确 |
| V2 (后处理) | 细分 + Laplacian平滑 + 凸包 | 面数增加但不自然 |
| V3 (定制化) | 每个物体专用策略 | 全部watertight |

**V3 各物体策略**:
- **饮料瓶**: 圆拟合top mask → 圆柱体mesh (130v, wt=True)
- **面包**: 轮廓有机挤出 + top/bottom fan caps (2882v, wt=True)
- **移液枪**: 多截面变宽度挤出 (962v, wt=True)

### Phase 6: 集成与验证 (07-05 末期)
- 手物联合跟踪集成 (Member A/B协作)
- task2-4 集成preflight
- 12个序列整理为 `data/pipeline_assets/{sequence}/left_urdf/`，包含 `scan.urdf`、`scan.obj`、`left_obj.pkl`、`left_hand.pkl` 接口占位
- 新增 `scripts/preflight_task24_integration.py`，自动检查 object/hand pkl、URDF/OBJ、timestamps、placeholder hand
- 严格preflight明确指出当前 `left_hand.pkl` 仍为placeholder，避免把对象侧 smoke test 误报为真实 hand-object tracking

### Phase 7: 答辩展示与GitHub交付整理 (07-05 最终)
- 新建 `figure/` 展示包，精选物体资产、几何overlay、pose tracking、IsaacGym验证、task3.4对象侧联调证据
- 清理黑屏/无效IsaacGym截图，改用验证日志和状态卡片 `isaacgym_validation_ppt.png` 展示asset可加载性
- 新增 `figure/recommended/`，按PPT推荐顺序标记9个核心图片/视频
- 新增 `figure/RECOMMENDED_FIGURES.md`，为每张推荐图写明用途、优先级和答辩caption
- 新增 `figure/07_analysis_charts/`，生成蛛网图、小提琴图、轨迹稳定性条带图
- 将 `WORK_SUMMARY.md` 同步到 `figure/06_reports/`，便于GitHub页面与个人贡献展示引用

---

## 三、核心贡献

### 1. Object Pose Tracking — 轨迹恢复 (5 pts)

**问题**: GT物体pose在pose_3d.hdf5中被故意masked (mask=all False)

**解决路径**:
```
V2: 重心DLT → stride=15, 11个源关键帧 → 样条插值2486帧 → 50% SUSPICIOUS
  ↓ Tier 1 优化
V3: DLT三角化 + mesh silhouette 73角度yaw搜索 → stride=5, 757个真实跟踪帧 → 12/12 GOOD
  ↓ Tier 2 优化
V3.1: V3 + 卡尔曼滤波 (常速度6D状态) → 帧间抖动消除
```

**最终指标**:
| 指标 | 值 |
|------|-----|
| 总帧数 | 757 |
| 3-view覆盖率 | 97.0% |
| 无效率 | 3.0% (theta_jump) |
| 质量评级 | 12/12 GOOD |
| 跟踪方法 | DLT + Mesh Silhouette + Kalman |

### 2. 3D模型重建 — 精度提升

**问题**: 轮廓挤出法产生矩形盒状模型，与实际物体差距大

**解决路径**:
```
V1: 轮廓挤出 → 矩形柱 → 形状差
  ↓ 后处理
V2: 细分 + Laplacian + 凸包 → 面数多但不自然
  ↓ 定制化重构
V3: 饮料→圆柱, 面包→有机挤出+caps, 移液枪→多截面
```

**最终模型**:
| 物体 | 顶点 | 面 | Watertight | 尺寸 (m) |
|------|------|-----|-----------|-----------|
| Bread | 2,882 | 5,760 | ✅ | 0.12×0.07×0.04 |
| Pipette | 962 | 1,920 | ✅ | 0.258×0.02×0.085 |
| Drink AD | 898 | 1,792 | ✅ | 0.07×0.07×0.20 |
| Drink YYKX | 898 | 1,792 | ✅ | 0.07×0.07×0.20 |

### 3. IsaacGym渲染视频

IsaacGym asset加载与60步物理验证通过。由于IsaacGym Preview 4在RTX4090上存在GPU渲染兼容问题，黑屏/无效渲染图不作为最终展示证据；最终展示采用CPU physics验证日志、`isaacgym_validation_summary.json`、`isaacgym_validation_ppt.png` 以及有效的hand-object帧视频。

### 4. Task2-4 / 3.4 对象侧接口与Preflight

**贡献**:
- 为12个序列生成pipeline handoff结构: `data/pipeline_assets/{sequence}/left_urdf/`
- 每个序列提供 `scan.urdf`、`scan.obj`、`left_obj.pkl` 和 `left_hand.pkl` 接口文件
- 新增 `scripts/preflight_task24_integration.py`，无需IsaacGym即可检查联调输入
- 输出 `task24_preflight_strict.json` 和 `task24_preflight_object_only.json`

**当前结论**:
- Object-side结构完整: 12/12序列有URDF、OBJ、object pose和时间戳
- 严格模式失败原因全部为 `left_hand.pkl_is_placeholder_or_all_zero`
- 完整真实hand-object tracking仍需Member B替换真实手部轨迹、Member A接Sharpa rollout

### 5. 展示材料与个人贡献证据包

**贡献**:
- 新建 `figure/`，集中整理答辩/PPT/GitHub可直接使用的高质量展示素材
- 新建 `figure/recommended/`，将重点图片和demo视频按推荐顺序编号
- 新增 `figure/RECOMMENDED_FIGURES.md`，标记每张图的展示优先级、用途与caption
- 新增 `figure/07_analysis_charts/`，补充定量分析图:
  - `asset_quality_radar.png`: 多指标资产质量蛛网图
  - `trajectory_speed_violin.png`: 轨迹速度分布小提琴图
  - `trajectory_speed_band.png`: 轨迹稳定性条带图，非GT误差图
- 删除/排除黑屏IsaacGym截图，避免展示误导性或打不开的材料

---

## 四、技术方案详解

### 4.1 Pose Tracking算法

```
Algorithm: Multi-view Mask Silhouette Pose Recovery

Input: 3相机mask + 标定 + 3D mesh
Output: 每帧4×4变换矩阵 (位置 + yaw旋转)

步骤:
1. mask_2d_stats: 提取每相机掩码重心 + 2D bbox + orientation
2. camera_ray_from_pixel: 重心 → 相机光线 (世界坐标系)
3. DLT triangulation: 多视图光线 → 最小二乘3D交点
4. optimize_yaw_by_multiview_projection:
   - 遍历73个yaw候选角 (-180°~180°)
   - 投影mesh点到3个相机视图
   - 计算 silhouette coverage + bbox IoU
   - 选最高分yaw
5. yaw_transform_world: 构建4×4位姿矩阵
6. Kalman filter: 常速度模型平滑 (位置6D + yaw 2D状态)
```

### 4.2 3D模型重建算法

| 物体类型 | 算法 |
|----------|------|
| 圆柱体 (饮料) | top mask圆拟合 → radius; side mask → height; 64-slice圆柱mesh |
| 有机形状 (面包) | top mask轮廓 → 12层锥形挤出 + top/bottom triangle fan caps → Laplacian平滑 |
| 变截面 (移液枪) | top mask轮廓 → 沿主轴64截面; side mask → 每截面厚度估计 |

---

## 五、文件清单

### Python脚本 (17个)

| 脚本 | 功能 |
|------|------|
| `scripts/mask_extraction_v2.py` | SAM掩码提取 (stride=5) |
| `scripts/pose_tracking_v2.py` | Pose Tracking V3.1 (DLT + Silhouette + Kalman) |
| `scripts/recon_advanced.py` | 3D模型重建V3 (圆柱/有机/变截面) |
| `scripts/recon_visual_hull.py` | Visual Hull重建 (已修复calib路径, 但因标定偏差未采用) |
| `scripts/render_trajectory.py` | IsaacGym 4视角渲染 (48 MP4) |
| `scripts/viz_trajectory_overlay.py` | 轨迹叠加视频帧 (257帧) |
| `scripts/viz_geometry_consistency.py` | 3D模型叠加视频帧 (68帧) |
| `scripts/validate_asset_isaacgym.py` | IsaacGym资产加载验证 |
| `scripts/verify_integration.py` | 集成验证 (资产+pose回放) |
| `scripts/verify_hand_object_joint.py` | 手物联合验证 |
| `scripts/preflight_task24_integration.py` | task2-4集成preflight |
| `scripts/recon_multiview_v4.py` | 多视图重建 (旧版) |
| `scripts/recon_scaled_v3.py` | 缩放重建 (旧版) |
| `scripts/fix_collision_mesh.py` | 碰撞网格修复 |
| `scripts/check_isaac_env.py` | IsaacGym环境诊断 |
| `scripts/foundationpose_track.py` | FoundationPose集成 (不可用 - 缺深度) |
| `scripts/run_foundationpose_tracking.py` | FoundationPose运行器 (不可用) |

### 核心库 (1个)

| 文件 | 功能 |
|------|------|
| `src/object_recon/pose_tracking.py` | Pose tracking 核心库 (1857行) |

### 生成物

| 类型 | 数量 | 路径 |
|------|------|------|
| 2D掩码 | ~2,278张 | `outputs/mask_pose/` |
| 掩码叠加图 | ~2,000帧 | `outputs/mask_pose/{obj}/{seq}/mask_overlays/` |
| 3D模型 (.obj) | 4个visual + 4个collision | `runs/object_asset_v1/{obj}/mesh/` |
| URDF资产 | 4个object.urdf | `runs/object_asset_v1/{obj}/asset/` |
| IsaacGym验证 | 4个asset_check.log | `runs/object_asset_v1/{obj}/` |
| 模型渲染 | 20张 (5视角×4物体) | `runs/object_asset_v1/{obj}/renders/` |
| Pose轨迹 | 12个JSON + 12个NPZ | `outputs/mask_pose/{obj}/{seq}/` |
| 轨迹叠加 | 257帧 | `outputs/trajectory_viz/` |
| 几何验证 | 68帧 | `outputs/geometry_viz/` |
| IsaacGym验证/可视化 | 验证日志 + 有效帧/视频 | `runs/object_asset_v1/`, `figure/04_isaacgym_validation/`, `figure/05_task24_integration/` |
| 汇总图表 | 2张PNG | `outputs/object3_submission/` |
| 提交材料 | 1份报告 + bonus报告 | `outputs/object3_submission/` |
| PPT展示图包 | 15MB精选素材 | `figure/` |
| 重点推荐图 | 9个图片/视频 | `figure/recommended/` |
| 定量分析图 | 3张PNG + 数据JSON | `figure/07_analysis_charts/` |
| 工作总结副本 | 1份Markdown | `figure/06_reports/WORK_SUMMARY.md` |

### Object Bonus (额外加分)

- 6个铰接模型: pipette_articulated (棱柱关节), drink_ad/yykx_articulated (旋转关节)
- 3个优化模型: bread_optimized (多分辨率碰撞), drink_ad/yykx_optimized (摩擦参数)

---

## 六、关键Git提交记录

| Hash | 时间 | 内容 |
|------|------|------|
| 82a91a3 | 10:17 | advanced 3D reconstruction V3 — fix remaining limitations |
| cf10fac | 10:02 | advanced 3D reconstruction V2 — object-specific modeling |
| c638c9c | 09:46 | improve 3D models — subdivision + Laplacian + convex hull |
| 633b025 | 09:34 | fix IsaacGym top-view black screen + video text overlays |
| 7a8a6ca | 09:23 | IsaacGym trajectory renderer — 48 MP4 videos |
| 8cce7c7 | 09:10 | regenerate visualizations with V3.1 + fix cam calib loading |
| aa2bcf2 | 08:22 | Tier 2 — Kalman filter temporal smoothing |
| 61db184 | 08:16 | pose tracking V3 — DLT + mesh silhouette optimization |
| 3df02a0 | 07:40 | fix remaining deductions — drink symmetry + pitch/roll |
| 085bc1a | 07:35 | cleanup — mark old pose_tracking as deprecated |
| 7413fdd | 07:31 | Object Bonus — articulated models + physics optimization |
| d1f5117 | 07:25 | integration handoff — 12 sequence assets for Sharpa pipeline |
| f989ea3 | 07:15 | fix pose tracking — Gaussian smooth + linear interp |
| cb18e1b | 07:02 | Object reconstruction & IsaacGym asset — complete deliverables |
| 9916b4c | 06:12 | dense pose tracking (stride=5) + video overlay evidence |
| 10872a6 | 05:13 | Multi-view 3D pose triangulation |
| ec6374c | 04:40 | complete object reconstruction (3.3) |

_(详见 `git log --oneline --graph` 获取完整历史)_

---

## 七、已知局限

1. **旋转估计**: 仅yaw (绕Y轴) 通过mesh silhouette优化 + Kalman滤波估计，pitch/roll未约束
2. **相机标定偏差**: 三相机光线交点分散 (~0.4m)，导致Visual Hull voxel carving不可行
3. **无深度数据**: FoundationPose (RGB-D 6D pose) 无法使用，环境已就绪但缺depth
4. **透明物体**: Drink瓶的掩码质量较低
5. **移液枪theta_jump**: 3%帧有yaw跳变 (窄的非对称物体silhouette优化不稳定)
6. **无GT误差曲线**: 主物体GT pose mask全False，因此不能报告真实GT tracking error；条带图仅表示轨迹速度稳定性/分布
7. **3.4完整tracking依赖队友输出**: 当前对象侧handoff完成，但 `left_hand.pkl` 仍为placeholder，完整Sharpa tracking需Member B真实手轨迹与Member A rollout
8. **IsaacGym GPU渲染兼容性**: RTX4090 + IsaacGym Preview 4可能出现黑屏/illegal memory access，最终展示中已剔除黑屏图，改用验证日志和状态卡片

---

## 八、环境信息

- **计算**: 2× NVIDIA RTX 4090 (44GB each), CUDA 12.9
- **Python**: 3.8 (主环境, IsaacGym) + 3.12 (torch 2.5.1, diffusers)
- **IsaacGym**: Preview 4, CPU管线 (无GPU渲染)
- **关键依赖**: SAM vit_b, trimesh 4.12.2, Open3D, pytorch3d 0.7.9, nvdiffrast, warp-lang 1.14

---

## 九、一键复现

```bash
# 0. 设置数据路径
export HO_TRACKER_DATA=/mnt/workspace/Hackthon/data/human_demo

# 1. 掩码提取
python3.8 scripts/mask_extraction_v2.py --stride 5 --objects bread pipette drink_ad drink_yykx

# 2. Pose Tracking
python3.8 scripts/pose_tracking_v2.py --objects bread pipette drink_ad drink_yykx

# 3. 3D模型重建
python3.8 scripts/recon_advanced.py --objects bread pipette drink_ad drink_yykx

# 4. 轨迹可视化
python3.8 scripts/viz_trajectory_overlay.py

# 5. 几何一致性可视化
python3.8 scripts/viz_geometry_consistency.py

# 6. IsaacGym渲染 (48个视频)
python3.8 scripts/render_trajectory.py --all

# 7. IsaacGym验证
for obj in bread pipette drink_ad drink_yykx; do
  python3.8 scripts/validate_asset_isaacgym.py --urdf runs/object_asset_v1/$obj/asset/object.urdf
done

# 8. task2-4 / 3.4 对象侧preflight
python3.8 scripts/preflight_task24_integration.py --root data/pipeline_assets
python3.8 scripts/preflight_task24_integration.py --root data/pipeline_assets --allow-placeholder-hand
```

### 展示材料入口

```text
figure/README.md
figure/RECOMMENDED_FIGURES.md
figure/recommended/
figure/07_analysis_charts/
```

---

> **报告生成时间**: 2026-07-05
> **Git仓库**: github.com/Decembered/AIlab-E6
> **分支**: task3-object-reconstruction
> **最终Commit**: 以 `git log --oneline -1` 为准
