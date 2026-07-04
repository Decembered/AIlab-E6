# 子任务 3.3 交付物：物体形状重建与 IsaacGym Asset

## 复现步骤

```bash
# 1. 环境依赖
# - Python 3.8, numpy, opencv-python, trimesh, segment-anything (vit_b)
# - IsaacGym Preview 4（仅 asset 验证需要）
# - 数据集：HO-Tracker-Challenge（HuggingFace）

# 2. 下载数据集
export HF_ENDPOINT=https://hf-mirror.com
python3.8 -c "
from huggingface_hub import snapshot_download
snapshot_download('kelvin34501/HO-Tracker-Challenge', repo_type='dataset',
    local_dir='data')
"

# 3. 提取 Object 2D Mask
python3.8 scripts/mask_extraction_v2.py --objects bread pipette drink_ad drink_yykx --stride 30

# 4. 物体 Pose 追踪
python3.8 scripts/pose_tracking_v2.py --objects bread pipette drink_ad drink_yykx

# 5. IsaacGym Asset 验证
python3.8 scripts/validate_object_assets_v1.py
```

## 交付物清单

| 文件/目录 | 说明 |
|-----------|------|
| `assets/` | 四个物体的 URDF、visual mesh、collision mesh |
| `renders/` | 每个物体的多视角渲染截图（.png） |
| `validation/` | IsaacGym 加载验证日志 |
| `geometry/` | 几何质量检查报告（watertight, manifold, 面数） |
| `masks/` | Mask 提取报告与样本（*详见 `outputs/mask_pose/`） |
| `trajectory/` | 物体运动轨迹 JSON + 概览图 |
| `member_c_task3_final_status.md` | 完整完成状态报告 |

## 主要结果截图

- 物体 3D 渲染：`renders/` 目录
- 轨迹概览：`trajectory/trajectory_overview.png`
- Mask overlay 样本：`outputs/mask_pose/{obj}/{seq}/mask_overlays/`

## 外部资源

- SAM (Segment Anything): https://github.com/facebookresearch/segment-anything (vit_b)
- HO-Tracker-Challenge 数据集: https://huggingface.co/datasets/kelvin34501/HO-Tracker-Challenge
- IsaacGym Preview 4: NVIDIA 官网
