# 3d-spatial-demo-planner

## Purpose
为 3D 空间智能/重建方向的 demo 做端到端规划。涵盖数据采集方案、重建方法选择、可视化部署，确保在黑客松时间约束内产出"看起来很强"的 3D demo。

## When to use
- 需要用 3DGS 或 NeRF 从视频/照片重建 3D 场景
- 需要规划数据采集方案（用手机、无人机还是仿真）
- 需要快速搭建一个可展示的 3D 可视化
- 需要将 3D 重建与语义查询结合（"红色的车在哪？"）
- 需要在 3D 重建上做 path planning

## Inputs
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `scene_type` | enum | ✅ | 场景类型: "indoor" / "outdoor" / "corridor" / "aerial" / "simulated" |
| `capture_device` | enum | ✅ | 采集设备: "phone" / "drone" / "simulation" / "existing_video" |
| `method` | enum | ✅ | 重建方法: "3dgs" / "nerf" / "dust3r" / "fast"（自动选择最快） |
| `output_format` | enum | ✅ | 展示形式: "web_viewer" / "video_render" / "point_cloud" / "interactive" / "ros_rviz" |
| `gpu_spec` | dict | ✅ | GPU 规格 |
| `time_budget_hours` | float | ✅ | 时间预算 (小时)，通常 1-4h |
| `add_semantic_query` | bool | ❌ | 是否添加语义查询，默认 false |
| `existing_data_path` | string | ❌ | 已有的视频/图片数据路径（如果有） |

**`gpu_spec` 详细规格:**
```yaml
gpu_spec:
  model: "RTX 3060"       # 或 "none" (CPU only)
  vram_gb: 12
  cuda_version: "12.1"
```

## Workflow
1. **数据采集方案设计**:
   - 相机轨迹规划（保证足够 overlap 和 coverage）
   - 帧率/分辨率建议
   - 数据预处理（去模糊、下采样、COLMAP 特征提取）
2. **方法选择**:
   - 根据 GPU + 时间预算选择重建方法
   - 提供安装命令
3. **重建执行**:
   - 运行重建 pipeline
   - 监控训练进度
   - 判断何时停止（质量够用即可，不追求极致）
4. **可视化部署**:
   - 根据 output_format 选择展示工具
   - 生成可展示的产物
5. **(可选) 语义查询**: 用 CLIP/OpenSeg 做 3D feature embedding

## Outputs
```yaml
data_capture_plan:
  device: "phone (iPhone 13)"
  trajectory: |
    围绕目标场景缓慢走一圈（360°），保持相机平视
    然后抬高相机 30° 再走一圈
    总录制时间: ~90 秒
  settings:
    resolution: "1920×1080"
    fps: 30
    stabilization: "on"
  tips:
    - "保持匀速运动，不要太快"
    - "避免纯白墙壁（无特征）"
    - "确保场景光照均匀"

preprocessing_steps:
  - "ffmpeg -i input.mp4 -r 3 frames/%04d.png  # 每 0.33s 抽一帧"
  - "检查抽帧质量: 应该清晰、无明显运动模糊"
  - "建议总共 200-500 帧"

method_recommendation:
  method: "3DGS"
  reason: "在 RTX 3060 12GB 上，200 帧场景约 30-45 分钟训练到可用质量"
  alternatives:
    - method: "Nerfstudio (splatfacto)"
      reason: "更方便的 CLI，但训练稍慢"
    - method: "DUSt3R"
      reason: "不需要 COLMAP，直接输出点云，最快但质量较低"
  
  install_commands: |
    pip install nerfstudio
    ns-install-cli  # 安装 COLMAP + 其他依赖

training_commands: |
  ns-train splatfacto --data data/my_scene/ --max-num-iterations 10000
  
monitoring: |
  每 1000 步检查 PSNR:
  - PSNR > 25: 基本可用
  - PSNR > 30: 质量不错
  - PSNR > 35: 非常好

visualization_plan:
  format: "web_viewer"
  tool: "viser (Nerfstudio 内置)"  # 或 "polycam web" / "rerun"
  deployment: |
    训练完成后:
    ns-viewer --load-config outputs/my_scene/splatfacto/config.yml
    # Web viewer 在 http://localhost:7007

semantic_query:
  enabled: true
  method: "LERF (Language Embedded Radiance Fields)"
  integration: |
    在 Nerfstudio 中使用 LERF:
    ns-train lerf --data data/my_scene/
  demo_script: |
    # 查询 "红色的椅子"
    python query_3d.py --text "red chair" --model outputs/lerf/

edge_cases:
  - "COLMAP 失败 (无特征场景)": 尝试 DUSt3R 替代，或调整拍摄（增加纹理）
  - "GPU OOM": 降低分辨率到 1K（Nerfstudio 支持 --pipeline.model.resolution-schedule）
  - "训练太慢": 降低 max-num-iterations 到 5000，早停也可以展示
  - "网页展示需要网络": 改用本地 polyscope viewer
```

## Guardrails
- ❌ 不要建议需要 >1h 训练的配置在时间预算 <2h 时
- ❌ 不要推荐需要 >16GB VRAM 的方法（3DGS 全分辨率需要，但可以降分辨率）
- ❌ 不要假设 COLMAP 一定成功——总是准备 DUSt3R 作为备选
- ✅ 优先选择有预训练模型或快速模式的方法
- ✅ 准备数据质量不够时的降级方案
- ✅ 标注每个步骤的预期时间

## Example prompt
> "我有一段 3 分钟的无人机航拍视频（1080p, 30fps），想用 3DGS 做重建然后网页展示。GPU 是 RTX 3060 12GB。时间预算 2 小时。另外，能不能在重建的场景里做简单的语义查询？比如'找到停车场里的红色车'。"

## Example prompt (简短版)
> "30 分钟能做一个看得过去的 3DGS demo 吗？我有一段 1 分钟的室内视频。GPU 是 RTX 4090。"

## Integration with SafeFly
虽然 SafeFly 主方向不是 3D 重建，但 3D 空间智能可以作为 SafeFly 的感知模块增强：
- 3DGS 离线重建飞行场景 → 作为高精度 reference map
- 在重建场景中标注安全区域和禁飞区
- 用语义查询辅助 VLA 理解 "飞到红门后面"
- 答辩时可以作为"SafeFly 的感知层可扩展性"展示
