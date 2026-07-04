# physical-ai-repo-scout

## Purpose
搜索和评估与 Physical AI 黑客松方向相关的开源代码仓库。核心目标是：在 24h 极限时间约束下，帮参赛者找到"一条命令能跑起来"的开源库，避免陷入环境配置的地狱。

## When to use
- 需要找某个功能的现成实现（如 occupancy mapping、VLA inference、drone simulation）
- 需要评估某个开源库是否适合在 24h 内集成
- 需要比较多个同类库的优缺点，做技术选型
- 准备 requirements.txt / environment.yml 时
- 不确定某个库是否需要 GPU 或特定 CUDA 版本时

## Inputs
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `functionality` | string | ✅ | 需要的功能，如 "3D occupancy mapping", "VLA inference", "potential field planner" |
| `language` | string | ❌ | 编程语言偏好，默认 "Python" |
| `license_filter` | string | ❌ | 许可证要求，默认 "permissive"（MIT/Apache/BSD） |
| `hardware_constraint` | string | ❌ | 硬件约束，如 "CPU only", "single GPU 8GB", "Jetson Orin" |
| `install_method` | string | ❌ | 安装偏好，默认 "pip"（也支持 "conda", "docker", "source"） |
| `ros_compatible` | bool | ❌ | 是否需要 ROS 兼容，默认 false |
| `max_deps` | int | ❌ | 可接受的最大依赖数量，默认 10 |

## Workflow
1. **搜索**: 在 GitHub (via gh CLI / API)、PyPI、RoboStack 中搜索
2. **检查**:
   - ⭐ GitHub stars（不是唯一指标，但是社区认可度的 proxy）
   - 📅 Last commit date（>1年未更新 → 标记为风险）
   - 📖 Documentation quality（README 是否清晰、有无 examples/）
   - 📦 Installation complexity（pip install vs 源码编译 vs 复杂依赖链）
   - 🧪 Test coverage（有无 CI、有无测试）
3. **快速验证**: 如果环境允许，尝试 `pip install` 并 import
4. **输出**: 结构化比较 + 最终推荐

## Outputs
```yaml
recommendations:
  - name: "library-name"
    url: "https://github.com/user/repo"
    stars: 1200
    last_update: "2026-06-15"
    one_liner: "一句话描述"
    install_cmd: "pip install library-name"
    integration_difficulty: "⭐"  # ⭐ easy, ⭐⭐ medium, ⭐⭐⭐ hard
    hardware_requirement: "CPU only"  # 或 "GPU 4GB", etc.
    pros:
      - "优点1"
      - "优点2"
    cons:
      - "缺点1"
    tested: true/false  # 是否实际验证过安装

comparison_table: |
  | 库 | Stars | Install | 集成难度 | 硬件需求 | 维护状态 |
  |----|-------|---------|----------|----------|----------|

final_pick:
  primary: "首选库名 + 理由"
  fallback: "备选库名 + 理由（当首选不可用时）"

installation_notes: |
  特殊注意事项，如 "需要 set CUDA_HOME", "Python>=3.10 only"
```

## Guardrails
- ❌ 不要推荐需要从源码编译且没有 CI 验证的库
- ❌ 不要推荐已停止维护（>1年无 commit）的库，除非它是唯一选择
- ❌ 不要推荐有已知安全漏洞的版本
- ✅ 优先推荐 `pip install` 一条命令能装好的库
- ✅ 标注需要 NVIDIA GPU / 特定 CUDA 版本 / 特定 OS 的库
- ✅ 如果库有 breaking changes，标注测试过的版本号
- ✅ 标注库的 license，避免商业化使用时的合规问题

## Example prompt (完整版)
> "找一个轻量的 Python 3D occupancy mapping 库。要求：1) 不需要 ROS，可以独立使用，2) 最好纯 Python 实现（或者有预编译 wheel），3) 能在 CPU 上实时运行（>20Hz 更新），4) 支持从深度图或点云更新。帮我比较 octomap-python、voxblox、和任何更好的替代方案。"

## Example prompt (简短版)
> "快速评估：open3d vs pcl-python vs pyntcloud，用于 3D 点云可视化，轻量优先。"

## Integration with SafeFly
当在 SafeFly 项目中使用时，会自动检查候选库与 SafeFly 技术栈的兼容性：
- 与 Octo/PyTorch 版本是否冲突
- 与 PyBullet/Gazebo 的仿真是否兼容
- 是否可以直接读取仿真传感器的输出格式
