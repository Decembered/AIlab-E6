# rl-experiment-analyzer

## Purpose
分析强化学习实验的训练日志和评估结果，生成可答辩的分析报告。帮助参赛者快速定位训练问题、提取实验亮点、准备答辩用的数据可视化。

## When to use
- 训练了一个 RL 策略，需要分析为什么 work 或不 work
- 需要对比多个 baseline（如 VLA-only vs SafeFly）
- 跑完实验需要提取关键数字用于答辩
- 训练过程中 reward curve 出现异常（如突然下降、震荡）
- 需要分析 failure cases 的模式

## Inputs
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `experiment_data` | path/url | ✅ | 实验数据路径或 WandB run URL |
| `data_format` | enum | ✅ | "wandb" / "tensorboard" / "csv" / "json" / "custom" |
| `experiment_config` | dict | ❌ | 实验配置（用于和结果交叉验证） |
| `baseline_data` | path/url | ❌ | Baseline 实验数据（用于对比） |
| `analysis_focus` | enum | ❌ | "training_dynamics" / "reward_decomposition" / "failure_analysis" / "comparison" / "all" |
| `episode_recordings` | path[] | ❌ | 评估 episode 的录制数据（视频、状态序列等） |

## Workflow
1. **数据加载与解析**: 统一接口处理不同格式的实验日志
2. **训练动态分析**:
   - Reward curve: 收敛速度、最终性能、稳定性
   - Loss curves: 是否健康下降
   - Entropy / exploration metrics: 是否充分探索
   - 检测异常: sudden drop, oscillation, plateau
3. **评估分析**:
   - Success rate by scene difficulty
   - 安全指标（碰撞率、干预率等）
   - Episode length / path efficiency
4. **Failure case 分析**:
   - 什么类型的场景容易失败？
   - 失败的 common pattern 是什么？
   - 失败的 root cause 推测
5. **对比分析**（如果有 baseline）:
   - 统计显著性检验
   - 各指标 vs baseline 的百分比改进
6. **生成可视化**: 答辩级别的图表
7. **萃取亮点**: 最适合答辩展示的 3-5 个数据点

## Outputs
```yaml
training_dynamics:
  convergence_speed: "在 X steps 内到达 peak performance"
  stability: "stable" / "oscillating" / "diverging"
  sample_efficiency: "相对于 baseline 提升了 X%"
  anomalies:
    - description: "reward 在 500k steps 处突然下降"
      possible_cause: "learning rate 过大导致 catastrophic forgetting"
      suggestion: "尝试降低 lr 或增加 batch size"

evaluation_results:
  overall_success_rate: 0.85
  per_scene:
    - scene: "scene_1_open"
      success_rate: 0.95
      avg_collision_rate: 0.00
    - scene: "scene_3_dynamic"
      success_rate: 0.70
      avg_collision_rate: 0.05
  
  safety_metrics:
    collision_rate: 0.00
    intervention_rate: 0.18
    min_obstacle_distance_m: 0.65
    emergency_brake_count: 2

failure_analysis:
  common_failure_modes:
    - mode: "窄通道中被势场困住"
      frequency: 0.40  # 40% 的失败
      suggestion: "增加 deadlock detection + 沿通道中线策略"
    - mode: "VLA 输出在障碍物后方"
      frequency: 0.35
      suggestion: "增加 VLA 的 depth 输入以理解遮挡"
  
  worst_case_episodes:
    - episode_id: 42
      failure: "动态障碍物从侧面快速接近"
      root_cause: "safety radius 不够大 (0.3m)"

comparison_summary:
  metric_improvements:
    - metric: "collision_rate"
      baseline: 0.30
      ours: 0.00
      improvement: "100% reduction"
    - metric: "task_success_rate"
      baseline: 0.80
      ours: 0.85
      improvement: "+6.25%"

visualizations:
  - type: "reward_curve"
    description: "训练过程中碰撞率下降曲线"
  - type: "bar_chart"
    description: "4 组实验条件在 3 个场景的成功率对比"
  - type: "pie_chart"
    description: "Safety intervention 原因分布"
  - type: "trajectory_3d"
    description: "SafeFly vs VLA-only 3D 轨迹对比"

答辩亮点萃取:
  - "碰撞率从 30% 降至 0%——意味着在 30 次飞行中，SafeFly 避免了全部 9 次潜在碰撞"
  - "安全干预仅发生在 18% 的 steps 中——说明 VLA 决策大部分合理，safety wrapper 精准不冗余"
  - "Safety wrapper 单次响应 <5ms，对 50Hz 控制零影响"
```

## Guardrails
- ❌ 不要过解读训练曲线中的噪声（不是每个抖动都有意义）
- ❌ 不要在数据不足时声称统计显著性（<5 seeds → 标注"preliminary"）
- ❌ 不要编造 failure case 的 root cause——如果数据不能确定，标注"需要进一步分析"
- ✅ 标注统计显著性（p-value、error bar、seed 数量）
- ✅ 对 failure case 的分析要诚实，不确定时标注推测 vs 确定
- ✅ 自动检测常见的训练问题（reward collapse, vanishing gradient 迹象）

## Example prompt
> "分析这个 WandB run (https://wandb.ai/...)，SafeFly 在 3 个场景各 10 episode 的评估结果。对比 4 组实验条件：random baseline, VLA-only, safety-only, SafeFly。帮我提取答辩用的 3 个最亮眼的数据点。"

## Example prompt (简短版)
> "训练日志在 experiments/logs/run_003.csv，reward curve 在 500k steps 后突然下降，帮我诊断可能的原因。"

## Integration with SafeFly
这个 skill 直接消费 SafeFly 实验模块的输出。SafeFly 的评估脚本应按照这个 skill 的输入格式记录数据，确保分析流程顺畅。建议 SafeFly 实验输出如下结构：
```
experiments/
├── logs/
│   ├── run_001_vla_only.csv
│   ├── run_002_safety_only.csv
│   └── run_003_safefly.csv
├── episodes/
│   ├── ep_001/
│   │   ├── states.jsonl
│   │   ├── rgb_frames/     # 可选
│   │   └── metadata.json
│   └── ...
└── config.yaml
```
