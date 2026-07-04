# vla-safety-wrapper-designer

## Purpose
为 VLA（Vision-Language-Action）驱动的机器人/无人机系统设计安全包装器（Safety Wrapper）。核心问题是：VLA 输出不一定物理安全——这个 skill 帮你设计一个轻量、实时、可解释的安全层，夹在 VLA 和底层控制之间。

## When to use
- 需要为 VLA 输出设计安全检查机制
- 需要调优 safety wrapper 的参数（安全半径、势场系数、速度上限）
- 需要分析 safety wrapper 的 failure mode 和 corner case
- 需要设计 safety wrapper 的状态机和仲裁逻辑
- 需要评估 safety wrapper 的计算复杂度是否满足实时性

## Inputs
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `vla_output_type` | enum | ✅ | VLA 输出格式: "waypoint" / "velocity" / "trajectory" / "joint_action" |
| `vla_output_spec` | dict | ✅ | VLA 输出的规格，如 `{"dim": 3, "range": [[-5,5], [-5,5], [0,3]], "frequency_hz": 2}` |
| `robot_type` | enum | ✅ | 机器人类型: "drone" / "mobile_robot" / "manipulator" / "legged" / "other" |
| `safety_constraints` | dict | ✅ | 安全约束规格 |
| `perception_spec` | dict | ✅ | 感知输入规格 |
| `control_frequency` | int | ✅ | 底层控制频率 (Hz)，如 50 |
| `compute_budget` | string | ❌ | 计算预算: "embedded" (Jetson/etc) / "laptop" / "server"，默认 "laptop" |

**`safety_constraints` 详细规格:**
```yaml
safety_constraints:
  collision_avoidance:
    enabled: true
    safety_radius_m: 0.5        # 无人机安全球半径
    prediction_horizon_s: 1.0   # 预测时域
  velocity_limits:
    enabled: true
    max_linear_velocity: 2.0    # m/s
    max_angular_velocity: 1.0   # rad/s
  workspace_bounds:
    enabled: true
    x_range: [-10, 10]          # m
    y_range: [-10, 10]
    z_range: [0.3, 5.0]         # 最低 0.3m, 最高 5m
  dynamics_constraints:
    enabled: false               # 无人机场景通常不需要额外约束（PX4 已处理）
    max_acceleration: null
```

**`perception_spec` 详细规格:**
```yaml
perception_spec:
  modality: "occupancy_grid"     # 或 "depth_map" / "point_cloud" / "voxel_grid"
  resolution_m: 0.2             # grid resolution
  local_window_m: [10, 10, 5]   # 局部窗口大小 [x, y, z]
  update_frequency_hz: 20       # 地图更新频率
```

## Workflow
1. **分析 VLA 输出语义**: 理解 VLA 输出的物理含义，确定需要检查哪些安全约束
2. **设计状态机**: 定义 safety wrapper 的状态和转移条件
   ```
   NOMINAL ──→ CORRECTED ──→ HOVER ──→ EMERGENCY
     │            │             │           │
     └── VLA 输出安全        └── 无法找到安全路径
                  └── VLA 输出需要修正
   ```
3. **设计每个安全检查**:
   - **碰撞检查**: 将 VLA 输出的 waypoint/trajectory 投影到 occupancy grid 上
   - **速度检查**: 将 VLA 输出 clamp 到安全速度范围内
   - **工作空间检查**: 确保目标点在安全飞行区域内
   - **动力学检查**: (可选) 确保 motion primitive 物理可行
4. **设计修正策略**:
   - **碰撞修正**: 人工势场法 (APF) 局部重规划
   - **速度修正**: 简单 clamp
   - **工作空间修正**: 投影到合法区域
5. **设计仲裁逻辑**: 多个修正如何融合
6. **复杂度分析**: 验证每个 safety check 的 wall-clock time < 1/control_frequency

## Outputs
```yaml
safety_wrapper_design:
  state_machine:
    states: ["NOMINAL", "CORRECTED", "HOVER", "EMERGENCY"]
    transitions: |
      NOMINAL→CORRECTED: VLA waypoint violates any constraint
      CORRECTED→NOMINAL: 修正后的 waypoint 安全 且 连续 N 个 tick 无冲突
      CORRECTED→HOVER:   修正失败（所有修正方向都被阻挡）
      HOVER→CORRECTED:   新的安全路径出现
      ANY→EMERGENCY:     传感器故障 或 电池临界

  safety_checks:
    - name: "collision_check"
      method: "ray_cast_on_occupancy_grid"
      complexity: "O(N) where N = ray_steps"
      worst_case_latency_ms: 2
      fallback: "inflate safety_radius by 2x if sensor noise detected"
    
    - name: "velocity_clamp"
      method: "element_wise_clamp"
      complexity: "O(1)"
      worst_case_latency_ms: 0.01
      fallback: null
    
    - name: "workspace_check"
      method: "boundary_projection"
      complexity: "O(1)"
      worst_case_latency_ms: 0.01
      fallback: null

  correction_strategies:
    - name: "potential_field_replanning"
      when: "collision_check fails"
      method: |
        F_total = K_att * (vla_waypoint - current_pos)   # 目标吸引力
                + Σ K_rep * (1/d - 1/d0) * (1/d²)       # 障碍物排斥力
      parameters:
        K_att: 1.0        # 吸引系数
        K_rep: 2.0        # 排斥系数
        d0: 1.0           # 排斥力影响距离 (m)
      complexity: "O(M*D) where M = nearby_obstacle_cells, D = 3"
      worst_case_latency_ms: 5
  
  parameter_recommendations:
    safety_radius_m: 0.5              # 建议 1.5-2× 无人机半径
    attraction_gain_K_att: 1.0        # 太高 → overshoot, 太低 → 保守
    repulsion_gain_K_rep: 2.0         # 太高 → 震荡, 太低 → 不够安全
    repulsion_range_d0_m: 1.0         # 障碍物开始排斥的距离
    consecutive_safe_ticks_for_nominal: 5  # 连续安全 tick 数才回到 NOMINAL

  code_skeleton: |
    class SafetyWrapper:
        def __init__(self, config): ...
        def check(self, vla_output, occupancy_grid, current_state) -> SafetyResult: ...
        def correct(self, vla_output, occupancy_grid, current_state) -> CorrectedOutput: ...
        def arbitrate(self, vla_output, corrected_output) -> FinalCommand: ...
        def step(self, vla_output, occupancy_grid, current_state) -> FinalCommand: ...
    
  edge_cases:
    - "传感器噪声导致假阳性障碍物 → 使用 temporal smoothing"
    - "狭窄通道中被两侧排斥力困住 → 检测 deadlock, 沿通道中线前进"
    - "VLA 连续输出相同的不安全 waypoint → 增加 goal attraction gain 逐步减小"
    - "动态障碍物快速接近 → 扩大安全半径, 优先横向避让"
```

## Guardrails
- ❌ 不要设计计算复杂度为 O(N²) 或更高的 safety check（无法实时）
- ❌ 不要声称 safety wrapper 提供 formal safety guarantee（它是 heuristic 的）
- ❌ 不要过度设计——黑客松场景下简单可靠 > 复杂完备
- ✅ Safety wrapper 的每个决策都必须是可解释的（记录干预原因）
- ✅ 明确标注各 safety check 的计算时间，确保总和 < 1/control_frequency
- ✅ 每个 safety check 必须有对应的 debug log 和可视化支持
- ✅ 提供参数调优的直觉（"如果碰撞太多 → 增大 safety_radius；如果过度保守 → 减小 K_rep"）

## Example prompt (完整版)
> "设计一个用于无人机的 safety wrapper。VLA (Octo) 输出 3D waypoint (x,y,z)，频率 2Hz。无人机底层控制频率 50Hz，通过 MAVROS 发送速度指令。使用 OctoMap（0.2m resolution, 10×10×5m 局部窗口）作为感知。安全约束：碰撞避免（0.5m 安全球）、速度上限 2m/s、飞行高度 0.3-5m。计算平台：笔记本 CPU。"

## Example prompt (简短版)
> "帮我调优 SafeFly 的 APF 参数。目前碰撞率 0% 但任务成功率只有 60%——怀疑 safety wrapper 太保守了。"

## Integration with SafeFly
这个 skill 是 SafeFly 项目的核心设计工具。输出的代码骨架可以直接映射到 `src/safety/safety_wrapper.py`。设计时假设：
- VLA 是异步的（低频更新 waypoint）
- Safety wrapper 是同步的（每个 control tick 独立运行）
- 两者通过共享变量通信（latest_waypoint + timestamp）
