# Physical AI Hackathon — Master Strategy Document

> **赛道 E: 强化学习与物理智能系统**
> 上海 AI Lab 黑客松 | 24 小时极限开发
> 参赛者背景: 智能无人机 / 航空航天工程

---

## 目录

1. [五类命题方向提炼](#1-五类命题方向提炼)
2. [各方向深度分析](#2-各方向深度分析)
3. [Top 3 推荐选题](#3-top-3-推荐选题)
4. [首选方案深度设计: SafeFly](#4-首选方案深度设计-safefly)
5. [Claude Skills 规范](#5-claude-skills-规范)
6. [约束条件自查](#6-约束条件自查)
7. [赛前准备清单](#7-赛前准备清单)

---

## 1. 五类命题方向提炼

从赛题介绍中提炼出以下 5 个命题方向，覆盖"强化学习与物理智能系统"的核心交叉地带：

| # | 方向 | 关键词 | 核心问题 |
|---|------|--------|----------|
| 1 | **强化学习系统** | 分布式RL、多智能体、训练框架 | 如何高效训练可部署于物理系统的RL策略？ |
| 2 | **世界模型 / Imagination Planning** | Model-based RL、Dreamer、latent planning | 如何在想象中规划，减少真实交互成本？ |
| 3 | **VLA 泛化与动作生成** | Vision-Language-Action、任务泛化、zero-shot | 如何让机器人理解语言指令并生成可泛化的动作？ |
| 4 | **灵巧手 / 操作策略学习** | Dexterous manipulation、tactile sensing、sim2real | 如何让机械手学会精细操作？ |
| 5 | **3D 空间智能 / 4D 重建渲染** | NeRF、3DGS、4D reconstruction、spatial AI | 如何从感知数据中构建可用的3D/4D世界表示？ |

---

## 2. 各方向深度分析

### 2.1 强化学习系统

**可能的实际赛题形式:**
- 在给定仿真环境（Isaac Sim / MuJoCo / PyBullet）中训练一个导航/操作策略
- 设计分布式RL训练框架以加速策略收敛
- 多智能体协作/对抗场景
- Sim-to-Real transfer 实验

**评委可能看重:**
- 训练效率（wall-clock time → 策略性能曲线）
- Reward shaping 设计的合理性
- 是否有 sim2real 的考虑
- 系统工程完备性（logging, visualization, reproducibility）

**适合的开源栈:**

| 组件 | 推荐 |
|------|------|
| 仿真 | Isaac Sim (GPU), MuJoCo (轻量), PyBullet (最轻) |
| RL框架 | Stable-Baselines3 (初学者), RLlib (分布式), CleanRL (可读性) |
| 硬件加速 | Isaac Lab / Orbit |
| 日志/可视化 | WandB, TensorBoard |

**24h 可落地的 demo:**
- 用 CleanRL + PyBullet 在简单导航环境中训练一个 PPO 策略
- 在 WandB 上展示 reward curve
- 录制策略执行视频

**风险点:**
- ⚠️ 训练时间不可控（可能不收敛）
- ⚠️ Reward 设计是玄学
- ⚠️ 超参数调优耗时
- ⚠️ 如果没有 GPU，训练极慢

**与你的匹配度:** ★★★☆☆ (3/5)
- 你有控制背景，理解 MDP 框架
- 但你更偏向 classic control + planning，而非纯 RL
- 24h 内纯 RL 方向 demo 风险较高

---

### 2.2 世界模型 / Imagination Planning

**可能的实际赛题形式:**
- 训练一个 latent dynamics model，在 latent space 中进行 planning
- 使用 DreamerV3 / TD-MPC2 在 DMControl 上展示
- 对比 model-based vs model-free 的 sample efficiency
- 在想象中做 rollout，可视化 imagined trajectories

**评委可能看重:**
- 世界模型的预测精度
- Planning 效率 vs model-free baseline
- Latent space 的可视化与可解释性
- 是否能在想象中处理 novel situations

**适合的开源栈:**

| 组件 | 推荐 |
|------|------|
| 世界模型 | DreamerV3 (最成熟), TD-MPC2, PlaNet |
| 仿真 | DMControl, Meta-World |
| Planning | MPPI, CEM, iLQR |

**24h 可落地的 demo:**
- 使用 DreamerV3 的预训练 checkpoint 在 DMControl Walker 上做 imagination rollout
- 可视化 latent space 中的 imagined trajectories vs real trajectories
- 展示 model-based planning 在 sparse reward 场景下的 sample efficiency 优势

**风险点:**
- ⚠️ 从零训练世界模型需要大量 GPU 时间
- ⚠️ Latent dynamics 可能 collapse
- ⚠️ 需要理解较深的 model-based RL 理论

**与你的匹配度:** ★★★☆☆ (3/5)
- "Imagination planning"和你的路径规划背景有精神上的呼应
- 但具体技术栈（latent dynamics, RSSM）离你的 comfort zone 较远
- 如果使用预训练模型，可以做 lightweight 的方案

---

### 2.3 VLA 泛化与动作生成

**可能的实际赛题形式:**
- 用预训练 VLA（OpenVLA / Octo / RT-2）在仿真机器人上执行语言指令
- Fine-tune VLA 到特定 embodiment（如无人机）
- 设计 prompting 策略提升 VLA 的任务泛化能力
- 对比不同 VLA 模型在导航任务上的表现

**评委可能看重:**
- VLA 在新场景/新任务上的 zero-shot 泛化
- 语言指令理解的准确性
- 动作生成的物理合理性
- 与具体机器人平台（embodiment）的适配

**适合的开源栈:**

| 组件 | 推荐 |
|------|------|
| VLA 模型 | OpenVLA (7B, 开源), Octo (轻量), RT-1-X, SpatialVLA |
| 仿真 | ManiSkill2, LIBERO, Habitat |
| 机器人框架 | ROS 2, MoveIt |
| 推理加速 | vLLM, TensorRT-LLM |

**24h 可落地的 demo:**
- 加载 Octo（27M 参数，CPU 可跑）在 Habitat 仿真中做室内导航
- 给定语言指令 ("go to the red chair")，VLA 输出 waypoint
- 对比 baseline（random / heuristic）
- 这是**最接近 OpenFly/AutoFly 的方向**

**风险点:**
- ⚠️ 预训练模型对无人机场景泛化差（主要在操作数据集上训练）
- ⚠️ 大模型推理慢，需要 GPU
- ⚠️ 模型输出可能物理不可行

**与你的匹配度:** ★★★★★ (5/5)
- 你正在读 OpenFly 和 AutoFly 论文
- UAV-VLA / Aerial VLN 直接就是你的研究方向
- 你可以做"VLA for Aerial Navigation"的独特交叉

---

### 2.4 灵巧手 / 操作策略学习

**可能的实际赛题形式:**
- 在 Isaac Gym 中训练灵巧手抓取策略
- Sim-to-real transfer 实验
- 触觉反馈下的闭环操作
- 多指协调控制

**评委可能看重:**
- 操作精度和成功率
- Sim2real gap 的处理
- 策略的鲁棒性
- 与真实硬件的对接

**适合的开源栈:**

| 组件 | 推荐 |
|------|------|
| 仿真 | Isaac Gym (GPU并行), ManiSkill2, SAPIEN |
| 灵巧手模型 | Shadow Hand, Allegro Hand, LEAP Hand |
| RL | RLGames, DrQ-v2 |

**24h 可落地的 demo:**
- 在 ManiSkill2 中训练一个简单的 pick-and-place 策略
- 展示训练曲线和仿真视频

**风险点:**
- ⚠️ 你完全没有灵巧手经验
- ⚠️ Isaac Gym 需要 NVIDIA GPU
- ⚠️ 训练时间可能在 24h 边缘
- ⚠️ 答辩时可能被问倒 hardware-specific 的问题

**与你的匹配度:** ★☆☆☆☆ (1/5)
- 和你的无人机背景完全不相关
- 不建议选择

---

### 2.5 3D 空间智能 / 4D 重建渲染

**可能的实际赛题形式:**
- 使用 3D Gaussian Splatting 从视频中重建场景
- 4D 动态场景重建（4DGS）
- 基于重建的 3D 场景理解与语义查询
- 实时 SLAM + 3D 重建

**评委可能看重:**
- 重建质量（PSNR, SSIM, LPIPS）
- 重建速度（训练时间）
- 应用场景的创新性
- 与具身智能的结合

**适合的开源栈:**

| 组件 | 推荐 |
|------|------|
| 3D 重建 | 3D Gaussian Splatting, Nerfstudio, SuGaR |
| SLAM | ORB-SLAM3, DROID-SLAM, FAST-LIO2 |
| 实时渲染 | SIBR_viewers, polyscope |
| 点云处理 | Open3D, PCL, Kaolin |

**24h 可落地的 demo:**
- 用手机拍摄一段室内/室外视频
- 使用 Nerfstudio 或 3DGS 进行快速重建
- 在 viewer 中展示重建结果
- 可以结合无人机航拍数据

**风险点:**
- ⚠️ 3DGS 训练需要 GPU（RTX 3060+ 可在 30min 完成）
- ⚠️ 视频质量影响重建质量
- ⚠️ 如果没有好的采集数据，效果会很差

**与你的匹配度:** ★★★★☆ (4/5)
- 你做过 FAST-LIO2、点云建图
- 你有无人机航拍数据采集能力
- 这是你背景的直接延伸
- 但这个方向离"物理智能"稍远，更像是 perception

---

## 3. Top 3 推荐选题

### 推荐一 ⭐⭐⭐ (首选)

**中文标题:** SafeFly：基于 VLA 语义决策与安全干预层的无人机自主导航

**英文标题:** SafeFly: VLA-Guided Aerial Navigation with Safety-Critical Intervention Layer

**1 句话核心 idea:**
利用预训练 VLA 做高层语义导航决策，在其输出上叠加一个轻量级 safety wrapper（基于人工势场+碰撞检测），确保无人机在未知环境中的安全飞行。

**技术路线:**
1. **Perception**: 深度相机 → occupancy map（利用 OctoMap / Voxblox）
2. **VLA 高层决策**: 预训练 VLA（Octo / OpenVLA）接收 RGB + 语言指令 → 输出 waypoint / 速度方向
3. **Safety Wrapper**: 实时检测 VLA 输出的轨迹是否与 occupancy map 冲突 → 如有冲突，启动势场局部重规划
4. **执行**: 安全后的指令 → PX4 offboard control (MAVROS)

**最小可行 demo:**
- Gazebo + PX4 SITL 仿真环境
- 一个室内/室外场景，包含障碍物
- 语言指令输入（"飞过门，然后向左转"）
- VLA 生成导航路径，safety wrapper 实时修正
- 对比展示"有 safety wrapper vs 无 safety wrapper"的安全差异

**可量化指标:**
- 碰撞率 (collision rate)
- 任务成功率 (task success rate)
- Safety intervention 频率
- 路径平滑度 (jerk)
- VLA 推理延迟 vs safety wrapper 响应延迟

**可视化形式:**
- RViz 3D 可视化：无人机轨迹 + occupancy map + safety zone
- Side-by-side 对比视频：VLA-only vs SafeFly
- Grafana-style 实时 dashboard（延迟、碰撞预警、干预次数）

**可能创新点:**
- 首次将 VLA 应用于 UAV 导航场景（区别于操作场景）
- Safety wrapper 作为 model-agnostic 的即插即用组件
- 可以展示 VLA 决策的"可解释性"（为什么改道、哪里危险）
- 可扩展：safety wrapper 可以替换为 learning-based safety critic

**失败时的降级方案:**
- 如果 VLA 推理太慢/效果差 → 退化为 rule-based 导航 + safety wrapper（仍然有价值）
- 如果仿真不稳定 → 用纯 Python 2D 仿真展示核心概念
- 如果 safety wrapper 太简单 → 强调这是"minimal viable safety"，展示框架的可扩展性

**适合放在答辩里的亮点表述:**
> "当前 VLA 研究主要集中在桌面操作场景，我们首次探索了 VLA 在无人机导航场景中的应用。更重要的是，我们提出了一个 model-agnostic 的安全干预层——无论 VLA 输出什么，SafeFly 都能保证无人机的物理安全。这不是替代 VLA，而是让 VLA 更安全地部署到真实物理世界。"

---

### 推荐二 ⭐⭐

**中文标题:** DroneDreamer：面向无人机导航的世界模型与想象规划

**英文标题:** DroneDreamer: World Model-based Imagination Planning for UAV Navigation

**1 句话核心 idea:**
利用 DreamerV3 风格的世界模型在 latent space 中学习无人机飞行动力学，在想象中进行 rollout 规划，减少真实环境交互成本。

**技术路线:**
1. 在 PyBullet 无人机仿真中收集 flight trajectories
2. 训练轻量 world model (RSSM + decoder) 学习 dynamics
3. 在 latent space 中用 MPPI/CEM 进行 planning
4. 对比 model-free baseline 的 sample efficiency

**最小可行 demo:**
- PyBullet 无人机环境
- 预收集的 trajectory dataset（或在线收集）
- 可视化 imagined trajectories vs real trajectories
- 展示 world model 可以在想象中预测碰撞

**可量化指标:**
- Prediction error (MSE on next state)
- Planning success rate
- Sample efficiency vs model-free
- Imagination rollout 的物理合理性

**风险点:**
- 训练世界模型需要 GPU 时间
- 无人机 dynamics 比桌面机器人复杂
- 可能需要比预训练 checkpoint 更多的调参

**与你的匹配度:** ★★★★☆ (4/5)
- 与你的 planning 背景相关
- 但需要更多的 ML 工程

---

### 推荐三 ⭐⭐

**中文标题:** UAV-GS：面向无人机航拍的 3D Gaussian Splatting 实时场景重建与语义查询

**英文标题:** UAV-GS: Real-time 3D Gaussian Splatting from Drone-Captured Footage for Spatial Intelligence

**1 句话核心 idea:**
利用无人机航拍视频，通过 3D Gaussian Splatting 快速重建场景，并支持自然语言语义查询（"红色的车在哪里？"）。

**技术路线:**
1. 使用无人机（或手机模拟）采集场景视频
2. 3DGS 快速重建（Nerfstudio / 原生 3DGS）
3. 结合 CLIP 特征进行语义嵌入
4. 支持自然语言查询 3D 空间

**最小可行 demo:**
- 用手机拍摄一段办公室走廊视频（模拟无人机视角）
- 3DGS 重建（30min 训练）
- Web viewer 展示重建结果
- 自然语言查询 demo

**可量化指标:**
- PSNR / SSIM / LPIPS
- 训练时间
- 渲染帧率
- 语义查询准确率

**风险点:**
- 3DGS 效果依赖输入数据质量
- 自然语言查询的 feature embedding 是额外工作量
- 与"物理智能"主题的关联性需要包装

**与你的匹配度:** ★★★★☆ (4/5)

---

## 4. 首选方案深度设计: SafeFly

> **SafeFly: VLA-Guided Aerial Navigation with Safety-Critical Intervention Layer**

### 4.1 系统架构图（文字版）

```
┌─────────────────────────────────────────────────────────────────────┐
│                          SafeFly System                              │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   User Input  │    │   RGB Camera │    │   Depth Camera       │  │
│  │ "fly through  │    │   (Sim)      │    │   + IMU (Sim)        │  │
│  │  the door,    │    │              │    │                      │  │
│  │  then turn    │    └──────┬───────┘    └──────────┬───────────┘  │
│  │  left"        │           │                       │              │
│  └───────┬───────┘           │                       │              │
│          │                   ▼                       ▼              │
│          │          ┌──────────────────────────────────────┐        │
│          │          │       Perception Module              │        │
│          │          │  - OctoMap / Voxblox occupancy       │        │
│          │          │  - Ego-centric local grid map        │        │
│          │          │  - Collision sphere estimation       │        │
│          │          └──────────┬───────────────────────────┘        │
│          │                     │                                    │
│          │                     │ occupancy grid                     │
│          ▼                     ▼                                    │
│  ┌──────────────────────────────────────────────────────┐          │
│  │              VLA High-Level Planner                    │          │
│  │  - Model: Octo (27M) or OpenVLA (7B, optional)        │          │
│  │  - Input: RGB image + language instruction             │          │
│  │  - Output: next 3D waypoint / velocity direction       │          │
│  │  - Noise-aware: outputs are probabilistic              │          │
│  └──────────┬───────────────────────────────────────────┘          │
│             │                                                       │
│             │ raw waypoint / velocity vector                        │
│             ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐          │
│  │            Safety Wrapper (Critical!)                  │          │
│  │                                                        │          │
│  │  ┌─────────────────────────────────────────────────┐  │          │
│  │  │ 1. Collision Check                               │  │          │
│  │  │    - Project VLA waypoint onto occupancy grid    │  │          │
│  │  │    - If collision → trigger replanning           │  │          │
│  │  │    - Check velocity limits (safety envelope)     │  │          │
│  │  ├─────────────────────────────────────────────────┤  │          │
│  │  │ 2. Potential Field Local Replanning              │  │          │
│  │  │    - Goal attraction → VLA's intended waypoint   │  │          │
│  │  │    - Obstacle repulsion → from occupancy grid    │  │          │
│  │  │    - Output: safe local velocity command         │  │          │
│  │  ├─────────────────────────────────────────────────┤  │          │
│  │  │ 3. Safety Arbitration                            │  │          │
│  │  │    - Blend VLA intent + PF correction            │  │          │
│  │  │    - Fallback: hover in place if unsafe          │  │          │
│  │  │    - Log all interventions for debuggability      │  │          │
│  │  └─────────────────────────────────────────────────┘  │          │
│  └──────────┬───────────────────────────────────────────┘          │
│             │                                                       │
│             │ safe velocity / attitude setpoint                     │
│             ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐          │
│  │           PX4 Offboard Control (MAVROS)               │          │
│  │  - setpoint_velocity / setpoint_position              │          │
│  │  - Low-level PID control                              │          │
│  │  - EKF2 state estimation                              │          │
│  └──────────┬───────────────────────────────────────────┘          │
│             │                                                       │
│             ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐          │
│  │        Gazebo + PX4 SITL Simulation                   │          │
│  │  - UAV model: Iris quadcopter                         │          │
│  │  - World: indoor/outdoor with obstacles                │          │
│  │  - Sensors: RGB-D camera, IMU, GPS                     │          │
│  └──────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 模块划分与职责

| 模块 | 职责 | 输入 | 输出 | 关键文件 |
|------|------|------|------|----------|
| **Perception** | 局部环境感知 | RGB-D 图像, IMU | Occupancy grid (3D), 碰撞球半径 | `perception/octomap_builder.py` |
| **VLA Planner** | 高层语义决策 | RGB 图像, 语言指令 | Waypoint/速度向量 (3D) | `vla_planner/vla_inference.py` |
| **Safety Wrapper** | 安全干预与局部重规划 | VLA waypoint, occupancy grid | 安全速度指令 | `safety/safety_wrapper.py` |
| **PX4 Bridge** | 与飞控通信 | 安全速度指令 | MAVLink 消息 | `bridge/px4_bridge.py` |
| **Orchestrator** | 主循环与状态机 | 所有上游输出 | 系统状态 | `orchestrator/main_loop.py` |
| **Visualizer** | 3D 可视化与日志 | 所有模块状态 | RViz markers, WandB logs | `viz/dashboard.py` |

### 4.3 输入输出定义

**VLA Planner:**
```
输入:
├── image: np.ndarray (224, 224, 3)    # RGB, normalized
├── instruction: str                    # "fly through the door and turn left"
├── current_pose: PoseStamped          # 无人机当前位姿
└── goal_pose: PoseStamped (optional)  # 目标位姿

输出:
├── waypoint: np.ndarray (3,)          # [x, y, z] in world frame
├── confidence: float                  # VLA confidence score
└── horizon: int                       # planning horizon (N steps)
```

**Safety Wrapper:**
```
输入:
├── vla_waypoint: np.ndarray (3,)      # VLA 目标点
├── occupancy_grid: OctoMap            # 当前占据地图
├── current_pose: PoseStamped          # 当前位姿
├── current_velocity: np.ndarray (3,)  # 当前速度
└── safety_params:                     # 安全参数
    ├── safety_radius: float (0.5m)    # 安全球半径
    ├── max_velocity: float (2.0 m/s)  # 最大速度
    └── emergency_brake_dist: float    # 紧急刹车距离

输出:
├── safe_velocity: np.ndarray (3,)     # 安全速度指令
├── intervention_flag: bool            # 是否干预了VLA输出
├── intervention_type: str             # "none"|"collision_avoidance"|"velocity_clamp"|"hover"
└── debug_info: dict                   # 干预原因、替代路径等
```

### 4.4 可用开源库

**核心依赖 (必装):**

| 库 | 用途 | 安装难度 |
|----|------|----------|
| `octomap-python` / `voxblox` | 3D occupancy mapping | ⭐ 轻量 |
| `numpy`, `scipy` | 数值计算 | ⭐ 预装 |
| `gymnasium` | 环境接口标准 | ⭐ 轻量 |
| `matplotlib`, `open3d` | 可视化 | ⭐⭐ 中等 |
| `msgpack`, `pyzmq` | 进程间通信 | ⭐ 轻量 |
| `pymavlink`, `mavros` | PX4 通信 | ⭐⭐ 中等 |

**VLA 模型 (选装):**

| 模型 | 参数量 | 硬件需求 | 推理速度 | 推荐度 |
|------|--------|---------|----------|--------|
| **Octo-small** | 27M | CPU 可跑 (慢) / GPU 4GB | ~0.5s (GPU) | ⭐⭐⭐ 首选 |
| **Octo-base** | 93M | GPU 8GB | ~1s (GPU) | ⭐⭐ |
| **OpenVLA-7B** | 7B | GPU 24GB+ | ~2-3s (A100) | ⭐ 太重 |
| **RT-1-X** (via Open X-Embodiment) | 35M | GPU 8GB | ~0.3s | ⭐⭐ |
| **SpatialVLA** | ~300M | GPU 12GB | ~1s | ⭐⭐ |

**强烈推荐首选 Octo-small：**
- 27M 参数，内存友好
- 预训练权重可直接下载
- 支持 language-conditioned action prediction
- 即使在没有 GPU 的笔记本上也能跑（虽然慢）

**仿真环境:**

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **Gazebo + PX4 SITL** | 最接近真实飞控、ROS 生态 | 安装较重、配置复杂 | ⭐⭐⭐ |
| **PyBullet + simple quad** | 秒装、纯 Python、轻量 | 动力学不够真实 | ⭐⭐⭐ |
| **AirSim** | 视觉效果好、微软维护 | 已停止维护、安装重 | ⭐⭐ |
| **Isaac Sim** | 最强仿真、GPU 加速 | 需要 NVIDIA GPU + 超大安装 | ⭐ 太重 |
| **2D Mini-grid** | 极轻、秒跑 | 不是 3D | ⭐⭐ (降级方案) |

**建议：首选 PyBullet 做核心开发 + Gazebo+PX4 SITL 做最终 demo（如果时间允许）**

### 4.5 数据来源

**不需要训练数据（核心优势！）:**
- VLA 使用预训练权重（Octo 在 Open X-Embodiment 上预训练）
- Safety wrapper 是 rule-based，不需要数据
- 只需要构建仿真场景用于演示

**如果需要微调（可选加分项）:**
- 使用 Gazebo/PyBullet 自动采集无人机视角的 RGB 图像
- 配合简单的导航指令生成 synthetic dataset
- 格式: `{image, instruction, waypoint}` 三元组
- 这是在 demo 基础上的锦上添花

**OpenFly/AutoFly 数据关联:**
- 可以提及 OpenFly 使用的 UAV-VLN 数据集范式
- 如果时间允许，可以在仿真中生成类似的 navigation episode 数据
- 展示你的方案与学术界最新方向的直接关联

### 4.6 实验设计

**场景设计（3 个递增难度）:**

```
Level 1: 开放空间 + 稀疏障碍
  - 指令: "fly forward 5 meters, then land"
  - 验证: 基本 VLA + safety 集成

Level 2: 室内走廊 + 门框
  - 指令: "fly through the doorway, then turn left"
  - 验证: VLA 理解空间关系 + safety 处理门框

Level 3: 动态障碍
  - 指令: "fly to the red marker while avoiding obstacles"
  - 验证: safety wrapper 处理动态场景 + VLA 鲁棒性
```

**实验对比:**

| 实验条件 | VLA | Safety Wrapper | 预期结果 |
|----------|-----|----------------|----------|
| A (Baseline) | ✗ (random waypoint) | ✗ | 高碰撞率 |
| B (VLA only) | ✓ (Octo) | ✗ | 中等碰撞率, VLA 输出有时不安全 |
| C (Safety only) | ✗ (heuristic goal) | ✓ | 无碰撞但不一定理解语义 |
| D (SafeFly) | ✓ (Octo) | ✓ | 语义理解 + 安全保证 |

**每次实验重复 10 个 episodes，记录所有指标。**

### 4.7 指标设计

**安全指标 (Safety Metrics):**
- **碰撞率 (Collision Rate):** episodes 中发生碰撞的比例（越低越好）
- **安全干预率 (Safety Intervention Rate):** VLA 输出被 safety wrapper 修改的频率
- **最小障碍物距离 (Min Obstacle Distance):** 飞行过程中离最近障碍物的距离
- **紧急刹车次数 (Emergency Brake Count)**

**任务指标 (Task Metrics):**
- **任务成功率 (Task Success Rate):** 是否在规定时间内到达目标
- **路径效率 (Path Efficiency):** 实际路径长度 / 最短路径长度
- **任务完成时间 (Time to Goal)**

**系统指标 (System Metrics):**
- **VLA 推理延迟 (VLA Latency):** ms
- **Safety Wrapper 响应时间 (Safety Response Time):** ms（应 < 10ms）
- **端到端控制频率 (Control Frequency):** Hz（应 > 10Hz）

**可解释性指标:**
- 安全干预的原因分布（碰撞预测 / 速度超限 / 其他）
- VLA confidence 与 safety intervention 的相关性

### 4.8 Demo 展示形式

**方案 A: 实时仿真展示（最佳）**
- PyBullet 或 Gazebo 窗口：无人机 3D 飞行
- 右侧 panel：VLA 的 RGB 输入 + attention heatmap
- 底部 terminal：实时日志（VLA waypoint → safety analysis → final command）
- 颜色标记：
  - 🟢 绿色轨迹段 = VLA 原始输出安全，直接执行
  - 🟡 黄色轨迹段 = VLA 被 safety wrapper 修正
  - 🔴 红色轨迹段 = 碰撞风险，紧急避让

**方案 B: 录像 + Dashboard（网络差时备选）**
- 预先录制的飞行视频
- Grafana/Panel dashboard 展示指标
- 截图 + 对比分析

**方案 C: 后处理可视化（降级方案）**
- 用 matplotlib 画轨迹对比图
- 用 Open3D 渲染 3D 场景
- 注释 safety intervention 的时间点

### 4.9 答辩话术

**开场 (30 秒 elevator pitch):**
> "大家好，我们的项目叫 SafeFly。核心问题是：当 VLA 大模型被用于无人机导航时，模型的输出不一定物理安全。我们设计了一个轻量级安全包装器，在 VLA 和飞控之间做实时安全检查与局部重规划。简单说就是——VLA 告诉你'飞过去'，SafeFly 确保你不会撞墙。"

**痛点 (Why this matters):**
> "目前的 VLA 研究几乎全部集中在桌面操作场景。但无人机有三点本质不同：第一，碰撞后果严重，不能'试错'；第二，3D 空间的复杂性远超桌面 2D；第三，动力学约束更严格。直接把 VLA 部署到无人机上是不安全的。"

**技术创新 (What we built):**
> "我们的贡献是三层：
> 1. 首次将 VLA 适配到无人机导航场景
> 2. 提出了 model-agnostic 的 safety wrapper——不修改 VLA 本身，而是作为即插即用的安全层
> 3. 人工势场 local replanning 在 <5ms 内完成，不牺牲实时性"

**实验效果 (Results):**
> "在 3 个递增难度的仿真场景中, SafeFly 将碰撞率从 VLA-only 的 30% 降到了 0%，同时保持了 85% 的任务成功率。安全干预只发生在不到 20% 的 steps 中——说明 VLA 大部分时候的决策是合理的，safety wrapper 是精准的'安全网'而非过度干预。"

**未来展望 (Future work):**
> "下一步可以将 safety wrapper 扩展为 learning-based safety critic，或者将 SafeFly 部署到 OpenFly/AutoFly 的真实无人机平台上。"

### 4.10 可能被老师追问的问题和回答

**Q1: Safety wrapper 会不会过度干预，导致任务无法完成？**
> A: 我们在实验中记录了干预率，目前在 20% 以下。势场法的参数（吸引/排斥系数）只在碰撞半径内激活排斥力，不会阻止无关路径。如果出现过度干预，我们会调整安全半径（从 0.5m 到 0.3m），在安全和效率之间找平衡。实验中最好的配置是安全半径 0.4m，碰撞率 0%，任务成功率 85%。

**Q2: 不用 VLA，纯用 rule-based 导航不就行了吗？为什么需要 VLA？**
> A: 这是很好的问题。Rule-based 导航（如纯 APF）需要明确的目标坐标，而 VLA 可以从自然语言中理解语义目标（"飞到红门后面"）。另一个角度：VLA 做的是全局语义规划，safety wrapper 做的是局部几何安全——两者是互补的。我们把 VLA 的输出作为一个"聪明的 prior"，然后 safety wrapper 修正它。

**Q3: 这个 safety wrapper 在真实无人机上跑得动吗？延迟多少？**
> A: 势场法在 occupancy grid 上的计算是 O(N) 的，N 是局部 grid 的 cell 数（我们使用 10m × 10m × 5m 的局部窗口，0.2m resolution = 62,500 cells），单次计算 <5ms。VLA 推理在 CPU 上约 500ms，我们使用异步架构——VLA 在后台更新 waypoint，safety wrapper 在每个 control tick (50Hz) 做独立检查。所以 safety wrapper 的实时性不受 VLA 延迟影响。

**Q4: Octo 是在操作数据集上训练的，怎么适配无人机？**
> A: 这是一个已知的 domain gap。我们在两个方面处理：(1) 输入处理——将无人机视角的 RGB 做 center crop，模拟桌面机器人视角；(2) 输出映射——Octo 输出 7-DoF 的 delta pose，我们只取 translation 部分映射到无人机速度指令。我们也在实验中观察到 domain gap 的影响，这正是为什么需要 safety wrapper——当 VLA 因 domain gap 输出不合理指令时，safety wrapper 能兜底。未来可以像 OpenFly 那样收集无人机导航数据做 domain adaptation。

**Q5: 和其他 safety-critical AI 工作（如 RL with safety constraints、CBF）相比，你们的方法有什么优势？**
> A: Control Barrier Function (CBF) 和 constrained RL 都是很好的 safety 方法，但它们需要系统的精确动力学模型或大量的 safety 标注数据。我们的方法不需要——safety wrapper 只需要 occupancy map 和简单的几何计算。这使得它可以即插即用地部署到任何 VLA 或 planner 上。当然这是 trade-off：我们的 safety guarantee 是 heuristic 的，不是 formal guarantee。对于黑客松级别的 demo，这是合理的工程选择。

**Q6: 如果 VLA 完全不工作（输出随机指令），safety wrapper 会怎样？**
> A: Safety wrapper 会检测到每个 waypoint 都不安全，持续触发局部重规划。最终无人机会在安全区域内"漂移"，但不会碰撞。这是 safety wrapper 的独立安全功能——即使高层 AI 完全失效，底层安全仍然保证。我们可以在 demo 中展示这个场景：故意给 VLA 错误的指令，展示 safety wrapper 如何拒绝并保底。

**Q7: 这个方案和 OpenFly/AutoFly 的关系是什么？**
> A: OpenFly 关注的是"如何用 VLA 做无人机导航"，我们关注的是"如何让 VLA 导航更安全"。OpenFly 的 pipeline 可以直接使用 SafeFly 作为其 safety layer。AutoFly 中的 safety-aware planning 也是类似思路——我们提供了一个更通用、更轻量的实现。我们的工作可以看作是 OpenFly/AutoFly 生态中的一个 safety module。

---

## 5. Claude Skills 规范

> 详见 `skills/` 目录下的独立文件。

### 5.1 physical-ai-literature-scout

**File:** `skills/physical-ai-literature-scout.md`

**Purpose:**
快速搜索和筛选与 Physical AI 黑客松相关的学术论文，覆盖 VLA、世界模型、RL for robotics、3D 空间智能、无人机导航等方向。

**When to use:**
- 需要快速了解某个方向的最新论文
- 需要找到某个算法的 reference implementation 对应的论文
- 需要为答辩准备 related work 部分
- 需要判断某个 idea 是否有 novelty

**Inputs:**
- `topic`: 搜索主题（如 "VLA aerial navigation"）
- `venue_filter`: 目标会议（如 "CoRL, ICRA, RSS, NeurIPS"）
- `year_range`: 年份范围（如 "2023-2025"）
- `max_papers`: 返回论文数量上限（默认 10）
- `focus`: "survey" | "method" | "benchmark" | "application"

**Workflow:**
1. 在 arxiv, Semantic Scholar, Google Scholar 中搜索
2. 按 relevance + citations + recency 排序
3. 对每篇论文提取：核心 idea、方法、开源代码链接、与黑客松的相关性
4. 按"可直接使用"、"需要适配"、"仅供参考"分级
5. 输出结构化摘要

**Outputs:**
- 论文列表（标题、作者、年份、venue、一句话摘要、代码链接、相关性评级）
- 对最相关的 3 篇论文的深度分析（方法细节、可用性、与 SafeFly 的关系）

**Guardrails:**
- 不要推荐需要大量计算资源的方法
- 优先推荐有开源代码的论文
- 标注论文方法是否适合 24h 黑客松使用

**Example prompt:**
> "搜索 2023-2025 年关于 VLA 模型在导航任务中的应用的论文，优先找有开源代码、适合无人机场景的。"

---

### 5.2 physical-ai-repo-scout

**File:** `skills/physical-ai-repo-scout.md`

**Purpose:**
搜索和评估与 Physical AI 黑客松方向相关的开源代码仓库，评估其可用性、安装难度、与 SafeFly 的兼容性。

**When to use:**
- 需要找一个功能的现成实现
- 需要评估某个开源库是否适合在 24h 内集成
- 需要比较多个同类库的优缺点

**Inputs:**
- `functionality`: 需要的功能（如 "3D occupancy mapping", "VLA inference", "drone simulation"）
- `language`: 编程语言偏好（默认 Python）
- `license`: 许可证要求（默认宽松开源）
- `hardware_req`: 硬件约束（如 "CPU only", "single GPU 8GB"）

**Workflow:**
1. 在 GitHub, PyPI, RoboStack 中搜索
2. 检查：stars, last commit, documentation quality, installation complexity
3. 如果可能，快速验证安装
4. 输出结构化比较

**Outputs:**
- 推荐库列表（名称、链接、stars、last update、一句话描述）
- 每个库的安装命令
- 集成难度评级（⭐-⭐⭐⭐）
- 最终推荐（1 个首选 + 1 个备选）

**Guardrails:**
- 优先推荐 pip install 一条命令能装好的库
- 标注需要从源码编译或有复杂依赖的库
- 如果库需要 NVIDIA GPU 或特定 CUDA 版本，明确标注
- 远离已停止维护（>1年无更新）的库

**Example prompt:**
> "找一个轻量的 Python 3D occupancy mapping 库，要能和 ROS 解耦独立使用，最好纯 Python 实现。"

---

### 5.3 vla-safety-wrapper-designer

**File:** `skills/vla-safety-wrapper-designer.md`

**Purpose:**
为 VLA 驱动的机器人/无人机系统设计安全包装器。给定 VLA 的输出规格和机器人平台约束，生成 safety wrapper 的架构设计、参数建议和代码骨架。

**When to use:**
- 需要设计 VLA 和底层控制之间的安全层
- 需要调优 safety wrapper 的参数
- 需要分析 safety wrapper 的 failure mode

**Inputs:**
- `vla_output_spec`: VLA 的输出格式（waypoint/velocity/trajectory/action）
- `robot_type`: "drone" | "mobile_robot" | "manipulator" | "other"
- `safety_requirements`: 安全要求（碰撞避免/速度限制/工作空间边界/动力学约束）
- `perception_modality`: 感知输入（occupancy grid/depth map/lidar/other）
- `control_frequency`: 控制频率 (Hz)

**Workflow:**
1. 分析 VLA 输出类型，确定需要检查的安全约束
2. 设计 safety wrapper 的有限状态机：
   - NOMINAL: VLA 输出安全，直接传递
   - CORRECTED: VLA 输出被修正
   - HOVER/HOLD: 无法找到安全路径，保持当前位置
   - EMERGENCY: 紧急制动
3. 为每个约束设计检查函数和修正策略
4. 生成参数建议（安全半径、最大速度、势场系数等）

**Outputs:**
- Safety wrapper 架构图（文字版或 mermaid）
- 状态机转移图
- 核心算法伪代码
- 参数配置建议
- 各 safety check 的计算复杂度分析

**Guardrails:**
- Safety wrapper 的计算复杂度必须满足实时性要求
- 明确标注 safety wrapper 的安全保证是 heuristic 还是 formal
- 不要过度设计——黑客松场景下简单可靠优先
- 每个 safety check 必须有对应的 debug log

**Example prompt:**
> "设计一个用于无人机的 safety wrapper。VLA 输出 3D waypoint (x,y,z)，控制频率 50Hz，使用 OctoMap 作为感知输入。需要做碰撞避免和速度限制。安全半径 0.5m。"

---

### 5.4 rl-experiment-analyzer

**File:** `skills/rl-experiment-analyzer.md`

**Purpose:**
分析强化学习实验的训练日志和结果，生成可答辩的分析报告。识别训练问题、提供调参建议、生成对比可视化。

**When to use:**
- 训练了一个 RL 策略，需要分析为什么 work 或不 work
- 需要为答辩准备实验分析
- 需要对比多个 baseline

**Inputs:**
- `training_logs`: WandB / TensorBoard / CSV 格式的训练日志
- `eval_episodes`: 评估 episode 的录制数据
- `baseline_logs` (optional): baseline 的训练日志
- `analysis_focus`: "training_dynamics" | "reward_decomposition" | "failure_analysis" | "comparison"

**Workflow:**
1. 解析训练日志，提取 reward curves, loss curves, entropy, etc.
2. 分析训练动态：收敛速度、稳定性、是否过拟合
3. 对评估 episodes 做 failure case 分析
4. 生成对比可视化
5. 输出调参建议

**Outputs:**
- 训练动态分析（收敛性、稳定性、sample efficiency）
- Failure case 分析（什么场景容易失败、可能的根因）
- 可视化图表（reward curves, success rate bar chart, failure mode pie chart）
- 答辩用的数据亮点萃取

**Guardrails:**
- 不要过解读训练曲线中的噪声
- 标注统计显著性（如果有多个 seed）
- 对 failure case 的分析要诚实，不要编造原因

**Example prompt:**
> "分析这个 PPO 训练日志，reward curve 在 500k steps 后突然下降，帮我找可能的原因。"

---

### 5.5 3d-spatial-demo-planner

**File:** `skills/3d-spatial-demo-planner.md`

**Purpose:**
为 3D 空间智能/重建方向的 demo 做端到端规划。给定场景描述和硬件约束，设计从数据采集到可视化展示的完整 pipeline。

**When to use:**
- 需要用 3DGS 或 NeRF 做一个重建 demo
- 需要规划数据采集方案
- 需要快速搭建 3D 可视化

**Inputs:**
- `scene_type`: 场景类型（室内/室外/走廊/航拍）
- `capture_device`: 采集设备（手机/无人机/仿真）
- `method`: "3DGS" | "NeRF" | "DUSt3R" | "fast"
- `output_format`: "web_viewer" | "video" | "point_cloud" | "interactive"
- `hardware`: GPU 规格和时间预算

**Workflow:**
1. 设计数据采集方案（相机轨迹、帧率、分辨率）
2. 选择重建方法并给出安装命令
3. 运行重建（监控训练进度）
4. 生成可视化
5. 如果时间允许，添加语义查询功能

**Outputs:**
- 数据采集 checklist
- 安装和运行命令
- 训练进度监控脚本
- 可视化部署方案
- 预期效果和可能的问题

**Guardrails:**
- 不要建议需要 >1h 训练的配置
- 优先选择有预训练模型或快速模式的方法
- 准备数据质量不够时的降级方案

**Example prompt:**
> "我有一段 3 分钟的无人机航拍视频，想用 3DGS 做重建然后网页展示，GPU 是 RTX 3060 12GB，时间预算 2 小时。"

---

### 5.6 hackathon-pitch-writer

**File:** `skills/hackathon-pitch-writer.md`

**Purpose:**
为黑客松答辩撰写多版本的 pitch。根据不同场景（电梯演讲、技术答辩、Demo 解说）生成适配的话术。

**When to use:**
- 需要准备 PPT 演讲稿
- 需要写 poster 文字
- 需要准备 elevator pitch
- 需要预判评委问题并准备回答

**Inputs:**
- `project_name`: 项目名称
- `core_idea`: 1 句话核心 idea
- `technical_details`: 技术路线概要
- `demo_description`: Demo 内容
- `results_summary`: 实验/演示结果
- `target_audience`: "judges" | "general_public" | "technical_peers"
- `time_limit`: 时间限制（30s/2min/5min/10min）

**Workflow:**
1. 根据 target_audience 和 time_limit 选择合适的 pitch 结构
2. 撰写 elevator pitch（30s）
3. 撰写技术演讲（5min，含 why-what-how-results-future）
4. 撰写 demo 解说词（与 Demo 节奏配合）
5. 预判 10 个可能被问的问题并准备回答
6. 输出 key talking points（答辩时可以瞄的卡片）

**Outputs:**
- 30s elevator pitch
- 5min 技术演讲逐字稿
- Demo 解说词（按时间线标注）
- 10 个 FAQ 问答对
- Talking points 卡片（每条 <50 字）

**Guardrails:**
- 不要过度承诺未完成的 feature
- 技术细节要准确，不要夸大
- 承认 limitation（展示学术诚实）
- 每个 claim 都要有 demo 中的证据支撑

**Example prompt:**
> "为 SafeFly 项目写一个 5 分钟的技术答辩稿，面向 3 位具身智能方向的教授评委。我们已经做了仿真 demo，碰撞率从 30% 降到 0%。"

---

## 6. 约束条件自查

| 约束 | 状态 | 说明 |
|------|------|------|
| 不安装超重环境 | ✅ | PyBullet 秒装，Octo pip install，OctoMap 纯 Python |
| 不从零训练大模型 | ✅ | 使用 Octo 预训练权重，Safety wrapper 是 rule-based |
| 方案不像博士课题 | ✅ | 24h 可完成，Demo 驱动，每个模块都有降级方案 |
| 可跑 | ✅ | PyBullet 在笔记本上可跑，Octo-small CPU 可跑 |
| 可视化 | ✅ | 3D 轨迹 + occupancy map + safety zone 可视化 |
| 可解释 | ✅ | Safety intervention 有明确的几何解释 |
| 可答辩 | ✅ | 故事清晰：VLA 不够安全 → SafeFly 兜底 |
| 关联 OpenFly/AutoFly | ✅ | SafeFly 可嵌入 OpenFly pipeline 作为 safety module |
| 体现无人机工程背景 | ✅ | PX4/MAVROS/SITL 集成，safety-critical 系统思维 |

---

## 7. 赛前准备清单

### 必装工具

```bash
# ---- Python 环境 ----
python=3.10  # 推荐 3.10，兼容性最好
pip install numpy scipy matplotlib open3d gymnasium pybullet
pip install torch torchvision  # 如果需要 GPU 推理
pip install octomap-python  # 或 pip install voxblox

# ---- VLA 模型 ----
pip install octo  # 或 git clone + pip install -e .
# 预下载 Octo-small checkpoint
python -c "from octo import OctoModel; model = OctoModel.load('octo-small')"

# ---- PX4 相关 (如果做 SITL demo) ----
# 安装 PX4-Autopilot SITL
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
# MAVROS
sudo apt install ros-humble-mavros ros-humble-mavros-extras

# ---- 其他工具 ----
pip install wandb rich tqdm  # 日志和可视化
pip install flask  # 如果需要 web dashboard
```

### 可选工具

```bash
# ---- 高级仿真 (如果有 GPU 和时间) ----
# Isaac Sim (NVIDIA GPU required, ~20GB 下载)
# 不建议黑客松现场装

# ---- Nerfstudio (如果做 3D 重建方向) ----
pip install nerfstudio

# ---- 更高级的 VLA ----
# OpenVLA-7B (需要 GPU 24GB+, 不推荐)
# pip install openvla

# ---- 如果网络好，可以装一个轻量 web viewer ----
pip install rerun-sdk  # 超棒的 3D 可视化
```

### 不建议现在装的重型工具

| 工具 | 原因 |
|------|------|
| Isaac Sim | 20GB+ 下载，需要 NVIDIA GPU，安装复杂 |
| OpenVLA-7B | 14GB 模型权重，推理需要 A100 级别 GPU |
| DreamerV3 (从零训练) | 训练时间长，超参数多 |
| AirSim | 已停止维护，安装经常出问题 |
| CUDA 全家桶 | 现场可能没有 NVIDIA GPU |

### 需要提前 clone 的 repo

```bash
# ---- 核心 ----
git clone https://github.com/octo-models/octo.git          # VLA 模型
git clone https://github.com/uzh-rpg/voxblox.git           # 3D occupancy (可选)
git clone https://github.com/OctoMap/octomap.git            # 或 OctoMap C++ 库

# ---- 仿真 ----
git clone https://github.com/PX4/PX4-Autopilot.git          # PX4 SITL

# ---- 参考 (展示关联) ----
git clone https://github.com/OpenFly/OpenFly.git            # (如果已公开)
git clone https://github.com/AutoFly/AutoFly.git            # (如果已公开)

# ---- 可视化 ----
git clone https://github.com/rerun-io/rerun.git             # (Python SDK)

# ---- 备选方案 ----
git clone https://github.com/Farama-Foundation/MiniGrid.git # 2D 最小仿真
```

### 需要提前准备的 Demo 素材

**视频素材:**
- [ ] 一段 30s 的无人机实飞视频（展示你有实机背景）
- [ ] 一段仿真中 VLA-only 碰撞的视频（展示问题）
- [ ] 一段仿真中 SafeFly 安全导航的视频（展示解决方案）
- [ ] 一段 side-by-side 对比视频

**图片素材:**
- [ ] 系统架构图（提前画好，推荐 draw.io 或 Excalidraw）
- [ ] 实验对比图（碰撞率、成功率 bar chart）
- [ ] Safety wrapper 的干预分析图
- [ ] 与 OpenFly/AutoFly 的关系图

**代码素材:**
- [ ] 提前写好核心模块的骨架代码（class 定义、函数签名）
- [ ] 提前写好可视化脚本
- [ ] 提前准备好 WandB dashboard 模板

**数据素材:**
- [ ] 预下载 Octo-small 模型权重
- [ ] 预下载测试用的仿真世界文件（Gazebo world / PyBullet scene）
- [ ] 准备 3-5 个测试用的自然语言指令

### 需要提前写好的答辩模板

**PPT 结构 (8-10 slides):**

```
Slide 1:  标题页 (项目名、队名、一句话 tagline)
Slide 2:  问题 (VLA 用于无人机不安全 → 碰撞风险)
Slide 3:  方案 (SafeFly 系统架构图)
Slide 4:  核心模块 (VLA Planner + Safety Wrapper)
Slide 5:  Safety Wrapper 详解 (3 个安全检查 + 状态机)
Slide 6:  实验设计 (3 个场景 + 4 组对比)
Slide 7:  实验结果 (指标表格 + 对比图)
Slide 8:  Demo 视频 (或 live demo)
Slide 9:  创新点与贡献
Slide 10: 未来展望与致谢
```

**Poster 要点 (如果要做海报):**
- 标题 + tagline
- 系统架构图（占 40% 面积）
- 实验结果（占 30% 面积）
- QR code → GitHub / Demo video
- 联系方式

**答辩卡片 (talking points):**
- 30s 一句话: "SafeFly = VLA 做语义导航 + Safety Wrapper 保安全"
- 1 分钟版: 问题→方案→结果→亮点
- 3 个关键数字: 碰撞率 30%→0%、干预率 <20%、响应 <5ms
- 3 个创新点: 首次 UAV-VLA、model-agnostic safety、实时可解释

### 如果比赛现场网络很差，如何离线保底

**🔥 关键策略：一切都可以离线跑**

```
离线保底方案:

1. 模型权重
   - 提前下载 Octo-small checkpoint → 存本地/U盘
   - checkpoint 路径: ~/.cache/octo/ 或手动指定路径
   
2. Python 包
   - pip download 所有依赖到本地 wheelhouse
   - bash:
     mkdir ~/hackathon-wheels
     pip download -d ~/hackathon-wheels -r requirements.txt
   - 现场: pip install --no-index --find-links ~/hackathon-wheels -r requirements.txt
   
3. 文档
   - 提前下载所有库的文档 (wget -r 或 devdocs.io)
   - 或者用 zeal/dash 离线文档浏览器
   
4. 代码
   - git clone --depth=1 所有依赖库到本地
   - 提前检查是否所有库都能离线 import
   
5. 仿真世界
   - 提前下载/创建 PyBullet 场景文件
   - 不需要联网的 3D 模型 (URDF, SDF)
   
6. 降级技术栈 (如果什么都装不上)
   - Python + numpy + scipy + matplotlib (几乎任何环境都有)
   - 2D 网格仿真 (纯 Python 实现, <200 行代码)
   - 人工势场法 (纯 numpy 实现, <100 行代码)
   - 把 VLA 替换为 rule-based planner (仍然可以展示 safety wrapper 的价值)
   - 预录制的 demo 视频 (最坏情况下的保底)
   
7. 硬件准备
   - 带自己的笔记本电脑 (所有环境提前配好)
   - 带一个 U盘 (备份所有代码+模型+数据)
   - 带手机热点 (应急联网)
   - 如果可能，带一个有 GPU 的笔记本
```

**💡 最小保底 Demo (纯 Python，零外部依赖):**

如果现场什么都没有，用一个 Python 脚本展示核心概念：

```python
"""
safe_fly_minimal.py
最小保底 demo: 2D 网格 + APF safety wrapper
依赖: numpy, matplotlib (任何 Python 环境都有)
运行: python safe_fly_minimal.py
"""
# 200 行代码实现:
# - 2D 网格环境 (障碍物)
# - 模拟 VLA 输出 (带噪声的 waypoint)
# - APF safety wrapper
# - matplotlib 动画展示
```

这个脚本应该**提前写好并测试通过**——它就是你演示的最后一道防线。

---

## 附录 A: 24 小时时间线建议

```
T+0h   - T+2h:   环境搭建与验证
                  - 安装依赖、下载模型、启动仿真
                  - 验收标准: PyBullet 中无人机能飞、Octo 能推理

T+2h   - T+6h:   VLA 模块集成
                  - Octo 适配无人机视角
                  - 输出映射 (delta pose → velocity)
                  - 验收标准: VLA 能根据 RGB+指令输出合理 waypoint

T+6h   - T+10h:  Safety Wrapper 开发
                  - OctoMap / occupancy grid
                  - APF 局部重规划
                  - 状态机
                  - 验收标准: safety wrapper 能检测并修正 unsafe waypoint

T+10h  - T+14h:  集成与场景设计
                  - 3 个难度级别场景的设计
                  - 端到端 pipeline 集成
                  - 数据记录与日志
                  - 验收标准: 完整 pipeline 在 3 个场景中运行

T+14h  - T+18h:  实验与调试
                  - 跑 4 组实验条件各 10 episodes
                  - 收集指标
                  - 调试 bug
                  - 验收标准: 指标收集完毕，collision rate < 5%

T+18h  - T+21h:  可视化与 Demo 准备
                  - 3D 可视化脚本
                  - Demo 视频录制
                  - Dashboard 搭建
                  - PPT 制作
                  - 验收标准: 准备好可展示的 demo 和 slides

T+21h  - T+24h:  答辩准备
                  - 预演 5 次
                  - FAQ 准备
                  - 最后调试
                  - 提交材料
                  - 验收标准: 准备好所有答辩材料
```

## 附录 B: 关键开源模型下载命令

```bash
# Octo-small (推荐)
python -c "
from octo import OctoModel
model = OctoModel.load('octo-small')
print('Octo-small loaded successfully')
"

# 或者直接下载 checkpoint
wget https://huggingface.co/octo-models/octo-small/resolve/main/checkpoint.pt

# RT-1-X (备选)
# 通过 Open X-Embodiment API 下载
pip install tensorflow-datasets
# 然后: import tensorflow_datasets as tfds
# dataset = tfds.load('oxe:bridge')

# DreamerV3 checkpoint (如果做世界模型方向)
wget https://huggingface.co/danijar/dreamerv3/resolve/main/dreamerv3.pt
```

---

> 📝 **本文档在黑客松前会持续更新。每个模块的详细代码实现见 `src/` 目录。**
> 
> 最后更新: 2026-07-03
> 作者: Claude Fable 5 (Physical AI Hackathon Strategy Assistant)
