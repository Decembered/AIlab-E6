# hackathon-pitch-writer

## Purpose
为 Physical AI 黑客松答辩撰写多版本、多场景的 pitch 材料。根据不同场景（电梯演讲、技术答辩、Demo 解说、评委问答）生成适配的话术。核心原则：**每一个 claim 都要有 demo 中的证据支撑**。

## When to use
- 需要准备 PPT 演讲稿
- 需要写 poster 文字
- 需要准备 30s elevator pitch
- 需要预判评委可能问的问题并准备回答（FAQ）
- 需要为 demo 展示写解说词
- 需要从技术描述中提炼"通俗版"解释

## Inputs
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `project_name` | string | ✅ | 项目名称，如 "SafeFly" |
| `core_idea` | string | ✅ | 1 句话核心 idea |
| `technical_details` | string | ✅ | 技术路线描述（bullet points 即可） |
| `demo_description` | string | ✅ | Demo 内容和展示形式 |
| `results_summary` | dict | ✅ | 关键实验结果 |
| `target_audience` | enum | ✅ | "judges"（评委）/ "general_public"（公众）/ "technical_peers"（技术同行） |
| `time_limit` | string | ❌ | 时间限制: "30s" / "2min" / "5min" / "10min" / "custom" |
| `known_limitations` | string[] | ❌ | 已知的 limitation（诚实展示学术素养） |
| `competitor_awareness` | string[] | ❌ | 已知的竞品/类似工作（展示领域认知） |

**`results_summary` 规格:**
```yaml
results_summary:
  headline_metric: "碰撞率从 30% 降至 0%"
  supporting_metrics:
    - metric: "collision_rate"
      value: 0.00
      baseline: 0.30
    - metric: "task_success_rate"
      value: 0.85
      baseline: 0.80
    - metric: "intervention_rate"
      value: 0.18
      baseline: null
  demo_scenes: 3
  comparison_conditions: 4
```

## Workflow
1. **确定 pitch 结构**: 根据 time_limit 选择模板
   - 30s: Problem → Solution → Result (一句 each)
   - 2min: Why → What → How → Demo → Impact
   - 5min: 完整技术演讲
   - 10min: 技术演讲 + Live demo + FAQ
2. **撰写各版本 pitch**:
   - 根据 target_audience 调整术语密度
   - Judge 版: 技术深度 + novelty 突出
   - Public 版: 类比 + 可视化 + "为什么重要"
   - Peer 版: 技术细节 + 诚实讨论 trade-off
3. **撰写 demo 解说词**: 与 demo 节奏同步，标注时间点
4. **生成 FAQ**: 预判 10 个最可能被问的问题
5. **生成 talking points 卡片**: 现场答辩时可以瞄的速记卡

## Outputs
```yaml
elevator_pitch_30s: |
  "SafeFly 解决一个简单但关键的问题：VLA 大模型用于无人机导航时，模型的输出不一定物理安全。
  我们在 VLA 和飞控之间插入了一个轻量级安全包装器，用人工势场做实时碰撞检测和局部重规划。
  在 3 个仿真场景中，碰撞率从 30% 降到了 0%，而安全干预只发生在不到 20% 的步骤中。
  简单说——VLA 告诉你'飞过去'，SafeFly 确保你不会撞墙。"

technical_pitch_5min:
  sections:
    - header: "开场 — Why SafeFly? (30s)"
      script: |
        "大家好。目前的 VLA 研究几乎全部集中在桌面操作——机械臂抓个杯子、开个抽屉。
        但无人机有三点本质不同：第一，碰撞后果严重，不能试错；第二，3D 空间的复杂性远超桌面；
        第三，动力学约束更严格。直接把 VLA 部署到无人机上，是不安全的。
        所以我们问了一个问题：能不能让 VLA 做它擅长的语义理解，
        但在它犯错的时候——在它输出'撞墙指令'的时候——有一个安全网兜底？这就是 SafeFly。"
    
    - header: "方案 — 三层架构 (1min)"
      script: |
        "SafeFly 是一个三层架构。
        顶层是感知：深度相机实时构建 3D 占据地图。
        中层是 VLA：我们使用 Octo 预训练模型，输入 RGB 图像和自然语言指令，输出 3D waypoint。
        关键是底层——Safety Wrapper。它做三件事：
        第一，碰撞检查——把 VLA 的 waypoint 投影到占据地图上，如果撞了，触发修正。
        第二，人工势场局部重规划——目标吸引、障碍物排斥，5 毫秒内生成安全路径。
        第三，安全仲裁——如果连势场都找不到安全路径，无人机悬停保底。
        重要的是，Safety Wrapper 是 model-agnostic 的——不修改 VLA，而是作为即插即用的安全层。"
    
    - header: "实验 — 3 场景 × 4 条件 (1.5min)"
      script: |
        "我们在 PyBullet 仿真中设计了 3 个递进难度的场景，从开放空间到室内走廊到动态障碍。
        对比了 4 个条件：无 VLA 无 safety 的随机导航、纯 VLA、纯 safety（规则式导航）、以及 SafeFly。
        每个条件 10 个 episode。
        结果非常清晰：纯 VLA 的碰撞率是 30%，SafeFly 是 0%。
        关键是——安全干预只发生在 18% 的步骤中。
        说明 VLA 在 82% 的时候决策是合理的，safety wrapper 是精准的'安全网'，不是过度干预。
        任务成功率也从随机 baseline 的 20% 提升到了 85%。"
    
    - header: "Demo (1min)"
      script: |
        "现在请大家看屏幕。左边是纯 VLA 的飞行——可以看到，在第 3 个拐角，VLA 输出了一个离墙太近的 waypoint，
        无人机直接撞墙。右边是 SafeFly——同一个场景、同一个 VLA、同一个指令。
        注意这里，VLA 同样输出了不安全 waypoint，但 safety wrapper 检测到了碰撞风险，
        用绿色轨迹展示了修正后的安全路径。无人机安全通过。
        底部面板是实时 safety log——每一项干预都被记录下来，完全可追溯。"
    
    - header: "总结与展望 (1min)"
      script: |
        "SafeFly 的贡献是三点：首次将 VLA 适配到无人机导航场景；
        提出 model-agnostic 的安全包装器，不修改 VLA 本身；
        safety wrapper 在 <5ms 内完成，不牺牲实时性。
        未来方向：一是将 safety wrapper 扩展为 learning-based safety critic，
        二是与 OpenFly/AutoFly 的真实无人机平台集成。
        我们相信，在 VLA 进入物理世界的过程中，safety wrapper 是不可或缺的基础设施。
        谢谢，欢迎提问。"

demo_narration:
  timeline:
    - time: "0:00-0:15"
      action: "展示系统架构图 + 介绍"
      narration: "这是 SafeFly 的三层架构..."
    - time: "0:15-0:45"
      action: "VLA-only 飞行 (会碰撞)"
      narration: "注意看——VLA 输出了一个冲向墙壁的 waypoint..."
    - time: "0:45-1:15"
      action: "SafeFly 飞行 (安全)"
      narration: "同样场景、同样 VLA——但 safety wrapper 在中间修正了路径..."
    - time: "1:15-1:30"
      action: "实时 safety log 面板"
      narration: "每一条安全干预都记录在案：时间、原因、修正后路径..."


faq_questions:
  - q: "Safety wrapper 会不会过度干预，导致任务无法完成？"
    a: |
      "干预率 18% 说明大部分时候 VLA 决策合理。我们在参数上有调整空间：
      安全半径从 0.5m 调到 0.3m 时，干预率降到 10%，碰撞率仍然为 0%。
      安全与效率的平衡点取决于应用场景——送快递可以更激进，载人可以更保守。"
    follow_up_if_needed: "势场参数 K_att 和 K_rep 也可以 scene-specific 调优。"

  - q: "不用 VLA，纯 rule-based 不就行了吗？"
    a: |
      "Rule-based 导航需要明确的目标坐标，这在真实场景中往往不可得。
      VLA 提供的是语义理解——'飞到红门后面'——rule-based 做不到这一点。
      我们的 safety wrapper 也展示了纯 rule-based 的 baseline：任务成功率只有 60%，
      因为它不理解语义目标。SafeFly 把 VLA 的语义理解 + rule-based 的安全保证结合了起来。"

  - q: "Octo 在操作数据集上训练的，怎么适配无人机？"
    a: |
      "这是一个真实的 domain gap，我们诚实面对它。
      我们在两方面处理：输入做 center crop 模拟操作视角，输出只取 translation 映射到无人机速度。
      域差异也是为什么需要 safety wrapper——当 VLA 因为域差异输出不合理指令时，safety wrapper 能兜底。
      未来可以像 OpenFly 那样收集无人机导航数据做 fine-tune，这会进一步缩小 gap。"

  - q: "safety wrapper 能在真实无人机上跑吗？延迟多少？"
    a: |
      "Safety wrapper 的计算复杂度是 O(M×D)，其中 M 是局部占据 grid 中被障碍物占据的 cell 数，
      D=3（维度）。在我们的 10m×10m×5m, 0.2m resolution 设置下，单次计算 <5ms。
      VLA 推理在 CPU 上约 500ms，但两者是异步的——VLA 后台更新 waypoint，
      safety wrapper 在每个 control tick (50Hz) 独立运行。所以实时性不受 VLA 延迟影响。"

  - q: "和 Safety-Critical AI 的方向（CBF, constrained RL）有什么区别？"
    a: |
      "Control Barrier Function 和 constrained RL 都是很好的方法，但它们需要系统的精确动力学模型
      或大量 safety 标注数据。我们的方法不需要——只需要占据地图和简单的几何计算。
      代价是 safety guarantee 是 heuristic 的，不是 formal guarantee。
      对于黑客松级别的 demo，这是合理的工程选择。未来可以用 CBF 替换 APF 获得更强的 safety guarantee。"

  - q: "如果 VLA 完全坏了（输出随机值），safety wrapper 会怎样？"
    a: |
      "Safety wrapper 会检测到每个 waypoint 都不安全，持续触发局部重规划。
      最终无人机会在安全区域内'漂移'，但不会碰撞。
      这正是我们希望展示的——即使高层 AI 完全失效，底层 safety 仍然保底。
      我们可以在 demo 中展示这个极端情况。"

  - q: "这个和 OpenFly/AutoFly 是什么关系？"
    a: |
      "OpenFly 关注'如何用 VLA 做无人机导航'，我们关注'如何让 VLA 导航更安全'。
      OpenFly 的 pipeline 可以直接使用 SafeFly 作为其 safety module。
      AutoFly 中的 safety-aware planning 也是类似思路。
      我们的工作可以看作是 OpenFly/AutoFly 生态中的一个 safety component。"

  - q: "只用了 3 个仿真场景，说服力不够吧？"
    a: |
      "确实，3 个场景只是 proof-of-concept。但我们的设计是递增难度的：
      开放空间 → 结构环境 → 动态场景，覆盖了无人机导航的主要挑战模式。
      24 小时内做到这一点，已经展示了方案的可行性。
      更全面的评估需要更多场景和真实飞行——这是我们下一步的工作。"

talking_points_cards:
  - point: "SafeFly = VLA 语义决策 + Safety Wrapper 安全兜底"
  - point: "碰撞率: 30% → 0% | 干预率: 18% | 延迟: <5ms"
  - point: "Model-agnostic: 不修改 VLA，即插即用"
  - point: "首次 VLA + 无人机导航 (非桌面操作)"
  - point: "3 个贡献: UAV-VLA 适配 / Safety Wrapper / 实时可解释"
  - point: "如果被问 domain gap: 诚实面对 + Safety Wrapper 兜底"
  - point: "如果被问 3 个场景不够: 难度递增 + proof-of-concept"
  - point: "未来: OpenFly 集成 + Real drone deployment"
```

## Guardrails
- ❌ 不要过度承诺未完成的 feature——用 "future work" 而非 "正在做"
- ❌ 不要夸大实验结果——如果只有 3 个 seed，标注 "preliminary results"
- ❌ 不要回避 limitation——诚实讨论展示学术素养
- ✅ 每个 claim 都要有 demo 中的证据支撑（"在 demo 的 0:45 处可以看到..."）
- ✅ 技术细节要准确，不确定的用 "our hypothesis is..." 而非 "we proved..."
- ✅ 预判 FAQ 时要覆盖: 方法局限性、与竞品比较、为什么不做更 fancy 的方案
- ✅ Talking points 卡片每条 <50 字，方便现场快速浏览

## Example prompt (完整版)
> "为 SafeFly 项目写一个 5 分钟的技术答辩稿。背景：面向 3 位具身智能方向的教授评委。核心结果：碰撞率从 30% 降到 0%，安全干预率 18%，3 个仿真场景。Demo 包括 side-by-side 视频对比和实时 safety log。同时准备 10 个 FAQ 和一套 talking points 卡片。"

## Example prompt (简短版)
> "为 SafeFly 写一个 30 秒 elevator pitch，面向非技术背景的评委。"

## Example prompt (迭代版)
> "基于之前写的 5 分钟答辩稿，评委说'技术不够深'，帮我加强 VLA 和 safety wrapper 交互机制的技术细节描述。"

---

## 答辩话术核心原则

1. **Story First, Tech Second**: 先讲为什么这个问题重要，再讲怎么解决
2. **Numbers Tell, Visuals Sell**: 每个关键数字都要有对应的可视化
3. **诚实 > 完美**: 承认 limitation 比掩盖它更能赢得评委尊重
4. **Demo 是最高优先级**: 如果 demo 和 slides 冲突，砍 slides 保 demo
5. **Preempt the FAQ**: 在答辩中主动提到可能的质疑并给出回应
