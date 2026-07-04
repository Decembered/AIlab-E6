# physical-ai-literature-scout

## Purpose
快速搜索和筛选与 Physical AI 黑客松相关的学术论文，覆盖 VLA、世界模型、RL for robotics、3D 空间智能、无人机导航等方向。帮助参赛者在有限时间内快速建立 related work 知识体系，找到可复用的开源方法。

## When to use
- 需要快速了解某个方向的最新论文
- 需要找到某个算法的 reference implementation 对应的论文
- 需要为答辩准备 related work 部分
- 需要判断某个 idea 在学术上是否有 novelty
- 需要确认某个技术路线是否有论文支撑

## Inputs
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `topic` | string | ✅ | 搜索主题，如 "VLA aerial navigation", "world model planning drone" |
| `venue_filter` | string[] | ❌ | 目标会议/期刊，如 ["CoRL", "ICRA", "RSS", "NeurIPS", "IROS"] |
| `year_range` | string | ❌ | 年份范围，如 "2023-2025"，默认 "2023-2026" |
| `max_papers` | int | ❌ | 返回论文数量上限，默认 10 |
| `focus` | enum | ❌ | "survey"（综述）、"method"（方法）、"benchmark"（基准）、"application"（应用） |
| `hardware_budget` | string | ❌ | 硬件预算约束，如 "CPU only", "single GPU 8GB"，用于过滤计算密集型方法 |

## Workflow
1. **搜索**: 在 arxiv API、Semantic Scholar API、Papers With Code 中搜索
2. **排序**: 按 relevance × citations × recency 综合排序
3. **提取**: 对每篇论文提取结构化信息
4. **分级**: 按实用性分为 3 级
   - 🟢 L1: 有开源代码，可直接在 24h 黑客松中使用
   - 🟡 L2: 有参考价值，但需要适配
   - 🔴 L3: 仅作学术参考，不适合直接使用
5. **深度分析**: 对最相关的 3 篇给出详细分析

## Outputs
```yaml
papers:
  - title: "Paper Title"
    authors: "First Author et al."
    year: 2024
    venue: "CoRL 2024"
    one_liner: "一句话描述核心贡献"
    code_url: "https://github.com/..."  # 或 null
    relevance_score: 8.5  # 1-10
    usability: "L1"  # L1/L2/L3
    hardware_requirement: "GPU 8GB"
    key_insight: "对这个黑客松有什么启发"

top3_deep_dive:
  - paper: "最相关论文"
    method_summary: "方法简述"
    architecture: "文字架构图"
    code_availability: "有/无/部分"
    integration_difficulty: "easy/medium/hard"
    relation_to_safefly: "与 SafeFly 的关系"

recommendation: "总结建议：最值得参考的 2-3 个方法，以及为什么"
```

## Guardrails
- ❌ 不要推荐需要 >100 GPU-hours 训练的方法
- ❌ 不要推荐没有开源代码且方法描述不清晰的工作
- ✅ 优先推荐有 pip install 或 Docker 镜像的方法
- ✅ 标注论文方法是否适合 24h 黑客松使用
- ✅ 如果某个重要论文没有代码，标注"需要从零实现，不建议黑客松尝试"

## Example prompt
> "搜索 2023-2025 年关于 VLA 模型在导航任务（非桌面操作）中的应用的论文。优先找有开源代码、适合无人机场景的。硬件预算是单 GPU 8GB。重点关注：1) VLA 如何输出 waypoint，2) 如何处理 safety，3) 有什么仿真环境。"

---

## Example prompt (简短版)
> "快速找 5 篇最相关的 VLA for drone navigation 论文，标注哪些有代码。"
