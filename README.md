# Video2Motion2Action: Dexterous Manipulation Skill Transfer

> **Track 1: Physical Intelligence | Topic T-1 | Team Challenge**
> Shanghai AI Lab Hackathon — July 2026

## Overview

From human manipulation demonstration videos, reconstruct hand motion, object shape, and hand-object interaction trajectories, then transfer to Sharpa dexterous hand tracking in IsaacGym simulation.

**Pipeline**: Video → Motion (hand + object reconstruction) → Action (Sharpa tracking in physics simulation)

## Team

| Member | Role | Key Deliverables |
|--------|------|-----------------|
| **A** | Sharpa Tracking + Tracker Bonus | Training pipeline, checkpoint, rollout, scoring |
| **B** | Hand Reconstruction + Hand Bonus | 2D mask, 3D MANO/mesh, trajectory, retargeting |
| **C** | Object Reconstruction + Object Bonus | 3D model, IsaacGym asset, pose trajectory |

## Scoring (100 points total)

| Module | Points | Owner |
|--------|--------|-------|
| 3.1 Sharpa Tracking | 25 | A |
| 3.2 Hand Reconstruction & Trajectory | 25 | B |
| 3.3 Object Reconstruction & IsaacGym Asset | 25 | C |
| 3.4 Comprehensive Task | 25 | All |

Bonus items: Tracker Bonus, Hand Bonus, Object Bonus (on top of 100 points).

## Quick Start

### Environment

```bash
# Check environment
python scripts/check_isaac_env.py

# For IsaacGym-based tasks, use the prepared venv
source .venvs/isaacgym-py38/bin/activate
```

### Data

Download HO-Tracker Challenge data from HuggingFace:
```bash
# From HF (may need proxy)
# https://huggingface.co/datasets/kelvin34501/HO-Tracker-Challenge

# Or from HF Mirror (China)
# https://hf-mirror.com/datasets/kelvin34501/HO-Tracker-Challenge
```

### References

- [HO-Tracker Baseline Repo](https://github.com/kelvin34501/HO-Tracker-Baseline-Challenge)
- [HO-Tracker Challenge Data](https://huggingface.co/datasets/kelvin34501/HO-Tracker-Challenge)
- IsaacGym Preview 4 (provided on cluster at `~/opt/isaac_gym/isaacgym`)
- [Cluster Usage Guide](https://aicarrier.feishu.cn/wiki/LNRiwNLRviogOfk30eHcv2Q9ngd)

## Critical Rules (ZERO-TOLERANCE)

1. **Do NOT modify** `eval_score.py` or physics parameters (friction, gravity, etc.)
2. **Do NOT** hardcode evaluation outputs or fabricate trajectories
3. **Do NOT** submit unchanged Inspire baseline only
4. Checkpoints must follow naming convention, saved under `runs/`
5. All reconstruction results must include visualization evidence
6. Object assets must be loadable in IsaacGym
7. All external resources must be cited in the report

## Deliverables

See [AGENTS.md](AGENTS.md) for the full per-member deliverable checklist.

### Key deliverables per member:
- **Code**: All code changes, scripts, commands (GitHub private repo)
- **README**: Environment dependencies, reproduction steps, main result screenshots
- **Report**: Method description, external resources, member contributions, failure analysis

## Project Structure

```text
AGENTS.md              # Full rules, roles, deliverables checklist
README.md              # This file
src/
  object_recon/        # Member C: object reconstruction
  hand_recon/          # Member B: hand reconstruction
  sharpa_tracking/     # Member A: Sharpa tracking
experiments/           # Timestamped experiment logs
scripts/               # Utility scripts
demos/                 # Final demos & visualizations
runs/                  # Sharpa checkpoints (required path!)
docs/                  # Additional documentation
```

## License

Private repository during competition. Access provided after submission.
