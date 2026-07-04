# rl-baseline-runner

## Purpose

Run small, reproducible RL or imitation-learning baselines for robotics simulation tasks using ManiSkill, Genesis, Isaac Lab, MuJoCo, or similar environments.

The emphasis is fast baseline evidence, not large-scale training. Prefer PPO, SAC, BC, random policy, scripted policy, or an existing repository baseline.

## When To Use

Use this skill when:

- A simulation task needs a quick baseline.
- The user wants success rate, reward curves, rollout videos, or failure modes.
- A repo provides existing PPO / SAC / BC configs.
- We need a language-conditioned manipulation baseline with minimal training.

## Inputs

- Simulator or benchmark name
- Task name
- Algorithm, default priority: PPO, SAC, BC, random/scripted smoke test
- Seed or seed list
- Time budget
- Hardware constraints
- Existing config path, if any

## Outputs

Each run must create an experiment folder containing:

- `README.md`
- `config.yaml`
- `command.sh`
- `metrics.json`
- `logs.txt`
- `figures/`
- `outputs/`

Required recorded fields:

- Simulator and task version
- Algorithm and policy architecture
- Seed
- Command
- Runtime
- Reward / success rate / episode length
- Curves or scalar metrics
- Failure reason if not successful

## Steps

1. Run environment diagnostics first:

   ```bash
   python scripts/check_physical_ai_env.py
   ```

2. Confirm the simulator can import without starting a long job.
3. Prefer an environment reset and random action rollout as the first smoke test.
4. Find the smallest official baseline config.
5. Reduce training steps, parallel env count, and evaluation episodes for a hackathon-scale run.
6. Save exact config and command before running.
7. Run with `tee` into `logs.txt`.
8. Export metrics to `metrics.json`.
9. Save curves and screenshots under `figures/`.
10. Save videos, policies, or debug output under `outputs/`.

## Constraints

- Do not run long training jobs by default.
- Do not use distributed RL until a single-machine baseline works.
- Do not download large assets or datasets without approval.
- Do not claim policy quality from a tiny smoke test; label it as smoke-test evidence.
- Keep seeds explicit.

## Failure Debugging

- If simulator import fails, use `sim-env-debugger`.
- If CUDA OOM occurs, reduce env count, image resolution, batch size, replay size, and model size.
- If reward is flat, verify action space, observation space, task reset, and reward scale.
- If evaluation fails but training runs, check wrappers and checkpoint format.
- If rendering fails, try headless / EGL / CPU rendering or disable video for metrics-only run.

## Minimum Runnable Demo

The minimum demo is:

- One reset and 10 random or scripted environment steps, or
- A tiny PPO / SAC / BC run with a very small step budget, or
- Evaluation of a provided checkpoint for 1 to 5 episodes

It must save command, config, logs, metrics, and at least one figure or video when rendering is available.

