# physical-ai-repo-scout

## Purpose

Rapidly read an unfamiliar Physical AI, robot learning, VLA, RL, simulation, 3D reconstruction, or spatial intelligence repository and produce a practical launch summary.

The goal is to answer: what does this repo do, how do we install it minimally, where is the demo or evaluation entrypoint, what can run quickly, and what will probably break.

## When To Use

Use this skill when:

- A new GitHub repository or local cloned repository needs evaluation.
- The user asks whether a repo is useful for the Physical AI hackathon.
- A repo has unclear installation, demos, training scripts, or dependencies.
- We need to decide whether to adopt, fork, or ignore a repo.

## Inputs

- Repository path or URL
- Target task, for example VLA inference, RL baseline, 3D scene demo, policy evaluation
- Hardware constraints, especially GPU memory and CUDA availability
- Time budget for smoke test

## Outputs

Create a concise repo scouting report, preferably under `docs/` or an experiment folder:

- Repo summary
- Main claims and likely hackathon relevance
- Install entrypoints
- Demo / evaluation entrypoints
- Training entrypoints, if relevant
- Key dependencies and version risks
- Required assets, checkpoints, or datasets
- Minimal runnable command
- Expected outputs
- Likely failure points
- Recommendation: use now, use later, or skip

## Steps

1. Inspect top-level files with `rg --files`, `ls`, and dependency manifests.
2. Read `README`, docs, examples, and installation files before source files.
3. Identify package manager: pip, conda, uv, poetry, Docker, submodules, custom scripts.
4. Search for entrypoints:

   ```bash
   rg -n "if __name__ == .__main__.|argparse|click.command|hydra.main|train|eval|demo|inference" .
   rg --files | rg "demo|example|train|eval|infer|policy|rollout|notebook"
   ```

5. Identify heavyweight requirements such as datasets, checkpoints, simulator assets, CUDA extensions, Isaac Sim, or ROS.
6. Find the smallest non-training run: inference on one image, one rollout, one scene render, one synthetic example, or one CLI help command.
7. Record exact commands without executing long jobs.
8. If running a smoke test, save outputs and logs under `experiments/`.

## Constraints

- Do not install large dependencies or download checkpoints without approval.
- Do not run training unless it is an explicitly tiny smoke test.
- Do not use `sudo`.
- Do not modify repo code during scouting unless the user asks for a patch.
- Do not trust README claims until matching entrypoints are found.

## Failure Debugging

- If install instructions are missing, inspect `pyproject.toml`, `setup.py`, `requirements*.txt`, `environment*.yml`, and Dockerfiles.
- If imports fail, map missing modules to optional extras or submodules.
- If checkpoints are required, identify size and source before downloading.
- If CUDA extensions fail, record compiler, CUDA, PyTorch, and driver versions.
- If simulator import fails, check whether assets, license, display, or EGL setup is required.

## Minimum Runnable Demo

A valid scout demo is one of:

- `python script.py --help` for the main train/eval/demo script
- One inference on a bundled sample
- One simulator reset and random action step
- One point cloud or mesh visualization from a tiny local sample

The demo must produce a saved command, log, and short conclusion.

