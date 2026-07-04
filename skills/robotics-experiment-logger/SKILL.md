# robotics-experiment-logger

## Purpose

Ensure every robotics, RL, VLA, simulation, or 3D experiment is reproducible and presentation-ready.

The logger creates structured experiment folders and records enough metadata to rerun, debug, and report results.

## When To Use

Use this skill before any experiment that:

- Runs model inference
- Runs a simulator
- Trains or evaluates a policy
- Generates visualizations
- Produces metrics for a report or demo
- Fails in a way that should be diagnosed later

## Inputs

- Experiment name
- Goal / hypothesis
- Command to run
- Config dictionary or config file
- Seed
- Expected output files
- Optional task tags: VLA, RL, 3D, sim, safety, planner

## Outputs

An experiment folder:

```text
experiments/YYYY-MM-DD_exp_name/
  README.md
  config.yaml
  command.sh
  metrics.json
  logs.txt
  figures/
  outputs/
```

## Steps

1. Create the folder before running the experiment.
2. Write `config.yaml` with task, model, simulator, seed, and hardware assumptions.
3. Write `command.sh` with exact commands.
4. Capture environment metadata:

   - Time and timezone
   - Git commit or `N/A`
   - Git dirty status or `N/A`
   - Python version
   - Conda or venv
   - CUDA availability
   - GPU name

5. Run commands with output captured to `logs.txt`.
6. Write scalar results to `metrics.json`.
7. Save screenshots, curves, and visualizations to `figures/`.
8. Save machine-readable outputs to `outputs/`.
9. Update `README.md` with result, limitation, failure reason, and next step.

## Constraints

- Do not run unlogged experiments when results may matter.
- Do not overwrite old experiment folders unless explicitly requested.
- Do not use vague commands such as "run notebook manually" as the only record.
- If a run fails, record the failure as a valid experiment result.

## Failure Debugging

- If metrics are missing, inspect logs and write a minimal `metrics.json` containing status, runtime, and error type.
- If git is unavailable, write `N/A`.
- If a figure cannot be generated, save a text explanation in `figures/README.md`.
- If output files are too large, save paths and metadata instead of copying large binaries.

## Minimum Runnable Demo

Create one experiment folder for a smoke test and record:

- Exact command
- Environment summary
- Status: success or failed
- One metric or diagnostic
- One next step

