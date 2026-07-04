# Experiments

Every experiment must live in its own timestamped folder:

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

## Required README Format

```markdown
# YYYY-MM-DD exp_name

## Goal

What question does this experiment answer?

## Setup

- Time:
- Timezone:
- Git commit:
- Git status:
- Python:
- Environment:
- CUDA:
- GPU:
- Seed:

## Command

See `command.sh`.

## Config

See `config.yaml`.

## Results

- Status:
- Main metric:
- Runtime:
- Output files:

## Failure Reason

Write `N/A` if successful. If failed, include the shortest useful diagnosis.

## Next Step

The next concrete action.
```

## Required config.yaml Fields

```yaml
task:
model_or_algorithm:
simulator:
dataset_or_asset:
seed:
hardware:
notes:
```

## Required metrics.json Fields

```json
{
  "status": "success_or_failed",
  "runtime_seconds": null,
  "seed": null,
  "metrics": {},
  "artifacts": [],
  "failure_reason": null,
  "next_step": null
}
```

## Rules

- Create the folder before running the command.
- Save exact commands in `command.sh`.
- Capture stdout and stderr in `logs.txt`.
- Save plots, screenshots, and curves under `figures/`.
- Save actions, trajectories, generated files, and videos under `outputs/`.
- If git is unavailable, record `N/A`.
- Failed experiments still count as experiments when they include logs and a next step.

