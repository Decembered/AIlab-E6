# Experiments

Experiment directories are created on-demand for new trials.
Each experiment follows the logging convention (README.md, config.yaml, command.sh, metrics.json, logs.txt).

## Recent Experiments

- `2026-07-05_obj_recon_drink_yykx/` — Mask debug for drink_yykx (incomplete)

## Convention

```bash
EXP=experiments/$(date +%F)_exp_name
mkdir -p "$EXP"/figures "$EXP"/outputs "$EXP"/models
touch "$EXP"/README.md "$EXP"/config.yaml "$EXP"/command.sh "$EXP"/metrics.json "$EXP"/logs.txt
```
