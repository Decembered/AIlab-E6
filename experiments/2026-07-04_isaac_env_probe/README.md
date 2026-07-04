# Isaac Environment Probe

## Goal

Prepare a local Isaac Gym / Isaac Sim deployment path for hackathon manipulation and dexterous hand projects without downloading large simulator packages by default.

## Result Summary

- Created project-local Python environments for both candidate stacks:
  - `.venvs/isaacgym-py38`
  - `.venvs/isaaclab-py310`
- Added reusable scripts:
  - `scripts/setup_isaac_stack.sh`
  - `scripts/check_isaac_env.py`
- Did not download Isaac Sim, Isaac Sim assets, or Isaac Gym packages.

## Diagnosis

Classic Isaac Gym Preview 4 is probably the wrong main path for this machine because the GPU is RTX 4060 Laptop / Ada-class. Prefer Isaac Sim 4.5 + Isaac Lab for active hackathon work.

## Next Step

Choose one:

1. Place the NVIDIA Isaac Gym Preview 4 package under `~/Downloads/` and run the legacy installer.
2. Approve the large Isaac Sim 4.5 RL bundle download and run `scripts/setup_isaac_stack.sh install-isaacsim45-rl --i-approve-large-download`.
