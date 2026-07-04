# Isaac Gym / Isaac Sim Deployment Notes

## Local Decision

This machine has Ubuntu 22.04, NVIDIA driver 580, and an RTX 4060 Laptop GPU with about 8 GB VRAM. The default system Python is not suitable for simulator work, so the project uses isolated environments under `.venvs/`.

For hackathon manipulation, dexterous hand, and RL baselines, prefer:

1. Isaac Sim 4.5 + Isaac Lab for current GPU simulation and RL workflows.
2. Classic Isaac Gym Preview 4 only for old repositories that have not migrated yet.

## Why Not Classic Isaac Gym First

Isaac Gym Preview 4 is legacy software. NVIDIA's archive lists Ubuntu 18.04/20.04 and Python 3.6-3.8 as prerequisites. This workstation is Ubuntu 22.04 with an Ada GPU, and old Isaac Gym GPU PhysX binaries are likely incompatible with SM 8.9 GPUs such as RTX 40 series.

The practical consequence is:

- Old `IsaacGymEnvs` projects can still be inspected and sometimes installed.
- GPU simulation may fail with `CUDA error: no kernel image is available for execution on the device`.
- For a black-box hackathon demo, Isaac Lab is the safer path.

## Prepared Environments

```bash
scripts/setup_isaac_stack.sh prepare
```

Creates:

- `.venvs/isaacgym-py38`: legacy Isaac Gym Preview 4 path.
- `.venvs/isaaclab-py310`: Isaac Sim / Isaac Lab 4.x path.

Check status:

```bash
scripts/setup_isaac_stack.sh check
```

## Install Classic Isaac Gym Preview 4

Manual step required: download the Isaac Gym Preview 4 package from NVIDIA after accepting the license.

Then run:

```bash
scripts/setup_isaac_stack.sh install-isaacgym ~/Downloads/IsaacGym_Preview_4_Package.tar.gz
```

After installation, use CPU smoke tests first. Treat GPU mode on RTX 4060 as experimental and likely unsupported.

## Install Isaac Sim 4.5 RL Bundle

This may download more than 5 GB, so it is deliberately guarded:

```bash
scripts/setup_isaac_stack.sh install-isaacsim45-rl --i-approve-large-download
```

Before first run, accept the NVIDIA Omniverse EULA if appropriate:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
source .venvs/isaaclab-py310/bin/activate
python -c 'import isaacsim; print("isaacsim import ok")'
```

For a full offline GUI or asset-heavy workflow, Isaac Sim 4.5 Linux is about 6.7 GB and the optional asset packs are much larger. Do not download those asset packs unless a demo specifically needs them.

## Hackathon Usage Recommendation

Use Isaac Lab for GPU RL or manipulation demos, then package only the smallest meaningful result:

- one reset and rollout smoke test,
- one robot arm or dexterous hand task,
- metrics saved under `experiments/`,
- screenshots or short videos under `figures/` or `outputs/`.

Any VLA or learned action output must be described as a proposal that needs a safety filter, planner, controller, state estimator, and emergency stop path before real robot execution.
