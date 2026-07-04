# Video2Motion2Action IsaacGym Setup

## What This Challenge Needs

The task page names `IsaacGym Preview 4` directly and gives this smoke-test path:

```bash
cd ~/opt/isaac_gym/isaacgym/python/examples
python3.8 joint_monkey.py
```

So the minimum simulator target is classic Isaac Gym Preview 4, not Isaac Sim 4.5. The relevant pieces are:

- Python 3.8 environment.
- Isaac Gym Preview 4 Python package and examples.
- PyTorch for HO-Tracker / ManipTrans / tracking policy code.
- Gym 0.23-era API compatibility.
- Mesh/URDF/asset tooling for Sharpa hand and object assets.
- Optional CUDA torch for training and rollout.

## Local Status

Prepared:

- `.venvs/isaacgym-py38`
- NumPy, Gym, OpenCV headless, Matplotlib, SciPy, TensorBoard
- Trimesh and mesh utility extras
- CPU PyTorch 1.13.1 and torchvision 0.14.1

Missing by design:

- Isaac Gym Preview 4 package. It requires either the NVIDIA licensed download or the contest image copy at `~/opt/isaac_gym/isaacgym`.
- CUDA PyTorch. The download from `download.pytorch.org` was too slow in this session; the CPU wheel is installed for import and asset checks.

## Commands

Prepare or repair the environment:

```bash
scripts/setup_isaac_stack.sh prepare
scripts/setup_isaac_stack.sh install-video2motion-deps
```

Check readiness:

```bash
.venvs/isaacgym-py38/bin/python scripts/check_video2motion_isaacgym.py
```

Install Isaac Gym after placing the NVIDIA package locally:

```bash
scripts/setup_isaac_stack.sh install-isaacgym ~/Downloads/IsaacGym_Preview_4_Package.tar.gz
```

Then run the official visual smoke test:

```bash
source .venvs/isaacgym-py38/bin/activate
cd ~/opt/isaac_gym/isaacgym/python/examples
python joint_monkey.py
```

## Important Hardware Note

This workstation uses an RTX 4060 Laptop GPU with compute capability 8.9. Classic Isaac Gym Preview 4 may fail in GPU PhysX mode on Ada GPUs. For scoring runs, the contest-provided DSW image or another known-compatible environment is safer. Locally, this setup is best for dependency checks, asset packaging, import debugging, and reportable preflight validation.
