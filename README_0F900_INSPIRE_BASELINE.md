# HO-Tracker 0f900@0 Inspire Baseline

This note documents the minimal reproducible baseline for the Physical Intelligence Center hackathon task:

```text
HO-Tracker sample 0f900@0
-> MANO hand/object tracking data
-> retarget to the official Inspire dexterous hand
-> train the official ResDexHand policy in IsaacGym Preview 4
-> rollout and evaluate
```

The baseline is not a provided pretrained checkpoint. It is the official pipeline run from the official sample and default Inspire hand configuration.

## Tested Machine

- Server: xmu75
- Conda env: `/home/autovla/miniconda3/envs/maniptrans`
- Repo path: `/data/autovla/ho_tracker_challenge/HO-Tracker-Baseline-Challenge`
- IsaacGym: official IsaacGym Preview 4 package
- Python: 3.8
- PyTorch: 1.13.1+cu117
- CUDA visible devices used for smoke tests: GPU 4 and GPU 5

## Important Notes

IsaacGym Preview 4 is required. The public `isaacgym-1.0rc4` wheel can import `gymapi`, but it may be missing `isaacgym/_bindings/src/gymtorch/gymtorch.cpp`. This baseline imports `gymtorch`, so install IsaacGym from the official Preview 4 package:

```bash
cd /data/autovla/ho_tracker_challenge/IsaacGym_Preview_4_Package/isaacgym/python
python -m pip install -e . --no-deps
```

Always export the conda library path before importing IsaacGym:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

Also import `isaacgym` before `torch` in standalone Python tests.

## Setup

From the repo root:

```bash
bash scripts/setup_maniptrans_env.sh
```

This script assumes:

- conda is installed at `/home/autovla/miniconda3`
- the official IsaacGym Preview 4 package is available at `/data/autovla/ho_tracker_challenge/IsaacGym_Preview_4_Package`
- the repo is checked out at `/data/autovla/ho_tracker_challenge/HO-Tracker-Baseline-Challenge`

## Prepare Minimal 0f900@0 Data

```bash
bash scripts/prepare_0f900_data.sh
```

This downloads only the files needed by `0f900@0`:

- `left_hand.pkl`
- `left_obj.pkl`
- `left_urdf/scan.ply`

It also creates a minimal `left_urdf/scan.urdf`, because the baseline code loads a URDF while the Hugging Face sample provides the mesh.

## Run Baseline

For a quick smoke test:

```bash
bash scripts/run_0f900_inspire_baseline.sh smoke
```

This runs:

1. `mano2dexhand.py` retargeting with the official `0f900@0` sample.
2. A short 2-epoch ResDexHand training smoke test.
3. A short rollout smoke test.
4. `eval_score.py` to verify the scoring entrypoint.

The smoke test is only meant to prove the environment and pipeline are connected. It is not expected to produce a high-quality policy.

For the official long training command:

```bash
bash scripts/run_0f900_inspire_baseline.sh train
```

This starts the default training command from the baseline README:

```bash
python main/rl/train.py task=ResDexHand dexhand=inspire side=LH headless=true \
  num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true \
  "dataIndices=[0f900@0]" actionsMovingAverage=0.4 experiment=baseline
```

The default config trains for many epochs and should be run in `tmux` or `nohup`.

## W&B

After logging in with:

```bash
wandb login
```

enable W&B during training by adding:

```bash
wandb_activate=true wandb_project=<project-name> wandb_entity=<entity-or-team>
```

For example:

```bash
python main/rl/train.py task=ResDexHand dexhand=inspire side=LH headless=true \
  num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true \
  "dataIndices=[0f900@0]" actionsMovingAverage=0.4 experiment=baseline \
  wandb_activate=true wandb_project=ho-tracker-baseline
```

## Outputs

Expected generated files:

```text
data/retargeting/HO-Tracker/mano2inspire_lh/test_sample/h1o1/0f900@0/opt.pkl
runs/baseline_0f900@0_inspire_lh__*/nn/*.pth
dumps/dump_baseline_0f900@0_inspire_lh__*/rollouts.hdf5
logs/*.log
```

Do not commit generated data, logs, checkpoints, rollout dumps, or IsaacGym packages to Git.

## Verified Smoke Result

On xmu75, the smoke test verified:

- IsaacGym Preview 4 `gymtorch` import and JIT build.
- GPU PhysX sim creation.
- `0f900@0` data load: 830 frames, Inspire LH 12 DoFs.
- Retargeting output `opt.pkl` shape:
  - `opt_wrist_pos`: `(830, 3)`
  - `opt_wrist_rot`: `(830, 3)`
  - `opt_dof_pos`: `(830, 12)`
  - `opt_joints_pos`: `(830, 18, 3)`
- 2-epoch smoke training produced a checkpoint under `runs/`.
- Rollout smoke produced an HDF5 file under `dumps/`.

