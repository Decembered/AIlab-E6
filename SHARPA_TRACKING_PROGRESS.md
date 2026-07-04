# Sharpa Tracking Progress Log

Date: 2026-07-04

This document records the work done for the Sharpa Tracking part of the Video2Motion2Action task, including what has been verified, what failed, and what remains blocked.

## Current Scope

Role A focuses on the Sharpa Tracking module:

1. Build the IsaacGym / ManipTrans / HO-Tracker baseline environment.
2. Run the official Inspire sample `0f900@0`.
3. Add or verify Sharpa URDF support.
4. Run Sharpa retargeting, training, rollout, and scoring.
5. Improve Sharpa tracking score.

The first two items have been completed on xmu75. The Sharpa-specific items are partially prepared but blocked by missing Sharpa URDF/mesh assets.

## GitHub Upload Status

Prepared branch on xmu75:

```text
repo: /data/autovla/ho_tracker_challenge/HO-Tracker-Baseline-Challenge
branch: lucas/0f900-inspire-baseline
commit: 5d8119a Add 0f900 Inspire baseline reproducibility scripts
target remote: https://github.com/Decembered/AIlab-E6.git
```

Push attempt result:

```text
remote: Permission to Decembered/AIlab-E6.git denied to Decembered.
fatal: unable to access 'https://github.com/Decembered/AIlab-E6.git/': The requested URL returned error: 403
```

Conclusion: the provided GitHub token did not have write permission to the target repo. A new token is required with `Contents: Read and write` access to `Decembered/AIlab-E6`.

Security note: because token input was echoed by the remote TTY during one attempt, the GitHub PAT should be revoked and regenerated.

## W&B Status

W&B was configured and verified on xmu75.

```text
account: tangy2462
entity: tangy2462-xiamen-university
project: ho-tracker-baseline
```

Project URL:

```text
https://wandb.ai/tangy2462-xiamen-university/ho-tracker-baseline
```

Verified W&B smoke run:

```text
run name: wandb_smoke_0f900@0_inspire_lh__07-04-09-55-26
```

Run URL:

```text
https://wandb.ai/tangy2462-xiamen-university/ho-tracker-baseline/runs/uid_wandb_smoke_0f900@0_inspire_lh__07-04-09-55-26
```

Note: `wandb` was upgraded from `0.12.21` to `0.24.2` because the old version did not accept the new `wandb_v1_...` API key format. `rl-games` declares `wandb<0.13.0`, but a 1-epoch training smoke test worked after the upgrade.

## Completed Inspire Baseline

Environment verified:

```text
Python: 3.8
PyTorch: 1.13.1+cu117
IsaacGym: official Preview 4 package
IsaacGym path: /data/autovla/ho_tracker_challenge/IsaacGym_Preview_4_Package/isaacgym/python/isaacgym
```

Important fix:

The public `isaacgym-1.0rc4` wheel was not enough because it missed:

```text
isaacgym/_bindings/src/gymtorch/gymtorch.cpp
```

The official IsaacGym Preview 4 package was installed in editable mode instead.

Minimal `0f900@0` data was prepared:

```text
data/HO-Tracker/data/test_sample/h1o1/0f900@0/left_hand.pkl
data/HO-Tracker/data/test_sample/h1o1/0f900@0/left_obj.pkl
data/HO-Tracker/data/test_sample/h1o1/0f900@0/left_urdf/scan.ply
data/HO-Tracker/data/test_sample/h1o1/0f900@0/left_urdf/scan.urdf
```

The `scan.urdf` file was generated because the baseline code expects an object URDF, while the Hugging Face minimal sample provides a mesh file.

Retargeting command:

```bash
python main/dataset/mano2dexhand.py \
  --data_idx 0f900@0 \
  --side left \
  --dexhand inspire \
  --headless \
  --iter 7000
```

Result:

```text
data/retargeting/HO-Tracker/mano2inspire_lh/test_sample/h1o1/0f900@0/opt.pkl
```

Retargeted shapes:

```text
opt_wrist_pos: (830, 3)
opt_wrist_rot: (830, 3)
opt_dof_pos: (830, 12)
opt_joints_pos: (830, 18, 3)
```

Training smoke test:

```bash
python main/rl/train.py task=ResDexHand dexhand=inspire side=LH headless=true \
  num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true \
  "dataIndices=[0f900@0]" actionsMovingAverage=0.4 experiment=baseline \
  max_iterations=2 rl_train.params.config.save_frequency=1
```

Result:

```text
runs/baseline_0f900@0_inspire_lh__07-04-09-21-37/nn/baseline.pth
```

Rollout smoke test produced:

```text
dumps/dump_baseline_0f900@0_inspire_lh__07-04-09-25-35/rollouts.hdf5
```

This smoke rollout contained failed rollouts only, which is expected after two epochs. It verifies the pipeline, not final performance.

## Sharpa Asset Search

Searched for Sharpa assets on xmu75:

```bash
find /data/autovla /home/autovla -iname "*sharpa*" -o -iname "*sharp*"
```

Result: no Sharpa URDF/mesh was found. Matches were unrelated packages such as image libraries and C# files.

Searched inside the challenge repo:

```bash
grep -RIn "sharpa\\|Sharpa\\|SHARPA\\|sharp" .
```

Result: only README references. There is no Sharpa dexhand class, URDF, mesh, or mapping.

Public web search did not find a direct official Sharpa URDF download. Public product/news material indicates Sharpa has around 22 DoF and mentions ROS2/MuJoCo/URDF/MJCF support, but the actual asset files are not available in the current workspace.

## Sharpa Work Plan

Once Sharpa assets are available, add them under:

```text
maniptrans_envs/assets/sharpa_hand/
```

Expected files:

```text
sharpa_hand_left.urdf
sharpa_hand_right.urdf
meshes/...
```

Then implement:

```text
maniptrans_envs/lib/envs/dexhands/sharpa.py
```

The Sharpa dexhand class must provide:

- `name = "sharpa"`
- `urdf_path`
- `body_names`
- `dof_names`
- `hand2dex_mapping`
- `dex2hand_mapping`
- `contact_body_names`
- `bone_links`
- `weight_idx`
- `relative_rotation`
- optional `relative_translation`
- PID reference gains

The most important and error-prone part is `hand2dex_mapping`: each relevant Sharpa rigid body used by tracking should map to a MANO hand segment. If this mapping is wrong, retargeting may converge numerically but produce poor hand-object contact and bad final tracking.

## Validation Order For Sharpa

Use this order instead of jumping directly into long training:

1. Verify factory registration:

```bash
python scripts/check_dexhand_support.py --dexhand sharpa --side left
```

2. Verify IsaacGym can load the Sharpa URDF:

```bash
python scripts/check_dexhand_support.py --dexhand sharpa --side left --load-asset
```

3. Run Sharpa retargeting:

```bash
python main/dataset/mano2dexhand.py \
  --data_idx 0f900@0 \
  --side left \
  --dexhand sharpa \
  --headless \
  --iter 7000
```

Expected result:

```text
data/retargeting/HO-Tracker/mano2sharpa_lh/test_sample/h1o1/0f900@0/opt.pkl
```

4. Run short Sharpa training smoke:

```bash
python main/rl/train.py task=ResDexHand dexhand=sharpa side=LH headless=true \
  num_envs=256 learning_rate=2e-4 test=false randomStateInit=true \
  "dataIndices=[0f900@0]" actionsMovingAverage=0.4 experiment=sharpa_smoke \
  max_iterations=1 rl_train.params.config.save_frequency=1 \
  wandb_activate=true wandb_project=ho-tracker-baseline
```

5. Run full Sharpa training:

```bash
python main/rl/train.py task=ResDexHand dexhand=sharpa side=LH headless=true \
  num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true \
  "dataIndices=[0f900@0]" actionsMovingAverage=0.4 experiment=sharpa_baseline \
  wandb_activate=true wandb_project=ho-tracker-baseline
```

6. Run rollout and scoring:

```bash
python main/rl/eval_rollout.py --tag sharpa_baseline --dexhand sharpa
python main/rl/eval_score.py
```

## Trial And Error Notes

1. IsaacGym wheel issue:
   - Symptom: `gymapi` worked, but importing `gymtorch` failed because `gymtorch.cpp` was missing.
   - Fix: install from official IsaacGym Preview 4 package.

2. IsaacGym import order:
   - Symptom: `ImportError: PyTorch was imported before isaacgym modules`.
   - Fix: import `isaacgym` before `torch` in standalone tests.

3. `LD_LIBRARY_PATH` issue:
   - Symptom: `libpython3.8.so.1.0` not found.
   - Fix: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}`.

4. GitHub direct clone instability on xmu75:
   - Symptom: GitHub clone timed out or failed.
   - Fix: download source archives locally and copy/install on xmu75.

5. W&B API key format:
   - Symptom: old `wandb 0.12.21` rejected `wandb_v1_...` key.
   - Fix: upgrade to `wandb 0.24.2`; smoke training synced successfully.

6. Eval score after short smoke:
   - Symptom: `eval_score.py` ended with division by zero.
   - Cause: short 2-epoch policy had no successful rollouts.
   - Interpretation: rollout file writing works; final score requires longer training and successful samples.

7. GitHub push:
   - Symptom: `403 Permission to Decembered/AIlab-E6.git denied`.
   - Cause: provided PAT does not have write access to target repo.
   - Required fix: regenerate a PAT with target repo `Contents: Read and write`.

## Current Blocker

Sharpa tracking cannot be fully run until the Sharpa URDF and mesh assets are available.

Required from organizers or team:

```text
maniptrans_envs/assets/sharpa_hand/sharpa_hand_left.urdf
maniptrans_envs/assets/sharpa_hand/sharpa_hand_right.urdf
maniptrans_envs/assets/sharpa_hand/meshes/*
```

After assets are available, the next concrete task is to implement `maniptrans_envs/lib/envs/dexhands/sharpa.py` and validate it with `scripts/check_dexhand_support.py`.

## Added Helper Scripts

Two helper scripts were added after the Inspire baseline commit:

```text
scripts/check_dexhand_support.py
scripts/run_sharpa_tracking.sh
```

`check_dexhand_support.py` validates:

- `DexHandFactory` registration;
- URDF path existence;
- `body_names` / `dof_names`;
- `hand2dex_mapping` / `dex2hand_mapping`;
- `contact_body_names`;
- `weight_idx`;
- optional IsaacGym URDF loading through `--load-asset`.

Validation result for Inspire:

```bash
python scripts/check_dexhand_support.py --dexhand inspire --side left
```

Result:

```text
dexhand: inspire_lh
urdf_exists: True
n_bodies: 18
n_dofs: 12
OK
```

Validation result for Sharpa before adding assets:

```bash
python scripts/check_dexhand_support.py --dexhand sharpa --side left
```

Result:

```text
FAILED: DexHandFactory.create_hand('sharpa', 'left')
ValueError: Hand type 'sharpa_lh' not registered.
```

This is the expected current failure because the repository has no Sharpa dexhand implementation yet.
