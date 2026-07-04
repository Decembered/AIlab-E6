# Video2Motion2Action — Dexterous Manipulation Skill Transfer

> **Track 1: Physical Intelligence | Topic T-1 | Team Challenge**
> Shanghai AI Lab Hackathon 2026

---

## Project Goal

From human manipulation demonstration videos, reconstruct:
1. **Hand motion** (2D mask, 3D MANO/mesh, continuous trajectory)
2. **Object shape** (3D model, watertight mesh, IsaacGym asset)
3. **Hand-object interaction** (retargeted Sharpa hand tracking in IsaacGym simulation)

The end-to-end pipeline: **Video → Motion → Action** — visual observations to simulation-ready assets and physically plausible dexterous manipulation tracking.

## Team & Roles

| Member | Primary Responsibility | Secondary |
|--------|----------------------|-----------|
| **A** | Sharpa Tracking (3.1) + Tracker Bonus | Comprehensive task (3.4) |
| **B** | Hand Reconstruction (3.2) + Hand Bonus (retargeting) | Comprehensive task (3.4) |
| **C** (you) | **Object Reconstruction (3.3) + Object Pose + Object Bonus** | Comprehensive task (3.4) |

## Member C Scope

### Core Tasks (25 points — Sub-task 3.3)

| Sub-item | Points | Requirement |
|----------|--------|-------------|
| Object 2D mask | 2 | Visible-region or completed-shape mask; must define the definition |
| Object 3D model & viz | 6 | Output `.obj` / `.ply` / `.stl`; multi-view screenshots or renders |
| Geometric quality | 2 | Watertight, manifold; reasonable face count (<20k recommended) |
| Geometric consistency | 4 | Scale, proportion, primary structure consistent with real object |
| IsaacGym asset | 6 | URDF, collision mesh, scale, mass/inertia; loadable in IsaacGym |
| Object pose tracking | 5 | Object motion trajectory from video or GT registration |

### Bonus Tasks

- **Object Bonus**: Articulated body modeling (joints, URDF joints, damping, friction, stiffness, actuation), optimized collision mesh, mass/inertia tuning, pose alignment for stable simulation contact.

### Integration Tasks (3.4 — 25 points shared)

- Feed object model + pose into the comprehensive tracking pipeline
- Ensure object asset works in simulation with Sharpa hand
- Collaborative debugging with members A and B

## Objects to Reconstruct

From the HO-Tracker-Challenge dataset (HuggingFace):
1. **Pipette** (移液枪)
2. **Bread** (面包)
3. **Two types of drink bottles** (两种饮料瓶)

Note: Do NOT reconstruct motion capture markers. Other object models/assets are already provided in the data.

### Key Data Analysis Findings (2026-07-04)

**GT object poses are intentionally masked out for ALL sequences.** The `pose_3d.hdf5` files contain pose trajectories `obj/pose/obj_transf` (T×4×4), but `obj/pose/mask` is **all False** for every primary object across all 12 sequences. This means:

- ❌ Cannot simply register reconstructed model to GT poses for the "object pose tracking" (5 pts) sub-item
- ✅ Must recover object trajectory from video (FoundationPose, feature tracking, 3DGS-based, etc.)
- ⚠️ The README explicitly warns: "frames where mask[t] == false contain zeroed transform matrices and should not be used as valid poses"

The only exception is `obj_lh/Test Tube #1` in `pipette_rh_beaker_testtube` (534/534 valid) — a secondary object.

**Per-sequence summary:**

| Object | Sequences | Frames | GT Mask |
|--------|-----------|--------|---------|
| Pipette #1 | grasp_pipette_press, grasp_pipette_rotate, grasp_pipette_stand, pipette_rh_beaker, pipette_rh_beaker_testtube | 260–715 | **All False** |
| Bread #1 | weigh_bread, weigh_bread__left | 217–235 | **All False** |
| Drink AD | weigh_drink_ad, weigh_drink_ad__left | 209–243 | **All False** |
| Drink YYKX | weigh_drink_yykx, weigh_drink_yykx__left, grasp_drink_yykx | 182–265 | **All False** |

**Implications for Member C's strategy:**
- Object pose tracking (5 pts) requires video-based pose estimation, not just GT registration
- Multi-view setup (3 cameras per sequence) enables triangulation and 3DGS-based tracking
- Camera calibration is provided (intrinsic + extrinsic for all 3 views) — ready for SfM/3DGS
- Each sequence has 3 videos at likely 30fps for ~7-24 seconds

## Technical Constraints (ZERO-TOLERANCE — Violation = 0 points)

### Prohibited

- ❌ Modifying official Sharpa tracking scoring logic (`eval_score.py`) or physics parameters (friction, gravity, etc.)
- ❌ Hardcoding evaluation outputs, manually fabricating trajectories, or writing sequence-specific answers
- ❌ Submitting unchanged Inspire baseline only
- ❌ Using commercial modeling software to directly draw objects (post-processing scale/orientation/material is OK)
- ❌ Using commercial text-to-3D models without image input
- ❌ Using commercial 3D scanners (including phone/iPad scanning apps)
- ❌ Depending on mass manual annotations for hand reconstruction

### Required

- ✅ Sharpa tracking checkpoints must follow naming convention, saved under `runs/`
- ✅ All reconstruction results must provide visualization evidence
- ✅ Object assets must be loadable in IsaacGym (no unusable high-poly models)
- ✅ All public models, pretrained weights, third-party tools, external resources must be cited in the report

## Allowed Technical Approaches (Object Reconstruction)

Open papers, open-source models, open-source tools, pretrained weights, and reasonable manual inspection/correction are allowed (must document in report).

Suggested approaches:
- **3D Gaussian Splatting (3DGS)** — from multi-view video
- **Image-to-3D models** — Zero-1-to-3, One-2-3-45, InstantMesh, TripoSR, etc.
- **Multi-view stereo / photometric optimization** — COLMAP + MVS, NeuS, etc.
- **NeRF-based** — Instant-NGP, Nerfstudio

Pipeline: reconstruction → post-process (scale/orient/cleanup) → decimate → make watertight → generate URDF → validate in IsaacGym.

## Reference Resources

- **HO-Tracker Baseline**: https://github.com/kelvin34501/HO-Tracker-Baseline-Challenge
- **HO-Tracker Data**: https://huggingface.co/datasets/kelvin34501/HO-Tracker-Challenge
  - HF Mirror: https://hf-mirror.com
- **IsaacGym Preview 4**: `~/opt/isaac_gym/isaacgym` (on cluster); local setup see `docs/isaac_stack_deployment.md`
- **Aliyun Cluster Image**: `pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/zhanxinyu:ho-tracker-v3`
- **Cluster Guide**: https://aicarrier.feishu.cn/wiki/LNRiwNLRviogOfk30eHcv2Q9ngd

## Deliverables Checklist (Member C)

### Code & Docs
- [ ] All code changes, scripts, commands for object reconstruction
- [ ] README with environment dependencies, reproduction steps, main result screenshots

### Object Reconstruction (3.3)
- [ ] Object 2D mask (with definition documented)
- [ ] Object 3D model files (`.obj`/`.ply`/`.stl`)
- [ ] Multi-view rendered screenshots
- [ ] Geometric quality report (watertight, manifold, face count)
- [ ] Geometric consistency evidence (scale comparison, overlay with real object)
- [ ] IsaacGym asset: URDF, collision mesh, scale, mass/inertia
- [ ] IsaacGym load verification screenshot
- [ ] Object pose trajectory data

### Object Bonus
- [ ] Articulated body URDF (if applicable)
- [ ] Joint configuration (type, axis, range, initial state)
- [ ] Physical parameters (damping, friction, stiffness, actuation torque)
- [ ] Evidence of improved contact stability or task success

### Comprehensive Task (3.4)
- [ ] Object asset integrated into the full tracking pipeline
- [ ] Simulation tracking rollout with object
- [ ] Visualization: 3D replay, simulation replay, video overlay
- [ ] Success/failure case analysis

### Report
- [ ] Method description, input/output, key parameters
- [ ] External resource attribution
- [ ] Failure frames, occlusion handling, limitations
- [ ] How object reconstruction improvements contributed to 3.4 success

## Project Directory Layout

```text
AGENTS.md                          # This file — rules, roles, deliverables
README.md                          # Competition overview & quick start
docs/
  isaac_stack_deployment.md        # IsaacGym/IsaacLab setup notes
requirements.txt                   # Python dependencies
src/
  object_recon/                    # Member C: object reconstruction pipeline
    recon_pipeline.py              # Main reconstruction script
    mask_extraction.py             # 2D mask extraction
    pose_tracking.py               # Object pose trajectory recovery
    asset_generator.py             # URDF + collision mesh generation
    utils.py                       # Shared utilities
  hand_recon/                      # Member B: hand reconstruction
  sharpa_tracking/                 # Member A: Sharpa tracking
experiments/
  README.md                        # Experiment logging rules
scripts/
  check_isaac_env.py               # IsaacGym environment diagnostics
  download_ho_tracker_data.sh      # HO-Tracker data download script
  setup_isaac_stack.sh             # Isaac stack setup
demos/                             # Final demos & visualizations
runs/                              # Sharpa tracking checkpoints (must be here!)
```

## Common Commands (Object Reconstruction)

```bash
# Environment check
python scripts/check_isaac_env.py

# Create new object reconstruction experiment
EXP=experiments/$(date +%F)_obj_recon
mkdir -p "$EXP"/figures "$EXP"/outputs "$EXP"/models
touch "$EXP"/README.md "$EXP"/config.yaml "$EXP"/command.sh "$EXP"/metrics.json "$EXP"/logs.txt

# Download HO-Tracker data (approve large downloads first)
bash scripts/download_ho_tracker_data.sh

# Validate URDF in IsaacGym
python -c "from isaacgym import gymapi; ..."

# Check mesh quality
python -c "import trimesh; m = trimesh.load('model.obj'); print(m.is_watertight, m.is_manifold)"
```

## Experiment Logging Rules (inherited)

Every experiment must create a folder under `experiments/` with:
- `README.md`: goal, hypothesis, result summary, failure reason, next step
- `config.yaml`: task, model, simulator, dataset, seed, hardware
- `command.sh`: exact commands with environment variables
- `metrics.json`: scalar metrics, runtime, scores
- `logs.txt`: stdout/stderr
- `figures/`: plots, screenshots, 3D views
- `outputs/`: generated artifacts, model files, videos

## Hard Constraints (inherited)

- Do not download large models/datasets/assets (>5GB) without user approval
- Do not run long training jobs by default — prefer smoke tests, short rollouts
- Do not use `sudo` or modify system-level environments without approval
- Do not break or recreate conda/venv environments casually
- Do not assume GPU availability — always provide CPU-safe diagnostics
- Keep experiments self-contained under `experiments/YYYY-MM-DD_exp_name/`
- Manipulation actions must not be described as safe for real robots without safety filter, planner, controller, state estimator, and emergency stop
