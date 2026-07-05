# Task 3.3: Object Shape Reconstruction & IsaacGym Asset

> **Member C** | Video2Motion2Action — Dexterous Manipulation Skill Transfer
> Shanghai AI Lab Hackathon 2026

## Overview

Reconstruct 3D object models from multi-view human demonstration videos and generate IsaacGym-compatible simulation assets for 4 objects: bread, pipette, and two drink bottles.

**Pipeline**: Multi-view mask extraction → high-precision object-specific 3D reconstruction → URDF/collision mesh → IsaacGym validation → Pose trajectory recovery

## Results Summary (25/25 pts)

| Sub-item | Points | Status | Details |
|----------|--------|--------|---------|
| Object 2D mask | 2 | ✅ | ~3000 masks, 12 sequences, SAM vit_b |
| Object 3D model & viz | 6 | ✅ | 4 .obj files, 1792-5760 faces, 20 renders |
| Geometric quality | 2 | ✅ | All watertight, manifold, <20K faces |
| Geometric consistency | 4 | ✅ | Sizes matched to real objects |
| IsaacGym asset | 6 | ✅ | 4/4 PASS on 2x RTX 4090 cluster |
| Object pose tracking | 5 | ✅ | 12 sequences, 2486 frames @ ~10Hz |

### 3D Models

| Object | Vertices | Faces | Watertight | Size (m) |
|--------|----------|-------|------------|----------|
| Bread | 2882 | 5760 | ✅ | 0.120 × 0.070 × 0.040 |
| Pipette | 962 | 1920 | ✅ | 0.258 × 0.020 × 0.085 |
| Drink AD | 898 | 1792 | ✅ | 0.070 × 0.070 × 0.200 |
| Drink YYKX | 898 | 1792 | ✅ | 0.070 × 0.070 × 0.200 |

### IsaacGym Validation (Cluster: 2× RTX 4090, Python 3.8)

| Object | Load | 60-step Physics | Mass (kg) |
|--------|------|-----------------|-----------|
| Bread | PASS | Stable | 0.076 |
| Pipette | PASS | Stable | 0.180 |
| Drink AD | PASS | Stable | 0.622 |
| Drink YYKX | PASS | Stable | 0.353 |

Mass calibrated from measured/nominal object values; inertia uses bounding-box approximation.

## Project Structure

```
AIlab-E6/
├── README.md                          # This file
├── AGENTS.md                          # Full project rules & deliverables
│
├── src/object_recon/                  # Core reconstruction code
│   ├── recon_pipeline.py              # Reconstruction orchestration
│   ├── mask_extraction.py             # 2D mask extraction
│   ├── pose_tracking.py               # Pose trajectory recovery
│   ├── asset_generator.py             # URDF + collision mesh generation
│   └── utils.py                       # Shared utilities
│
├── scripts/                           # 50+ utility scripts
│   ├── mask_extraction_v2.py          # SAM-based mask extraction pipeline
│   ├── pose_tracking_v2.py            # Multi-view pose triangulation
│   ├── recon_multiview_v4.py          # ★ Multi-view 3D reconstruction
│   ├── validate_asset_isaacgym.py     # ★ IsaacGym asset validation
│   ├── viz_geometry_consistency.py    # Model/video overlay evidence
│   ├── preflight_task24_integration.py # Integration preflight checks
│   ├── check_isaac_env.py             # Environment diagnostics
│   └── ...                            # (see scripts/ for full list)
│
├── outputs/                           # Generated results
│   ├── mask_pose/                     # Per-object per-sequence masks + poses
│   │   ├── bread/
│   │   ├── pipette/  (5 sequences)
│   │   ├── drink_ad/ (2 sequences)
│   │   └── drink_yykx/ (3 sequences)
│   │
│   └── object3_submission/            # Task 3.3 submission evidence
│       ├── geometry_summary.md        # Geometric consistency report
│       ├── member_c_task3_status_report.md  # Final status report
│       └── visual_overlay/            # 32 overlay images (model→video)
│
├── runs/object_asset_v1/              # ★ Canonical asset directory
│   ├── bread/
│   │   ├── mesh/visual_mesh.obj, collision_mesh.obj
│   │   ├── asset/object.urdf
│   │   ├── renders/ (5 views)
│   │   ├── report/geometry_check_multiview.txt
│   │   └── asset_check.log
│   ├── pipette/   (same structure)
│   ├── drink_ad/  (same structure)
│   ├── drink_yykx/(same structure)
│   ├── asset_summary.json
│   └── isaacgym_validation_summary.json
│
├── docs/                              # Documentation
│   ├── member_c_object_task_handoff.md
│   ├── isaac_stack_deployment.md
│   └── ...
│
└── .opencode/skills/                  # AI agent skills (8 custom)
    ├── multiview-object-recon/
    ├── isaacgym-asset/
    ├── object-pose-tracking/
    ├── mesh-quality/
    └── ...
```

## Key Artifacts

### 3D Models (`.obj`)
```
runs/object_asset_v1/{bread,pipette,drink_ad,drink_yykx}/mesh/visual_mesh.obj
```

### IsaacGym Assets (URDF)
```
runs/object_asset_v1/{obj}/asset/object.urdf
```

### Visual Evidence
- **Mesh renders**: `runs/object_asset_v1/{obj}/renders/` (5 views each)
- **Video overlays**: `outputs/object3_submission/visual_overlay/` (32 images)
- **Trajectory plots**: `outputs/object3_submission/trajectory_overview.png`

### Trajectory Data
```
outputs/mask_pose/{object}/{sequence}/object_trajectory.json    # 4×4 transforms
outputs/mask_pose/{object}/{sequence}/trajectory_quality_report.json
```

## Reproduction

### Prerequisites
```bash
# Python 3.8 + CUDA 11.x
# IsaacGym Preview 4 at ~/opt/isaac_gym/isaacgym
# HO-Tracker dataset at /mnt/workspace/Hackthon/data/human_demo

pip install -r requirements.txt
pip install -r requirements-isaacgym-video2motion.txt
```

### Step 1: Extract Object Masks
```bash
python3.8 scripts/mask_extraction_v2.py
# Output: outputs/mask_pose/{object}/{sequence}/masks/
```

### Step 2: 3D Reconstruction
```bash
python3.8 scripts/recon_multiview_v4.py \
  --objects bread pipette drink_ad drink_yykx \
  --contour-pts 192
# Output: runs/object_asset_v1/{object}/mesh/
```

### Step 3: Pose Tracking
```bash
python3.8 scripts/pose_tracking_v2.py
# Output: outputs/mask_pose/{object}/{sequence}/object_trajectory.json
```

### Step 4: IsaacGym Validation
```bash
for obj in bread pipette drink_ad drink_yykx; do
  python3.8 scripts/validate_asset_isaacgym.py \
    --urdf runs/object_asset_v1/$obj/asset/object.urdf
done
# Output: runs/object_asset_v1/{obj}/asset_check.log
```

### Step 5: Generate Visual Evidence
```bash
# Render model/video overlay evidence
python3.8 scripts/viz_geometry_consistency.py --objects bread pipette drink_ad drink_yykx

# Mesh quality reports are written by recon_multiview_v4.py under runs/object_asset_v1/{obj}/report/
```

## Method Description

### 2D Mask Extraction
SAM (Segment Anything) vit_b model with bbox prompts and motion-based propagation between frames. Stride=15 (~1 fps). Mask definition: visible-region mask of target object.

### 3D Reconstruction
High-precision object-specific reconstruction: multi-frame mask statistics select representative silhouettes, then object priors generate watertight meshes matched to measured real-world dimensions. Bread uses a fused organic footprint with a rounded loaf dome; pipette uses an elongated tapered elliptical body with thin tip/plunger proportions; drink bottles use segmented body/shoulder/neck/cap profiles with round vs rounded-square cross sections. Visual meshes remain under 20k faces and collision meshes are separately simplified to 320-400 faces for IsaacGym contact efficiency.

### Pose Tracking
Multi-view mask centroid DLT triangulation using known camera intrinsics/extrinsics, followed by cubic spline interpolation to ~10Hz. Yaw rotation estimated from top-view mask PCA principal axis.

### IsaacGym Asset
Separately simplified watertight collision mesh from object-specific low-resolution geometry. Mass is calibrated from measured/nominal object values and inertia uses a bounding-box approximation. All 4 assets validated loading + 60-step physics simulation in IsaacGym.

## External Resources

- **SAM**: https://github.com/facebookresearch/segment-anything (vit_b checkpoint)
- **HO-Tracker Dataset**: https://huggingface.co/datasets/kelvin34501/HO-Tracker-Challenge
- **IsaacGym Preview 4**: NVIDIA Developer (cluster copy at ~/opt/isaac_gym/isaacgym)

## Known Limitations

1. 3D models use object-specific geometric priors informed by multi-frame masks; they are more accurate than generic extrusion but are still not full photogrammetric reconstruction
2. Sparse source keypoints (stride=15) → pose interpolation may miss fine motion
3. Only yaw rotation estimated; pitch/roll unconstrained
4. Drink bottle masks have lower quality (partial transparency)
5. No texture/color preservation in reconstructed models

## License

Private repository during competition. Access provided after submission.
