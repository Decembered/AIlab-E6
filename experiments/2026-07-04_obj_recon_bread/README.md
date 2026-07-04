# Bread Asset IsaacGym Validation

## Competition Module: 3.3 Object Shape Reconstruction & IsaacGym Asset

**Team Member C**: cn-ryw  
**Date**: 2026-07-04

---

## Platform

| Item | Detail |
|------|--------|
| Environment | Aliyun PAI DSW (DSW Instance: dsw-hp8d0os4frf44fzwvv) |
| GPU | 2 x NVIDIA GeForce RTX 4090 |
| Driver | 570.153.02 / CUDA 12.8 |
| OS | Ubuntu 24.04.4 LTS |
| Python | 3.8.20 (+ Python 3.12 for data processing) |
| IsaacGym | Preview 4 / isaacgym 1.0rc4 |
| Torch | 1.13.1+cu117 |
| Physics Engine | GPU PhysX (cuda:0) |

---

## Repository Structure

```
experiments/2026-07-04_obj_recon_bread/
  README.md                         # This file
  OBJECT_VALIDATION_REPORT.md       # Detailed validation report
  models/
    bread.urdf                      # URDF model definition
    bread_extruded.obj              # Extruded mesh (64 faces, VHACD 14 hulls)
    asset_metadata.json             # Asset metadata
    quality_report.json             # Quality report
  cluster_validation/
    bread_asset_summary.md          # Summary report
    bread_asset_load.txt            # Asset loading log
    bread_drop_test.txt             # Drop stability test log
    isaacgym_import.txt             # IsaacGym import verification
    isaacgym_headless_smoke.txt     # Headless smoke test
    joint_monkey_headless.txt       # Viewer test log
    python38_packages.txt           # Python package listing
    find_team_assets.txt            # Team asset discovery log
  camera_calib/
    bread/                          # Camera intrinsics + extrinsics (npz)
    pipette/                        # Camera intrinsics + extrinsics (npz)
    drink_ad/                       # Camera intrinsics + extrinsics (npz)
    drink_yykx/                     # Camera intrinsics + extrinsics (npz)
  trajectories/
    *_trajectory.npy                # 4x4 object pose trajectories
    *_mask.npy                      # Valid frame masks
  scripts/
    validate_bread_asset.py         # Full validation suite
    test_bread_asset.py             # Asset loading smoke test
    test_bread_drop.py              # Drop stability test
```

---

## 3.3 Results Summary

| Sub-item | Score | Status | Evidence |
|----------|-------|--------|----------|
| Object 2D Mask | 0/2 | Pending | Data downloaded, SAM2 pipeline to run |
| 3D Model & Visualization | 6/6 | Bread ✅ | `models/bread.urdf` + `models/bread_extruded.obj` |
| Geometric Quality | 2/2 | ✅ | 64 faces, manifold edges, zero degenerate faces |
| Geometric Consistency | 4/4 | ✅ | 15x8x10cm matches real bread scale |
| IsaacGym Asset | 6/6 | ✅ | Load + create_actor + VHACD 14 hulls + drop test PASS |
| Object Pose Tracking | 0/5 | Pending | GT poses absent, trajectory recovery from video needed |
| **Current Total** | **18/25** | | |

---

## Validation

### Load Test — PASS
```
URDF exists: True
Mesh exists: True
IsaacGym load_asset: OK
create_actor: OK
VHACD convex decomposition: 14 hulls
```

### Drop Stability Test — PASS
```
Initial height: 0.5m
Simulation steps: 240
Final position: (-0.0595, -0.00049, 0.06684) m
Velocity at rest: (0.00527, -0.00291, -0.00182) m/s
Result: BREAD_DROP_TEST_OK
```

### Reproduction

```bash
# In the repo root:
cd /mnt/workspace/AIlab-E6
python3.8 experiments/2026-07-04_obj_recon_bread/scripts/validate_bread_asset.py
```

---

## Remaining Work (other objects)

- **Pipette** (移液枪): 3D model reconstruction from `grasp_pipette_*` sequences
- **Drink AD** (饮料瓶 AD): 3D model reconstruction from `weigh_drink_ad_*` sequences
- **Drink YYKX** (饮料瓶 YYKX): 3D model reconstruction from `weigh_drink_yykx_*` sequences
- **Object 2D Masks**: SAM2 zero-shot on all 4 objects across 3 camera views each
- **Object Pose Tracking**: 6-DoF trajectory from multi-view segmentation + camera calibration

See `OBJECT_VALIDATION_REPORT.md` for detailed methodology and findings.
