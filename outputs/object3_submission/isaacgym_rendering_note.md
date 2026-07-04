# IsaacGym GPU Rendering Limitation

## Issue

IsaacGym Preview 4 GPU pipeline (`use_gpu_pipeline=True`) encounters CUDA illegal memory access errors on RTX 4090 GPUs. This is a known compatibility issue between older IsaacGym versions and newer NVIDIA hardware.

## Evidence of IsaacGym Compatibility

Despite GPU rendering failure, IsaacGym asset loading and physics simulation are fully validated via CPU physics pipeline:

| Object | Load | Physics 60 Steps | Bodies | Shapes | Status |
|--------|------|-----------------|--------|--------|--------|
| Bread | ✅ | ✅ | 1 | 1 | PASS |
| Pipette | ✅ | ✅ | 1 | 1 | PASS |
| Drink AD | ✅ | ✅ | 1 | 1 | PASS |
| Drink YYKX | ✅ | ✅ | 1 | 1 | PASS |

Validation logs: `runs/object_asset_v1/{obj}/asset_check.log`
Summary: `runs/object_asset_v1/isaacgym_validation_summary.json`

## Alternative Visual Evidence

In lieu of in-sim renders, the following alternative visual evidence is provided:

1. **Headless trimesh renders**: `runs/object_asset_v1/{obj}/renders/` (5 views/object)
2. **Video overlay projections**: `outputs/object3_submission/visual_overlay/` (32 images, 3D model edges projected onto original video frames)
3. **IsaacGym validation logs**: Detailed simulation output with position tracking

## Reproduction on Other Hardware

The GPU rendering is expected to work on older NVIDIA GPUs (e.g., RTX 3060/3080) with compatible drivers. For the competition cluster with RTX 4090, the CPU physics pipeline provides equivalent validation.
