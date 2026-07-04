# Member C Task 3.3 Submission Evidence Status

## Summary

This package organizes the current object reconstruction deliverables for the Video2Motion2Action task. The current data has RGB/calibration/HDF5, but primary-object GT pose masks are all false and no raw depth frames are available, so the pose output is video/mask-driven rather than GT or RGB-D ICP.

## Asset Completion

| Object | Mesh/URDF | Geometry | Renders | IsaacGym local probe | Current pose evidence |
|---|---|---|---|---|---|
| Bread #1 | yes | watertight, 1280 visual faces | `runs/object_asset_v1/bread/renders/` | failed_local_env: FAIL: isaacgym import failed: No module named 'isaacgym' | available sparse multi-view |
| Pipette #1 | yes | watertight, 472 visual faces | `runs/object_asset_v1/pipette/renders/` | failed_local_env: FAIL: isaacgym import failed: No module named 'isaacgym' | not yet; usable masks 0/5 |
| Drink AD | yes | watertight, 512 visual faces | `runs/object_asset_v1/drink_ad/renders/` | failed_local_env: FAIL: isaacgym import failed: No module named 'isaacgym' | not yet; usable masks 0/3 |
| Drink YYKX | yes | watertight, 512 visual faces | `runs/object_asset_v1/drink_yykx/renders/` | failed_local_env: FAIL: isaacgym import failed: No module named 'isaacgym' | not yet; usable masks 0/3 |

## Current Task 3.3 Score Readiness

| Sub-item | Status | Evidence | Remaining work |
|---|---|---|---|
| Object 2D mask | partial | Bread v4.1 masks and debug overlays exist | Generate stable dynamic masks for pipette/drinks and continuous bread masks |
| Object 3D model & viz | mostly ready | Four `.obj` assets plus generated render views | Add video-overlay/scale comparison for non-bread objects |
| Geometric quality | ready | Watertight checks, face counts, extents | Keep reports in final README |
| Geometric consistency | partial | Dimensions and bread evidence exist | Add mesh projection or visual comparison for pipette/drinks |
| IsaacGym asset | blocked locally if probe fails | URDF/collision files and local probe logs | Re-run on cluster IsaacGym if local import/sim fails |
| Object pose tracking | partial | Prompt-free Phase 5 pipeline and sparse bread multi-view pose | Need stable masks for every target object/sequence |

## Pose Tracking Status

- Dataset inventory: `/home/ruan/research/Hackthon/outputs/dataset_inventory.csv`
- Mask audit: `outputs/object3_submission/mask_audit.csv`
- Bread method: `multi_view_mask_pose`
- Bread valid frames: 1/3
- Bread limitation: Depth-free calibrated multi-view mask fitting. This is an approximate video-derived pose, not RGB-D ICP or GT.

## Why Some Items Remain Incomplete

- The official HDF5 primary-object trajectories are present but invalid because the primary object masks are all false.
- The current repository data has no discoverable depth frames, so real `mask_depth_icp` cannot be run on these sequences yet.
- Existing fixed-coordinate SAM prompts fail on long sequences; future masks must use dynamic bbox/centroid prompts or mask propagation.
- IsaacGym validation must be treated as an environment-dependent step; local failures are logged and should be repeated on the cluster image.

## Next Execution Defaults

- Keep `runs/object_asset_v1` as the canonical asset root.
- Keep `scripts/phase5_mask_depth_pose.py` as the prompt-free pose tracking entrypoint.
- Use `multi_view_mask_pose` for current RGB-only calibrated data; use `mask_depth_icp` only if depth or reconstructed per-frame point clouds are added.
- For drinks, mark yaw as ambiguous and prioritize translation continuity.
