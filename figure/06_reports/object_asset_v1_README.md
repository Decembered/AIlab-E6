# Object Asset v1 for Video2Motion2Action

This folder contains the current Member C object assets for task 3.3. The assets are simulation-first, low-face-count, watertight meshes with separate visual and collision geometry.

## Objects

| Folder | Object | Visual Mesh | Collision Mesh | URDF | Renders | IsaacGym Probe |
|---|---|---|---|---|---|---|
| bread | Bread #1 | `mesh/visual_mesh.obj` | `mesh/collision_mesh.obj` | `asset/object.urdf` | `renders/` | cluster_pass_cpu_physics |
| pipette | Pipette #1 | `mesh/visual_mesh.obj` | `mesh/collision_mesh.obj` | `asset/object.urdf` | `renders/` | cluster_pass_cpu_physics |
| drink_ad | Drink AD | `mesh/visual_mesh.obj` | `mesh/collision_mesh.obj` | `asset/object.urdf` | `renders/` | cluster_pass_cpu_physics |
| drink_yykx | Drink YYKX | `mesh/visual_mesh.obj` | `mesh/collision_mesh.obj` | `asset/object.urdf` | `renders/` | cluster_pass_cpu_physics |

## Units and Coordinate Convention

- Units are meters and kilograms.
- Drink bottles: mesh centered near origin; z-axis is vertical.
- Bread: mesh centered near origin; x-axis is the long side.
- Pipette: mesh centered near origin; x-axis is the length direction.

## Validation Summary

- Geometry summary: `asset_summary.json`, `geometry_summary.csv`, and per-object `report/` files.
- Render evidence: per-object `renders/front.png`, `side.png`, `top.png`, `angle.png`, `collision_angle.png`.
- IsaacGym cluster validation: `isaacgym_validation_summary.json` and per-object `asset_check.log` record 4/4 load plus 60-step CPU physics PASS on the 2x RTX 4090 cluster Python 3.8 environment.
- Local IsaacGym probe logs remain under `report/asset_check_local.log`. A local failure only means the legacy `isaacgym` Python module is unavailable in that local environment; use the cluster validation as the submission evidence.
