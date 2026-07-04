# Object Asset v1 for Video2Motion2Action

This folder contains the current Member C object assets for task 3.3. The assets are simulation-first, low-face-count, watertight meshes with separate visual and collision geometry.

## Objects

| Folder | Object | Visual Mesh | Collision Mesh | URDF | Renders | Local IsaacGym Probe |
|---|---|---|---|---|---|---|
| bread | Bread #1 | `mesh/visual_mesh.obj` | `mesh/collision_mesh.obj` | `asset/object.urdf` | `renders/` | failed_local_env |
| pipette | Pipette #1 | `mesh/visual_mesh.obj` | `mesh/collision_mesh.obj` | `asset/object.urdf` | `renders/` | failed_local_env |
| drink_ad | Drink AD | `mesh/visual_mesh.obj` | `mesh/collision_mesh.obj` | `asset/object.urdf` | `renders/` | failed_local_env |
| drink_yykx | Drink YYKX | `mesh/visual_mesh.obj` | `mesh/collision_mesh.obj` | `asset/object.urdf` | `renders/` | failed_local_env |

## Units and Coordinate Convention

- Units are meters and kilograms.
- Drink bottles: mesh centered near origin; z-axis is vertical.
- Bread: mesh centered near origin; x-axis is the long side.
- Pipette: mesh centered near origin; x-axis is the length direction.

## Validation Summary

- Geometry summary: `asset_summary.json`, `geometry_summary.csv`, and per-object `report/` files.
- Render evidence: per-object `renders/front.png`, `side.png`, `top.png`, `angle.png`, `collision_angle.png`.
- IsaacGym local probe: per-object `report/asset_check_local.log`. If local IsaacGym is unavailable, rerun the same validation on the competition cluster image.
