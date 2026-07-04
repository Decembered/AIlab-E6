# Object Asset v1 Geometry Summary

All units are meters and kilograms. These assets are simulation-first, low-face-count meshes intended for IsaacGym loading and tracking integration.

| Object | Visual faces | Collision faces | Watertight | Extents (m) | Mass (kg) | URDF |
|---|---:|---:|---|---|---:|---|
| Bread #1 | 1280 | 12 | yes | 0.120 x 0.070 x 0.040 | 0.08 | `/home/ruan/research/Hackthon/runs/object_asset_v1/bread/asset/object.urdf` |
| Pipette #1 | 472 | 12 | yes | 0.258 x 0.020 x 0.085 | 0.08 | `/home/ruan/research/Hackthon/runs/object_asset_v1/pipette/asset/object.urdf` |
| Drink AD | 512 | 64 | yes | 0.070 x 0.070 x 0.200 | 0.3 | `/home/ruan/research/Hackthon/runs/object_asset_v1/drink_ad/asset/object.urdf` |
| Drink YYKX | 512 | 64 | yes | 0.070 x 0.070 x 0.200 | 0.3 | `/home/ruan/research/Hackthon/runs/object_asset_v1/drink_yykx/asset/object.urdf` |

## Evidence Layout

- Per-object renders: `runs/object_asset_v1/<object>/renders/`
- Render contact sheet: `outputs/object3_submission/render_contact_sheet.png`
- Per-object geometry checks: `runs/object_asset_v1/<object>/report/`
- IsaacGym probe logs: `runs/object_asset_v1/<object>/report/asset_check_local.log`
- Pose inventory: `outputs/dataset_inventory.csv`
- Bread current pose output: `outputs/mask_pose/bread/weigh_bread__2026_0701_0044_30/`
