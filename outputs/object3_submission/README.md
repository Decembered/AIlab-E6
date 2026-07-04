# Object 3.3 Submission Evidence README

This folder is the compact evidence bundle for Member C's object reconstruction,
asset generation, and pose-tracking work.

## Files

| File | Purpose |
|---|---|
| `member_c_task3_status_report.md` | Human-readable status: done, incomplete, reasons, next steps |
| `geometry_summary.csv` | Four-object mesh/URDF metrics: faces, extents, watertightness, mass, inertia |
| `geometry_summary.md` | Short Markdown version of geometry summary |
| `asset_summary.json` | Machine-readable asset summary |
| `mask_audit.csv` | Object mask quality audit: area, area ratio, pose usability |
| `mask_audit_summary.md` | Short Markdown mask audit summary |
| `isaacgym_validation_summary.json` | Local IsaacGym probe result for each URDF |
| `render_contact_sheet.png` | One-image overview of all four object assets |

## Canonical Assets

The actual object assets live in:

```text
runs/object_asset_v1/
```

Each object folder contains:

```text
asset/object.urdf
mesh/visual_mesh.obj
mesh/collision_mesh.obj
renders/
report/
```

## Canonical Pose Output

The current best pose output is the sparse bread multi-view result:

```text
outputs/mask_pose/bread/weigh_bread__2026_0701_0044_30/object_trajectory_multiview_pose.json
```

This is not GT and not RGB-D ICP. It is an approximate pose from calibrated
multi-view masks and the reconstructed mesh.

## Rebuild

Run:

```bash
/home/ruan/miniconda3/envs/objasset/bin/python \
  scripts/prepare_object3_submission_evidence.py \
  --python /home/ruan/miniconda3/envs/objasset/bin/python
```

This regenerates this folder and updates `runs/object_asset_v1/`.

## Caveats

- Primary-object GT pose masks are all false in the provided HDF5 files.
- No raw depth frames are currently available.
- Local IsaacGym validation fails in this environment because the legacy
  `isaacgym` module is missing; rerun on the cluster IsaacGym Python 3.8 setup.
- Pipette/drink masks are currently too large for reliable pose tracking.
