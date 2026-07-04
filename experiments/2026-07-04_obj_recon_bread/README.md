# Bread Asset IsaacGym Validation

## Platform

- **Environment**: Aliyun PAI DSW
- **GPU**: 2 x NVIDIA GeForce RTX 4090
- **Driver**: 570.153.02 / CUDA 12.8
- **OS**: Ubuntu 24.04.4 LTS

## Software Stack

- **IsaacGym**: Preview 4 / isaacgym 1.0rc4
- **Python**: 3.8.20
- **Torch**: 1.13.1+cu117
- **Physics**: GPU PhysX (cuda:0)

## Asset

| File | Description |
|------|-------------|
| `models/bread.urdf` | URDF model definition |
| `models/bread_extruded.obj` | Extruded mesh (VHACD 14 hulls) |
| `models/asset_metadata.json` | Asset metadata |
| `models/quality_report.json` | Quality report |

## Cluster IsaacGym Validation

### Load Test — PASS

- URDF exists: True
- Mesh exists: True
- IsaacGym load_asset: OK
- create_actor: OK
- Simulation steps: 120
- VHACD convex decomposition: 14 hulls

### Drop Stability Test — PASS

- Initial height: z = 0.5 m
- Simulation steps: 240
- Final position: x = -0.0595, y = -0.00049, z = 0.06684
- Result flag: `BREAD_DROP_TEST_OK`

## Validation Logs

See `cluster_validation/` for full logs:

- `bread_asset_summary.md` — Summary report
- `bread_asset_load.txt` — Asset loading log
- `bread_drop_test.txt` — Drop stability test log
- `isaacgym_import.txt` — IsaacGym import verification
- `isaacgym_headless_smoke.txt` — Headless smoke test
- `joint_monkey_headless.txt` — Viewer joint monkey test
- `python38_packages.txt` — Python package listing
- `find_team_assets.txt` — Team asset discovery log

## Limitation

Smoke test only. Does not yet verify:
- Physical interaction quality with Sharpa hand
- Real-world scale accuracy
- Hand-object contact stability
- Tracking task success
