# 2026-07-04 Object Reconstruction — Bread

## Goal

Reconstruct 3D model of bread from `weigh_bread__2026_0701_0044_30` sequence and generate IsaacGym-ready asset.

## Setup

- Time: 2026-07-04T17:00+08
- Python: 3.12.9
- CUDA: TBD
- GPU: RTX 4060 Laptop (8GB)
- Sequence: weigh_bread__2026_0701_0044_30

## Video Properties

| Camera | Resolution | FPS | Frames | Duration |
|--------|-----------|-----|--------|----------|
| side_1 | 1280×720 | 15.0 | 235 | 15.7s |
| side_2 | 1280×720 | 15.0 | 235 | 15.7s |
| top | 1280×720 | 15.0 | 235 | 15.7s |

## Camera Calibration

All 3 cameras have valid 3×3 intrinsics and 4×4 extrinsics.

Camera world positions:
- side_1: (0.512, 0.906, 0.481)
- side_2: (-0.536, 0.913, 0.421)
- top: (-0.011, 1.198, -0.348)

## Frames Extracted

141 frames total (47 per camera, every 5th frame)

## Next Step

Phase 2: SAM2 2D mask extraction
