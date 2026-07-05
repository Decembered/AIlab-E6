# Geometric Consistency Summary

*Generated: 2026-07-05T00:00:00Z

| Object | Vertices | Faces | Collision Faces | Extents (m) | Target (m) | Deviation |
|--------|----------|-------|-----------------|-------------|------------|-----------|
| bread | 2882 | 5760 | 320 | 0.120×0.070×0.040 | 0.120×0.070×0.040 | 0%/0%/0% |
| pipette | 962 | 1920 | 400 | 0.258×0.020×0.085 | 0.258×0.020×0.085 | 0%/0%/0% |
| drink_ad | 898 | 1792 | 384 | 0.070×0.070×0.200 | 0.070×0.070×0.200 | 0%/0%/0% |
| drink_yykx | 898 | 1792 | 384 | 0.070×0.070×0.200 | 0.070×0.070×0.200 | 0%/0%/0% |

### Key Fixes (2026-07-05)
- Replaced generic contour extrusion with high-precision object-specific parametric reconstruction.
- Bread: multi-frame top-mask footprint plus rounded loaf dome; width now matches 0.070 m target.
- Pipette: tapered elliptical body, thin tip, and plunger/body proportions; width now fixed at 0.020 m target.
- Drink bottles: segmented body/shoulder/neck/cap profiles; `drink_ad` uses round cross-section and `drink_yykx` uses rounded-square cross-section.
- Collision meshes are separately simplified to 320-400 faces while staying watertight for IsaacGym contact efficiency.
