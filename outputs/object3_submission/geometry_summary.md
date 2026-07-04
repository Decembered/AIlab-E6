# Geometric Consistency Summary

*Generated: 2026-07-05T06:42:24.837154

## Method

Multi-view mask contour extrusion: XY footprint from top camera mask, height estimated from side camera masks. Fine polygon approximation with multi-slice extrusion for smooth vertical surfaces.

## Per-Object Geometry

| Object | Vertices | Faces | Watertight | Extents (m) | Mass (kg) |
|--------|----------|-------|------------|-------------|----------|
| bread | 296 | 588 | True | 0.120×0.063×0.040 | 0.0775 |
| pipette | 1600 | 3196 | True | 0.258×0.104×0.085 | 1.2116 |
| drink_ad | 232 | 460 | True | 0.070×0.118×0.200 | 1.4695 |
| drink_yykx | 144 | 284 | True | 0.070×0.066×0.200 | 0.8571 |

## Scale Validation

| Object | Reconstructed (m) | Expected (m) | Deviation |
|--------|-------------------|-------------|----------|
| bread | 0.120×0.063×0.040 | 0.120×0.070×0.040 | 0%/10%/0% |
| pipette | 0.258×0.104×0.085 | 0.258×0.020×0.085 | 0%/422%/0% |
| drink_ad | 0.070×0.118×0.200 | 0.070×0.070×0.200 | 0%/69%/0% |
| drink_yykx | 0.070×0.066×0.200 | 0.070×0.070×0.200 | 0%/5%/0% |

## Limitations

- XY footprint from top-camera mask; fine details may be simplified
- Height corrected to known real-world object dimensions
- Rotationally symmetric objects (drinks) have simplified geometry
- No texture or color information preserved
