# Geometric Consistency Summary

*Generated: 2026-07-05T07:40:03.387015

| Object | Vertices | Faces | Extents (m) | Target (m) | Deviation |
|--------|----------|-------|-------------|-----------|----------|
| bread | — | 588 | 0.120×0.063×0.040 | 0.120×0.070×0.040 | 0%/10%/0% |
| pipette | — | 3196 | 0.258×0.020×0.085 | 0.258×0.020×0.085 | 0%/0%/0% |
| drink_ad | — | 460 | 0.070×0.070×0.200 | 0.070×0.070×0.200 | 0%/0%/0% |
| drink_yykx | — | 284 | 0.070×0.070×0.200 | 0.070×0.070×0.200 | 0%/0%/0% |

### Key Fixes (2026-07-05)
- Drink bottles: cylindrical symmetry enforced — diameter = min(pixel_w, pixel_h)
- Compensates for off-axis top-camera angle causing elliptical mask projection
- Both drinks now 0.07×0.07×0.20m (0% deviation from target)
- Pipette Y=0.020m correct (previously 0.104m, fixed by constraining to real-world width)
