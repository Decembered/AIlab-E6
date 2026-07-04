# 2026-07-04 Object Reconstruction — Bread

## Goal

Reconstruct 3D model of bread from `weigh_bread__2026_0701_0044_30` sequence and generate IsaacGym-ready asset.

## Setup

- Time: 2026-07-04T17:00+08
- Python: 3.12.9
- CUDA: CPU-only (PyTorch 2.12.1+cpu, RTX 4060 CUDA 13 bus error)
- GPU: RTX 4060 Laptop (8GB) — GPU not usable due to driver/CUDA mismatch
- Sequence: weigh_bread__2026_0701_0044_30

## Video Properties

| Camera | Resolution | FPS | Frames | Duration |
|--------|-----------|-----|--------|----------|
| side_1 | 1280x720 | 15.0 | 235 | 15.7s |
| side_2 | 1280x720 | 15.0 | 235 | 15.7s |
| top | 1280x720 | 15.0 | 235 | 15.7s |

## Camera Calibration

All 3 cameras have valid 3x3 intrinsics and 4x4 extrinsics.

Camera world positions:
- side_1: (0.512, 0.906, 0.481)
- side_2: (-0.536, 0.913, 0.421)
- top: (-0.011, 1.198, -0.348)

## Frames Extracted

141 frames total (47 per camera, every 5th frame)

---

## Phase 2: 2D Mask — Iteration History

### v1: GrabCut (FAILED) — [phase2_mask_grabcut.py](../../scripts/phase2_mask_grabcut.py)

**Method:** OpenCV GrabCut with center-rect initialization (GC_INIT_WITH_RECT, 5 iterations).

**Result:** Severe over-segmentation. Mask covered entire work surface (green mat + hand + bread),
~100k foreground pixels but essentially the whole table area.

**Failure reason:**
1. Bread color is similar to the green work mat — GrabCut cannot semantically distinguish them
2. Hand above bread was included in mask (hand-bread adhesion)
3. Mask boundary did not follow bread edges — affected by color/brightness, not semantic target
4. Scale area partially included, creating mixed foreground

**Conclusion:** Classical CV segmentation cannot solve this. Requires semantic segmentation (SAM).

### v2: SAM with center prompt (PARTIAL FAILURE) — [phase2_sam_mask.py](../../scripts/phase2_sam_mask.py)

**Method:** SAM vit_b on CPU, 3 positive point prompts at [520,340], [560,320], [480,360].
Score: 0.984, 157,994 fg pixels (17.1%).

**Result:** Better than GrabCut but prompt points were placed on the green mat area instead of
the bread surface. SAM segmented the green work surface, not the bread.

**Failure reason:**
1. Prompt points at [520,340], [560,320], [480,360] were on green mat, not bread
2. Bread is in the upper-left area of the center region, points missed it
3. No negative prompts to exclude the mat and hand
4. No box prompt to constrain the region

### v3: SAM with corrected prompts (CURRENT) — [phase2_sam_refined.py](../../scripts/phase2_sam_refined.py)

**Method:** SAM vit_b on CPU with:
- 4 positive points on bread surface: [545,375], [585,385], [620,365], [640,300]
- 5 negative points on green mat/hand/scale: [500,350], [470,390], [560,290], [585,320], [680,340]
- Tight box prompt: [520, 255, 665, 405]
- Post-processing: keep only connected component nearest to positive centroid

| Metric | v1 GrabCut | v2 SAM (bad prompts) | v3 SAM (corrected) |
|--------|-----------|---------------------|-------------------|
| Foreground pixels | ~100k (over-seg) | 157,994 (17.1%) | **20,291 (2.2%)** |
| SAM score | N/A | 0.984 | 0.889 |
| Components | N/A | N/A | 7 → kept 1 (37px from centroid) |
| Green mat excluded? | ❌ | ❌ | ✅ |
| Hand excluded? | ❌ | ❌ | ✅ |
| Scale excluded? | ❌ | ❌ | ✅ |

**Key improvements in v3:**
1. Positive points placed ON the bread visible surface (not the mat)
2. Negative points explicitly exclude green mat, hand, and electronic scale
3. Tight bounding box constrains SAM's search region
4. Connected-component post-processing removes stray mat pixels

**Output files:**
- `masks/camera_top_frame_000115_overlay_sam_refined.jpg` — visualization with color-coded prompts
- `masks/camera_top_frame_000115_mask_sam_refined.png` — binary mask

### v4: Over-shrunk (CRITICAL FIX) — [phase2_sam_v4.py](../../scripts/phase2_sam_v4.py)

**Method:** Shifted positives up/right onto bread texture, added bottom-left negative points, tightened box bottom to 380.

**Result:** 7,455 px (0.8%). Background clean but over-shrunk — bread bottom and lower-right clipped.

**Failure reason:** Box bottom too tight (380), negative point [545,390] too close to bread bottom edge.

### v4.1: Restore bottom edge (FINAL for Phases 3-5) — [phase2_sam_v41.py](../../scripts/phase2_sam_v41.py)

**Method:** Micro-adjustments from v4 — box bottom 380→395, moved dangerous negative [545,390]→[535,405], added positive [595,360] to anchor bread lower region.

**Result:** 12,030 px (1.3%), SAM score 0.843. Hits target range 1.0-1.5%. Best balance of completeness and cleanliness.

**Output files:**
- `masks/camera_top_frame_000115_mask_sam_v41.png` — binary mask
- `masks/camera_top_frame_000115_overlay_sam_v41.jpg` — visualization
- `masks/camera_top_frame_000115_compare_v3_v4_v41.jpg` — 3-version comparison

---

## Phase 3-5: Re-run with v4.1 Mask

### Phase 3: 3D Reconstruction

| Metric | GrabCut (old) | SAM v4.1 (new) |
|--------|--------------|----------------|
| Contour | 220→16 pts (over-simplified) | **388→90 pts** |
| Vertices | 34 | **182** |
| Faces | 64 | **360** |
| Dimensions | 15×10×8 cm | 15×10×8 cm |

Output: `models/bread_v41.obj`

### Phase 4: IsaacGym Asset

- Mass: 240g (200 kg/m³ bread density)
- Inertia: (0.000328, 0.000650, 0.000578)
- Output: `models/bread_v41.urdf`

### Phase 5: Pose Tracking

**Method:** SAM v4.1 on 7 key frames, linear interpolation for remaining 40 frames.

**Known issue:** Fixed-coordinate SAM prompts don't adapt to object motion.
Frames 30, 40 showed inflated masks (18-21k px) with large orientation jumps (47-57°),
likely because prompts hit background/hand at those timestamps.

**Mitigation:** Trajectory smoothed with 3-frame moving average.
Total displacement: 2.1 cm (bread essentially stationary during weighing).

Output: `object_trajectory_v41.npz`

## Next Steps

- [x] SAM mask iterated to v4.1 (best balance)
- [x] Phase 3-5 re-run with v4.1 mask
- [ ] Phase 6: Validate URDF on cluster with IsaacGym
- [ ] Replace fixed-coordinate SAM tracking with FoundationPose or optical flow
- [ ] Apply same pipeline to pipette, drink AD, drink YYKX objects
