#!/usr/bin/env python3
"""Phase 2 v4: SAM mask for pipette — wider but flat box for elongated object."""
import sys
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage

from segment_anything import sam_model_registry, SamPredictor

EXP_DIR = Path('/home/ruan/research/Hackthon/experiments/2026-07-05_obj_recon_pipette')
FRAME_PATH = EXP_DIR / 'frames' / 'camera_side_1' / 'frame_000000.jpg'
DEBUG_DIR = EXP_DIR / 'mask_debug'
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

SAM_CHECKPOINT = Path('/home/ruan/.cache/sam/sam_vit_b_01ec64.pth')


def analyze_mask(mask, label=""):
    """Print mask diagnostics."""
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n+1))
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    ar = (xmax - xmin) / max(1, ymax - ymin)
    print(f"  {label}: {mask.sum()}px, {n} components, "
          f"bbox ({xmin},{ymin})-({xmax},{ymax}), "
          f"{xmax-xmin}x{ymax-ymin}, AR={ar:.1f}")


def main():
    print("Loading SAM...")
    sam = sam_model_registry['vit_b'](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device='cpu')
    predictor = SamPredictor(sam)

    frame_bgr = cv2.imread(str(FRAME_PATH))
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(frame)

    # Pipette analysis:
    # - Long thin object, should span horizontally across the center
    # - White/light blue body with darker grip
    # - v1 was too wide (14% area), v3 too narrow (fragmented)
    #
    # Strategy v4: wide box with tight vertical bounds
    # The pipette body is at ~y=340-380 (from pixel analysis of center region)
    # It spans from ~x=300 to x=900

    runs = []

    # v4a: wide horizontal band, center region
    pos_4a = np.array([[640, 360], [480, 355], [800, 365]])
    neg_4a = np.array([[640, 300], [640, 430], [300, 355], [980, 370],
                       [640, 500], [640, 600], [200, 600], [1080, 600]])
    box_4a = np.array([350, 300, 930, 420])

    masks_4a, scores_4a, _ = predictor.predict(
        point_coords=np.vstack([pos_4a, neg_4a]),
        point_labels=np.array([1]*len(pos_4a) + [0]*len(neg_4a)),
        box=box_4a[None, :],
        multimask_output=True,
    )
    best = masks_4a[np.argmax(scores_4a)]
    analyze_mask(best, "v4a")

    # Keep only largest connected component
    labeled, n = ndimage.label(best)
    sizes = ndimage.sum(best, labeled, range(1, n+1))
    best_clean = labeled == (np.argmax(sizes) + 1)
    analyze_mask(best_clean, "v4a clean")

    # Check aspect ratio of clean mask
    rows = np.any(best_clean, axis=1)
    cols = np.any(best_clean, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    ar = (xmax - xmin) / max(1, ymax - ymin)
    print(f"  v4a clean AR: {ar:.1f}")

    if ar >= 3.0:
        print("  ✅ v4a is elongated — looks like a pipette!")
        best_mask = best_clean
    else:
        # v4b: longer but same height band
        print("\n  Trying v4b: wider horizontal span...")
        pos_4b = np.array([[640, 360], [400, 350], [880, 370],
                           [520, 355], [760, 365]])
        neg_4b = np.array([[640, 290], [640, 440], [640, 520],
                           [250, 350], [1030, 370]])
        box_4b = np.array([250, 280, 1030, 430])

        masks_4b, scores_4b, _ = predictor.predict(
            point_coords=np.vstack([pos_4b, neg_4b]),
            point_labels=np.array([1]*len(pos_4b) + [0]*len(neg_4b)),
            box=box_4b[None, :],
            multimask_output=True,
        )
        best_b = masks_4b[np.argmax(scores_4b)]
        analyze_mask(best_b, "v4b")

        # Clean
        labeled_b, n_b = ndimage.label(best_b)
        sizes_b = ndimage.sum(best_b, labeled_b, range(1, n_b+1))
        best_b_clean = labeled_b == (np.argmax(sizes_b) + 1)
        analyze_mask(best_b_clean, "v4b clean")

        # Check AR
        rows_b = np.any(best_b_clean, axis=1)
        cols_b = np.any(best_b_clean, axis=0)
        ymin_b, ymax_b = np.where(rows_b)[0][[0, -1]]
        xmin_b, xmax_b = np.where(cols_b)[0][[0, -1]]
        ar_b = (xmax_b - xmin_b) / max(1, ymax_b - ymin_b)
        print(f"  v4b clean AR: {ar_b:.1f}")

        if ar_b >= 3.0:
            print("  ✅ v4b is elongated!")
            best_mask = best_b_clean
        else:
            # Just use v4a clean, it's the best we have
            best_mask = best_clean
            print(f"  ⚠️ Using v4a clean (AR={ar:.1f})")

    # Save best mask
    out_path = DEBUG_DIR / 'pipette_mask_best.npy'
    np.save(out_path, best_mask)
    cv2.imwrite(str(DEBUG_DIR / 'pipette_mask_best.png'),
               (best_mask.astype(np.uint8) * 255))

    # Final summary
    rows = np.any(best_mask, axis=1)
    cols = np.any(best_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    print(f"\n=== Final mask ===")
    print(f"  Area: {best_mask.sum()}px ({best_mask.sum()/(720*1280)*100:.1f}%)")
    print(f"  BBox: ({xmin},{ymin})-({xmax},{ymax}), {xmax-xmin}x{ymax-ymin}")
    print(f"  Aspect ratio: {(xmax-xmin)/(ymax-ymin):.1f}")


if __name__ == '__main__':
    main()
