#!/usr/bin/env python3
"""Phase 2: Multi-view SAM mask extraction for pipette.

Extract masks from all 3 camera views, then combine for 3D reconstruction.
Each view constrains one profile of the pipette:
  - camera_side_1: length (X) vs height (Y)
  - camera_side_2: width (Z) vs height (Y)
  - camera_top: length (X) vs width (Z)
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage

from segment_anything import sam_model_registry, SamPredictor

EXP_DIR = Path('/home/ruan/research/Hackthon/experiments/2026-07-05_obj_recon_pipette')
FRAMES_DIR = EXP_DIR / 'frames'
DEBUG_DIR = EXP_DIR / 'mask_debug'
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

SAM_CHECKPOINT = Path('/home/ruan/.cache/sam/sam_vit_b_01ec64.pth')

# Target area: pipette is ~15-20cm long, ~2-3cm wide
# At camera distance, should be ~1-2% of frame
AREA_MIN, AREA_MAX = 5000, 30000


def load_sam():
    sam = sam_model_registry['vit_b'](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device='cpu')
    return SamPredictor(sam)


def extract_mask(predictor, frame, pos, neg, box, name):
    """Extract and clean SAM mask."""
    all_pts = np.vstack([pos, neg])
    all_labels = np.array([1]*len(pos) + [0]*len(neg))

    masks, scores, _ = predictor.predict(
        point_coords=all_pts,
        point_labels=all_labels,
        box=box[None, :],
        multimask_output=True,
    )

    best = masks[np.argmax(scores)]
    score = float(scores[np.argmax(scores)])

    # Keep only largest connected component
    labeled, n = ndimage.label(best)
    if n > 0:
        sizes = ndimage.sum(best, labeled, range(1, n+1))
        best = labeled == (np.argmax(sizes) + 1)

    # Get stats
    rows = np.any(best, axis=1)
    cols = np.any(best, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    area = best.sum()
    ar = (xmax - xmin) / max(1, ymax - ymin)

    print(f"  {name}: area={area}px ({area/(frame.shape[0]*frame.shape[1])*100:.1f}%), "
          f"bbox ({xmin},{ymin})-({xmax},{ymax}), {xmax-xmin}x{ymax-ymin}, AR={ar:.1f}, score={score:.3f}")

    return best, score, area


def main():
    print("Loading SAM...")
    predictor = load_sam()

    # ===== Frame 0 from each camera =====
    views = {}
    for cam in ['camera_side_1', 'camera_side_2', 'camera_top']:
        frame_path = FRAMES_DIR / cam / 'frame_000000.jpg'
        if frame_path.exists():
            bgr = cv2.imread(str(frame_path))
            views[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            print(f"{cam}: {views[cam].shape}")

    masks = {}

    # ===== camera_side_1: side profile (length × height) =====
    print("\n=== camera_side_1: side profile ===")
    predictor.set_image(views['camera_side_1'])

    # Pipette is horizontal-ish, white body with blue accents
    # Focus on the central white elongated region
    pos_s1 = np.array([
        [640, 360],    # center of body
        [480, 350],    # left end (grip)
        [800, 370],    # right end (tip)
        [560, 355],    # body midpoint left
        [720, 365],    # body midpoint right
    ])
    neg_s1 = np.array([
        [640, 280],    # above pipette
        [640, 460],    # below pipette (hand area)
        [640, 550],    # far below (hand)
    ])
    box_s1 = np.array([300, 280, 980, 440])

    mask_s1 = None
    for attempt in range(3):
        if attempt == 0:
            p, n, b = pos_s1, neg_s1, box_s1
        elif attempt == 1:
            # Tighter
            p = np.array([[640, 360], [500, 348], [780, 368]])
            n = np.array([[640, 280], [640, 450], [500, 440], [780, 440]])
            b = np.array([350, 290, 930, 430])
        else:
            # Even tighter
            p = np.array([[640, 355], [520, 345], [760, 365]])
            n = np.array([[640, 290], [640, 430]])
            b = np.array([380, 300, 900, 420])

        mask, score, area = extract_mask(predictor, views['camera_side_1'],
                                          p, n, b, f"side1_v{attempt+1}")
        if mask_s1 is None or (AREA_MIN <= area <= AREA_MAX):
            mask_s1 = mask
        if AREA_MIN <= area <= AREA_MAX:
            break

    masks['camera_side_1'] = mask_s1

    # ===== camera_side_2: other side (width × height) =====
    print("\n=== camera_side_2: other side profile ===")
    predictor.set_image(views['camera_side_2'])

    pos_s2 = np.array([[640, 360], [500, 350], [780, 370]])
    neg_s2 = np.array([[640, 280], [640, 460], [640, 550]])
    box_s2 = np.array([300, 280, 980, 440])

    mask_s2, score_s2, area_s2 = extract_mask(predictor, views['camera_side_2'],
                                                pos_s2, neg_s2, box_s2, "side2")
    masks['camera_side_2'] = mask_s2

    # ===== camera_top: top-down view (length × width) =====
    print("\n=== camera_top: top-down view ===")
    predictor.set_image(views['camera_top'])

    pos_top = np.array([[640, 360], [500, 355], [780, 365]])
    neg_top = np.array([[640, 200], [640, 550], [200, 360], [1080, 360]])
    box_top = np.array([250, 220, 1030, 500])

    mask_top, score_top, area_top = extract_mask(predictor, views['camera_top'],
                                                   pos_top, neg_top, box_top, "top")
    masks['camera_top'] = mask_top

    # ===== Save all masks =====
    print("\n=== Saving masks ===")
    for cam, mask in masks.items():
        if mask is not None:
            np.save(DEBUG_DIR / f'mask_{cam}.npy', mask)
            cv2.imwrite(str(DEBUG_DIR / f'mask_{cam}.png'),
                       (mask.astype(np.uint8) * 255))
            print(f"  {cam}: {mask.sum()}px saved")

    # ===== Multi-view consistency check =====
    print("\n=== Multi-view consistency ===")
    # Check: side_1 and side_2 should have similar height
    if masks['camera_side_1'] is not None and masks['camera_side_2'] is not None:
        h1 = np.any(masks['camera_side_1'], axis=1).sum()
        h2 = np.any(masks['camera_side_2'], axis=1).sum()
        print(f"  Height side1={h1}px, side2={h2}px, diff={abs(h1-h2)/max(h1,h2)*100:.1f}%")

    # Check: side_1 and top should have similar length
    if masks['camera_side_1'] is not None and masks['camera_top'] is not None:
        w1 = np.any(masks['camera_side_1'], axis=0).sum()
        w_top = np.any(masks['camera_top'], axis=0).sum()
        print(f"  Width side1={w1}px, top={w_top}px, diff={abs(w1-w_top)/max(w1,w_top)*100:.1f}%")

    print("\nDone! Multi-view masks ready for 3D reconstruction.")


if __name__ == '__main__':
    main()
