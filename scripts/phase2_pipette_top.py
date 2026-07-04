#!/usr/bin/env python3
"""Phase 2: SAM mask for pipette from TOP view — should be clearly elongated."""
import sys
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage

from segment_anything import sam_model_registry, SamPredictor

EXP_DIR = Path('/home/ruan/research/Hackthon/experiments/2026-07-05_obj_recon_pipette')
FRAME_PATH = EXP_DIR / 'frames' / 'camera_top' / 'frame_000000.jpg'
DEBUG_DIR = EXP_DIR / 'mask_debug'
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
SAM_CHECKPOINT = Path('/home/ruan/.cache/sam/sam_vit_b_01ec64.pth')


def analyze(mask, label):
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n+1))
    if n == 0:
        print(f"  {label}: EMPTY")
        return mask, 0, 0
    # Keep largest component
    largest = labeled == (np.argmax(sizes) + 1)
    rows = np.any(largest, axis=1)
    cols = np.any(largest, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    ar = (xmax - xmin) / max(1, ymax - ymin)
    print(f"  {label}: {largest.sum()}px, {n} comps, "
          f"bbox ({xmin},{ymin})-({xmax},{ymax}), {xmax-xmin}x{ymax-ymin}, AR={ar:.1f}")
    return largest, xmax-xmin, ymax-ymin


def main():
    sam = sam_model_registry['vit_b'](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device='cpu')
    predictor = SamPredictor(sam)

    frame_bgr = cv2.imread(str(FRAME_PATH))
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    print(f"Top frame: {w}x{h}")
    predictor.set_image(frame)

    # Top view: pipette should be a long thin rectangle or line
    # It typically rests on a table, seen from above
    # Should be clearly elongated

    # v1: wide box, center of frame
    print("\n--- Top v1: wide center ---")
    pos1 = np.array([[640, 360], [500, 350], [780, 350]])
    neg1 = np.array([[640, 100], [640, 600], [100, 360], [1100, 360],
                     [100, 100], [1100, 600]])
    box1 = np.array([200, 200, 1080, 520])

    masks1, scores1, _ = predictor.predict(
        point_coords=np.vstack([pos1, neg1]),
        point_labels=np.array([1]*len(pos1) + [0]*len(neg1)),
        box=box1[None, :],
        multimask_output=True,
    )
    best1 = masks1[np.argmax(scores1)]
    m1, w1, h1 = analyze(best1, "top_v1")
    ar1 = w1 / max(1, h1)

    # v2: try different positive points if v1 not elongated
    print("\n--- Top v2: refined points ---")
    # If pipette is more left-right oriented
    pos2 = np.array([[640, 360], [400, 350], [880, 350],
                     [550, 355], [720, 355]])
    neg2 = np.array([[640, 200], [640, 520], [640, 600],
                     [200, 350], [1080, 350]])
    box2 = np.array([150, 220, 1130, 500])

    masks2, scores2, _ = predictor.predict(
        point_coords=np.vstack([pos2, neg2]),
        point_labels=np.array([1]*len(pos2) + [0]*len(neg2)),
        box=box2[None, :],
        multimask_output=True,
    )
    best2 = masks2[np.argmax(scores2)]
    m2, w2, h2 = analyze(best2, "top_v2")
    ar2 = w2 / max(1, h2)

    # v3: try top-right diagonal orientation
    print("\n--- Top v3: diagonal ---")
    pos3 = np.array([[500, 400], [600, 350], [700, 300], [800, 250]])
    neg3 = np.array([[640, 100], [640, 600], [200, 360], [1080, 360]])
    box3 = np.array([250, 150, 1050, 500])

    masks3, scores3, _ = predictor.predict(
        point_coords=np.vstack([pos3, neg3]),
        point_labels=np.array([1]*len(pos3) + [0]*len(neg3)),
        box=box3[None, :],
        multimask_output=True,
    )
    best3 = masks3[np.argmax(scores3)]
    m3, w3, h3 = analyze(best3, "top_v3")
    ar3 = w3 / max(1, h3)

    # Pick best (most elongated with reasonable area)
    candidates = [
        (m1, ar1, w1, h1, "top_v1"),
        (m2, ar2, w2, h2, "top_v2"),
        (m3, ar3, w3, h3, "top_v3"),
    ]
    valid = [(m, ar, w, h, n) for m, ar, w, h, n in candidates if m is not None and m.sum() > 1000]

    if valid:
        # Prefer elongated
        best = max(valid, key=lambda x: x[1] if x[1] >= 3 else 0)
        if best[1] < 2:
            # Just pick by area
            best = max(valid, key=lambda x: x[0].sum())

        best_mask, best_ar, bw, bh, bname = best
        print(f"\n=== Best: {bname} ===")
        print(f"  Area: {best_mask.sum()}px, AR: {best_ar:.1f}, {bw}x{bh}")
        np.save(DEBUG_DIR / 'pipette_mask_best.npy', best_mask)
        cv2.imwrite(str(DEBUG_DIR / 'pipette_mask_best.png'),
                   (best_mask.astype(np.uint8) * 255))

        # Save debug images
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for m, ar, w, h, name in candidates:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.imshow(frame)
            ax1.imshow(np.ma.masked_where(~m, np.ones_like(m)), cmap='Greens', alpha=0.4)
            ax1.set_title(f'{name} | {m.sum()}px | AR={ar:.1f}')
            ax1.axis('off')
            ax2.imshow(m, cmap='gray')
            ax2.set_title(f'Mask only')
            ax2.axis('off')
            plt.tight_layout()
            plt.savefig(DEBUG_DIR / f'{name}.jpg', dpi=100)
            plt.close()
            print(f"  Debug saved: {name}.jpg")
    else:
        print("No valid masks found!")


if __name__ == '__main__':
    main()
