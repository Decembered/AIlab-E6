#!/usr/bin/env python3
"""Phase 2: SAM mask extraction for pipette — iterative prompt refinement."""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

EXP_DIR = Path('/home/ruan/research/Hackthon/experiments/2026-07-05_obj_recon_pipette')
FRAME_PATH = EXP_DIR / 'frames' / 'camera_side_1' / 'frame_000000.jpg'
DEBUG_DIR = EXP_DIR / 'mask_debug'
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

SAM_CHECKPOINT = Path('/home/ruan/.cache/sam/sam_vit_b_01ec64.pth')
DEVICE = 'cpu'

# Target area range for pipette: thin elongated object
AREA_MIN, AREA_MAX = 3000, 15000


def load_sam():
    sam = sam_model_registry['vit_b'](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=DEVICE)
    return SamPredictor(sam)


def show_mask(mask, ax, color, alpha=0.5):
    h, w = mask.shape[-2:]
    mask_img = np.zeros((h, w, 4))
    mask_img[mask] = np.array(color + [alpha])
    ax.imshow(mask_img)


def save_debug(frame, masks, scores, points, box, name):
    """Save debug visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax in axes:
        ax.imshow(frame)
        ax.axis('off')

    if masks is not None and len(masks) > 0:
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]

        show_mask(best_mask, axes[0], [0.2, 0.8, 0.2], alpha=0.4)

        mask_display = np.zeros((*best_mask.shape, 3), dtype=np.uint8)
        mask_display[best_mask] = [100, 200, 100]
        axes[1].imshow(mask_display)

        area_px = best_mask.sum()
        axes[0].set_title(f'{name} | Area: {area_px}px ({area_px/(frame.shape[0]*frame.shape[1])*100:.1f}%) | Score: {best_score:.3f}', fontsize=11)
        axes[1].set_title(f'Mask {area_px} px', fontsize=11)
    else:
        axes[0].set_title(f'{name} | NO MASK', fontsize=11)

    # Draw points
    if points is not None:
        all_pts, all_labels = points
        pos_pts = all_pts[all_labels == 1] if len(all_pts) > 0 else np.array([]).reshape(0, 2)
        neg_pts = all_pts[all_labels == 0] if len(all_pts) > 0 else np.array([]).reshape(0, 2)
        if len(pos_pts) > 0:
            axes[0].scatter(pos_pts[:, 0], pos_pts[:, 1], c='lime', s=80, marker='*',
                          edgecolors='white', linewidths=1)
        if len(neg_pts) > 0:
            axes[0].scatter(neg_pts[:, 0], neg_pts[:, 1], c='red', s=60, marker='x', linewidths=2)

    # Draw box
    if box is not None:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='yellow', linewidth=2)
        axes[0].add_patch(rect)

    out_path = DEBUG_DIR / f'{name}.jpg'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path.name}')

    if masks is not None and len(masks) > 0:
        return masks[np.argmax(scores)], float(scores[np.argmax(scores)])
    return None, 0.0


def run_variant(predictor, frame, pos, neg, box, name):
    """Run one SAM variant and save results."""
    all_pts = np.vstack([pos, neg])
    all_labels = np.array([1]*len(pos) + [0]*len(neg))
    masks, scores, _ = predictor.predict(
        point_coords=all_pts,
        point_labels=all_labels,
        box=box[None, :],
        multimask_output=True,
    )
    best_mask, best_score = save_debug(frame, masks, scores,
                                        (all_pts, all_labels), box, name)
    if best_mask is not None:
        area = best_mask.sum()
        print(f'  {name}: area={area}px ({area/(frame.shape[0]*frame.shape[1])*100:.1f}%), score={best_score:.3f}')
        if AREA_MIN <= area <= AREA_MAX:
            print(f'  ✅ {name} in target range!')
        return best_mask, best_score, area
    return None, 0.0, 0


def main():
    if not SAM_CHECKPOINT.exists():
        print(f"ERROR: SAM checkpoint not found at {SAM_CHECKPOINT}")
        sys.exit(1)

    print(f"Loading SAM vit_b on {DEVICE}...")
    predictor = load_sam()
    print("SAM loaded.")

    frame_bgr = cv2.imread(str(FRAME_PATH))
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")
    predictor.set_image(frame)

    # ====== v1: initial guess (wide coverage) ======
    print("\n--- v1: initial wide coverage ---")
    mask_v1, score_v1, area_v1 = run_variant(
        predictor, frame,
        pos=np.array([[640, 360], [500, 340], [780, 380], [640, 340], [640, 380]]),
        neg=np.array([[100, 360], [1100, 360], [640, 100], [640, 600],
                      [100, 100], [1100, 100], [100, 600], [1100, 600]]),
        box=np.array([400, 250, 900, 480]),
        name="pipette_v1_initial"
    )
    if mask_v1 is not None:
        np.save(DEBUG_DIR / 'pipette_mask_v1.npy', mask_v1)

    # ====== v2: tighter box, more negatives near pipette ======
    print("\n--- v2: tighter box, narrow focus ---")
    mask_v2, score_v2, area_v2 = run_variant(
        predictor, frame,
        pos=np.array([[640, 360], [550, 350], [730, 370]]),
        neg=np.array([[640, 280], [640, 440], [400, 300], [880, 300],
                      [400, 420], [880, 420], [600, 500], [700, 500],
                      [200, 360], [1080, 360]]),
        box=np.array([420, 290, 860, 410]),
        name="pipette_v2_tighter"
    )

    # ====== v3: ultra-narrow box ======
    print("\n--- v3: ultra-narrow box ---")
    mask_v3, score_v3, area_v3 = run_variant(
        predictor, frame,
        pos=np.array([[640, 355], [560, 348], [720, 362]]),
        neg=np.array([[640, 310], [640, 400], [500, 340], [780, 380], [640, 500]]),
        box=np.array([430, 310, 850, 390]),
        name="pipette_v3_narrow"
    )

    # ====== Pick best ======
    print("\n=== Summary ===")
    best = None
    for name, mask, score, area in [
        ("v1", mask_v1, score_v1, area_v1),
        ("v2", mask_v2, score_v2, area_v2),
        ("v3", mask_v3, score_v3, area_v3),
    ]:
        if mask is not None:
            in_range = "✅" if AREA_MIN <= area <= AREA_MAX else "⚠️"
            print(f"  {name}: area={area}px, score={score:.3f} {in_range}")
            if AREA_MIN <= area <= AREA_MAX:
                if best is None or score > best[1]:
                    best = (mask, score, name)

    if best is None:
        # Pick the one closest to range
        candidates = [(mask_v1, score_v1, area_v1, "v1"),
                      (mask_v2, score_v2, area_v2, "v2"),
                      (mask_v3, score_v3, area_v3, "v3")]
        valid = [(m, s, a, n) for m, s, a, n in candidates if m is not None]
        if valid:
            best = min(valid, key=lambda x: min(abs(x[2]-AREA_MIN), abs(x[2]-AREA_MAX)))
            best = (best[0], best[1], best[3])
            print(f"  No mask in target range. Using closest: {best[2]} (area={valid[0][2]})")

    if best:
        mask_best, score_best, name_best = best
        np.save(DEBUG_DIR / 'pipette_mask_best.npy', mask_best)
        cv2.imwrite(str(DEBUG_DIR / 'pipette_mask_best.png'),
                   (mask_best.astype(np.uint8) * 255))
        print(f"\nBest mask: {name_best} (score={score_best:.3f}, area={mask_best.sum()}px)")
        print(f"Saved: pipette_mask_best.npy")


if __name__ == '__main__':
    main()
