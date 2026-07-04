#!/usr/bin/env python3
"""Phase 2: Multi-view SAM mask extraction — unified for all objects.

Uses all 3 camera views (side_1, side_2, top) with hand-aware negative prompts.
Outputs clean masks ready for multi-view 3D reconstruction.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

SAM_CHECKPOINT = Path('/home/ruan/.cache/sam/sam_vit_b_01ec64.pth')


def load_sam():
    sam = sam_model_registry['vit_b'](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device='cpu')
    return SamPredictor(sam)


def get_skin_mask(frame_rgb):
    """Detect skin-colored pixels for use as negative prompts."""
    r, g, b = frame_rgb[:,:,0].astype(float), frame_rgb[:,:,1].astype(float), frame_rgb[:,:,2].astype(float)
    # Skin detection in RGB
    skin = (r > 95) & (g > 40) & (b > 20) & \
           (r > g) & (r > b) & \
           (np.abs(r - g) > 15) & \
           (r > g + 10) & (r > b + 10)
    return skin


def extract_clean_mask(predictor, frame_rgb, pos, neg, box, name, debug_dir):
    """Extract mask, keep largest component, save debug image."""
    all_pts = np.vstack([pos, neg])
    all_labels = np.array([1]*len(pos) + [0]*len(neg))

    masks, scores, _ = predictor.predict(
        point_coords=all_pts,
        point_labels=all_labels,
        box=box[None, :],
        multimask_output=True,
    )

    best_idx = np.argmax(scores)
    best = masks[best_idx]
    score = float(scores[best_idx])

    # Clean: keep largest component
    labeled, n = ndimage.label(best)
    if n > 0:
        sizes = ndimage.sum(best, labeled, range(1, n+1))
        best = labeled == (np.argmax(sizes) + 1)

    # Stats
    rows = np.any(best, axis=1)
    cols = np.any(best, axis=0)
    if rows.sum() == 0 or cols.sum() == 0:
        print(f"  {name}: EMPTY MASK")
        return None, 0, 0

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    area = best.sum()
    ar = (xmax - xmin) / max(1, ymax - ymin)

    print(f"  {name}: area={area}px ({area/(frame_rgb.shape[0]*frame_rgb.shape[1])*100:.1f}%), "
          f"bbox W={xmax-xmin} H={ymax-ymin}, AR={ar:.1f}, score={score:.3f}")

    # Save debug
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(frame_rgb)

    # Mask overlay
    overlay = np.zeros((*best.shape, 4))
    overlay[best] = [0.2, 0.8, 0.2, 0.4]
    ax1.imshow(overlay)

    # Draw prompts
    ax1.scatter(pos[:, 0], pos[:, 1], c='lime', s=60, marker='*', edgecolors='white', linewidths=0.5)
    ax1.scatter(neg[:, 0], neg[:, 1], c='red', s=40, marker='x', linewidths=1.5)
    if box is not None:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='yellow', linewidth=2)
        ax1.add_patch(rect)
    ax1.set_title(f'{name} | Area={area}px | AR={ar:.1f} | Score={score:.3f}')
    ax1.axis('off')

    ax2.imshow(best, cmap='gray')
    ax2.set_title('Cleaned Mask')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(debug_dir / f'{name}.jpg', dpi=100)
    plt.close()

    return best, score, area


def find_best_frame(video_path, n_samples=10):
    """Sample frames and pick the one with least skin-colored pixels in center."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total // n_samples)

    best_frame, best_skin_ratio = None, 1.0
    best_idx = 0

    for i in range(0, total, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # Check skin ratio in center region
        center = rgb[h//4:3*h//4, w//4:3*w//4]
        skin = get_skin_mask(center)
        skin_ratio = skin.sum() / center.shape[0] / center.shape[1]

        if skin_ratio < best_skin_ratio:
            best_skin_ratio = skin_ratio
            best_frame = rgb
            best_idx = i

    cap.release()
    print(f"  Best frame: {best_idx} (skin ratio: {best_skin_ratio:.2%})")
    return best_frame, best_idx


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase2_multiview_all.py <object_name>")
        print("  object_name: bread | pipette | drink_ad | drink_yykx")
        sys.exit(1)

    obj = sys.argv[1]

    # Map object to sequence
    SEQ_MAP = {
        'bread': 'weigh_bread__2026_0701_0044_30',
        'pipette': 'grasp_pipette_stand__2026_0701_0019_19',
        'drink_ad': 'weigh_drink_ad__2026_0701_0047_56',
        'drink_yykx': 'weigh_drink_yykx__2026_0701_0051_12',
    }

    if obj not in SEQ_MAP:
        print(f"Unknown object: {obj}")
        print(f"Options: {list(SEQ_MAP.keys())}")
        sys.exit(1)

    seq = SEQ_MAP[obj]
    exp_dir = Path(f'/home/ruan/research/Hackthon/experiments/2026-07-05_obj_recon_{obj}')
    video_dir = Path(f'/home/ruan/research/Hackthon/data/human_demo/{seq}/video')
    debug_dir = exp_dir / 'mask_debug'
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Multi-view SAM: {obj} ===")
    print(f"Sequence: {seq}")

    predictor = load_sam()
    masks = {}

    for cam in ['camera_side_1', 'camera_side_2', 'camera_top']:
        video_path = video_dir / f'{cam}.mkv'
        if not video_path.exists():
            print(f"  {cam}: no video")
            continue

        print(f"\n--- {cam} ---")

        # Use frame 0 for all views (object is stable on table at start)
        # Only use adaptive frame selection for pipette (hand-held, need minimal occlusion)
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        if obj == 'pipette':
            frame, frame_idx = find_best_frame(video_path)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame_bgr = cap.read()
            if ret:
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                print(f"  ERROR: Cannot read frame 0")
                continue
        cap.release()

        # Detect skin regions for smart negative points
        skin = get_skin_mask(frame)
        h, w = frame.shape[:2]

        # Generate skin-based negative points (sample from skin regions)
        skin_ys, skin_xs = np.where(skin)
        neg_from_skin = []
        if len(skin_ys) > 0:
            # Sample evenly from skin regions
            indices = np.linspace(0, len(skin_ys)-1, min(6, len(skin_ys)), dtype=int)
            for idx in indices:
                neg_from_skin.append([skin_xs[idx], skin_ys[idx]])

        predictor.set_image(frame)

        # Common negative points: corners + skin samples
        neg_common = np.array([
            [10, 10], [w-10, 10], [10, h-10], [w-10, h-10],  # corners
        ])

        # Object-specific prompts
        if obj == 'bread':
            # Bread: large blob, ~15% of frame
            pos = np.array([
                [int(w*0.50), int(h*0.48)],
                [int(w*0.46), int(h*0.50)],
                [int(w*0.54), int(h*0.50)],
            ])
            neg = np.array([
                [int(w*0.50), int(h*0.25)],
                [int(w*0.50), int(h*0.72)],
                [int(w*0.30), int(h*0.50)],
                [int(w*0.70), int(h*0.50)],
                [int(w*0.15), int(h*0.50)],
                [int(w*0.85), int(h*0.50)],
            ])
            box = np.array([int(w*0.25), int(h*0.28), int(w*0.75), int(h*0.68)])

        elif obj == 'pipette':
            # Pipette: thin elongated, white/light blue, horizontal-ish
            # Centered in frame, spanning ~30-70% width
            cx, cy = w//2, h//2
            pos = np.array([
                [cx, cy],
                [int(cx - w*0.12), int(cy - 5)],
                [int(cx + w*0.12), int(cy + 5)],
                [int(cx - w*0.20), int(cy - 3)],
                [int(cx + w*0.20), int(cy + 3)],
            ])
            neg = np.vstack([
                neg_common,
                np.array([[cx, int(cy - h*0.10)],     # above pipette
                         [cx, int(cy + h*0.15)],       # below (hand)
                         [cx, int(cy + h*0.25)],       # far below
                         [int(cx - w*0.25), cy],
                         [int(cx + w*0.25), cy]]),
            ])
            if len(neg_from_skin) > 0:
                neg = np.vstack([neg, np.array(neg_from_skin)])
            box = np.array([int(w*0.22), int(h*0.38), int(w*0.78), int(h*0.58)])

        elif obj in ('drink_ad', 'drink_yykx'):
            # Drink bottles: tall cylindrical, ~5-10% of frame
            pos = np.array([
                [int(w*0.50), int(h*0.45)],
                [int(w*0.48), int(h*0.30)],
                [int(w*0.52), int(h*0.60)],
                [int(w*0.46), int(h*0.50)],
                [int(w*0.54), int(h*0.50)],
            ])
            neg = np.array([
                [int(w*0.50), int(h*0.15)],
                [int(w*0.50), int(h*0.80)],
                [int(w*0.20), int(h*0.45)],
                [int(w*0.80), int(h*0.45)],
            ])
            box = np.array([int(w*0.30), int(h*0.18), int(w*0.70), int(h*0.72)])

        mask, score, area = extract_clean_mask(
            predictor, frame, pos, neg, box, f"{obj}_{cam}", debug_dir
        )

        if mask is not None:
            masks[cam] = (mask, frame_idx)
            np.save(debug_dir / f'mask_{cam}.npy', mask)
            cv2.imwrite(str(debug_dir / f'mask_{cam}.png'), (mask.astype(np.uint8) * 255))

    # ===== Summary =====
    print(f"\n=== Multi-view Summary: {obj} ===")
    for cam in ['camera_side_1', 'camera_side_2', 'camera_top']:
        if cam in masks:
            mask, fidx = masks[cam]
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            h_span = rows.sum()
            w_span = cols.sum()
            ar = w_span / max(1, h_span)
            print(f"  {cam}: {mask.sum()}px, {w_span}x{h_span}, AR={ar:.1f}, frame={fidx}")
        else:
            print(f"  {cam}: MISSING")

    print("\nDone!")


if __name__ == '__main__':
    main()
