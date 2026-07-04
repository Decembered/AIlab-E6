"""Phase 2: Object 2D mask extraction using OpenCV GrabCut.

No PyTorch/SAM required. GrabCut is built into OpenCV.
Bread on scale is a well-defined foreground object — GrabCut works well here.
"""
import os, json
import numpy as np
import cv2

FRAMES_DIR = 'experiments/2026-07-04_obj_recon_bread/frames'
OUT_DIR = 'experiments/2026-07-04_obj_recon_bread/masks'
os.makedirs(OUT_DIR, exist_ok=True)

MASK_DEFINITION = (
    "Visible-region mask: pixels belonging to the object in the current frame. "
    "Generated using OpenCV GrabCut with center-region initialization. "
    "Occluded portions are not hallucinated."
)

def grabcut_mask(image, center_frac=0.4, border_frac=0.1):
    """Run GrabCut with center-as-probable-foreground initialization."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Probable foreground: center rectangle
    ch, cw = int(h * center_frac), int(w * center_frac)
    cx, cy = (w - cw) // 2, (h - ch) // 2
    rect = (cx, cy, cw, ch)

    # Run GrabCut
    mask, _, _ = cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5,
                              cv2.GC_INIT_WITH_RECT)
    # mask: 0=BG, 1=FG, 2=PROB_BG, 3=PROB_FG
    result = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    return result, rect

# Process all three cameras at the middle frame
results = {}
for cam in ['camera_side_1', 'camera_side_2', 'camera_top']:
    frame_path = f'{FRAMES_DIR}/{cam}/frame_000115.jpg'
    image = cv2.imread(frame_path)
    print(f"\nProcessing {cam}...")
    print(f"  Image: {image.shape}")

    # Run GrabCut
    mask, rect = grabcut_mask(image)
    n_fg = (mask > 0).sum()
    print(f"  Foreground pixels: {n_fg} ({n_fg / mask.size:.1%})")
    print(f"  Rectangle: {rect}")

    # Save mask
    cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_mask.png', mask)

    # Create overlay visualization
    overlay = image.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255, 0, 0), 2)
    cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_overlay.jpg', overlay)

    results[cam] = {
        "mask_pixels": int(n_fg),
        "mask_ratio": float(n_fg / mask.size),
        "rect": rect,
    }

# Also process first and last frames for trajectory visualization
for cam in ['camera_top']:
    for frame_name in ['frame_000000', 'frame_000230']:
        frame_path = f'{FRAMES_DIR}/{cam}/{frame_name}.jpg'
        if not os.path.exists(frame_path):
            continue
        image = cv2.imread(frame_path)
        mask, rect = grabcut_mask(image)
        cv2.imwrite(f'{OUT_DIR}/{cam}_{frame_name}_mask.png', mask)
        overlay = image.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        cv2.imwrite(f'{OUT_DIR}/{cam}_{frame_name}_overlay.jpg', overlay)
        print(f"\n  {cam}/{frame_name}: {int(mask.sum())} fg pixels")

# Save metadata
metadata = {
    "mask_definition": MASK_DEFINITION,
    "method": "OpenCV GrabCut, GC_INIT_WITH_RECT, 5 iterations, center-rect init",
    "results": results,
}
with open(f'{OUT_DIR}/mask_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*60}")
print(f"Masks saved to: {OUT_DIR}/")
print(f"Files: {sorted(os.listdir(OUT_DIR))}")
print(f"\nMask definition: {MASK_DEFINITION}")
print("Done! Phase 2 complete.")
