"""Phase 2 v4.1: Micro-adjustment — restore bread bottom edge after v4 over-shrunk.

Changes from v4:
- Box bottom: 380→395 (moderate relax, not back to 405)
- Dangerous negative [545,390] → [535,405] (moved away from bread bottom)
- Added positive [595,360] to anchor bread lower region
- All other prompts unchanged
"""
import os, json
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

FRAMES_DIR = 'experiments/2026-07-04_obj_recon_bread/frames'
OUT_DIR = 'experiments/2026-07-04_obj_recon_bread/masks'
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_PATH = os.path.expanduser('~/.cache/sam/sam_vit_b_01ec64.pth')
MODEL_TYPE = 'vit_b'
DEVICE = 'cpu'

# v4.1: v4 positives + one extra at bread lower region
POSITIVE_POINTS = np.array([
    [555, 300],
    [590, 305],
    [625, 300],
    [640, 325],
    [600, 345],
    [595, 360],   # NEW: anchor bread bottom
])
POSITIVE_LABELS = np.ones(len(POSITIVE_POINTS), dtype=int)

# v4.1: moved dangerous [545,390] → [535,405] (safer, further from bread)
NEGATIVE_POINTS = np.array([
    [500, 350],
    [500, 390],
    [535, 405],   # CHANGED from [545,390] — moved away from bread bottom edge
    [560, 285],
    [585, 320],
    [680, 340],
])
NEGATIVE_LABELS = np.zeros(len(NEGATIVE_POINTS), dtype=int)

# v4.1: Box bottom 380→395 (moderate relax, between v3=405 and v4=380)
BBOX = np.array([515, 250, 670, 395])
USE_BOX = True

print("=" * 60)
print("Phase 2 v4.1: Restore Bread Bottom Edge")
print("=" * 60)

# --- Load SAM ---
print(f"Loading SAM {MODEL_TYPE} on {DEVICE}...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CKPT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

cam = 'camera_top'
frame_path = f'{FRAMES_DIR}/{cam}/frame_000115.jpg'
image = cv2.imread(frame_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"Image: {image.shape}")
predictor.set_image(image_rgb)

# Combine prompts
all_points = np.vstack([POSITIVE_POINTS, NEGATIVE_POINTS])
all_labels = np.concatenate([POSITIVE_LABELS, NEGATIVE_LABELS])

print(f"Positive ({len(POSITIVE_POINTS)}): {POSITIVE_POINTS.tolist()}")
print(f"Negative ({len(NEGATIVE_POINTS)}): {NEGATIVE_POINTS.tolist()}")
print(f"Box: {BBOX.tolist()}")

# Run SAM
masks, scores, _ = predictor.predict(
    point_coords=all_points,
    point_labels=all_labels,
    box=BBOX[None, :],
    multimask_output=True,
)
best_idx = np.argmax(scores)
mask_raw = masks[best_idx]
print(f"SAM scores: {[f'{s:.4f}' for s in scores]}")
print(f"Selected: idx={best_idx}, score={scores[best_idx]:.4f}")
print(f"Raw mask: {mask_raw.sum()} px ({mask_raw.mean():.1%})")

# --- Post-processing: keep only main component near positive centroid ---
num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
    mask_raw.astype(np.uint8), connectivity=8
)
pos_centroid = POSITIVE_POINTS.mean(axis=0)
best_comp, best_dist = 0, float('inf')

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    cx, cy = centroids[i]
    dist = np.sqrt((cx - pos_centroid[0])**2 + (cy - pos_centroid[1])**2)
    if area > 1000 and dist < 300:
        if dist < best_dist:
            best_dist = dist
            best_comp = i

mask_refined = (labels_im == best_comp) if best_comp > 0 else np.zeros_like(mask_raw, dtype=bool)
refined_px = mask_refined.sum()
refined_pct = mask_refined.mean()

print(f"\nPost-processing:")
print(f"  Components: {num_labels - 1}")
print(f"  Positive centroid: ({pos_centroid[0]:.0f}, {pos_centroid[1]:.0f})")
print(f"  Selected: comp {best_comp}, dist={best_dist:.0f}px, area={refined_px}px ({refined_pct:.1%})")

# --- Save mask ---
mask_img = (mask_refined * 255).astype(np.uint8)
cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_mask_sam_v41.png', mask_img)

# --- Overlay ---
overlay = image.copy()
overlay[mask_refined] = (overlay[mask_refined] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

if USE_BOX:
    cv2.rectangle(overlay, (BBOX[0], BBOX[1]), (BBOX[2], BBOX[3]), (0, 255, 255), 2)

for pt in POSITIVE_POINTS:
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 7, (255, 0, 0), -1)
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 1)

for pt in NEGATIVE_POINTS:
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 7, (0, 0, 255), -1)
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 1)

y0 = 25
for text, color in [
    ("GREEN  = Bread Mask (v4.1)", (0, 255, 0)),
    ("BLUE   = Positive prompts (bread)", (255, 0, 0)),
    ("RED    = Negative prompts (exclude)", (0, 0, 255)),
    ("YELLOW = Box prompt", (0, 255, 255)),
]:
    cv2.putText(overlay, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    y0 += 20

overlay_path = f'{OUT_DIR}/{cam}_frame_000115_overlay_sam_v41.jpg'
cv2.imwrite(overlay_path, overlay)
print(f"\nSaved: {overlay_path}")

# --- Comparison: v3 | v4 | v4.1 ---
v3_path = f'{OUT_DIR}/{cam}_frame_000115_overlay_sam_refined.jpg'
v4_path = f'{OUT_DIR}/{cam}_frame_000115_overlay_sam_v4.jpg'
images = []
labels = []
for label, path in [("v3", v3_path), ("v4", v4_path), ("v4.1", overlay_path)]:
    if os.path.exists(path):
        img = cv2.imread(path)
        h_target = min(img.shape[0], 400)  # crop for compact comparison
        images.append(img[:h_target])
        labels.append(label)

if len(images) == 3:
    comp = np.hstack(images)
    x = 10
    for lbl in labels:
        cv2.putText(comp, lbl, (x, comp.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        x += images[0].shape[1]
    comp_path = f'{OUT_DIR}/{cam}_frame_000115_compare_v3_v4_v41.jpg'
    cv2.imwrite(comp_path, comp)
    print(f"Comparison: {comp_path}")

# --- Metadata ---
prev_metrics = {
    "v3_px": 20291, "v3_score": 0.8893,
    "v4_px": 7455,  "v4_score": 0.8758,
}
metadata = {
    "version": "v4.1",
    "changes_from_v4": {
        "box_bottom": "380→395 (moderate relax)",
        "negative_fix": "[545,390]→[535,405] (away from bread bottom)",
        "positive_added": "[595,360] (anchor bread lower region)",
    },
    "positive_points": POSITIVE_POINTS.tolist(),
    "negative_points": NEGATIVE_POINTS.tolist(),
    "box": BBOX.tolist(),
    "sam_score": float(scores[best_idx]),
    "mask_pixels_refined": int(refined_px),
    "mask_ratio_refined": float(refined_pct),
    "components_found": int(num_labels - 1),
    "selected_component": int(best_comp),
    "vs_v4_delta": int(refined_px) - prev_metrics["v4_px"],
    "comparison": prev_metrics,
}
with open(f'{OUT_DIR}/mask_metadata_sam_v41.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved: {OUT_DIR}/mask_metadata_sam_v41.json")
print(f"\nProgress: v3={prev_metrics['v3_px']}px → v4={prev_metrics['v4_px']}px → v4.1={refined_px}px")
print(f"Target range: 1.0%-1.5% (~9k-14k px)")
if 0.01 <= refined_pct <= 0.015:
    print("IN TARGET RANGE ✓")
elif refined_pct < 0.01:
    print("Still below target — may need box bottom → 400")
else:
    print("Above target — check for background leakage")
print("Done!")
