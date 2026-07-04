"""Phase 2 v4: Further refined SAM mask — tighten bread coverage, suppress bottom-left background.

Changes from v3:
- Positive points shifted up/right onto bread texture (away from edges/shadows)
- Negative points strengthened in bottom-left background area
- Box bottom edge tightened (405→380) to exclude background below bread
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

# v4: Positive points shifted up/right onto orange-yellow bread texture
POSITIVE_POINTS = np.array([
    [555, 300],   # upper-left of bread
    [590, 305],   # upper-center
    [625, 300],   # upper-right
    [640, 325],   # right edge
    [600, 345],   # center of bread
])
POSITIVE_LABELS = np.ones(len(POSITIVE_POINTS), dtype=int)

# v4: Strengthened bottom-left background suppression + hand + scale
NEGATIVE_POINTS = np.array([
    [500, 350],   # left background
    [500, 390],   # bottom-left background (NEW)
    [545, 390],   # below bread background (NEW)
    [560, 285],   # hand
    [585, 320],   # hand/occlusion
    [680, 340],   # electronic scale
])
NEGATIVE_LABELS = np.zeros(len(NEGATIVE_POINTS), dtype=int)

# v4: Tightened box, especially bottom edge (405→380)
BBOX = np.array([515, 250, 670, 380])
USE_BOX = True

print("=" * 60)
print("Phase 2 v4: Refined SAM Mask — Tighter Bread Coverage")
print("=" * 60)

# --- Load SAM ---
print(f"Loading SAM {MODEL_TYPE} on {DEVICE}...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CKPT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# --- Process frame ---
cam = 'camera_top'
frame_path = f'{FRAMES_DIR}/{cam}/frame_000115.jpg'
image = cv2.imread(frame_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]
print(f"Image: {image.shape}")

predictor.set_image(image_rgb)

# Combine prompts
all_points = np.vstack([POSITIVE_POINTS, NEGATIVE_POINTS])
all_labels = np.concatenate([POSITIVE_LABELS, NEGATIVE_LABELS])

print(f"Positive ({len(POSITIVE_POINTS)}): {POSITIVE_POINTS.tolist()}")
print(f"Negative ({len(NEGATIVE_POINTS)}): {NEGATIVE_POINTS.tolist()}")
print(f"Box: {BBOX.tolist()} (bottom tightened to {BBOX[3]})")

# Run SAM
if USE_BOX:
    masks, scores, logits = predictor.predict(
        point_coords=all_points,
        point_labels=all_labels,
        box=BBOX[None, :],
        multimask_output=True,
    )
else:
    masks, scores, _ = predictor.predict(
        point_coords=all_points,
        point_labels=all_labels,
        multimask_output=True,
    )

best_idx = np.argmax(scores)
mask_raw = masks[best_idx]
print(f"\nSAM scores: {[f'{s:.4f}' for s in scores]}")
print(f"Selected: idx={best_idx}, score={scores[best_idx]:.4f}")
print(f"Raw mask: {mask_raw.sum()} px ({mask_raw.mean():.1%})")

# --- Post-processing: keep component nearest to positive centroid ---
num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
    mask_raw.astype(np.uint8), connectivity=8
)

pos_centroid = POSITIVE_POINTS.mean(axis=0)
best_comp = 0
best_dist = float('inf')

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    cx, cy = centroids[i]
    dist = np.sqrt((cx - pos_centroid[0])**2 + (cy - pos_centroid[1])**2)
    if area > 1000 and dist < 300:
        if dist < best_dist:
            best_dist = dist
            best_comp = i

mask_refined = np.zeros_like(mask_raw, dtype=bool)
if best_comp > 0:
    mask_refined = (labels_im == best_comp)

refined_px = mask_refined.sum()
refined_pct = mask_refined.mean()
print(f"\nPost-processing:")
print(f"  Components: {num_labels - 1}")
print(f"  Positive centroid: ({pos_centroid[0]:.0f}, {pos_centroid[1]:.0f})")
print(f"  Selected: comp {best_comp}, dist={best_dist:.0f}px, area={refined_px}px ({refined_pct:.1%})")

# --- Save mask ---
mask_img = (mask_refined * 255).astype(np.uint8)
cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_mask_sam_v4.png', mask_img)

# --- Visualization overlay ---
overlay = image.copy()

# Green overlay for bread mask
overlay[mask_refined] = (overlay[mask_refined] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

# Yellow box
if USE_BOX:
    cv2.rectangle(overlay, (BBOX[0], BBOX[1]), (BBOX[2], BBOX[3]), (0, 255, 255), 2)

# Blue positive points
for pt in POSITIVE_POINTS:
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 7, (255, 0, 0), -1)
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 1)

# Red negative points
for pt in NEGATIVE_POINTS:
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 7, (0, 0, 255), -1)
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 1)

# Legend
y0 = 25
for text, color in [
    ("GREEN  = Bread Mask (v4)", (0, 255, 0)),
    ("BLUE   = Positive prompts (bread)", (255, 0, 0)),
    ("RED    = Negative prompts (exclude)", (0, 0, 255)),
    ("YELLOW = Box prompt", (0, 255, 255)),
]:
    cv2.putText(overlay, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    y0 += 22

# Version tag
cv2.putText(overlay, "v4", (w - 60, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Save
overlay_path = f'{OUT_DIR}/{cam}_frame_000115_overlay_sam_v4.jpg'
cv2.imwrite(overlay_path, overlay)
print(f"\nSaved: {overlay_path}")

# --- Comparison with v3 ---
v3_overlay_path = f'{OUT_DIR}/{cam}_frame_000115_overlay_sam_refined.jpg'
if os.path.exists(v3_overlay_path):
    # Create side-by-side comparison
    v3_img = cv2.imread(v3_overlay_path)
    h_crop = min(overlay.shape[0], v3_img.shape[0])
    comparison = np.hstack([v3_img[:h_crop], overlay[:h_crop]])
    cv2.putText(comparison, "v3", (10, h_crop - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(comparison, "v4", (overlay.shape[1] + 10, h_crop - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    comp_path = f'{OUT_DIR}/{cam}_frame_000115_compare_v3_v4.jpg'
    cv2.imwrite(comp_path, comparison)
    print(f"Comparison: {comp_path}")

# --- Metadata ---
v3_px = 20291  # from v3 log
metadata = {
    "version": "v4",
    "changes_from_v3": {
        "positive_shifted": "up/right onto bread texture (avg y: 315 vs 356)",
        "negative_strengthened": "added [500,390] and [545,390] for bottom-left background",
        "box_tightened": "bottom edge 405→380",
        "postprocess": "same: keep component nearest positive centroid",
    },
    "positive_points": POSITIVE_POINTS.tolist(),
    "negative_points": NEGATIVE_POINTS.tolist(),
    "box": BBOX.tolist(),
    "sam_score": float(scores[best_idx]),
    "mask_pixels_raw": int(mask_raw.sum()),
    "mask_pixels_refined": int(refined_px),
    "mask_ratio_refined": float(refined_pct),
    "components_found": int(num_labels - 1),
    "selected_component": int(best_comp),
    "v3_v4_pixel_change": int(refined_px) - v3_px,
}
with open(f'{OUT_DIR}/mask_metadata_sam_v4.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved: {OUT_DIR}/mask_metadata_sam_v4.json")
print(f"\nv3 pixels: {v3_px} → v4 pixels: {refined_px} (delta: {int(refined_px) - v3_px:+d})")
print(f"Done!")
