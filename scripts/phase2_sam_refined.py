"""Phase 2 v3: Refined SAM mask with corrected prompts.

Positive points: on bread surface only
Negative points: green mat, hand, scale
Box prompt: tight bounding box around bread
Post-processing: keep only the largest connected component near positive points
"""
import os, json
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

FRAMES_DIR = 'experiments/2026-07-04_obj_recon_bread/frames'
OUT_DIR = 'experiments/2026-07-04_obj_recon_bread/masks'
os.makedirs(OUT_DIR, exist_ok=True)

# --- Config ---
CKPT_PATH = os.path.expanduser('~/.cache/sam/sam_vit_b_01ec64.pth')
MODEL_TYPE = 'vit_b'
DEVICE = 'cpu'

# Positive points: on bread visible surface
POSITIVE_POINTS = np.array([
    [545, 375],
    [585, 385],
    [620, 365],
    [640, 300],
])
POSITIVE_LABELS = np.ones(len(POSITIVE_POINTS), dtype=int)  # label=1

# Negative points: green mat, hand, scale
NEGATIVE_POINTS = np.array([
    [500, 350],   # green mat
    [470, 390],   # green mat
    [560, 290],   # hand
    [585, 320],   # hand
    [680, 340],   # scale
])
NEGATIVE_LABELS = np.zeros(len(NEGATIVE_POINTS), dtype=int)  # label=0

# Tight bounding box around bread
BBOX = np.array([520, 255, 665, 405])  # [x1, y1, x2, y2]
USE_BOX = True

print("=" * 60)
print("Phase 2 v3: Refined SAM Mask — Bread Only")
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

# Combine all points
all_points = np.vstack([POSITIVE_POINTS, NEGATIVE_POINTS])
all_labels = np.concatenate([POSITIVE_LABELS, NEGATIVE_LABELS])

print(f"Positive points ({len(POSITIVE_POINTS)}): {POSITIVE_POINTS.tolist()}")
print(f"Negative points ({len(NEGATIVE_POINTS)}): {NEGATIVE_POINTS.tolist()}")
if USE_BOX:
    print(f"Box: {BBOX.tolist()}")

# Run SAM
if USE_BOX:
    masks, scores, logits = predictor.predict(
        point_coords=all_points,
        point_labels=all_labels,
        box=BBOX[None, :],
        multimask_output=True,
    )
else:
    masks, scores, logits = predictor.predict(
        point_coords=all_points,
        point_labels=all_labels,
        multimask_output=True,
    )

# Pick best mask
best_idx = np.argmax(scores)
mask = masks[best_idx]
print(f"\nSAM scores: {scores}")
print(f"Selected mask: idx={best_idx}, score={scores[best_idx]:.4f}")
print(f"Raw mask pixels: {mask.sum()} ({mask.mean():.1%})")

# --- Post-processing: keep only largest component near positive points ---
# Step 1: Find connected components
num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
    mask.astype(np.uint8), connectivity=8
)

# Step 2: Find the component closest to the positive point centroid
pos_centroid = POSITIVE_POINTS.mean(axis=0)
best_comp = 0  # 0 = background
best_dist = float('inf')
kept_pixels = 0

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    cx, cy = centroids[i]
    dist = np.sqrt((cx - pos_centroid[0])**2 + (cy - pos_centroid[1])**2)
    # Prefer components that: (a) are near positive points, (b) are reasonably sized
    # Bread at this distance should be ~50k-200k pixels
    if area > 1000 and dist < 300:  # within 300px of positive centroid
        if dist < best_dist:
            best_dist = dist
            best_comp = i
            kept_pixels = area

print(f"\nPost-processing:")
print(f"  Connected components found: {num_labels - 1}")
print(f"  Positive centroid: ({pos_centroid[0]:.0f}, {pos_centroid[1]:.0f})")
print(f"  Selected component {best_comp}: area={kept_pixels}px, dist={best_dist:.0f}px")

# Build refined mask: only the selected component
mask_refined = np.zeros_like(mask, dtype=bool)
if best_comp > 0:
    mask_refined = (labels_im == best_comp)
print(f"  Refined mask pixels: {mask_refined.sum()} ({mask_refined.mean():.1%})")

# --- Save masks ---
mask_img = (mask_refined * 255).astype(np.uint8)
cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_mask_sam_refined.png', mask_img)

# --- Build visualization overlay ---
overlay = image.copy()

# Green overlay for mask
overlay[mask_refined] = (overlay[mask_refined] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

# Draw box
if USE_BOX:
    cv2.rectangle(overlay, (BBOX[0], BBOX[1]), (BBOX[2], BBOX[3]), (255, 255, 0), 2)

# Draw positive points: BLUE filled circles
for pt in POSITIVE_POINTS:
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 7, (255, 0, 0), -1)
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 1)

# Draw negative points: RED filled circles
for pt in NEGATIVE_POINTS:
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 7, (0, 0, 255), -1)
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 1)

# Legend
cv2.putText(overlay, "GREEN = Bread Mask", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.putText(overlay, "BLUE dots = Positive prompts (bread)", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.putText(overlay, "RED dots = Negative prompts (exclude)", (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.putText(overlay, "YELLOW rect = box prompt", (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

overlay_path = f'{OUT_DIR}/{cam}_frame_000115_overlay_sam_refined.jpg'
cv2.imwrite(overlay_path, overlay)
print(f"\nSaved: {overlay_path}")

# --- Save metadata ---
metadata = {
    "method": "SAM vit_b, point+box prompt, connected-component filtering",
    "positive_points": POSITIVE_POINTS.tolist(),
    "negative_points": NEGATIVE_POINTS.tolist(),
    "box": BBOX.tolist(),
    "sam_score": float(scores[best_idx]),
    "mask_pixels_raw": int(mask.sum()),
    "mask_pixels_refined": int(mask_refined.sum()),
    "mask_ratio_refined": float(mask_refined.mean()),
    "components_found": int(num_labels - 1),
    "selected_component": int(best_comp),
    "postprocess": "Keep only connected component nearest to positive point centroid (dist<300px, area>1000px)",
}
with open(f'{OUT_DIR}/mask_metadata_sam_refined.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved: {OUT_DIR}/mask_metadata_sam_refined.json")
print(f"\nDone! Check: {overlay_path}")
