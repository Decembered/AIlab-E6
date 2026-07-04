"""Phase 2: Object 2D mask extraction using SAM.

Strategy: Use the top-down camera view (best visibility of bread on scale),
manually define a bounding box on the first frame, let SAM propagate.
"""
import sys, os, pickle, json
import numpy as np
import cv2
from pathlib import Path

# Ensure torch + SAM are installed
try:
    import torch
except ImportError:
    print("Installing PyTorch...")
    os.system(f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q")
    import torch

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Installing segment-anything...")
    os.system(f"{sys.executable} -m pip install segment-anything -q")
    from segment_anything import sam_model_registry, SamPredictor

# ============================================================
# Config
# ============================================================
SEQ = 'data/human_demo/weigh_bread__2026_0701_0044_30'
FRAMES_DIR = 'experiments/2026-07-04_obj_recon_bread/frames'
OUT_DIR = 'experiments/2026-07-04_obj_recon_bread/masks'
SAM_CHECKPOINT = os.path.expanduser('~/.cache/sam/sam_vit_h_4b8939.pth')
MODEL_TYPE = "vit_h"  # vit_h (best), vit_l, vit_b (fast)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Download SAM checkpoint if needed
# ============================================================
if not os.path.exists(SAM_CHECKPOINT):
    print(f"SAM checkpoint not found at {SAM_CHECKPOINT}")
    print("Downloading SAM vit_b (lightweight, faster)...")
    SAM_CHECKPOINT = os.path.expanduser('~/.cache/sam/sam_vit_b_01ec64.pth')
    MODEL_TYPE = "vit_b"
    os.makedirs(os.path.dirname(SAM_CHECKPOINT), exist_ok=True)
    if not os.path.exists(SAM_CHECKPOINT):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        print(f"Downloading from {url}...")
        # Try wget first, then urllib
        ret = os.system(f"wget -q -O {SAM_CHECKPOINT} {url}")
        if ret != 0:
            import urllib.request
            urllib.request.urlretrieve(url, SAM_CHECKPOINT)
        print("Download complete.")

# ============================================================
# Load SAM
# ============================================================
print(f"Loading SAM {MODEL_TYPE}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)

# ============================================================
# Define mask scope for each object type (for report)
# ============================================================
MASK_DEFINITION = (
    "Visible-region mask: pixels belonging to the object in the current frame. "
    "Occluded portions are NOT hallucinated. This is a per-frame segmentation of "
    "the visible object surface from each camera view."
)

# ============================================================
# Run SAM on a sample frame with auto-prompt via center box
# ============================================================
# First, let's test on the top camera (best view of bread on scale)
cam = 'camera_top'
sample_frame = f'{FRAMES_DIR}/{cam}/frame_000115.jpg'
print(f"\nProcessing: {sample_frame}")
image = cv2.imread(sample_frame)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb)

# Auto-prompt: use center region as initial guess (the object is usually centered)
h, w = image.shape[:2]
# Bread is likely in the center area — use a generous center box
cx, cy = w // 2, h // 2
box_w, box_h = w // 3, h // 3
input_box = np.array([cx - box_w//2, cy - box_h//2, cx + box_w//2, cy + box_h//2])

print(f"  Input box: {input_box}")
masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=True,
)

# Pick best mask
best_idx = np.argmax(scores)
mask = masks[best_idx]
print(f"  Mask scores: {scores}, selected: {best_idx}")

# Save mask
mask_img = (mask * 255).astype(np.uint8)
cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_mask.png', mask_img)

# Save mask overlay for visualization
overlay = image.copy()
overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
# Draw box
cv2.rectangle(overlay, (int(input_box[0]), int(input_box[1])),
              (int(input_box[2]), int(input_box[3])), (255, 0, 0), 2)
cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_overlay.jpg', overlay)

# ============================================================
# Batch process: apply to all three cameras at the middle frame
# ============================================================
results = {}
for cam in ['camera_side_1', 'camera_side_2', 'camera_top']:
    frame_path = f'{FRAMES_DIR}/{cam}/frame_000115.jpg'
    image = cv2.imread(frame_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    box_w, box_h = w // 3, h // 3
    input_box = np.array([cx - box_w//2, cy - box_h//2, cx + box_w//2, cy + box_h//2])

    masks, scores, _ = predictor.predict(
        box=input_box[None, :],
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_mask.png', (mask * 255).astype(np.uint8))
    overlay = image.copy()
    overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    cv2.rectangle(overlay, (int(input_box[0]), int(input_box[1])),
                  (int(input_box[2]), int(input_box[3])), (255, 0, 0), 2)
    cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_overlay.jpg', overlay)

    results[cam] = {
        "mask_pixels": int(mask.sum()),
        "mask_ratio": float(mask.mean()),
        "score": float(scores[best_idx]),
    }
    print(f"  {cam}: mask={mask.sum()} px ({mask.mean():.1%}), score={scores[best_idx]:.3f}")

# ============================================================
# Save mask metadata
# ============================================================
metadata = {
    "mask_definition": MASK_DEFINITION,
    "method": f"SAM {MODEL_TYPE}, center-box prompt, multimask_output=True",
    "sequence": SEQ,
    "frame": "frame_000115 (middle frame)",
    "results": results,
}
with open(f'{OUT_DIR}/mask_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nMasks saved to: {OUT_DIR}/")
print("Files: ", sorted(os.listdir(OUT_DIR)))
print("\nDone! Phase 2 mask extraction complete.")
