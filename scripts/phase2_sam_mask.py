"""Phase 2 v2: Bread mask extraction using SAM with point prompt.

User clicks one point on the bread, SAM segments it accurately.
This replaces the failed GrabCut approach.
"""
import os, sys, json
import numpy as np
import cv2

FRAMES_DIR = 'experiments/2026-07-04_obj_recon_bread/frames'
OUT_DIR = 'experiments/2026-07-04_obj_recon_bread/masks'
os.makedirs(OUT_DIR, exist_ok=True)

# Ensure SAM is available
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Installing SAM...")
    os.system(f"{sys.executable} -m pip install segment-anything git+https://github.com/facebookresearch/segment-anything.git -q")
    import torch
    from segment_anything import sam_model_registry, SamPredictor

print("=" * 60)
print("Phase 2 v2: SAM-based Bread Mask Extraction")
print("=" * 60)

# --- Download SAM checkpoint ---
CHECKPOINT_DIR = os.path.expanduser('~/.cache/sam')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Use vit_b (lightweight, faster to download, ~370MB)
CKPT_PATH = f'{CHECKPOINT_DIR}/sam_vit_b_01ec64.pth'
MODEL_TYPE = 'vit_b'

if not os.path.exists(CKPT_PATH):
    print(f"Downloading SAM {MODEL_TYPE} checkpoint...")
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    ret = os.system(f"wget -q --show-progress -O {CKPT_PATH} {url}")
    if ret != 0:
        # Try mirror
        ret = os.system(f"wget -q --show-progress -O {CKPT_PATH} https://hf-mirror.com/facebook/sam-vit-base/resolve/main/sam_vit_b_01ec64.pth")
    if ret != 0:
        print("ERROR: Could not download SAM checkpoint. Please download manually.")
        sys.exit(1)

print(f"Checkpoint: {CKPT_PATH}")

# --- Load SAM ---
print("Loading SAM...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CKPT_PATH)
sam.to(device=device)
predictor = SamPredictor(sam)

# --- Process top camera middle frame with point prompt ---
# The bread is in the center-left of the top camera view.
# Use a point prompt at the approximate bread center.
cam = 'camera_top'
frame_path = f'{FRAMES_DIR}/{cam}/frame_000115.jpg'
print(f"\nProcessing: {frame_path}")
image = cv2.imread(frame_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image_rgb)

# Point prompt: click on the bread
# In the top view, the bread is roughly at (520, 340) — center-left area
# We'll try a few points and pick the best mask
input_points = np.array([
    [520, 340],   # Center of bread
    [560, 320],   # Slightly right
    [480, 360],   # Slightly left
])
input_labels = np.array([1, 1, 1])  # All foreground points

print(f"  Input points: {input_points.tolist()}")

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)

best_idx = np.argmax(scores)
mask = masks[best_idx]
print(f"  Mask scores: {scores}, selected: {best_idx}")
print(f"  Mask pixels: {mask.sum()} ({mask.mean():.1%})")

# Save
mask_img = (mask * 255).astype(np.uint8)
cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_mask_sam.png', mask_img)

# Overlay
overlay = image.copy()
overlay[mask] = (overlay[mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
# Draw prompt points
for pt in input_points:
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 8, (255, 0, 0), -1)
cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_overlay_sam.jpg', overlay)

print(f"  Saved: {OUT_DIR}/{cam}_frame_000115_mask_sam.png")
print(f"  Saved: {OUT_DIR}/{cam}_frame_000115_overlay_sam.jpg")

# --- Also process first and last frames with same point ---
for fname in ['frame_000000', 'frame_000230']:
    frame_path = f'{FRAMES_DIR}/{cam}/{fname}.jpg'
    if not os.path.exists(frame_path):
        continue
    image = cv2.imread(frame_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=input_points[:1],  # Use just the center point for other frames
        point_labels=np.array([1]),
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    cv2.imwrite(f'{OUT_DIR}/{cam}_{fname}_mask_sam.png', (mask * 255).astype(np.uint8))
    overlay = image.copy()
    overlay[mask] = (overlay[mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    cv2.imwrite(f'{OUT_DIR}/{cam}_{fname}_overlay_sam.jpg', overlay)
    print(f"  {fname}: {mask.sum()} fg pixels ({mask.mean():.1%})")

# --- Now process side cameras ---
# Side camera bread position is different — use approximate points
side_points = {
    'camera_side_1': np.array([[500, 380]]),  # Center-ish
    'camera_side_2': np.array([[500, 380]]),  # Center-ish
}

for cam in ['camera_side_1', 'camera_side_2']:
    frame_path = f'{FRAMES_DIR}/{cam}/frame_000115.jpg'
    if not os.path.exists(frame_path):
        continue
    image = cv2.imread(frame_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    pts = side_points[cam]
    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=np.ones(len(pts)),
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_mask_sam.png', (mask * 255).astype(np.uint8))
    overlay = image.copy()
    overlay[mask] = (overlay[mask] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    cv2.imwrite(f'{OUT_DIR}/{cam}_frame_000115_overlay_sam.jpg', overlay)
    print(f"  {cam}: {mask.sum()} fg pixels ({mask.mean():.1%})")

# --- Update mask metadata ---
metadata = {
    "mask_definition": "Visible-region mask: SAM vit_b with point prompt on bread center",
    "method": "SAM vit_b, point prompt, multimask_output=True",
    "prompt_points": input_points.tolist(),
    "note": "Replaces GrabCut which failed due to bread/table color similarity",
}
with open(f'{OUT_DIR}/mask_metadata_sam.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone! SAM masks saved to {OUT_DIR}/")
print("Files with _sam suffix are the improved masks.")
