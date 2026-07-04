"""Object 2D mask extraction from video frames.

Uses SAM1 for zero-shot object segmentation with prompt-based masks.
SAM2 and 3DGS-based extraction are planned upgrades.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import json


def extract_masks_sam(
    frames_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    model_type: str = "vit_b",
    checkpoint_path: Optional[Path] = None,
    positive_points: Optional[List[Tuple[int, int]]] = None,
    negative_points: Optional[List[Tuple[int, int]]] = None,
    prompt_box: Optional[Tuple[int, int, int, int]] = None,
    crop_n_layers: int = 0,
) -> dict:
    """Extract per-frame object masks using SAM1 + explicit prompt.

    Prompts can be boxes, positive/negative points, or both.

    Args:
        frames_dir: Directory containing video frames (.png or .jpg).
        output_dir: Directory to save mask images (.png).
        device: 'cuda' or 'cpu'.
        model_type: 'vit_h', 'vit_l', or 'vit_b'.
        checkpoint_path: Path to SAM checkpoint. Auto-downloads if None.
        positive_points: List of (x, y) positive prompt coordinates.
        negative_points: List of (x, y) negative prompt coordinates.
        prompt_box: (x1, y1, x2, y2) bounding box prompt.
        crop_n_layers: SAM crop layers (0 = no cropping).

    Returns:
        dict with keys:
          - mask_paths: list of Path to saved mask images
          - mask_definition: str describing mask type
          - num_frames: int
          - method: str
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        import torch
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError(
            "segment-anything required. Install: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    if checkpoint_path is None:
        from urllib.request import urlretrieve
        ckpt_dir = Path.home() / ".cache/sam"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / f"sam_{model_type}_01ec64.pth"
        if not checkpoint_path.exists():
            url = f"https://dl.fbaipublicfiles.com/segment_anything/sam_{model_type}_01ec64.pth"
            urlretrieve(url, checkpoint_path)

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {frames_dir}")

    mask_paths = []
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        input_points = []
        input_labels = []
        if positive_points:
            input_points.extend(positive_points)
            input_labels.extend([1] * len(positive_points))
        if negative_points:
            input_points.extend(negative_points)
            input_labels.extend([0] * len(negative_points))

        input_point = np.array(input_points) if input_points else None
        input_label = np.array(input_labels) if input_labels else None
        box = np.array(prompt_box) if prompt_box else None

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=box,
            multimask_output=False,
        )

        mask = masks[0].astype(np.uint8) * 255
        out_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(out_path), mask)
        mask_paths.append(out_path)

    return {
        "mask_paths": [str(p) for p in mask_paths],
        "mask_definition": "binary visible-region instance mask (SAM1, box+point prompts)",
        "num_frames": len(mask_paths),
        "method": f"SAM {model_type}",
    }


def extract_masks_sam2(
    frames_dir: Path,
    output_dir: Path,
    object_prompt: str = "object",
    device: str = "cuda",
) -> dict:
    """Extract per-frame object masks using SAM2 (video segmentation).

    SAM2 extends SAM to video by considering images as a video with a
    single frame. Uses streaming memory for real-time processing.

    NOTE: This is a planned upgrade. Currently routes to SAM1 with
    generic prompt for backward compatibility.

    Args:
        frames_dir: Directory containing video frames.
        output_dir: Directory to save mask images.
        object_prompt: Text prompt for the object.
        device: 'cuda' or 'cpu'.

    Returns:
        dict with keys: mask_paths, mask_definition, num_frames, method
    """
    # SAM2 native video segmentation not yet available in this environment.
    # Falls back to SAM1 with center-box heuristic for now.
    import cv2

    img = cv2.imread(str(frames_dir.glob("*.png").__next__()))
    if img is None:
        img = cv2.imread(str(frames_dir.glob("*.jpg").__next__()))
    h, w = img.shape[:2]
    center_box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

    return extract_masks_sam(
        frames_dir=frames_dir,
        output_dir=output_dir,
        device=device,
        prompt_box=center_box,
    )


def masks_from_3dgs(
    gaussian_splat: object,
    camera_views: list,
    output_dir: Path,
) -> dict:
    """Extract object masks by rendering 3DGS alpha and thresholding.

    This is a fallback method when SAM is unavailable.

    NOTE: Not yet implemented. Requires 3DGS model with renderer.

    Args:
        gaussian_splat: Trained 3DGS model instance.
        camera_views: List of camera parameters.
        output_dir: Directory to save mask images.

    Returns:
        dict with keys: mask_paths, mask_definition, num_frames, method
    """
    raise NotImplementedError(
        "3DGS mask extraction not yet implemented. "
        "Use extract_masks_sam() with explicit prompts instead."
    )
