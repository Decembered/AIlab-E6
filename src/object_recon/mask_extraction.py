"""Object 2D mask extraction from video frames.

Uses SAM2 or similar foundation model for zero-shot object segmentation.
"""

from pathlib import Path
from typing import Optional

import numpy as np


def extract_masks_sam2(
    frames_dir: Path,
    output_dir: Path,
    object_prompt: str = "object",
    device: str = "cuda",
) -> dict:
    """Extract per-frame object masks using SAM2.

    Args:
        frames_dir: Directory containing video frames (per camera).
        output_dir: Directory to save mask images.
        object_prompt: Text prompt or point/box prompt for the object.
        device: 'cuda' or 'cpu'.

    Returns:
        dict with keys: mask_paths, mask_definition, num_frames, method
    """
    # TODO: Implement SAM2 mask extraction
    raise NotImplementedError("SAM2 mask extraction not yet implemented")


def extract_masks_sam(
    frames_dir: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    """Extract per-frame object masks using SAM1 + prompt.

    Args:
        frames_dir: Directory containing video frames.
        output_dir: Directory to save mask images.
        device: 'cuda' or 'cpu'.

    Returns:
        dict with keys: mask_paths, mask_definition, num_frames, method
    """
    # TODO: Implement SAM mask extraction
    raise NotImplementedError("SAM mask extraction not yet implemented")


def masks_from_3dgs(
    gaussian_splat: "GaussianModel",
    camera_views: list,
    output_dir: Path,
) -> dict:
    """Extract object masks by rendering 3DGS alpha and thresholding.

    This is a fallback method when SAM is unavailable or performs poorly.

    Args:
        gaussian_splat: Trained 3DGS model.
        camera_views: List of camera parameters.
        output_dir: Directory to save mask images.

    Returns:
        dict with keys: mask_paths, mask_definition, num_frames, method
    """
    # TODO: Implement 3DGS-based mask extraction
    raise NotImplementedError("3DGS mask extraction not yet implemented")
