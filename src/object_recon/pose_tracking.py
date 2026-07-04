"""Object pose trajectory recovery.

Methods:
1. GT registration — if accurate object trajectory is provided, register model to GT
2. Feature-based tracking — FoundationPose, feature matching + PnP
3. 3DGS-based tracking — track object in 3DGS reconstruction
"""

from pathlib import Path
from typing import Optional

import numpy as np


def register_to_gt(
    reconstructed_mesh: Path,
    gt_trajectory: Path,
    output_dir: Path,
) -> dict:
    """Register reconstructed object model to ground-truth trajectory.

    Args:
        reconstructed_mesh: Path to .obj/.ply mesh.
        gt_trajectory: Path to GT object pose data.
        output_dir: Output directory.

    Returns:
        dict with keys: poses (Nx4x4), timestamps, registration_error
    """
    # TODO: Implement GT registration
    raise NotImplementedError("GT registration not yet implemented")


def track_foundationpose(
    frames_dir: Path,
    object_mesh: Path,
    camera_params: Path,
    output_dir: Path,
) -> dict:
    """Track object pose using FoundationPose or similar.

    Args:
        frames_dir: Video frames.
        object_mesh: Reconstructed object mesh.
        camera_params: Camera intrinsics/extrinsics.
        output_dir: Output directory.

    Returns:
        dict with keys: poses (Nx4x4), timestamps, confidence
    """
    # TODO: Implement FoundationPose tracking
    raise NotImplementedError("FoundationPose tracking not yet implemented")


def track_from_3dgs(
    gaussian_splat: "GaussianModel",
    camera_views: list,
    output_dir: Path,
) -> dict:
    """Extract object pose trajectory from 3DGS reconstruction.

    If 3DGS was used for reconstruction, we can recover the object transform
    directly from the training process or by rendering and matching.

    Args:
        gaussian_splat: Trained 3DGS model.
        camera_views: Camera parameters.
        output_dir: Output directory.

    Returns:
        dict with keys: poses (Nx4x4), timestamps, method
    """
    # TODO: Implement 3DGS-based pose tracking
    raise NotImplementedError("3DGS pose tracking not yet implemented")


def poses_to_trajectory(
    poses: np.ndarray,
    timestamps: np.ndarray,
    output_path: Path,
    format: str = "npy",
) -> Path:
    """Save pose trajectory in specified format.

    Args:
        poses: (N, 4, 4) homogeneous transformation matrices.
        timestamps: (N,) timestamps.
        output_path: Output file path.
        format: 'npy' or 'csv' or 'json'.

    Returns:
        Path to saved trajectory file.
    """
    data = {"poses": poses, "timestamps": timestamps}
    if format == "npy":
        np.save(output_path, data)
    elif format == "json":
        import json
        with open(output_path, "w") as f:
            json.dump({
                "poses": poses.tolist(),
                "timestamps": timestamps.tolist(),
            }, f, indent=2)
    elif format == "csv":
        # Flatten to (N, 16) for CSV
        flat = poses.reshape(len(poses), 16)
        np.savetxt(output_path, np.hstack([timestamps[:, None], flat]), delimiter=",")
    return output_path
