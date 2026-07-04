"""Main object 3D reconstruction pipeline.

Supported approaches:
1. 3D Gaussian Splatting (3DGS) — gsplat / nerfstudio
2. Image-to-3D — InstantMesh, TripoSR, Zero-1-to-3
3. COLMAP + MVS — classical SfM + dense reconstruction
"""

from pathlib import Path
from typing import Optional, Literal

import numpy as np


def reconstruct_3dgs(
    frames_dir: Path,
    camera_params: Path,
    output_dir: Path,
    iterations: int = 7000,
) -> Path:
    """Reconstruct object using 3D Gaussian Splatting.

    Args:
        frames_dir: Multi-view video frames.
        camera_params: Camera intrinsics/extrinsics (COLMAP or given format).
        output_dir: Output directory for model and renders.
        iterations: Training iterations.

    Returns:
        Path to the exported mesh (.ply).
    """
    # TODO: Implement 3DGS reconstruction
    raise NotImplementedError("3DGS reconstruction not yet implemented")


def reconstruct_image_to_3d(
    input_image: Path,
    output_dir: Path,
    method: Literal["instantmesh", "triposr", "zero123"] = "instantmesh",
) -> Path:
    """Reconstruct object from single/few images using image-to-3D model.

    Args:
        input_image: Input image path.
        output_dir: Output directory.
        method: Which model to use.

    Returns:
        Path to the exported mesh (.obj).
    """
    # TODO: Implement image-to-3D reconstruction
    raise NotImplementedError("Image-to-3D reconstruction not yet implemented")


def reconstruct_colmap_mvs(
    frames_dir: Path,
    output_dir: Path,
    quality: Literal["low", "medium", "high"] = "medium",
) -> Path:
    """Reconstruct object using COLMAP SfM + MVS pipeline.

    Args:
        frames_dir: Multi-view video frames.
        output_dir: Output directory.
        quality: Reconstruction quality preset.

    Returns:
        Path to the dense point cloud / mesh.
    """
    # TODO: Implement COLMAP + MVS pipeline
    raise NotImplementedError("COLMAP + MVS reconstruction not yet implemented")


def postprocess_mesh(
    input_mesh: Path,
    output_mesh: Path,
    target_faces: int = 15000,
    make_watertight: bool = True,
    scale: Optional[float] = None,
) -> Path:
    """Post-process reconstructed mesh for IsaacGym compatibility.

    Steps:
    1. Scale calibration
    2. Orientation alignment
    3. Decimation to target face count
    4. Hole filling / make watertight
    5. Export as .obj

    Args:
        input_mesh: Raw reconstructed mesh.
        output_mesh: Output path for processed mesh.
        target_faces: Target face count (recommend <20k).
        make_watertight: Attempt to make mesh watertight.
        scale: Optional scale factor for size calibration.

    Returns:
        Path to processed mesh.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required for mesh post-processing")

    mesh = trimesh.load(input_mesh)

    # Scale
    if scale is not None:
        mesh.apply_scale(scale)

    # Decimate
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)

    # Watertight
    if make_watertight and not mesh.is_watertight:
        mesh.fill_holes()

    # Export
    mesh.export(output_mesh)

    # Quality report
    quality = {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "is_watertight": mesh.is_watertight,
        "is_manifold": mesh.is_manifold,
        "bounding_box": mesh.bounding_box.extents.tolist(),
    }
    return output_mesh, quality
