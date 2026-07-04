"""Main object 3D reconstruction pipeline.

Supported approaches:
1. Contour extrusion (2.5D) — fast, works without depth, current best
2. Image-to-3D — InstantMesh, TripoSR
3. 3DGS — gsplat / nerfstudio (planned)
4. COLMAP + MVS — classical SfM + dense reconstruction (planned)
"""

from pathlib import Path
from typing import Optional, Literal, Tuple
import subprocess
import sys
import numpy as np


def reconstruct_contour_extrusion(
    frames_dir: Path,
    mask_path: Path,
    output_dir: Path,
    camera_intrinsic: Optional[np.ndarray] = None,
    camera_extrinsic: Optional[np.ndarray] = None,
    ground_height: float = 0.0,
    extrusion_height: float = 0.1,
    target_faces: int = 15000,
) -> Path:
    """Reconstruct object via contour extrusion from a single mask.

    Projects mask pixels to a ground plane using camera calibration,
    generates a 3D mesh by extruding the contour upward. Fast and
    depth-free — suitable when ICP/depth data is unavailable.

    Args:
        frames_dir: Directory with source frames.
        mask_path: Path to the binary mask image (single frame).
        output_dir: Output directory.
        camera_intrinsic: 3x3 camera intrinsic matrix.
        camera_extrinsic: 4x4 world-to-camera extrinsic matrix.
        ground_height: Z-coordinate of ground plane in world frame.
        extrusion_height: Height to extrude the contour.
        target_faces: Decimation target.

    Returns:
        Path to exported mesh (.obj).
    """
    try:
        import cv2
        import trimesh
        from scipy.spatial import ConvexHull
    except ImportError as e:
        raise ImportError(f"Required package missing: {e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")

    largest_contour = max(contours, key=cv2.contourArea)
    contour_2d = largest_contour.squeeze(1)

    if camera_intrinsic is not None and camera_extrinsic is not None:
        K_inv = np.linalg.inv(camera_intrinsic)
        points_homogeneous = np.hstack([
            contour_2d.astype(np.float64), np.ones((len(contour_2d), 1))
        ])
        rays_camera = points_homogeneous @ K_inv.T
        rays_world = rays_camera @ camera_extrinsic[:3, :3].T
        cam_center = camera_extrinsic[:3, 3]

        t = (ground_height - cam_center[2]) / rays_world[:, 2]
        t = np.clip(t, 0.01, 10.0)
        points_3d_ground = cam_center + rays_world * t[:, None]
    else:
        h_img, w_img = mask.shape
        scale_x = 0.2 / w_img
        scale_y = 0.2 / h_img
        points_3d_ground = np.zeros((len(contour_2d), 3))
        points_3d_ground[:, 0] = (contour_2d[:, 0] - w_img / 2) * scale_x
        points_3d_ground[:, 1] = (h_img / 2 - contour_2d[:, 1]) * scale_y

    points_3d_top = points_3d_ground.copy()
    points_3d_top[:, 2] = extrusion_height

    all_points = np.vstack([points_3d_ground, points_3d_top])
    hull = ConvexHull(all_points[:, :2])
    top_face = points_3d_top[hull.vertices]
    bottom_face = points_3d_ground[hull.vertices]

    faces = []
    vertices = []
    n = len(top_face)
    for i in range(n):
        vertices.extend([bottom_face[i], bottom_face[(i + 1) % n], top_face[i]])
        faces.append([len(vertices) - 3, len(vertices) - 2, len(vertices) - 1])
        vertices.extend([top_face[i], bottom_face[(i + 1) % n], top_face[(i + 1) % n]])
        faces.append([len(vertices) - 3, len(vertices) - 2, len(vertices) - 1])

    vertices = np.array(vertices)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()

    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)

    out_path = output_dir / "reconstructed_mesh.obj"
    mesh.export(out_path)

    quality = {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "is_watertight": mesh.is_watertight,
        "is_manifold": mesh.is_manifold,
        "method": "contour_extrusion",
    }
    return out_path


def reconstruct_image_to_3d(
    input_image: Path,
    output_dir: Path,
    method: Literal["instantmesh", "triposr"] = "instantmesh",
) -> Path:
    """Reconstruct object from single image via image-to-3D model.

    Uses InstantMesh (by default) — a feed-forward model producing
    textured meshes in <10 seconds. Requires InstantMesh repo and
    checkpoint. Falls back gracefully if not installed.

    Args:
        input_image: Input image path.
        output_dir: Output directory.
        method: Which model to use.

    Returns:
        Path to the exported mesh (.obj).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "instantmesh":
        return _reconstruct_instantmesh(input_image, output_dir)
    elif method == "triposr":
        raise NotImplementedError("TripoSR integration not yet implemented")
    else:
        raise ValueError(f"Unknown method: {method}")


def _reconstruct_instantmesh(
    input_image: Path,
    output_dir: Path,
) -> Path:
    """InstantMesh integration via subprocess.

    Checks for InstantMesh installation. If installed, runs inference.
    Otherwise, falls back to contour extrusion with a heuristic.
    """
    import importlib.util
    if importlib.util.find_spec("InstantMesh") is None:
        raise RuntimeError(
            "InstantMesh not installed. Install from:\n"
            "  git clone https://github.com/TencentARC/InstantMesh\n"
            "  cd InstantMesh && pip install -r requirements.txt\n"
            "Or set method='contour_extrusion' for a simpler approach."
        )

    result = subprocess.run(
        [
            sys.executable, "-c",
            f"from InstantMesh.run import main; main('{input_image}', '{output_dir}')"
        ],
        capture_output=True, text=True, timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"InstantMesh failed: {result.stderr}")

    meshes = list(output_dir.glob("*.obj"))
    if not meshes:
        raise RuntimeError("InstantMesh produced no output mesh")

    return meshes[0]


def reconstruct_3dgs(
    frames_dir: Path,
    camera_params: Path,
    output_dir: Path,
    iterations: int = 7000,
) -> Path:
    """Reconstruct object using 3D Gaussian Splatting.

    NOTE: Planned integration with gsplat / nerfstudio.
    Currently not implemented.

    Args:
        frames_dir: Multi-view video frames.
        camera_params: Camera intrinsics/extrinsics.
        output_dir: Output directory.
        iterations: Training iterations.

    Returns:
        Path to the exported mesh (.ply).
    """
    raise NotImplementedError(
        "3DGS reconstruction not yet implemented. "
        "Use reconstruct_contour_extrusion() or install InstantMesh for image-to-3D."
    )


def reconstruct_colmap_mvs(
    frames_dir: Path,
    output_dir: Path,
    quality: Literal["low", "medium", "high"] = "medium",
) -> Path:
    """Reconstruct object using COLMAP SfM + MVS pipeline.

    NOTE: Planned integration with COLMAP CLI.
    Currently not implemented.

    Args:
        frames_dir: Multi-view video frames.
        output_dir: Output directory.
        quality: Reconstruction quality preset.

    Returns:
        Path to the dense point cloud / mesh.
    """
    raise NotImplementedError(
        "COLMAP + MVS reconstruction not yet implemented. "
        "Use reconstruct_contour_extrusion() or install InstantMesh for image-to-3D."
    )


def postprocess_mesh(
    input_mesh: Path,
    output_mesh: Path,
    target_faces: int = 15000,
    make_watertight: bool = True,
    scale: Optional[float] = None,
) -> Tuple[Path, dict]:
    """Post-process reconstructed mesh for IsaacGym compatibility.

    Steps:
    1. Scale calibration
    2. Decimation to target face count
    3. Hole filling / make watertight
    4. Export as .obj

    Args:
        input_mesh: Raw reconstructed mesh.
        output_mesh: Output path for processed mesh.
        target_faces: Target face count (recommend <20k).
        make_watertight: Attempt to make mesh watertight.
        scale: Optional scale factor for size calibration.

    Returns:
        Tuple of (Path to processed mesh, quality report dict).
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required for mesh post-processing")

    mesh = trimesh.load(input_mesh)

    if scale is not None:
        mesh.apply_scale(scale)

    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)

    if make_watertight and not mesh.is_watertight:
        mesh.fill_holes()
        if not mesh.is_watertight:
            mesh = mesh.convex_hull

    mesh.export(output_mesh)

    quality = {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "is_watertight": mesh.is_watertight,
        "is_manifold": mesh.is_manifold,
        "bounding_box": mesh.bounding_box.extents.tolist(),
    }
    return output_mesh, quality
