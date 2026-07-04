"""Mask-driven object pose trajectory recovery.

This module is intentionally prompt-free: it consumes masks, depth, camera
calibration, optional GT trajectories, and optional reconstructed object meshes.
The old fixed-coordinate SAM scripts remain as provenance artifacts, but they
are not used by this pipeline.
"""

from __future__ import annotations

import csv
import io
import json
import math
import pickle
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


CAMERAS = ("camera_top", "camera_side_1", "camera_side_2")
INVENTORY_FIELDS = (
    "object_name",
    "sequence_name",
    "has_rgb",
    "has_depth",
    "has_mask",
    "has_gt_pose",
    "num_frames",
    "recommended_tracking_mode",
)
YAW_AMBIGUOUS_OBJECTS = {"drink_ad", "drink_yykx", "bottle", "can"}


def _np():
    import numpy as np

    return np


def _cv2():
    import cv2

    return cv2


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _h5py():
    import h5py

    return h5py


@dataclass
class SequenceInventory:
    object_name: str
    sequence_name: str
    sequence_dir: Path
    has_rgb: bool
    has_depth: bool
    has_mask: bool
    has_gt_pose: bool
    num_frames: int
    recommended_tracking_mode: str
    camera_names: list[str] = field(default_factory=list)
    mask_paths: list[Path] = field(default_factory=list)
    depth_paths: list[Path] = field(default_factory=list)

    def to_csv_row(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in INVENTORY_FIELDS}


@dataclass
class TrackingConfig:
    camera: str = "camera_top"
    depth_scale: float = 0.001
    min_mask_area: int = 50
    min_depth_valid_ratio: float = 0.2
    max_centroid_jump_m: float = 0.12
    max_theta_jump_deg: float = 45.0
    max_pose_jump_m: float = 0.18
    min_point_count: int = 64
    min_icp_fitness: float = 0.15
    max_icp_rmse: float = 0.04
    max_image_mask_area_ratio: float = 0.10
    min_multiview_views: int = 2
    plane_axis: int = 1
    plane_value: float = 0.02
    yaw_search_degrees: float = 180.0
    yaw_search_steps: int = 37
    min_silhouette_score: float = 0.02
    percentile_clip: tuple[float, float] = (2.0, 98.0)
    voxel_size: float = 0.005
    max_debug_overlays: int = 12
    hold_invalid_pose: bool = True
    output_world_pose: bool = False
    enable_icp: bool = True


def infer_object_name(sequence_name: str) -> str:
    task = sequence_name.split("__2026_")[0]
    if "drink_ad" in task:
        return "drink_ad"
    if "drink_yykx" in task:
        return "drink_yykx"
    if "bread" in task:
        return "bread"
    if "pipette" in task:
        return "pipette"
    return task


def recommended_mode(has_gt_pose: bool, has_rgb: bool, has_mask: bool, has_depth: bool) -> str:
    if has_gt_pose:
        return "use_gt_pose"
    if has_mask and has_depth:
        return "mask_depth_icp"
    if has_rgb and has_mask:
        return "image_plane_pose"
    return "need_segmentation_or_manual"


def count_video_frames(video_path: Path) -> int:
    """Count frames without importing OpenCV.

    ffprobe is available on the hackathon machine and handles Matroska streams
    whose metadata does not expose nb_frames.
    """
    if not video_path.exists():
        return 0
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        return int(out) if out and out != "N/A" else 0
    except Exception:
        return 0


def count_hdf5_frames(hdf5_path: Path) -> int:
    if not hdf5_path.exists():
        return 0
    try:
        h5py = _h5py()
    except ImportError:
        return 0
    try:
        with h5py.File(hdf5_path, "r") as f:
            candidates = (
                "obj/pose/mask",
                "obj_rh/pose/mask",
                "obj_lh/pose/mask",
                "obj/pose/obj_transf",
                "obj_rh/pose/obj_transf",
                "obj_lh/pose/obj_transf",
            )
            for key in candidates:
                if key in f and len(f[key].shape) >= 1:
                    return int(f[key].shape[0])
    except Exception:
        return 0
    return 0


def discover_frame_files(root: Path, camera: str, stem: str) -> list[Path]:
    patterns = [
        f"{stem}/{camera}/frame_*.png",
        f"{stem}/{camera}/frame_*.jpg",
        f"{stem}/{camera}/frame_*.jpeg",
        f"{stem}/{camera}/frame_*.npy",
        f"{stem}/{camera}/frame_*.npz",
        f"{stem}/{camera}_frame_*",
        f"{stem}/frame_*",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted({p for p in files if p.is_file()})


def discover_related_experiment_dirs(
    experiments_root: Path,
    object_name: str,
    sequence_name: str,
) -> list[Path]:
    if not experiments_root.exists():
        return []
    task_name = sequence_name.split("__2026_")[0]
    candidates = []
    for path in experiments_root.glob("*"):
        if not path.is_dir():
            continue
        configured_sequence = read_experiment_sequence(path)
        if configured_sequence:
            if configured_sequence == sequence_name:
                candidates.append(path)
            continue
        name = path.name.lower()
        if task_name.lower() in name:
            candidates.append(path)
    return sorted(candidates)


def read_experiment_sequence(experiment_dir: Path) -> Optional[str]:
    config = experiment_dir / "config.yaml"
    if not config.exists():
        return None
    for raw_line in config.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("sequence:"):
            continue
        return line.split(":", 1)[1].strip().strip("'\"")
    return None


def discover_masks(
    sequence_dir: Path,
    object_name: str,
    sequence_name: str,
    camera: str = "camera_top",
    experiments_root: Path = Path("experiments"),
) -> list[Path]:
    roots = [sequence_dir] + discover_related_experiment_dirs(
        experiments_root, object_name, sequence_name
    )
    mask_files: list[Path] = []
    for root in roots:
        mask_files.extend(discover_frame_files(root, camera, "masks"))
        mask_files.extend(discover_frame_files(root, camera, "mask"))
    return sorted({p for p in mask_files if _looks_like_mask(p)})


def discover_depths(
    sequence_dir: Path,
    object_name: str,
    sequence_name: str,
    camera: str = "camera_top",
    experiments_root: Path = Path("experiments"),
) -> list[Path]:
    roots = [sequence_dir] + discover_related_experiment_dirs(
        experiments_root, object_name, sequence_name
    )
    depth_files: list[Path] = []
    for root in roots:
        depth_files.extend(discover_frame_files(root, camera, "depth"))
        depth_files.extend(discover_frame_files(root, camera, "depths"))
    allowed = {".png", ".npy", ".npz", ".tif", ".tiff", ".exr"}
    return sorted({p for p in depth_files if p.suffix.lower() in allowed})


def discover_masks_by_camera(
    sequence_dir: Path,
    object_name: str,
    sequence_name: str,
    experiments_root: Path = Path("experiments"),
    cameras: tuple[str, ...] = CAMERAS,
) -> dict[str, list[Path]]:
    return {
        cam: select_one_mask_per_frame(
            discover_masks(sequence_dir, object_name, sequence_name, cam, experiments_root)
        )
        for cam in cameras
    }


def group_multiview_masks(masks_by_camera: dict[str, list[Path]]) -> dict[int, dict[str, Path]]:
    frames: dict[int, dict[str, Path]] = {}
    for cam, paths in masks_by_camera.items():
        for path in paths:
            idx = frame_index_from_path(path)
            if idx is None:
                continue
            frames.setdefault(idx, {})[cam] = path
    return dict(sorted(frames.items()))


def has_multiview_mask_frame(
    sequence_dir: Path,
    object_name: str,
    sequence_name: str,
    experiments_root: Path = Path("experiments"),
    min_views: int = 2,
) -> bool:
    grouped = group_multiview_masks(
        discover_masks_by_camera(sequence_dir, object_name, sequence_name, experiments_root)
    )
    return any(len(paths) >= min_views for paths in grouped.values())


def _looks_like_mask(path: Path) -> bool:
    name = path.name.lower()
    if "overlay" in name or "compare" in name or "metadata" in name:
        return False
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy", ".npz"} and "mask" in name


def select_one_mask_per_frame(mask_paths: list[Path]) -> list[Path]:
    by_frame: dict[int, list[Path]] = {}
    no_frame: list[Path] = []
    for path in mask_paths:
        idx = frame_index_from_path(path)
        if idx is None:
            no_frame.append(path)
        else:
            by_frame.setdefault(idx, []).append(path)
    selected = [sorted(paths, key=mask_preference_key)[0] for _, paths in sorted(by_frame.items())]
    return selected + sorted(no_frame, key=mask_preference_key)


def mask_preference_key(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if "v41" in name:
        rank = 0
    elif "refined" in name:
        rank = 1
    elif "sam" in name:
        rank = 2
    else:
        rank = 3
    return rank, name


def has_valid_gt_pose(hdf5_path: Path, object_name: str) -> bool:
    if not hdf5_path.exists():
        return False
    try:
        h5py = _h5py()
    except ImportError:
        return False
    with h5py.File(hdf5_path, "r") as f:
        for group_name in relevant_object_groups(f, object_name):
            if f"{group_name}/pose/mask" not in f:
                continue
            mask = f[f"{group_name}/pose/mask"][()]
            if bool(mask.any()):
                return True
    return False


def relevant_object_groups(h5_file: Any, object_name: str) -> list[str]:
    groups: list[str] = []
    for name in ("obj", "obj_rh", "obj_lh"):
        if name in h5_file and object_group_matches(h5_file[name], object_name, default=(name == "obj")):
            groups.append(name)
    if "obj_other" in h5_file:
        for child in h5_file["obj_other"].keys():
            group_name = f"obj_other/{child}"
            if object_group_matches(h5_file[group_name], object_name, default=False):
                groups.append(group_name)
    return groups


def object_group_matches(group: Any, object_name: str, default: bool = False) -> bool:
    obj_id = str(read_hdf5_scalar(group.get("obj_id", None))).strip()
    if not obj_id:
        return default
    aliases = object_aliases(object_name)
    normalized_id = normalize_object_label(obj_id)
    return any(alias in normalized_id for alias in aliases)


def object_aliases(object_name: str) -> set[str]:
    normalized = normalize_object_label(object_name)
    aliases = {normalized}
    if normalized == "drinkad":
        aliases.add("drinkad")
    elif normalized == "drinkyykx":
        aliases.add("drinkyykx")
    elif normalized == "pipette":
        aliases.add("pipette")
    elif normalized == "bread":
        aliases.add("bread")
    return aliases


def normalize_object_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def read_hdf5_scalar(dataset: Any) -> Any:
    if dataset is None:
        return ""
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def inventory_dataset(
    data_root: Path,
    output_csv: Optional[Path] = None,
    experiments_root: Path = Path("experiments"),
    camera: str = "camera_top",
) -> list[SequenceInventory]:
    rows: list[SequenceInventory] = []
    for sequence_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        sequence_name = sequence_dir.name
        object_name = infer_object_name(sequence_name)
        videos = sorted((sequence_dir / "video").glob("*.mkv"))
        camera_names = [p.stem for p in videos]
        has_rgb = bool(videos)
        video_for_count = sequence_dir / "video" / f"{camera}.mkv"
        if not video_for_count.exists() and videos:
            video_for_count = videos[0]
        num_frames = count_hdf5_frames(sequence_dir / "pose_3d.hdf5")
        if num_frames <= 0:
            num_frames = count_video_frames(video_for_count)
        mask_paths = discover_masks(
            sequence_dir, object_name, sequence_name, camera, experiments_root
        )
        depth_paths = discover_depths(
            sequence_dir, object_name, sequence_name, camera, experiments_root
        )
        gt = has_valid_gt_pose(sequence_dir / "pose_3d.hdf5", object_name)
        mode = recommended_mode(gt, has_rgb, bool(mask_paths), bool(depth_paths))
        if mode == "image_plane_pose" and has_multiview_mask_frame(
            sequence_dir, object_name, sequence_name, experiments_root
        ):
            mode = "multi_view_mask_pose"
        rows.append(
            SequenceInventory(
                object_name=object_name,
                sequence_name=sequence_name,
                sequence_dir=sequence_dir,
                has_rgb=has_rgb,
                has_depth=bool(depth_paths),
                has_mask=bool(mask_paths),
                has_gt_pose=gt,
                num_frames=num_frames,
                recommended_tracking_mode=mode,
                camera_names=camera_names,
                mask_paths=mask_paths,
                depth_paths=depth_paths,
            )
        )
    if output_csv:
        write_inventory_csv(rows, output_csv)
    return rows


def write_inventory_csv(rows: Iterable[SequenceInventory], output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
    return output_csv


def load_intrinsics(calib_dir: Path, camera: str):
    return load_pickle_compat(calib_dir / camera / "cam_intr.pkl")


def load_extrinsics(calib_dir: Path, camera: str):
    return load_pickle_compat(calib_dir / camera / "cam_extr.pkl")


def load_pickle_compat(path: Path):
    """Load pickle files produced with NumPy 2 from NumPy 1.x environments."""

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    with open(path, "rb") as f:
        data = f.read()
    return _CompatUnpickler(io.BytesIO(data)).load()


def load_mask(path: Path):
    np = _np()
    if path.suffix.lower() in {".npy", ".npz"}:
        arr = np.load(path)
        if hasattr(arr, "files"):
            arr = arr[arr.files[0]]
        return arr.astype(bool)
    cv2 = _cv2()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read mask: {path}")
    return img > 127


def load_depth(path: Path, depth_scale: float = 0.001):
    np = _np()
    if path.suffix.lower() in {".npy", ".npz"}:
        arr = np.load(path)
        if hasattr(arr, "files"):
            arr = arr[arr.files[0]]
        return arr.astype("float64")
    cv2 = _cv2()
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Could not read depth: {path}")
    is_integer = depth.dtype.kind in "ui"
    depth = depth.astype("float64")
    if is_integer or float(depth.max(initial=0.0)) > 20.0:
        depth = depth * depth_scale
    return depth


def frame_index_from_path(path: Path) -> Optional[int]:
    match = re.search(r"frame[_-](\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def align_mask_depth_pairs(mask_paths: list[Path], depth_paths: list[Path]) -> list[tuple[int, Path, Path]]:
    depth_by_idx = {frame_index_from_path(p): p for p in depth_paths}
    pairs = []
    for mask_path in mask_paths:
        idx = frame_index_from_path(mask_path)
        if idx is not None and idx in depth_by_idx:
            pairs.append((idx, mask_path, depth_by_idx[idx]))
    return sorted(pairs, key=lambda item: item[0])


def backproject_mask_depth(mask, depth, K, config: TrackingConfig):
    np = _np()
    ys, xs = np.nonzero(mask)
    mask_area = int(len(xs))
    if mask_area < config.min_mask_area:
        return np.empty((0, 3)), {
            "mask_area": mask_area,
            "depth_valid_ratio": 0.0,
            "point_count": 0,
            "invalid_reason": f"mask_area_too_small ({mask_area}<{config.min_mask_area})",
        }

    z = depth[ys, xs].astype("float64")
    finite = np.isfinite(z) & (z > 0)
    depth_valid_ratio = float(finite.sum() / max(mask_area, 1))
    if finite.any():
        low, high = np.percentile(z[finite], config.percentile_clip)
        finite &= (z >= low) & (z <= high)
    if depth_valid_ratio < config.min_depth_valid_ratio:
        return np.empty((0, 3)), {
            "mask_area": mask_area,
            "depth_valid_ratio": depth_valid_ratio,
            "point_count": int(finite.sum()),
            "invalid_reason": (
                f"depth_valid_ratio_low "
                f"({depth_valid_ratio:.3f}<{config.min_depth_valid_ratio:.3f})"
            ),
        }

    xs = xs[finite].astype("float64")
    ys = ys[finite].astype("float64")
    z = z[finite]
    if len(z) < config.min_point_count:
        return np.empty((0, 3)), {
            "mask_area": mask_area,
            "depth_valid_ratio": depth_valid_ratio,
            "point_count": int(len(z)),
            "invalid_reason": f"point_count_low ({len(z)}<{config.min_point_count})",
        }
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    return pts, {
        "mask_area": mask_area,
        "depth_valid_ratio": depth_valid_ratio,
        "point_count": int(len(pts)),
        "invalid_reason": None,
    }


def pca_pose(points, yaw_ambiguous: bool = False):
    np = _np()
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    dominant = axes[:, 0]
    theta = float(math.atan2(dominant[0], dominant[2]))
    if yaw_ambiguous:
        theta = 0.0
    T = np.eye(4)
    T[:3, :3] = yaw_to_rotation(theta)
    T[:3, 3] = centroid
    return T, theta, eigvals[order], axes


def yaw_to_rotation(theta: float):
    np = _np()
    c, s = math.cos(theta), math.sin(theta)
    R = np.eye(3)
    R[0, 0] = c
    R[0, 2] = s
    R[2, 0] = -s
    R[2, 2] = c
    return R


def rotation_matrix_to_quaternion_xyzw(R):
    np = _np()
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype="float64")
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else np.array([0.0, 0.0, 0.0, 1.0])


def angle_delta_deg(a: Optional[float], b: float) -> float:
    if a is None:
        return 0.0
    delta = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


def invert_transform(T):
    np = _np()
    inv = np.eye(4)
    inv[:3, :3] = T[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ T[:3, 3]
    return inv


def require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required for mask_depth_icp. Install it in the runtime "
            "environment, e.g. `/home/ruan/miniconda3/envs/objasset/bin/python -m pip install open3d`."
        ) from exc
    return o3d


def load_canonical_point_cloud(mesh_path: Optional[Path], config: TrackingConfig, required: bool = False):
    if not config.enable_icp:
        if required:
            raise ValueError("mask_depth_icp requires ICP; remove --no-icp or use image_plane_pose fallback.")
        return None
    if not mesh_path or not mesh_path.exists():
        if required:
            raise FileNotFoundError(f"Canonical mesh not found for mask_depth_icp: {mesh_path}")
        return None
    if required:
        o3d = require_open3d()
    else:
        try:
            import open3d as o3d
        except ImportError:
            return None
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        if required:
            raise ValueError(f"Canonical mesh is empty or unreadable: {mesh_path}")
        return None
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    if config.voxel_size > 0:
        pcd = pcd.voxel_down_sample(config.voxel_size)
    return pcd


def run_icp(points, canonical_pcd, init_T, config: TrackingConfig):
    np = _np()
    if canonical_pcd is None:
        raise ValueError("mask_depth_icp requires a canonical point cloud.")
    o3d = require_open3d()

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(points)
    if config.voxel_size > 0:
        target = target.voxel_down_sample(config.voxel_size)
    if len(target.points) < 8:
        return init_T, 0.0, float("inf")

    result = o3d.pipelines.registration.registration_icp(
        canonical_pcd,
        target,
        max_correspondence_distance=max(config.voxel_size * 4.0, 0.02),
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    return np.asarray(result.transformation), float(result.fitness), float(result.inlier_rmse)


def choose_canonical_mesh(object_name: str, asset_root: Path = Path("runs/object_asset_v1")) -> Optional[Path]:
    candidates = [
        asset_root / object_name / "mesh" / "visual_mesh.obj",
        asset_root / object_name / "mesh" / "collision_mesh.obj",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_mesh_points(mesh_path: Path, sample_count: int = 2500):
    np = _np()
    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not mesh.is_empty() and len(mesh.triangles) > 0:
            pcd = mesh.sample_points_uniformly(number_of_points=sample_count)
            return np.asarray(pcd.points, dtype="float64")
        if not mesh.is_empty():
            return np.asarray(mesh.vertices, dtype="float64")
    except Exception:
        pass

    vertices = []
    with open(mesh_path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No vertices could be loaded from mesh: {mesh_path}")
    pts = np.asarray(vertices, dtype="float64")
    if len(pts) > sample_count:
        idx = np.linspace(0, len(pts) - 1, sample_count).astype(int)
        pts = pts[idx]
    return pts


def yaw_transform_world(translation, yaw: float):
    np = _np()
    T = np.eye(4)
    T[:3, :3] = yaw_to_rotation(yaw)
    T[:3, 3] = translation
    return T


def camera_ray_from_pixel(K, E, uv):
    np = _np()
    K = np.asarray(K, dtype="float64")
    E = np.asarray(E, dtype="float64")
    R = E[:3, :3]
    t = E[:3, 3]
    C = -R.T @ t
    pix = np.array([uv[0], uv[1], 1.0], dtype="float64")
    ray_cam = np.linalg.inv(K) @ pix
    ray_cam = ray_cam / max(np.linalg.norm(ray_cam), 1e-12)
    ray_world = R.T @ ray_cam
    ray_world = ray_world / max(np.linalg.norm(ray_world), 1e-12)
    return C, ray_world


def intersect_ray_with_axis_plane(origin, direction, axis: int, value: float):
    denom = float(direction[axis])
    if abs(denom) < 1e-8:
        return None
    alpha = (value - float(origin[axis])) / denom
    if alpha <= 0:
        return None
    return origin + alpha * direction


def project_world_points(points_world, K, E, image_shape):
    np = _np()
    K = np.asarray(K, dtype="float64")
    E = np.asarray(E, dtype="float64")
    h, w = image_shape[:2]
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    pts_cam = (E @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    front = z > 1e-6
    uvw = (K @ pts_cam[front].T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    in_bounds = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    return uv[in_bounds]


def bbox_iou_xyxy(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0 + 1), max(0, iy1 - iy0 + 1)
    inter = iw * ih
    area_a = max(0, ax1 - ax0 + 1) * max(0, ay1 - ay0 + 1)
    area_b = max(0, bx1 - bx0 + 1) * max(0, by1 - by0 + 1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def score_projected_points_against_mask(uv, mask, mask_bbox) -> tuple[float, dict[str, float]]:
    np = _np()
    if len(uv) == 0:
        return 0.0, {"coverage": 0.0, "bbox_iou": 0.0, "projected_points": 0}
    h, w = mask.shape[:2]
    xy = np.round(uv).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
    inside = mask[xy[:, 1], xy[:, 0]].astype(bool)
    coverage = float(inside.mean()) if len(inside) else 0.0
    proj_bbox = [
        int(xy[:, 0].min()),
        int(xy[:, 1].min()),
        int(xy[:, 0].max()),
        int(xy[:, 1].max()),
    ]
    iou = bbox_iou_xyxy(proj_bbox, mask_bbox)
    score = 0.7 * coverage + 0.3 * iou
    return score, {
        "coverage": coverage,
        "bbox_iou": iou,
        "projected_points": int(len(uv)),
    }


def estimate_multiview_translation(view_records: list[dict[str, Any]], config: TrackingConfig):
    np = _np()
    hits = []
    for rec in view_records:
        if not rec["view_valid"]:
            continue
        hit = intersect_ray_with_axis_plane(
            rec["camera_center"],
            rec["centroid_ray"],
            config.plane_axis,
            config.plane_value,
        )
        if hit is not None:
            hits.append(hit)
    if not hits:
        return None
    return np.mean(np.stack(hits, axis=0), axis=0)


def optimize_yaw_by_multiview_projection(
    mesh_points,
    translation,
    view_records: list[dict[str, Any]],
    yaw_ambiguous: bool,
    config: TrackingConfig,
) -> tuple[float, float, list[dict[str, Any]]]:
    np = _np()
    if yaw_ambiguous:
        yaw_candidates = np.array([0.0], dtype="float64")
    else:
        yaw_candidates = np.deg2rad(
            np.linspace(-config.yaw_search_degrees, config.yaw_search_degrees, config.yaw_search_steps)
        )
    best_yaw = 0.0
    best_score = -1.0
    best_view_scores: list[dict[str, Any]] = []
    for yaw in yaw_candidates:
        T = yaw_transform_world(translation, float(yaw))
        pts_world = (T[:3, :3] @ mesh_points.T).T + T[:3, 3]
        view_scores = []
        scores = []
        for rec in view_records:
            if not rec["view_valid"]:
                continue
            uv = project_world_points(pts_world, rec["K"], rec["E"], rec["mask"].shape)
            score, detail = score_projected_points_against_mask(uv, rec["mask"], rec["bbox_xyxy"])
            detail.update({"camera": rec["camera"], "score": score})
            view_scores.append(detail)
            scores.append(score)
        mean_score = float(np.mean(scores)) if scores else 0.0
        if mean_score > best_score:
            best_score = mean_score
            best_yaw = float(yaw)
            best_view_scores = view_scores
    return best_yaw, best_score, best_view_scores


def track_mask_depth_icp_sequence(
    sequence_dir: Path,
    output_dir: Path,
    object_name: Optional[str] = None,
    camera: str = "camera_top",
    mask_paths: Optional[list[Path]] = None,
    depth_paths: Optional[list[Path]] = None,
    mesh_path: Optional[Path] = None,
    config: Optional[TrackingConfig] = None,
) -> dict[str, Any]:
    np = _np()
    config = config or TrackingConfig(camera=camera)
    object_name = object_name or infer_object_name(sequence_dir.name)
    yaw_ambiguous = object_name in YAW_AMBIGUOUS_OBJECTS
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = mask_paths or discover_masks(sequence_dir, object_name, sequence_dir.name, camera)
    mask_paths = select_one_mask_per_frame(mask_paths)
    depth_paths = depth_paths or discover_depths(sequence_dir, object_name, sequence_dir.name, camera)
    pairs = align_mask_depth_pairs(mask_paths, depth_paths)
    if not pairs:
        raise ValueError(f"No aligned mask/depth frame pairs found for {sequence_dir}")

    K = load_intrinsics(sequence_dir / "camera_calib", camera)
    K = np.asarray(K, dtype="float64")
    E = np.asarray(load_extrinsics(sequence_dir / "camera_calib", camera), dtype="float64")
    mesh_path = mesh_path or choose_canonical_mesh(object_name)
    canonical = load_canonical_point_cloud(mesh_path, config, required=True)
    canonical_points = np.asarray(canonical.points)
    canonical_pose, _, _, _ = pca_pose(canonical_points, yaw_ambiguous=yaw_ambiguous)

    poses_raw = []
    poses_out = []
    frame_records = []
    prev_valid_pose = None
    prev_valid_centroid = None
    prev_valid_theta = None
    debug_dir = output_dir / "debug_overlay"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_stride = max(1, len(pairs) // max(config.max_debug_overlays, 1))

    for n, (frame_idx, mask_path, depth_path) in enumerate(pairs):
        mask = load_mask(mask_path)
        depth = load_depth(depth_path, config.depth_scale)
        points, metrics = backproject_mask_depth(mask, depth, K, config)
        invalid_reason = metrics["invalid_reason"]
        pose_raw = np.eye(4)
        theta = 0.0
        icp_fitness = None
        icp_rmse = None
        centroid_jump = 0.0
        theta_jump = 0.0
        pose_jump = 0.0

        if invalid_reason is None:
            observed_pose, theta, _, _ = pca_pose(points, yaw_ambiguous=yaw_ambiguous)
            if prev_valid_pose is not None:
                init_T = prev_valid_pose
                centroid_jump = float(np.linalg.norm(observed_pose[:3, 3] - prev_valid_centroid))
                theta_jump = angle_delta_deg(prev_valid_theta, theta)
            else:
                init_T = observed_pose @ invert_transform(canonical_pose)
            if centroid_jump > config.max_centroid_jump_m:
                invalid_reason = (
                    f"centroid_jump ({centroid_jump:.4f}>{config.max_centroid_jump_m:.4f}m)"
                )
            if invalid_reason is None and not yaw_ambiguous and theta_jump > config.max_theta_jump_deg:
                invalid_reason = (
                    f"theta_jump ({theta_jump:.1f}>{config.max_theta_jump_deg:.1f}deg)"
                )
            if invalid_reason is None:
                pose_raw, icp_fitness, icp_rmse = run_icp(points, canonical, init_T, config)
                if prev_valid_pose is not None:
                    pose_jump = float(np.linalg.norm(pose_raw[:3, 3] - prev_valid_pose[:3, 3]))
                    if pose_jump > config.max_pose_jump_m:
                        invalid_reason = (
                            f"pose_jump ({pose_jump:.4f}>{config.max_pose_jump_m:.4f}m)"
                        )
                if icp_fitness is not None and icp_fitness < config.min_icp_fitness:
                    invalid_reason = (
                        f"icp_fitness_low ({icp_fitness:.3f}<{config.min_icp_fitness:.3f})"
                    )
                if icp_rmse is not None and icp_rmse > config.max_icp_rmse:
                    invalid_reason = (
                        f"icp_rmse_high ({icp_rmse:.4f}>{config.max_icp_rmse:.4f}m)"
                    )

        valid = invalid_reason is None
        if valid:
            prev_valid_pose = pose_raw.copy()
            prev_valid_centroid = pose_raw[:3, 3].copy()
            prev_valid_theta = theta
            pose_out = pose_raw.copy()
        elif config.hold_invalid_pose and prev_valid_pose is not None:
            pose_out = prev_valid_pose.copy()
        else:
            pose_out = pose_raw.copy()

        if config.output_world_pose:
            pose_out = E @ pose_out
            pose_raw_export = E @ pose_raw
            transform_name = "T_world_object"
        else:
            pose_raw_export = pose_raw
            transform_name = "T_camera_object"

        poses_raw.append(pose_raw_export)
        poses_out.append(pose_out)
        frame_records.append(
            {
                "frame_id": int(frame_idx),
                "mask_path": str(mask_path),
                "depth_path": str(depth_path),
                "valid": bool(valid),
                "invalid_reason": invalid_reason,
                "mask_area": int(metrics["mask_area"]),
                "depth_valid_ratio": float(metrics["depth_valid_ratio"]),
                "point_count": int(metrics["point_count"]),
                "centroid": pose_out[:3, 3].tolist(),
                "translation": pose_out[:3, 3].tolist(),
                "rotation_matrix": pose_out[:3, :3].tolist(),
                "quaternion_xyzw": rotation_matrix_to_quaternion_xyzw(pose_out[:3, :3]).tolist(),
                "theta_rad": float(theta),
                "theta_deg": float(math.degrees(theta)),
                "centroid_jump_m": float(centroid_jump),
                "theta_jump_deg": float(theta_jump),
                "pose_jump_m": float(pose_jump),
                "icp_fitness": icp_fitness,
                "icp_rmse": icp_rmse,
            }
        )
        if n % debug_stride == 0:
            save_mask_depth_debug_overlay(mask, depth, debug_dir / f"frame_{frame_idx:06d}.png")

    poses_raw_np = np.stack(poses_raw)
    poses_out_np = np.stack(poses_out)
    timestamps = np.array([rec["frame_id"] for rec in frame_records], dtype="float64")
    return save_tracking_outputs(
        output_dir=output_dir,
        object_name=object_name,
        sequence_name=sequence_dir.name,
        method="mask_depth_icp",
        transform_name=transform_name,
        poses=poses_out_np,
        raw_poses=poses_raw_np,
        timestamps=timestamps,
        frame_records=frame_records,
        yaw_ambiguous=yaw_ambiguous,
        mesh_path=mesh_path,
        output_stem="object_trajectory_mask_depth_icp",
    )


def track_mask_depth_sequence(*args, **kwargs) -> dict[str, Any]:
    """Backward-compatible alias for the explicit ICP baseline."""
    return track_mask_depth_icp_sequence(*args, **kwargs)


def save_mask_depth_debug_overlay(mask, depth, output_path: Path) -> None:
    np = _np()
    cv2 = _cv2()
    d = depth.copy().astype("float64")
    valid = np.isfinite(d) & (d > 0)
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        d = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
    else:
        d[:] = 0
    gray = (d * 255).astype("uint8")
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color[mask.astype(bool)] = (0.35 * color[mask.astype(bool)] + [0, 180, 255]).astype("uint8")
    cv2.imwrite(str(output_path), color)


def track_multi_view_mask_sequence(
    sequence_dir: Path,
    output_dir: Path,
    object_name: Optional[str] = None,
    cameras: tuple[str, ...] = CAMERAS,
    mesh_path: Optional[Path] = None,
    config: Optional[TrackingConfig] = None,
    experiments_root: Path = Path("experiments"),
) -> dict[str, Any]:
    np = _np()
    config = config or TrackingConfig()
    object_name = object_name or infer_object_name(sequence_dir.name)
    yaw_ambiguous = object_name in YAW_AMBIGUOUS_OBJECTS
    output_dir.mkdir(parents=True, exist_ok=True)

    masks_by_camera = discover_masks_by_camera(
        sequence_dir, object_name, sequence_dir.name, experiments_root, cameras
    )
    grouped = group_multiview_masks(masks_by_camera)
    if not grouped:
        raise ValueError(f"No masks found for multi-view pose fitting: {sequence_dir}")

    calib = {}
    for cam in cameras:
        calib[cam] = {
            "K": _np().asarray(load_intrinsics(sequence_dir / "camera_calib", cam), dtype="float64"),
            "E": _np().asarray(load_extrinsics(sequence_dir / "camera_calib", cam), dtype="float64"),
        }

    mesh_path = mesh_path or choose_canonical_mesh(object_name)
    if mesh_path is None or not mesh_path.exists():
        raise FileNotFoundError(f"Canonical mesh not found for multi_view_mask_pose: {mesh_path}")
    mesh_points = load_mesh_points(mesh_path)

    poses = []
    raw_poses = []
    frame_records = []
    prev_valid_pose = None
    prev_valid_theta = None
    debug_dir = output_dir / "debug_overlay"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_stride = max(1, len(grouped) // max(config.max_debug_overlays, 1))

    for n, (frame_idx, cam_paths) in enumerate(grouped.items()):
        view_records = []
        for cam in cameras:
            path = cam_paths.get(cam)
            if path is None:
                continue
            mask = load_mask(path)
            stats = mask_2d_stats(mask)
            area_ratio = stats["mask_area"] / float(mask.shape[0] * mask.shape[1])
            invalid_reason = stats["invalid_reason"]
            if invalid_reason is None and area_ratio > config.max_image_mask_area_ratio:
                invalid_reason = (
                    f"mask_area_ratio_too_large "
                    f"({area_ratio:.3f}>{config.max_image_mask_area_ratio:.3f})"
                )
            K = calib[cam]["K"]
            E = calib[cam]["E"]
            center = None
            ray = None
            if stats["centroid_x"] is not None:
                center, ray = camera_ray_from_pixel(K, E, [stats["centroid_x"], stats["centroid_y"]])
            view_records.append(
                {
                    "camera": cam,
                    "mask_path": str(path),
                    "mask": mask,
                    "K": K,
                    "E": E,
                    "view_valid": invalid_reason is None,
                    "view_invalid_reason": invalid_reason,
                    "mask_area": int(stats["mask_area"]),
                    "mask_area_ratio": float(area_ratio),
                    "centroid_2d": (
                        [float(stats["centroid_x"]), float(stats["centroid_y"])]
                        if stats["centroid_x"] is not None
                        else None
                    ),
                    "bbox_xyxy": stats["bbox_xyxy"],
                    "theta_2d_deg": (
                        float(math.degrees(stats["theta_rad"])) if stats["theta_rad"] is not None else None
                    ),
                    "camera_center": center,
                    "centroid_ray": ray,
                }
            )

        valid_views = [rec for rec in view_records if rec["view_valid"]]
        invalid_reason = None
        if len(valid_views) < config.min_multiview_views:
            invalid_reason = (
                f"insufficient_valid_views ({len(valid_views)}<{config.min_multiview_views})"
            )

        translation = None
        yaw = 0.0
        silhouette_score = 0.0
        per_view_scores: list[dict[str, Any]] = []
        centroid_jump = 0.0
        theta_jump = 0.0
        pose_jump = 0.0
        pose_raw = np.eye(4)

        if invalid_reason is None:
            translation = estimate_multiview_translation(view_records, config)
            if translation is None:
                invalid_reason = "plane_intersection_failed"

        if invalid_reason is None:
            yaw, silhouette_score, per_view_scores = optimize_yaw_by_multiview_projection(
                mesh_points, translation, view_records, yaw_ambiguous, config
            )
            pose_raw = yaw_transform_world(translation, yaw)
            if prev_valid_pose is not None:
                centroid_jump = float(np.linalg.norm(translation - prev_valid_pose[:3, 3]))
                theta_jump = angle_delta_deg(prev_valid_theta, yaw)
                pose_jump = centroid_jump
            if centroid_jump > config.max_centroid_jump_m:
                invalid_reason = (
                    f"centroid_jump ({centroid_jump:.4f}>{config.max_centroid_jump_m:.4f}m)"
                )
            if invalid_reason is None and not yaw_ambiguous and theta_jump > config.max_theta_jump_deg:
                invalid_reason = (
                    f"theta_jump ({theta_jump:.1f}>{config.max_theta_jump_deg:.1f}deg)"
                )
            if invalid_reason is None and pose_jump > config.max_pose_jump_m:
                invalid_reason = f"pose_jump ({pose_jump:.4f}>{config.max_pose_jump_m:.4f}m)"
            if invalid_reason is None and silhouette_score < config.min_silhouette_score:
                invalid_reason = (
                    f"silhouette_score_low ({silhouette_score:.4f}<{config.min_silhouette_score:.4f})"
                )

        valid = invalid_reason is None
        if valid:
            prev_valid_pose = pose_raw.copy()
            prev_valid_theta = yaw
            pose_out = pose_raw.copy()
        elif config.hold_invalid_pose and prev_valid_pose is not None:
            pose_out = prev_valid_pose.copy()
        else:
            pose_out = pose_raw.copy()

        poses.append(pose_out)
        raw_poses.append(pose_raw)
        serializable_views = []
        for rec in view_records:
            serializable_views.append(
                {
                    "camera": rec["camera"],
                    "mask_path": rec["mask_path"],
                    "view_valid": rec["view_valid"],
                    "view_invalid_reason": rec["view_invalid_reason"],
                    "mask_area": rec["mask_area"],
                    "mask_area_ratio": rec["mask_area_ratio"],
                    "centroid_2d": rec["centroid_2d"],
                    "bbox_xyxy": rec["bbox_xyxy"],
                    "theta_2d_deg": rec["theta_2d_deg"],
                }
            )
        total_mask_area = sum(int(rec["mask_area"]) for rec in view_records)
        max_mask_area_ratio = max(
            (float(rec["mask_area_ratio"]) for rec in view_records),
            default=0.0,
        )
        frame_records.append(
            {
                "frame_id": int(frame_idx),
                "valid": bool(valid),
                "invalid_reason": invalid_reason,
                "mask_area": int(total_mask_area),
                "max_mask_area_ratio": float(max_mask_area_ratio),
                "valid_view_count": int(len(valid_views)),
                "view_count": int(len(view_records)),
                "views": serializable_views,
                "silhouette_score": float(silhouette_score),
                "per_view_scores": per_view_scores,
                "plane_axis": int(config.plane_axis),
                "plane_value": float(config.plane_value),
                "translation": pose_out[:3, 3].tolist(),
                "rotation_matrix": pose_out[:3, :3].tolist(),
                "quaternion_xyzw": rotation_matrix_to_quaternion_xyzw(pose_out[:3, :3]).tolist(),
                "theta_rad": float(yaw),
                "theta_deg": float(math.degrees(yaw)),
                "centroid_jump_m": float(centroid_jump),
                "theta_jump_deg": float(theta_jump),
                "pose_jump_m": float(pose_jump),
            }
        )

        if n % debug_stride == 0:
            save_multiview_debug_overlay(
                mesh_points,
                pose_out,
                view_records,
                debug_dir / f"frame_{frame_idx:06d}.png",
            )

    poses_np = np.stack(poses)
    raw_np = np.stack(raw_poses)
    timestamps = np.array([rec["frame_id"] for rec in frame_records], dtype="float64")
    return save_tracking_outputs(
        output_dir=output_dir,
        object_name=object_name,
        sequence_name=sequence_dir.name,
        method="multi_view_mask_pose",
        transform_name="T_world_object",
        poses=poses_np,
        raw_poses=raw_np,
        timestamps=timestamps,
        frame_records=frame_records,
        yaw_ambiguous=yaw_ambiguous,
        mesh_path=mesh_path,
        limitation=(
            "Depth-free calibrated multi-view mask fitting. This is an approximate "
            "video-derived pose, not RGB-D ICP or GT."
        ),
        output_stem="object_trajectory_multiview_pose",
    )


def save_multiview_debug_overlay(mesh_points, pose, view_records: list[dict[str, Any]], output_path: Path) -> None:
    np = _np()
    cv2 = _cv2()
    panels = []
    pts_world = (pose[:3, :3] @ mesh_points.T).T + pose[:3, 3]
    for rec in view_records:
        mask = rec["mask"]
        panel = np.zeros((*mask.shape, 3), dtype="uint8")
        panel[..., 1] = 50
        panel[mask.astype(bool)] = (0, 180, 255)
        uv = project_world_points(pts_world, rec["K"], rec["E"], mask.shape)
        if len(uv):
            xy = np.round(uv).astype(int)
            h, w = mask.shape[:2]
            xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
            panel[xy[:, 1], xy[:, 0]] = (255, 40, 40)
        cv2.putText(
            panel,
            rec["camera"],
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    if not panels:
        return
    target_h = min(panel.shape[0] for panel in panels)
    resized = []
    for panel in panels:
        scale = target_h / panel.shape[0]
        resized.append(cv2.resize(panel, (int(panel.shape[1] * scale), target_h)))
    canvas = np.concatenate(resized, axis=1)
    cv2.imwrite(str(output_path), canvas)


def track_image_plane_sequence(
    sequence_dir: Path,
    output_dir: Path,
    object_name: Optional[str] = None,
    camera: str = "camera_top",
    mask_paths: Optional[list[Path]] = None,
    config: Optional[TrackingConfig] = None,
) -> dict[str, Any]:
    np = _np()
    config = config or TrackingConfig(camera=camera)
    object_name = object_name or infer_object_name(sequence_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_paths = mask_paths or discover_masks(sequence_dir, object_name, sequence_dir.name, camera)
    mask_paths = select_one_mask_per_frame(mask_paths)
    if not mask_paths:
        raise ValueError(f"No masks found for image-plane tracking: {sequence_dir}")

    poses = []
    records = []
    prev_centroid = None
    debug_dir = output_dir / "debug_overlay"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_stride = max(1, len(mask_paths) // 12)
    for mask_path in sorted(mask_paths):
        mask = load_mask(mask_path)
        stats_2d = mask_2d_stats(mask)
        frame_idx = frame_index_from_path(mask_path) or len(records)
        area_ratio = stats_2d["mask_area"] / float(mask.shape[0] * mask.shape[1])
        invalid_reason = stats_2d["invalid_reason"]
        if invalid_reason is None and area_ratio > config.max_image_mask_area_ratio:
            invalid_reason = (
                f"mask_area_ratio_too_large "
                f"({area_ratio:.3f}>{config.max_image_mask_area_ratio:.3f})"
            )
        valid = invalid_reason is None
        if stats_2d["centroid_x"] is None:
            raw_centroid = np.zeros(3)
        else:
            raw_centroid = np.array([stats_2d["centroid_x"], stats_2d["centroid_y"], 0.0])
        if valid:
            centroid = raw_centroid
            jump = float(np.linalg.norm(centroid[:2] - prev_centroid[:2])) if prev_centroid is not None else 0.0
            prev_centroid = centroid
        else:
            centroid = prev_centroid if prev_centroid is not None else np.zeros(3)
            jump = 0.0
        pose = np.eye(4)
        pose[:3, 3] = centroid
        poses.append(pose)
        records.append(
            {
                "frame_id": int(frame_idx),
                "mask_path": str(mask_path),
                "valid": bool(valid),
                "invalid_reason": invalid_reason,
                "mask_area": int(stats_2d["mask_area"]),
                "mask_area_ratio": float(area_ratio),
                "depth_valid_ratio": None,
                "centroid": centroid.tolist(),
                "raw_centroid": raw_centroid.tolist(),
                "bbox_xyxy": stats_2d["bbox_xyxy"],
                "theta_rad": float(stats_2d["theta_rad"] or 0.0),
                "theta_deg": float(math.degrees(stats_2d["theta_rad"] or 0.0)),
                "centroid_jump_px": jump,
                "theta_jump_deg": 0.0,
                "icp_fitness": None,
                "icp_rmse": None,
            }
        )
        if len(records) == 1 or (len(records) - 1) % debug_stride == 0:
            save_mask_debug_overlay(mask, debug_dir / f"frame_{frame_idx:06d}.png")
    poses_np = np.stack(poses)
    timestamps = np.array([rec["frame_id"] for rec in records], dtype="float64")
    return save_tracking_outputs(
        output_dir=output_dir,
        object_name=object_name,
        sequence_name=sequence_dir.name,
        method="image_plane_pose",
        transform_name="T_image_object",
        poses=poses_np,
        raw_poses=poses_np,
        timestamps=timestamps,
        frame_records=records,
        yaw_ambiguous=object_name in YAW_AMBIGUOUS_OBJECTS,
        mesh_path=None,
        limitation="No depth was available; this is not a full 6D pose. Translation is reported in image pixels.",
        output_stem="object_trajectory_image_plane",
    )


def mask_2d_stats(mask) -> dict[str, Any]:
    np = _np()
    ys, xs = np.nonzero(mask)
    area = int(len(xs))
    if area == 0:
        return {
            "mask_area": 0,
            "centroid_x": None,
            "centroid_y": None,
            "bbox_xyxy": None,
            "theta_rad": None,
            "invalid_reason": "empty_mask",
        }
    cx = float(xs.mean())
    cy = float(ys.mean())
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x_centered = xs.astype("float64") - cx
    y_centered = ys.astype("float64") - cy
    mu20 = float((x_centered * x_centered).sum())
    mu02 = float((y_centered * y_centered).sum())
    mu11 = float((x_centered * y_centered).sum())
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02) if (mu20 + mu02) > 0 else 0.0
    return {
        "mask_area": area,
        "centroid_x": cx,
        "centroid_y": cy,
        "bbox_xyxy": [x0, y0, x1, y1],
        "theta_rad": theta,
        "invalid_reason": None,
    }


def save_mask_debug_overlay(mask, output_path: Path) -> None:
    np = _np()
    cv2 = _cv2()
    canvas = np.zeros((*mask.shape, 3), dtype="uint8")
    canvas[..., 1] = 60
    canvas[mask.astype(bool)] = (0, 180, 255)
    cv2.imwrite(str(output_path), canvas)


def extract_gt_trajectory(sequence_dir: Path, object_name: str, output_dir: Path) -> dict[str, Any]:
    np = _np()
    h5py = _h5py()
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = sequence_dir / "pose_3d.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        for group in relevant_object_groups(f, object_name):
            mask_path = f"{group}/pose/mask"
            pose_path = f"{group}/pose/obj_transf"
            ts_path = f"{group}/pose/timestamp"
            if mask_path not in f or pose_path not in f:
                continue
            valid_mask = f[mask_path][()].astype(bool)
            if not valid_mask.any():
                continue
            poses = f[pose_path][()]
            timestamps = f[ts_path][()] if ts_path in f else np.arange(len(poses))
            records = []
            for i, pose in enumerate(poses):
                records.append(
                    {
                        "frame_id": i,
                        "valid": bool(valid_mask[i]),
                        "invalid_reason": None if valid_mask[i] else "gt_pose_mask_false",
                        "mask_area": None,
                        "depth_valid_ratio": None,
                        "centroid": pose[:3, 3].tolist(),
                        "theta_rad": 0.0,
                        "theta_deg": 0.0,
                        "centroid_jump_m": 0.0,
                        "theta_jump_deg": 0.0,
                        "icp_fitness": None,
                        "icp_rmse": None,
                    }
                )
            return save_tracking_outputs(
                output_dir=output_dir,
                object_name=object_name,
                sequence_name=sequence_dir.name,
                method="use_gt_pose",
                transform_name="T_world_object",
                poses=poses,
                raw_poses=poses,
                timestamps=timestamps,
                frame_records=records,
                yaw_ambiguous=object_name in YAW_AMBIGUOUS_OBJECTS,
                mesh_path=None,
            )
    raise ValueError(f"No valid relevant GT pose found in {hdf5_path}")


def save_tracking_outputs(
    output_dir: Path,
    object_name: str,
    sequence_name: str,
    method: str,
    transform_name: str,
    poses,
    raw_poses,
    timestamps,
    frame_records: list[dict[str, Any]],
    yaw_ambiguous: bool,
    mesh_path: Optional[Path],
    limitation: Optional[str] = None,
    output_stem: Optional[str] = None,
) -> dict[str, Any]:
    np = _np()
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_count = sum(1 for rec in frame_records if rec.get("valid"))
    invalid_reasons: dict[str, int] = {}
    for rec in frame_records:
        reason = rec.get("invalid_reason")
        if reason:
            key = str(reason).split(" (", 1)[0]
            invalid_reasons[key] = invalid_reasons.get(key, 0) + 1

    if output_stem is None:
        output_stem = output_stem_for_method(method)

    npz_path = output_dir / f"{output_stem}.npz"
    np.savez(
        npz_path,
        poses=poses,
        raw_poses=raw_poses,
        timestamps=timestamps,
        valid=np.array([rec.get("valid", False) for rec in frame_records], dtype=bool),
    )

    json_path = output_dir / f"{output_stem}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "object_name": object_name,
                "sequence_name": sequence_name,
                "method": method,
                "transform_name": transform_name,
                "num_frames": len(frame_records),
                "num_valid": valid_count,
                "yaw_ambiguous": yaw_ambiguous,
                "limitation": limitation,
                "canonical_mesh": str(mesh_path) if mesh_path else None,
                "external_references": method_references(method),
                "poses": poses.tolist(),
                "timestamps": timestamps.tolist(),
                "frames": frame_records,
            },
            f,
            indent=2,
        )

    report_path = output_dir / "trajectory_quality_report.json"
    report = {
        "object_name": object_name,
        "sequence_name": sequence_name,
        "method": method,
        "num_frames": len(frame_records),
        "num_valid": valid_count,
        "valid_ratio": valid_count / max(len(frame_records), 1),
        "invalid_reasons": invalid_reasons,
        "yaw_ambiguity": (
            "Object is approximately axisymmetric; yaw is reported but not trusted."
            if yaw_ambiguous
            else None
        ),
        "limitation": limitation,
        "external_references": method_references(method),
        "metrics": summarize_frame_metrics(frame_records),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    plot_path = output_dir / "trajectory_plot.png"
    save_trajectory_plot(poses, frame_records, plot_path, method)

    return {
        "json_path": json_path,
        "npz_path": npz_path,
        "report_path": report_path,
        "plot_path": plot_path,
        "num_frames": len(frame_records),
        "num_valid": valid_count,
        "method": method,
    }


def output_stem_for_method(method: str) -> str:
    if method == "mask_depth_icp":
        return "object_trajectory_mask_depth_icp"
    if method == "multi_view_mask_pose":
        return "object_trajectory_multiview_pose"
    if method == "image_plane_pose":
        return "object_trajectory_image_plane"
    if method == "use_gt_pose":
        return "object_trajectory_gt"
    return "object_trajectory_mask_pose"


def method_references(method: str) -> list[dict[str, str]]:
    refs = {
        "mask_depth_icp": [
            {
                "name": "Open3D ICP registration",
                "url": "https://www.open3d.org/docs/latest/python_api/open3d.pipelines.registration.registration_icp.html",
                "role": "Point-cloud ICP registration backend; reports transformation, fitness, and inlier RMSE.",
            },
            {
                "name": "FoundationPose",
                "url": "https://github.com/NVlabs/FoundationPose",
                "role": "Future high-end RGB-D + mesh 6D pose tracking option, not used in this baseline.",
            },
        ],
        "image_plane_pose": [
            {
                "name": "SAM2",
                "url": "https://github.com/facebookresearch/sam2",
                "role": "Future mask propagation option; current fallback only consumes existing masks.",
            },
            {
                "name": "Track-Anything",
                "url": "https://github.com/gaomingqi/Track-Anything",
                "role": "Future video mask propagation option, not direct pose estimation.",
            },
            {
                "name": "CoTracker",
                "url": "https://github.com/facebookresearch/co-tracker",
                "role": "Future dynamic prompt/point tracking option, not direct 6D pose.",
            },
            {
                "name": "TAPIR",
                "url": "https://deepmind-tapir.github.io/",
                "role": "Future point tracking option, not direct 6D pose.",
            },
        ],
        "multi_view_mask_pose": [
            {
                "name": "Calibrated multi-view silhouette fitting",
                "url": "https://en.wikipedia.org/wiki/Visual_hull",
                "role": "Depth-free pose evidence from synchronized masks and calibrated cameras; used here as the current-data fallback.",
            },
            {
                "name": "3D Gaussian Splatting",
                "url": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/",
                "role": "Future multi-view video reconstruction/tracking option; not used in this lightweight baseline.",
            },
            {
                "name": "Nerfstudio",
                "url": "https://docs.nerf.studio/",
                "role": "Future NeRF/3DGS tooling option for longer multi-view fitting jobs.",
            },
        ],
    }
    return refs.get(method, [])


def summarize_frame_metrics(frame_records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = (
        "mask_area",
        "mask_area_ratio",
        "max_mask_area_ratio",
        "depth_valid_ratio",
        "point_count",
        "centroid_jump_m",
        "centroid_jump_px",
        "theta_jump_deg",
        "pose_jump_m",
        "icp_fitness",
        "icp_rmse",
        "valid_view_count",
        "view_count",
        "silhouette_score",
    )
    summary: dict[str, Any] = {}
    for key in numeric_keys:
        vals = [rec.get(key) for rec in frame_records if rec.get(key) is not None]
        if not vals:
            summary[key] = None
            continue
        summary[key] = {
            "min": float(min(vals)),
            "max": float(max(vals)),
            "mean": float(sum(vals) / len(vals)),
        }
    return summary


def save_trajectory_plot(poses, frame_records: list[dict[str, Any]], output_path: Path, method: str) -> None:
    np = _np()
    plt = _plt()
    valid = np.array([rec.get("valid", False) for rec in frame_records], dtype=bool)
    xyz = poses[:, :3, 3]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    y_axis = 1 if method == "image_plane_pose" else 2
    y_label = "y_px" if method == "image_plane_pose" else "z"
    axes[0].plot(xyz[:, 0], xyz[:, y_axis], color="0.65", linewidth=1)
    axes[0].scatter(xyz[valid, 0], xyz[valid, y_axis], c="green", s=18, label="valid")
    axes[0].scatter(xyz[~valid, 0], xyz[~valid, y_axis], c="red", marker="x", s=35, label="invalid")
    axes[0].set_title("Trajectory X/Y" if method == "image_plane_pose" else "Trajectory X/Z")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(y_label)
    if method == "image_plane_pose":
        axes[0].invert_yaxis()
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    frames = [rec["frame_id"] for rec in frame_records]
    mask_area = [rec.get("mask_area") or 0 for rec in frame_records]
    colors = ["green" if rec.get("valid") else "red" for rec in frame_records]
    axes[1].bar(frames, mask_area, color=colors, alpha=0.7)
    axes[1].set_title("Mask Area")
    axes[1].set_xlabel("frame")
    axes[1].grid(True, alpha=0.3)

    theta = [abs(rec.get("theta_deg") or 0.0) for rec in frame_records]
    axes[2].plot(frames, theta, color="0.3", linewidth=1)
    axes[2].scatter(frames, theta, c=colors, s=18)
    axes[2].set_title("Theta")
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("deg")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Mask-driven pose tracking: {method}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def interpolate_invalid_poses(poses, valid):
    """Linearly fill translations for invalid frames between valid neighbors."""
    np = _np()
    out = poses.copy()
    valid = np.asarray(valid, dtype=bool)
    idx = np.arange(len(out))
    valid_idx = idx[valid]
    if len(valid_idx) < 2:
        return out
    for axis in range(3):
        out[:, axis, 3] = np.interp(idx, valid_idx, out[valid_idx, axis, 3])
    return out


def poses_to_trajectory(
    poses,
    timestamps,
    output_path: Path,
    format: str = "npy",
) -> Path:
    np = _np()
    data = {"poses": poses, "timestamps": timestamps}
    if format == "npy":
        np.save(output_path, data)
    elif format == "json":
        with open(output_path, "w") as f:
            json.dump(
                {
                    "poses": poses.tolist(),
                    "timestamps": timestamps.tolist(),
                },
                f,
                indent=2,
            )
    elif format == "csv":
        flat = poses.reshape(len(poses), 16)
        np.savetxt(output_path, np.hstack([timestamps[:, None], flat]), delimiter=",")
    return output_path


def register_to_gt(
    reconstructed_mesh: Path,
    gt_trajectory: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Compatibility wrapper for older callers."""
    object_name = infer_object_name(gt_trajectory.parent.name)
    return extract_gt_trajectory(gt_trajectory.parent, object_name, output_dir)


def track_foundationpose(
    frames_dir: Path,
    object_mesh: Path,
    camera_params: Path,
    output_dir: Path,
) -> dict[str, Any]:
    raise NotImplementedError(
        "FoundationPose is not the primary method in Phase 5. Use mask/depth tracking."
    )


def track_from_3dgs(
    gaussian_splat: "GaussianModel",
    camera_views: list,
    output_dir: Path,
) -> dict[str, Any]:
    raise NotImplementedError("3DGS pose tracking is outside this mask-driven baseline.")
