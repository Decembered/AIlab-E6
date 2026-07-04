#!/usr/bin/env python3
"""Lightweight synthetic checks for the mask/depth pose baseline."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.object_recon.pose_tracking import (
    TrackingConfig,
    backproject_mask_depth,
    group_multiview_masks,
    has_valid_gt_pose,
    mask_2d_stats,
    pca_pose,
    project_world_points,
    recommended_mode,
    run_icp,
    score_projected_points_against_mask,
)


def test_recommended_modes() -> None:
    assert recommended_mode(True, True, False, False) == "use_gt_pose"
    assert recommended_mode(False, True, True, True) == "mask_depth_icp"
    assert recommended_mode(False, True, True, False) == "image_plane_pose"
    assert recommended_mode(False, False, False, False) == "need_segmentation_or_manual"


def test_backproject() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    depth = np.ones((4, 4), dtype=np.float64)
    K = np.array([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]])
    pts, metrics = backproject_mask_depth(
        mask, depth, K, TrackingConfig(min_mask_area=1, min_point_count=1)
    )
    assert pts.shape == (4, 3)
    assert metrics["invalid_reason"] is None
    assert np.allclose(pts[:, 2], 1.0)


def test_pca_pose() -> None:
    x = np.linspace(-1.0, 1.0, 20)
    pts = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    T, theta, eigvals, _ = pca_pose(pts)
    assert T.shape == (4, 4)
    assert eigvals[0] > eigvals[1]
    assert abs(abs(theta) - np.pi / 2.0) < 1e-6


def test_quality_gate_depth_ratio() -> None:
    mask = np.ones((10, 10), dtype=bool)
    depth = np.zeros((10, 10), dtype=np.float64)
    K = np.eye(3)
    pts, metrics = backproject_mask_depth(
        mask, depth, K, TrackingConfig(min_mask_area=1, min_depth_valid_ratio=0.5)
    )
    assert pts.shape == (0, 3)
    assert metrics["invalid_reason"].startswith("depth_valid_ratio_low")


def test_hdf5_all_false_mask_if_h5py_available() -> None:
    try:
        import h5py
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pose_3d.hdf5"
        with h5py.File(path, "w") as f:
            pose = f.create_group("obj").create_group("pose")
            pose.create_dataset("mask", data=np.zeros(3, dtype=bool))
            pose.create_dataset("obj_transf", data=np.zeros((3, 4, 4)))
        assert has_valid_gt_pose(path, "bread") is False


def test_secondary_gt_not_assigned_to_primary_if_h5py_available() -> None:
    try:
        import h5py
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pose_3d.hdf5"
        with h5py.File(path, "w") as f:
            rh = f.create_group("obj_rh")
            rh.create_dataset("obj_id", data=b"Pipette #1")
            rh_pose = rh.create_group("pose")
            rh_pose.create_dataset("mask", data=np.zeros(3, dtype=bool))
            rh_pose.create_dataset("obj_transf", data=np.zeros((3, 4, 4)))
            lh = f.create_group("obj_lh")
            lh.create_dataset("obj_id", data=b"Test Tube #1")
            lh_pose = lh.create_group("pose")
            lh_pose.create_dataset("mask", data=np.ones(3, dtype=bool))
            lh_pose.create_dataset("obj_transf", data=np.zeros((3, 4, 4)))
        assert has_valid_gt_pose(path, "pipette") is False


def test_image_mask_stats_and_area_ratio_gate() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    stats = mask_2d_stats(mask)
    assert stats["mask_area"] == 12
    assert stats["bbox_xyxy"] == [3, 2, 6, 4]
    ratio = stats["mask_area"] / mask.size
    assert ratio > TrackingConfig(max_image_mask_area_ratio=0.1).max_image_mask_area_ratio


def test_multiview_projection_score() -> None:
    K = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    E = np.eye(4)
    pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]])
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True
    uv = project_world_points(pts, K, E, mask.shape)
    score, detail = score_projected_points_against_mask(uv, mask, [4, 4, 7, 7])
    assert score > 0.0
    assert detail["projected_points"] == 3


def test_group_multiview_masks() -> None:
    grouped = group_multiview_masks(
        {
            "camera_top": [Path("camera_top_frame_000115_mask.png")],
            "camera_side_1": [Path("camera_side_1_frame_000115_mask.png")],
        }
    )
    assert 115 in grouped
    assert set(grouped[115]) == {"camera_top", "camera_side_1"}


def test_icp_recovery_if_open3d_available() -> None:
    try:
        import open3d as o3d
    except ImportError:
        return
    source = np.array(
        [
            [-0.05, -0.02, 0.0],
            [0.05, -0.02, 0.0],
            [-0.05, 0.02, 0.0],
            [0.05, 0.02, 0.0],
            [-0.05, -0.02, 0.08],
            [0.05, -0.02, 0.08],
            [-0.05, 0.02, 0.08],
            [0.05, 0.02, 0.08],
        ],
        dtype=np.float64,
    )
    T = np.eye(4)
    T[:3, 3] = [0.12, -0.03, 0.4]
    target = source + T[:3, 3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(source)
    result, fitness, rmse = run_icp(target, pcd, T, TrackingConfig(voxel_size=0.0))
    assert fitness > 0.9
    assert rmse < 1e-6
    assert np.allclose(result[:3, 3], T[:3, 3], atol=1e-5)


def main() -> None:
    test_recommended_modes()
    test_backproject()
    test_pca_pose()
    test_quality_gate_depth_ratio()
    test_hdf5_all_false_mask_if_h5py_available()
    test_secondary_gt_not_assigned_to_primary_if_h5py_available()
    test_image_mask_stats_and_area_ratio_gate()
    test_multiview_projection_score()
    test_group_multiview_masks()
    test_icp_recovery_if_open3d_available()
    print("mask/depth pose synthetic tests passed")


if __name__ == "__main__":
    main()
