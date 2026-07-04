#!/usr/bin/env python3
"""
Multi-view 3D pose tracking from SAM masks on 3 calibrated cameras.

Uses triangulation from 3 camera views to estimate object 3D position.
Robust to single-view occlusion — falls back to 2-view or 1-view.

Usage:
  python3.8 scripts/multiview_pose_tracking.py \
    --object bread \
    --output outputs/multiview_pose/bread
"""

import argparse, json, sys
from pathlib import Path
import numpy as np


OUTPUT_DIR = Path("/mnt/workspace/AIlab-E6/outputs/pose_tracking")
CALIB_DIR = Path("/mnt/workspace/AIlab-E6/data/calib")
CAMERAS = ["camera_top", "camera_side_1", "camera_side_2"]


def load_2d_keypoints(obj, cam):
    """Load 2D keypoints from unified_pose_pipeline output."""
    path = OUTPUT_DIR / f"{obj}_{cam}" / "object_trajectory_2d.json"
    if not path.exists():
        # Try alternate naming
        alt = OUTPUT_DIR / obj / "object_trajectory_2d.json"
        if cam == "camera_top" and alt.exists():
            path = alt
        else:
            return None
    with open(path) as f:
        data = json.load(f)
    return data["keypoints"]


def load_calibration(obj, cam):
    """Load camera intrinsics (3x3) and extrinsics (4x4)."""
    path = CALIB_DIR / obj / f"{cam}.npz"
    data = np.load(path)
    return data["K"], data["P"]


def pixel_to_ray(K, P, cx, cy):
    """Convert pixel coordinate to world-space ray (origin, direction)."""
    K_inv = np.linalg.inv(K)
    pixel = np.array([cx, cy, 1.0])
    ray_cam = K_inv @ pixel
    ray_world = P[:3, :3].T @ ray_cam
    ray_world = ray_world / np.linalg.norm(ray_world)
    origin = P[:3, 3]
    return origin, ray_world


def triangulate_from_rays(rays):
    """Triangulate 3D point from multiple (origin, direction) rays.
    
    Uses DLT-like midpoint-of-closest-approach for 2+ rays.
    Falls back to ground-plane projection for single ray.
    """
    if len(rays) < 1:
        return None

    if len(rays) == 1:
        # Single ray: project to ground plane (z=0)
        origin, direction = rays[0]
        if abs(direction[2]) < 1e-6:
            t = 10.0  # arbitrary far distance
        else:
            t = -origin[2] / direction[2]
        return origin + direction * t

    # 2+ rays: find closest point to all rays via least squares
    # For each pair of rays, find the midpoint of their closest approach
    points = []
    for i in range(len(rays)):
        o1, d1 = rays[i]
        for j in range(i + 1, len(rays)):
            o2, d2 = rays[j]
            # Closest approach between two lines
            n = np.cross(d1, d2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue  # parallel rays, skip
            n = n / n_norm
            # Solve: o1 + t1*d1 + s*n = o2 + t2*d2
            # => [d1  -d2  n] * [t1 t2 s]^T = o2 - o1
            A = np.column_stack([d1, -d2, n])
            try:
                params = np.linalg.solve(A, o2 - o1)
                p1 = o1 + params[0] * d1
                p2 = o2 + params[1] * d2
                points.append((p1 + p2) / 2)
            except np.linalg.LinAlgError:
                continue

    if not points:
        return rays[0][0] + rays[0][1] * 10.0  # fallback
    return np.mean(points, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-image-mask-area-ratio", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all 3 camera calibrations
    calibs = {}
    for cam in CAMERAS:
        try:
            K, P = load_calibration(args.object, cam)
            calibs[cam] = {"K": K, "P": P}
        except Exception as e:
            print(f"  Warning: no calibration for {cam}: {e}")

    # Load 2D keypoints for all cameras
    kps_by_cam = {}
    for cam in CAMERAS:
        kps = load_2d_keypoints(args.object, cam)
        if kps is not None:
            kps_by_cam[cam] = kps
            print(f"  {cam}: {len(kps)} keypoints")

    if not kps_by_cam:
        print(f"ERROR: No keypoints found for {args.object}")
        sys.exit(1)

    # Find the maximum frame count
    max_frames = max(len(v) for v in kps_by_cam.values())

    poses_4x4 = []
    valid_mask = []
    details = []

    for frame_idx in range(max_frames):
        rays = []
        active_cams = []
        areas = []

        for cam in CAMERAS:
            if cam not in kps_by_cam:
                continue
            if frame_idx >= len(kps_by_cam[cam]):
                continue
            kp = kps_by_cam[cam][frame_idx]
            if kp.get("area", -1) < 0:
                continue  # invalid frame
            if cam not in calibs:
                continue

            K = calibs[cam]["K"]
            P = calibs[cam]["P"]
            origin, direction = pixel_to_ray(K, P, kp["cx"], kp["cy"])
            rays.append((origin, direction))
            active_cams.append(cam)
            areas.append(kp["area"])

        pt_3d = triangulate_from_rays(rays)

        if pt_3d is not None:
            T = np.eye(4)
            T[:3, 3] = pt_3d
            poses_4x4.append(T)
            valid_mask.append(True)
            details.append({
                "frame": frame_idx,
                "cameras_used": active_cams,
                "position_xyz": pt_3d.tolist(),
            })
        else:
            valid_mask.append(False)
            details.append({"frame": frame_idx, "cameras_used": [], "error": "no valid rays"})

    poses_4x4 = np.stack(poses_4x4) if poses_4x4 else np.zeros((0, 4, 4))
    valid_mask = np.array(valid_mask)

    # Save
    np.savez(output_dir / "object_trajectory_multiview.npz",
             obj_transf=poses_4x4, mask=valid_mask)

    with open(output_dir / "object_trajectory_multiview.json", "w") as f:
        json.dump({
            "object": args.object,
            "method": "3-camera triangulation from SAM masks",
            "num_frames": max_frames,
            "num_valid": int(valid_mask.sum()),
            "cameras_available": list(kps_by_cam.keys()),
            "details": details,
        }, f, indent=2)

    # Report
    n_valid = int(valid_mask.sum())
    print(f"\n[{args.object}] Multi-view trajectory:")
    print(f"  Valid frames: {n_valid}/{max_frames} ({n_valid/max_frames*100:.0f}%)")
    if n_valid > 0:
        valid_poses = poses_4x4[valid_mask]
        print(f"  Position range X: [{valid_poses[:,0,3].min():.3f}, {valid_poses[:,0,3].max():.3f}]")
        print(f"  Position range Y: [{valid_poses[:,1,3].min():.3f}, {valid_poses[:,1,3].max():.3f}]")
        print(f"  Position range Z: [{valid_poses[:,2,3].min():.3f}, {valid_poses[:,2,3].max():.3f}]")

    with open(output_dir / "trajectory_quality_report.json", "w") as f:
        json.dump({
            "object": args.object, "method": "multi_view_triangulation",
            "num_frames_total": max_frames, "num_frames_valid": n_valid,
            "valid_ratio": n_valid / max(max_frames, 1), "cameras": list(kps_by_cam.keys()),
        }, f, indent=2)

    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
