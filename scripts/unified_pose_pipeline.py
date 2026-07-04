#!/usr/bin/env python3
"""
Unified Object Pose Tracking Pipeline (Member C, Sub-task 3.3)
-
Takes video frames + SAM checkpoint → masks → multi-view pose trajectory.

Usage:
  python3.8 scripts/unified_pose_pipeline.py \
    --object bread \
    --frames /mnt/workspace/AIlab-E6/data/frames/bread_top \
    --output /mnt/workspace/AIlab-E6/outputs/pose_tracking/bread

Dependencies: segment-anything, opencv-python, numpy, scipy, matplotlib
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import ConvexHull


def load_sam(model_type="vit_b", checkpoint=None):
    from segment_anything import sam_model_registry, SamPredictor
    if checkpoint is None:
        checkpoint = os.path.expanduser("~/.cache/sam/sam_vit_b_01ec64.pth")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    return SamPredictor(sam)


def generate_masks(predictor, frames_dir, output_dir, prompt_config, frame_stride=3):
    """Generate SAM masks for all frames using dynamic prompt tracking."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    masks = []
    valid = []
    last_centroid = None
    last_bbox = None

    area_range = prompt_config.get("area_range", [3000, 35000])
    max_centroid_jump = prompt_config.get("max_centroid_jump_px", 50)

    for i, img_path in enumerate(image_paths):
        if i % frame_stride != 0:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            valid.append(False)
            masks.append(None)
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        # Build prompts from object-specific config (fraction-of-image → pixels)
        h, w = image.shape[:2]
        pos_rel = prompt_config.get("positive_points", [[0.5, 0.5]])
        neg_rel = prompt_config.get("negative_points", [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
        box_rel = prompt_config.get("box", [0.25, 0.25, 0.75, 0.75])

        input_points = [[int(r[0]*w), int(r[1]*h)] for r in pos_rel]
        input_labels = [1] * len(input_points)
        neg_pts = [[int(r[0]*w), int(r[1]*h)] for r in neg_rel]
        input_points.extend(neg_pts)
        input_labels.extend([0] * len(neg_pts))

        # Skin detection negatives (for hand-held objects like pipette)
        if prompt_config.get("use_skin_neg", False):
            r = image_rgb[:,:,0].astype(float)
            g = image_rgb[:,:,1].astype(float)
            b = image_rgb[:,:,2].astype(float)
            skin = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (np.abs(r-g) > 15)
            skin_ys, skin_xs = np.where(skin)
            if len(skin_ys) > 0:
                indices = np.linspace(0, len(skin_ys)-1, min(8, len(skin_ys)), dtype=int)
                for idx in indices:
                    input_points.append([skin_xs[idx], skin_ys[idx]])
                    input_labels.append(0)

        prompt_box = np.array([int(box_rel[0]*w), int(box_rel[1]*h),
                               int(box_rel[2]*w), int(box_rel[3]*h)])
        sam_masks, scores, _ = predictor.predict(
            point_coords=np.array(input_points) if input_points else None,
            point_labels=np.array(input_labels) if input_labels else None,
            box=prompt_box,
            multimask_output=True,
        )

        # Select best mask: prefer within area_range, then highest score
        areas = np.array([m.sum() for m in sam_masks])
        in_range = (areas >= area_range[0]) & (areas <= area_range[1])
        if np.any(in_range):
            # Pick highest-scoring mask within range
            best_idx = np.argmax(scores * in_range.astype(float))
        else:
            # Fallback: pick highest-scoring mask
            best_idx = np.argmax(scores)
        mask = sam_masks[best_idx].astype(np.uint8) * 255
        area = sam_masks[best_idx].sum()  # pixel count (before * 255)

        # Quality gate
        if area < area_range[0] or area > area_range[1]:
            valid.append(False)
            # Don't update tracking for bad frames
        else:
            valid.append(True)
            # Update tracking state
            ys, xs = np.where(mask > 127)
            if len(xs) > 10:
                last_centroid = (int(xs.mean()), int(ys.mean()))
                last_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        masks.append(mask)

        # Save mask image
        out_path = output_dir / f"{img_path.stem}_mask.png"
        if mask is not None:
            cv2.imwrite(str(out_path), mask)

    return masks, valid


def masks_to_centroid_trajectory(masks, valid, image_shape):
    """Convert per-frame masks to 2D centroid trajectory."""
    h, w = image_shape
    centroids = []
    last_valid_centroid = None

    for i, (mask, v) in enumerate(zip(masks, valid)):
        if v and mask is not None:
            ys, xs = np.where(mask > 127)
            if len(xs) > 10:
                cx, cy = float(xs.mean()), float(ys.mean())
                centroids.append([i, cx, cy, float(mask.sum()), 1])
                last_valid_centroid = (cx, cy)
                continue

        if last_valid_centroid is not None:
            centroids.append([i, last_valid_centroid[0], last_valid_centroid[1], -1, 0])
        else:
            centroids.append([i, w/2, h/2, -1, 0])

    return centroids


def image_plane_to_3d(centroids, K, cam_extrinsic, ground_height=0.0):
    """Backproject 2D centroids to 3D using ground plane assumption."""
    K_inv = np.linalg.inv(K)
    poses = []
    valid_mask = []

    for c in centroids:
        frame_idx, cx, cy, area, v = c
        if v == 0:
            valid_mask.append(False)
            if len(poses) > 0:
                poses.append(poses[-1].copy())
            else:
                poses.append(np.eye(4))
            continue

        # Backproject to ground plane
        pixel_homogeneous = np.array([cx, cy, 1.0])
        ray_camera = K_inv @ pixel_homogeneous
        ray_world = cam_extrinsic[:3, :3].T @ ray_camera
        cam_center = cam_extrinsic[:3, 3]

        t = (ground_height - cam_center[2]) / max(ray_world[2], 1e-6)
        world_point = cam_center + ray_world * t

        T = np.eye(4)
        T[:3, 3] = world_point
        poses.append(T)
        valid_mask.append(True)

    return np.stack(poses, axis=0), np.array(valid_mask)


def smooth_trajectory(poses, valid_mask, window=5):
    """Smooth trajectory with a moving average filter."""
    smoothed = poses.copy()
    for i in range(len(poses)):
        start = max(0, i - window // 2)
        end = min(len(poses), i + window // 2 + 1)
        valid_window = [j for j in range(start, end) if valid_mask[j]]
        if len(valid_window) >= 2:
            smoothed[i, :3, 3] = np.mean([poses[j, :3, 3] for j in valid_window], axis=0)
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True, choices=["bread", "pipette", "drink_ad", "drink_yykx"])
    parser.add_argument("--frames", required=True, type=Path, help="Directory with frame_*.png")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--ground-height", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Object-specific SAM prompts (from phase2_multiview_all.py — manually optimized)
    configs = {
        "bread": {
            "area_range": [3000, 35000],
            "max_centroid_jump_px": 50,
            "positive_points": [[0.50, 0.48], [0.46, 0.50], [0.54, 0.50]],
            "negative_points": [[0.50, 0.25], [0.50, 0.72], [0.30, 0.50], [0.70, 0.50], [0.15, 0.50], [0.85, 0.50]],
            "box": [0.25, 0.28, 0.75, 0.68],
            "use_skin_neg": False,
        },
        "pipette": {
            "area_range": [20000, 150000],
            "max_centroid_jump_px": 100,
            "positive_points": [[0.50, 0.50], [0.38, 0.49], [0.62, 0.51], [0.30, 0.50], [0.70, 0.50]],
            "negative_points": [[0.01, 0.01], [0.99, 0.01], [0.01, 0.99], [0.99, 0.99],
                                [0.50, 0.40], [0.50, 0.65], [0.50, 0.75], [0.25, 0.50], [0.75, 0.50]],
            "box": [0.22, 0.38, 0.78, 0.58],
            "use_skin_neg": True,
        },
        "drink_ad": {
            "area_range": [60000, 160000],
            "max_centroid_jump_px": 80,
            "positive_points": [[0.50, 0.45], [0.48, 0.30], [0.52, 0.60], [0.46, 0.50], [0.54, 0.50]],
            "negative_points": [[0.50, 0.15], [0.50, 0.80], [0.20, 0.45], [0.80, 0.45]],
            "box": [0.30, 0.18, 0.70, 0.72],
            "use_skin_neg": False,
        },
        "drink_yykx": {
            "area_range": [60000, 160000],
            "max_centroid_jump_px": 80,
            "positive_points": [[0.50, 0.45], [0.48, 0.30], [0.52, 0.60], [0.46, 0.50], [0.54, 0.50]],
            "negative_points": [[0.50, 0.15], [0.50, 0.80], [0.20, 0.45], [0.80, 0.45]],
            "box": [0.30, 0.18, 0.70, 0.72],
            "use_skin_neg": False,
        },
    }

    cfg = configs[args.object]
    print(f"[{args.object}] Loading SAM...")
    predictor = load_sam()

    print(f"[{args.object}] Generating masks...")
    masks, valid = generate_masks(predictor, args.frames, mask_dir, cfg, args.frame_stride)
    n_valid = sum(valid)
    print(f"  Masks: {n_valid}/{len(masks)} valid")

    # Get image shape from first valid mask or first image
    sample_img = cv2.imread(str(sorted(Path(args.frames).glob("frame_*.png"))[0]))
    h, w = sample_img.shape[:2]

    print(f"[{args.object}] Computing centroid trajectory...")
    centroids = masks_to_centroid_trajectory(masks, valid, (h, w))

    # Save 2D trajectory
    poses_2d = []
    for c in centroids:
        if c[4] == 1:
            poses_2d.append({"frame": int(c[0]), "cx": float(c[1]), "cy": float(c[2]), "area": float(c[3])})

    with open(output_dir / "object_trajectory_2d.json", "w") as f:
        json.dump({"object": args.object, "method": "SAM_dynamic_tracking", "num_valid_frames": n_valid,
                    "num_total_frames": len(masks), "keypoints": poses_2d}, f, indent=2)

    # Try 3D backprojection if camera calibration exists
    calib_dir = output_dir.parent.parent / "data" / "camera_calib" / args.object
    if not calib_dir.exists():
        calib_dir = Path("/mnt/workspace/AIlab-E6/experiments/2026-07-04_obj_recon_bread/camera_calib") / args.object

    if calib_dir.exists():
        try:
            npz_path = calib_dir / "camera_top.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                K, P = data["K"], data["P"]
            else:
                import pickle
                with open(calib_dir / "camera_top" / "cam_intr.pkl", "rb") as f:
                    K = pickle.load(f)
                with open(calib_dir / "camera_top" / "cam_extr.pkl", "rb") as f:
                    P = pickle.load(f)
                K = np.array(K); P = np.array(P)

            print(f"[{args.object}] Backprojecting to 3D...")
            poses_3d, valid_3d = image_plane_to_3d(centroids, K, P, args.ground_height)
            poses_3d_smooth = smooth_trajectory(poses_3d, valid_3d, window=5)

            np.savez(output_dir / "object_trajectory_3d.npz",
                     obj_transf=poses_3d_smooth, mask=valid_3d)
            print(f"  3D trajectory: {valid_3d.sum()}/{len(valid_3d)} valid frames")
        except Exception as e:
            print(f"  3D backprojection skipped: {e}")

    # Summary
    report = {
        "object": args.object,
        "method": "SAM_dynamic_tracking + image_plane_backprojection",
        "num_frames_total": len(masks),
        "num_masks_valid": n_valid,
        "mask_valid_ratio": n_valid / max(len(masks), 1),
        "mode": "image_plane_pose" if n_valid > 0 else "failed",
    }
    with open(output_dir / "trajectory_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[{args.object}] DONE. n_valid={n_valid}/{len(masks)}")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
