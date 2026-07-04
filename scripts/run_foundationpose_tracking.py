#!/usr/bin/env python3
"""
FoundationPose 6D Pose Tracking Wrapper for Video2Motion2Action

Integrates NVIDIA FoundationPose (CVPR 2024 Highlight) for model-based
6D object pose estimation and tracking from video.

Requirements:
  - FoundationPose repo cloned at /mnt/workspace/foundationpose
  - Model weights downloaded to /mnt/workspace/foundationpose/weights/
  - Python 3.12 environment with torch 2.5.1, pytorch3d, nvdiffrast

Usage:
  python3.12 scripts/run_foundationpose_tracking.py \\
    --object bread \\
    --mesh runs/object_asset_v1/bread/mesh/visual_mesh.obj \\
    --data-dir data/human_demo/weigh_bread__2026_0701_0044_30 \\
    --output outputs/foundationpose/bread/
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="FoundationPose 6D tracking")
    parser.add_argument("--object", required=True, help="Object name (bread/pipette/drink_ad/drink_yykx)")
    parser.add_argument("--mesh", required=True, type=Path, help="Path to CAD mesh (.obj)")
    parser.add_argument("--data-dir", required=True, type=Path, help="Sequence directory with video/ and camera_calib/")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for trajectory")
    parser.add_argument("--camera", default="camera_top", help="Camera to use (camera_side_1/camera_side_2/camera_top)")
    parser.add_argument("--fp-dir", default="/mnt/workspace/foundationpose", type=Path, help="FoundationPose repo path")
    parser.add_argument("--refiner-ckpt", default="2023-10-28-18-33-37", help="Refiner checkpoint name")
    parser.add_argument("--scorer-ckpt", default="2024-01-11-20-02-45", help="Scorer checkpoint name")
    return parser.parse_args()


def check_weights(fp_dir: Path, refiner_ckpt: str, scorer_ckpt: str) -> bool:
    """Check if FoundationPose weights are downloaded."""
    refiner_path = fp_dir / "weights" / refiner_ckpt
    scorer_path = fp_dir / "weights" / scorer_ckpt
    ok = refiner_path.exists() and scorer_path.exists()
    if not ok:
        print(f"[ERROR] FoundationPose weights not found.")
        print(f"  Refiner: {refiner_path}  -> {'EXISTS' if refiner_path.exists() else 'MISSING'}")
        print(f"  Scorer:  {scorer_path}  -> {'EXISTS' if scorer_path.exists() else 'MISSING'}")
        print(f"\nDownload from Google Drive and place in {fp_dir}/weights/:")
        print(f"  https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i")
    return ok


def extract_frames(video_dir: Path, output_dir: Path, max_frames: int = 0) -> list:
    """Extract frames from MKV video files using ffmpeg."""
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    video_files = sorted(video_dir.glob("*.mkv"))
    if not video_files:
        raise FileNotFoundError(f"No MKV files in {video_dir}")

    video_path = video_files[0]
    frame_paths = []

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", "fps=5",
        "-q:v", "2",
        "-y",
        str(output_dir / "frame_%06d.png")
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    frame_paths = sorted(output_dir.glob("frame_*.png"))
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]

    return frame_paths


def load_camera_calib(calib_dir: Path, camera: str) -> tuple:
    """Load camera intrinsics (3x3) and extrinsics (4x4) from pkl/npz."""
    import pickle

    # Try npz first (our exported format)
    npz_path = calib_dir / f"{camera}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        return data["K"], data["P"]

    # Fall back to pkl
    intr_path = calib_dir / camera / "cam_intr.pkl"
    extr_path = calib_dir / camera / "cam_extr.pkl"
    if intr_path.exists() and extr_path.exists():
        with open(intr_path, "rb") as f:
            K = pickle.load(f)
        with open(extr_path, "rb") as f:
            P = pickle.load(f)
        return np.array(K), np.array(P)

    raise FileNotFoundError(f"Camera calibration not found at {calib_dir}/{camera}")


def run_foundationpose_tracking(
    mesh_path: Path,
    frame_paths: list,
    K: np.ndarray,
    P: np.ndarray,
    fp_dir: Path,
    refiner_ckpt: str,
    scorer_ckpt: str,
    output_dir: Path,
) -> dict:
    """Run FoundationPose registration + tracking on a sequence.

    Returns trajectory as dict with keys:
      - poses: (T, 4, 4) np.ndarray of object-to-world transforms
      - timestamps: (T,) frame indices
      - quality: metadata dict
    """
    import sys
    sys.path.insert(0, str(fp_dir))
    sys.path.insert(0, str(fp_dir / "mycpp"))

    import torch
    import cv2
    from estimater import FoundationPose
    from Utils import draw_posed_3d_box, set_seed

    set_seed(0)

    # Initialize FoundationPose
    glctx = None  # Will be auto-created in model-based mode
    refiner_path = str(fp_dir / "weights" / refiner_ckpt)
    scorer_path = str(fp_dir / "weights" / scorer_ckpt)

    est = FoundationPose(
        model_pts_file=str(mesh_path),
        model_normals_file=str(mesh_path),
        mesh_file=str(mesh_path),
        scorer_path=scorer_path,
        refiner_path=refiner_path,
        glctx=glctx,
    )

    poses = []
    timestamps = []
    mask = []

    for i, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))
        if image is None:
            mask.append(False)
            continue

        if i == 0:
            # First frame: full registration
            pose = est.register(K=K, rgb=image, ob_in_cam=None, iteration=5)
            if pose is not None:
                est.last_pose = pose
        else:
            # Subsequent frames: tracking
            pose = est.track_one(rgb=image, K=K, iteration=2)

        if pose is not None:
            poses.append(pose)
            timestamps.append(i)
            mask.append(True)
        else:
            mask.append(False)
            poses.append(np.eye(4))  # placeholder

    poses = np.stack(poses, axis=0)
    mask = np.array(mask)
    timestamps = np.array(timestamps)

    return {
        "poses": poses,
        "mask": mask,
        "timestamps": timestamps,
        "num_valid": int(mask.sum()),
        "num_frames": len(frame_paths),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check weights
    if not check_weights(args.fp_dir, args.refiner_ckpt, args.scorer_ckpt):
        sys.exit(1)

    # 2. Extract frames
    video_dir = Path(args.data_dir) / "video"
    frames_dir = output_dir / "frames"
    frame_paths = extract_frames(video_dir, frames_dir)
    print(f"[INFO] Extracted {len(frame_paths)} frames")

    # 3. Load calibration
    calib_dir = Path(args.data_dir) / "camera_calib"
    K, P = load_camera_calib(calib_dir, args.camera)
    print(f"[INFO] Camera K:\n{K[:1]}...")

    # 4. Run tracking
    print(f"[INFO] Running FoundationPose tracking on {args.object}...")
    result = run_foundationpose_tracking(
        mesh_path=Path(args.mesh),
        frame_paths=frame_paths,
        K=K,
        P=P,
        fp_dir=args.fp_dir,
        refiner_ckpt=args.refiner_ckpt,
        scorer_ckpt=args.scorer_ckpt,
        output_dir=output_dir,
    )

    # 5. Save outputs
    np.savez(output_dir / "object_trajectory_foundationpose.npz",
             obj_transf=result["poses"], mask=result["mask"], timestamps=result["timestamps"])

    # Save as JSON-compatible list
    trajectory_json = []
    for i in range(len(result["poses"])):
        if result["mask"][i]:
            trajectory_json.append({
                "frame": int(result["timestamps"][i]),
                "pose_4x4": result["poses"][i].tolist(),
            })

    with open(output_dir / "object_trajectory_foundationpose.json", "w") as f:
        json.dump({
            "object": args.object,
            "method": "FoundationPose (CVPR 2024)",
            "num_valid_frames": result["num_valid"],
            "num_total_frames": result["num_frames"],
            "trajectory": trajectory_json,
        }, f, indent=2)

    print(f"[DONE] Saved to {output_dir}/")
    print(f"  Valid frames: {result['num_valid']}/{result['num_frames']}")


if __name__ == "__main__":
    main()
