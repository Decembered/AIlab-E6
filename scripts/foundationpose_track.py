#!/usr/bin/env python3
"""
FoundationPose tracking runner for Video2Motion2Action.

Weights from hf-mirror: gpue/foundationpose-weights
"""

import argparse, sys, os, json, logging
from pathlib import Path
import numpy as np
import cv2
import torch
import trimesh
import nvdiffrast.torch as dr

# FoundationPose path
_fp_dir = "/mnt/workspace/foundationpose"
sys.path.insert(0, _fp_dir)
sys.path.insert(0, _fp_dir + "/mycpp")

from estimater import FoundationPose
from Utils import draw_posed_3d_box, set_seed
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor

logging.basicConfig(level=logging.INFO)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--object", required=True)
    p.add_argument("--mesh", required=True, type=Path)
    p.add_argument("--frames", required=True, type=Path)
    p.add_argument("--K", type=float, nargs=9, help="Camera intrinsics (row-major 3x3)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--weights", default="/mnt/workspace/foundationpose/weights", type=Path)
    return p.parse_args()


def load_mesh(path):
    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.geometry[list(mesh.geometry.keys())[0]]
    mesh.vertices = mesh.vertices.astype(np.float32)
    if mesh.vertex_normals is None:
        mesh.vertex_normals = np.zeros_like(mesh.vertices)
    return mesh


def main():
    args = parse_args()
    set_seed(0)

    # 1. Load mesh
    logging.info(f"Loading mesh: {args.mesh}")
    mesh = load_mesh(args.mesh)
    logging.info(f"  vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

    # 2. Create predictor instances (weights auto-loaded from config)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()

    # 3. Initialize FoundationPose
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
    )
    logging.info("FoundationPose initialized with weights")

    # 4. Camera intrinsics
    if args.K:
        K = np.array(args.K).reshape(3, 3)
    else:
        # Default camera_top calibration for weigh_bread
        K = np.array([[910.08, 0, 645.75], [0, 908.41, 379.71], [0, 0, 1]])

    # 5. Run tracking on frames
    frame_paths = sorted(Path(args.frames).glob("frame_*.png"))
    logging.info(f"Processing {len(frame_paths)} frames")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = []
    valid = []

    for i, fp in enumerate(frame_paths):
        image = cv2.imread(str(fp))
        if image is None:
            continue

        if i == 0:
            pose = est.register(K=K, rgb=image, ob_in_cam=None, iteration=5)
        else:
            pose = est.track_one(rgb=image, K=K, iteration=2)

        if pose is not None:
            poses.append(pose)
            valid.append(True)
        else:
            valid.append(False)
            poses.append(np.eye(4))

        if i % 10 == 0:
            logging.info(f"  frame {i}/{len(frame_paths)}: {'valid' if pose is not None else 'miss'}")

    poses_np = np.stack(poses)
    valid_np = np.array(valid)
    n_valid = valid_np.sum()

    np.savez(output_dir / "object_trajectory_foundationpose.npz",
             obj_transf=poses_np, mask=valid_np)
    logging.info(f"DONE: {n_valid}/{len(frame_paths)} valid frames")

    # Report
    report = {
        "object": args.object,
        "method": "FoundationPose (CVPR 2024)",
        "num_frames": len(frame_paths),
        "num_valid": int(n_valid),
        "validation_ratio": float(n_valid / max(len(frame_paths), 1)),
    }
    with open(output_dir / "trajectory_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[{args.object}] FoundationPose: {n_valid}/{len(frame_paths)} ({n_valid/max(len(frame_paths),1)*100:.0f}%)")


if __name__ == "__main__":
    main()
