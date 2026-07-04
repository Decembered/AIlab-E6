#!/usr/bin/env python3
"""Render a MediaPipe-21 3D hand skeleton trajectory from hand_traj.npz."""

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "task2" / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.hand_schema import HAND_CONNECTIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize 3D hand skeleton from task2 hand_traj.npz.")
    parser.add_argument("--traj_npz", required=True)
    parser.add_argument("--out_video", default="task2/outputs/videos/hand_3d_skeleton.mp4")
    parser.add_argument("--out_keyframes", default="task2/outputs/figures/hand_3d_skeleton_keyframes.png")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--title", default="Task2 3D Hand Skeleton")
    return parser.parse_args()


def array_to_text(value: np.ndarray | str) -> str:
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr)


def get_keypoints(data: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("keypoints3d", "hand_landmarks_3d", "world_landmarks"):
        if key in data:
            points = np.asarray(data[key], dtype=np.float32)
            if points.ndim == 3 and points.shape[1:] == (21, 3):
                return points
    raise KeyError("traj_npz must contain keypoints3d, hand_landmarks_3d, or world_landmarks with shape [T,21,3]")


def compute_limits(points: np.ndarray) -> tuple[np.ndarray, float]:
    valid = np.isfinite(points).all(axis=-1)
    if not valid.any():
        return np.zeros(3, dtype=np.float32), 1.0
    flat = points[valid]
    center = (flat.min(axis=0) + flat.max(axis=0)) / 2.0
    radius = float(np.max(flat.max(axis=0) - flat.min(axis=0)) / 2.0)
    return center, max(radius, 1e-3)


def draw_frame(points: np.ndarray, frame_id: int, title: str, center: np.ndarray, radius: float) -> np.ndarray:
    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{title}\nframe {frame_id}")
    for start, end in HAND_CONNECTIONS:
        seg = points[[start, end]]
        if np.isfinite(seg).all():
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#00aa55", linewidth=2)
    valid = np.isfinite(points).all(axis=1)
    if valid.any():
        ax.scatter(points[valid, 0], points[valid, 1], points[valid, 2], c="#dd2222", s=18)
        ax.scatter(points[0:1, 0], points[0:1, 1], points[0:1, 2], c="#2255ff", s=36, label="wrist")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-65)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def write_keyframes(path: Path, points: np.ndarray, frame_ids: np.ndarray, title: str, center: np.ndarray, radius: float) -> None:
    if len(points) == 0:
        return
    indices = np.linspace(0, len(points) - 1, min(6, len(points)), dtype=int)
    images = [draw_frame(points[i], int(frame_ids[i]), title, center, radius) for i in indices]
    rows = []
    for i in range(0, len(images), 3):
        row = np.concatenate(images[i:i + 3], axis=1)
        rows.append(row)
    canvas = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), canvas)


def main() -> None:
    args = parse_args()
    traj_npz = Path(args.traj_npz)
    out_video = Path(args.out_video)
    out_keyframes = Path(args.out_keyframes)
    if not traj_npz.is_file():
        raise FileNotFoundError(f"traj_npz does not exist: {traj_npz}")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    data = np.load(traj_npz, allow_pickle=True)
    points = get_keypoints(data)[:: args.stride]
    frame_ids = np.asarray(data["frame_ids"] if "frame_ids" in data else np.arange(len(get_keypoints(data))))[:: args.stride]
    if args.max_frames > 0:
        points = points[: args.max_frames]
        frame_ids = frame_ids[: args.max_frames]
    fps = float(args.fps if args.fps is not None else data["fps"] if "fps" in data else 30.0)
    source = array_to_text(data["source"]) if "source" in data else "unknown"
    title = f"{args.title} ({source})"
    center, radius = compute_limits(points)

    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_video), fps=fps, codec="libx264", macro_block_size=1)
    try:
        for i in tqdm(range(len(points)), desc="render 3D skeleton"):
            writer.append_data(draw_frame(points[i], int(frame_ids[i]), title, center, radius))
    finally:
        writer.close()
    write_keyframes(out_keyframes, points, frame_ids, title, center, radius)
    print(f"[OK] out_video: {out_video}")
    print(f"[OK] out_keyframes: {out_keyframes}")
    print("[NOTE] This is a MediaPipe-world skeleton visualization, not metric MANO mesh rendering.")


if __name__ == "__main__":
    main()
