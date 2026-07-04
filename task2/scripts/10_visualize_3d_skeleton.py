#!/usr/bin/env python3
"""生成 MediaPipe 3D hand skeleton 可视化图和视频。"""

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
    parser = argparse.ArgumentParser(description="Visualize 3D hand skeleton from hand_traj.npz.")
    parser.add_argument("--traj_npz", default="task2/outputs/trajectories/hand_traj.npz", help="输入 hand_traj.npz")
    parser.add_argument("--out_dir", default="task2/outputs/figures/hand_3d_skeleton", help="输出图片目录")
    parser.add_argument("--out_video", default="task2/outputs/videos/hand_3d_skeleton.mp4", help="输出 3D skeleton 视频")
    parser.add_argument("--fps", type=float, default=None, help="输出视频 fps，默认读取 npz")
    parser.add_argument("--max_frames", type=int, default=0, help="可选最大可视化帧数，0 表示全部")
    return parser.parse_args()


def set_equal_axes(ax, points: np.ndarray) -> None:
    if not np.any(np.isfinite(points)):
        center = np.zeros((3,), dtype=np.float32)
        radius = 0.05
    else:
        center = np.nanmean(points, axis=(0, 1))
        radius = np.nanmax(np.linalg.norm(points.reshape(-1, 3) - center, axis=1))
    radius = max(float(radius), 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_frame(points: np.ndarray, frame_id: int, valid: bool, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for start, end in HAND_CONNECTIONS:
        seg = points[[start, end]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="tab:blue", linewidth=2)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="tab:red", s=18)
    ax.set_title(f"frame {frame_id} valid={valid}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_equal_axes(ax, points[None, ...])
    ax.view_init(elev=20, azim=-70)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    traj_npz = Path(args.traj_npz)
    if not traj_npz.is_file():
        raise FileNotFoundError(f"轨迹文件不存在: {traj_npz}")
    data = np.load(traj_npz, allow_pickle=True)
    keypoints3d = data["keypoints3d"] if "keypoints3d" in data else data["hand_landmarks_3d"]
    valid = data["valid"] if "valid" in data else np.ones((keypoints3d.shape[0],), dtype=bool)
    frame_ids = data["frame_ids"] if "frame_ids" in data else np.arange(keypoints3d.shape[0])
    fps = float(args.fps if args.fps is not None else data["fps"] if "fps" in data else 15.0)
    if fps <= 0:
        raise ValueError("fps 必须为正数")
    count = keypoints3d.shape[0] if args.max_frames <= 0 else min(args.max_frames, keypoints3d.shape[0])
    if count <= 0:
        raise ValueError("没有可视化帧")

    out_dir = Path(args.out_dir)
    out_video = Path(args.out_video)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_png in out_dir.glob("*.png"):
        old_png.unlink()
    out_video.parent.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for idx in tqdm(range(count), desc="render 3d skeleton"):
        out_path = out_dir / f"{int(frame_ids[idx]):06d}.png"
        draw_frame(keypoints3d[idx], int(frame_ids[idx]), bool(valid[idx]), out_path)
        image_paths.append(out_path)

    writer = imageio.get_writer(str(out_video), fps=fps, codec="libx264", macro_block_size=1)
    try:
        for path in image_paths:
            writer.append_data(imageio.imread(path))
    finally:
        writer.close()
    print(f"[OK] figures: {out_dir}")
    print(f"[OK] video: {out_video}")


if __name__ == "__main__":
    main()
