#!/usr/bin/env python3
"""生成不依赖真实视频的 dummy 手部样例，用于验证后半段 pipeline。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dummy frames and MediaPipe-like landmarks JSON.")
    parser.add_argument("--frames_dir", default="task2/data/samples/dummy_frames", help="输出 dummy 帧目录")
    parser.add_argument("--out_json", default="task2/outputs/trajectories/dummy_mediapipe_landmarks.json", help="输出 dummy landmarks JSON")
    parser.add_argument("--num_frames", type=int, default=60, help="生成帧数")
    parser.add_argument("--width", type=int, default=640, help="图像宽度")
    parser.add_argument("--height", type=int, default=480, help="图像高度")
    return parser.parse_args()


def make_hand_points(cx: float, cy: float, scale: float) -> np.ndarray:
    # 简单 21 点手形模板，不代表真实 MediaPipe 拓扑尺度，只用于自检 pipeline。
    template = np.array(
        [
            [0.00, 0.00],
            [-0.18, -0.10], [-0.28, -0.22], [-0.36, -0.34], [-0.45, -0.47],
            [-0.10, -0.20], [-0.12, -0.40], [-0.14, -0.58], [-0.16, -0.75],
            [0.02, -0.22], [0.03, -0.45], [0.04, -0.66], [0.05, -0.86],
            [0.15, -0.19], [0.20, -0.38], [0.24, -0.55], [0.28, -0.72],
            [0.26, -0.14], [0.34, -0.29], [0.41, -0.42], [0.48, -0.56],
        ],
        dtype=np.float32,
    )
    points = template * scale
    points[:, 0] += cx
    points[:, 1] += cy
    return points


def draw_points(image: np.ndarray, points: np.ndarray) -> None:
    for point in points:
        cv2.circle(image, tuple(point.astype(int)), 4, (0, 0, 255), -1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if args.num_frames <= 0:
        raise ValueError("--num_frames 必须为正数")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width 和 --height 必须为正数")

    frames_dir = Path(args.frames_dir)
    out_json = Path(args.out_json)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for frame_id in range(args.num_frames):
        t = frame_id / max(1, args.num_frames - 1)
        cx = args.width * (0.30 + 0.40 * t)
        cy = args.height * (0.72 - 0.10 * np.sin(2 * np.pi * t))
        scale = min(args.width, args.height) * 0.22
        points = make_hand_points(cx, cy, scale)
        points[:, 0] = np.clip(points[:, 0], 0, args.width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, args.height - 1)

        image = np.full((args.height, args.width, 3), 245, dtype=np.uint8)
        cv2.putText(image, "dummy hand sample", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
        draw_points(image, points)
        file_name = f"{frame_id:06d}.jpg"
        cv2.imwrite(str(frames_dir / file_name), image)

        world = np.zeros((21, 3), dtype=np.float32)
        world[:, :2] = (points - np.array([[cx, cy]], dtype=np.float32)) / scale
        world[:, 2] = -0.03 * np.arange(21, dtype=np.float32)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        records.append(
            {
                "frame_id": frame_id,
                "file_name": file_name,
                "image_size": [args.height, args.width],
                "hands": [
                    {
                        "hand_index": 0,
                        "handedness": "Right",
                        "handedness_score": 0.99,
                        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "landmarks_2d": points.astype(float).tolist(),
                        "landmarks_normalized": np.column_stack([points[:, 0] / args.width, points[:, 1] / args.height, np.zeros(21)]).astype(float).tolist(),
                        "world_landmarks": world.astype(float).tolist(),
                    }
                ],
            }
        )

    data = {
        "schema_version": "task2_dummy_mediapipe_like_v1",
        "frames_dir": str(frames_dir),
        "num_frames": args.num_frames,
        "valid_frames": args.num_frames,
        "valid_ratio": 1.0,
        "frames": records,
    }
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] dummy frames: {frames_dir}")
    print(f"[OK] dummy landmarks: {out_json}")


if __name__ == "__main__":
    main()
