#!/usr/bin/env python3
"""将 MediaPipe 关键点画回原图并生成 mp4。"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "task2" / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.hand_schema import HAND_CONNECTIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MediaPipe landmark overlay video.")
    parser.add_argument("--frames_dir", required=True, help="输入帧目录")
    parser.add_argument("--landmarks_json", required=True, help="02_run_mediapipe_hands.py 输出 JSON")
    parser.add_argument("--out_video", default="task2/outputs/videos/mediapipe_overlay.mp4", help="输出 mp4 路径")
    parser.add_argument("--fps", type=float, default=30.0, help="输出视频帧率")
    return parser.parse_args()


def draw_hand(image, points_2d: list[list[float]], label: str) -> None:
    for start, end in HAND_CONNECTIONS:
        p1 = tuple(int(v) for v in points_2d[start])
        p2 = tuple(int(v) for v in points_2d[end])
        cv2.line(image, p1, p2, (0, 220, 0), 2, cv2.LINE_AA)
    for idx, point in enumerate(points_2d):
        center = tuple(int(v) for v in point)
        cv2.circle(image, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
        if idx == 0:
            cv2.putText(image, label, (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    landmarks_json = Path(args.landmarks_json)
    out_video = Path(args.out_video)

    if not frames_dir.is_dir():
        raise FileNotFoundError(f"输入帧目录不存在: {frames_dir}")
    if not landmarks_json.is_file():
        raise FileNotFoundError(f"关键点 JSON 不存在: {landmarks_json}")
    if args.fps <= 0:
        raise ValueError("--fps 必须为正数")

    data = json.loads(landmarks_json.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"JSON 中没有 frames: {landmarks_json}")

    out_video.parent.mkdir(parents=True, exist_ok=True)

    first_image = None
    for frame_record in frames:
        candidate = cv2.imread(str(frames_dir / frame_record["file_name"]))
        if candidate is not None:
            first_image = candidate
            break
    if first_image is None:
        raise RuntimeError(f"无法从帧目录读取任何图片: {frames_dir}")
    height, width = first_image.shape[:2]

    writer = imageio.get_writer(str(out_video), fps=args.fps, codec="libx264", macro_block_size=1)
    try:
        for frame_record in tqdm(frames, desc="write overlay video"):
            frame_path = frames_dir / frame_record["file_name"]
            image = cv2.imread(str(frame_path))
            if image is None:
                image = first_image.copy()
                image[:] = 0
                cv2.putText(image, f"missing frame: {frame_path.name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            for hand in frame_record.get("hands", []):
                label = f"{hand.get('handedness', 'Unknown')} {hand.get('handedness_score', 0.0):.2f}"
                draw_hand(image, hand["landmarks_2d"], label)
            writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()

    print(f"[OK] out_video: {out_video}")


if __name__ == "__main__":
    main()
