#!/usr/bin/env python3
"""从 overlay 视频/帧目录抽取成功和失败案例截图。"""

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract representative screenshots for task2 report.")
    parser.add_argument("--frames_dir", required=True, help="overlay 帧目录或原始帧目录")
    parser.add_argument("--out_dir", default="task2/outputs/figures/report_cases", help="输出截图目录")
    parser.add_argument("--frames", default="0,30,60,90,122,124,127,150,181", help="逗号分隔帧号")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"帧目录不存在: {frames_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    frame_ids = [int(item) for item in args.frames.split(",") if item.strip()]
    for frame_id in frame_ids:
        src = frames_dir / f"{frame_id:06d}.jpg"
        if not src.is_file():
            src = frames_dir / f"{frame_id:06d}.png"
        if not src.is_file():
            print(f"[WARN] missing frame {frame_id}: {src}")
            continue
        image = cv2.imread(str(src))
        if image is None:
            print(f"[WARN] cannot read frame {src}")
            continue
        label = "keyframe"
        out_path = out_dir / f"{frame_id:06d}_{label}.jpg"
        ok = cv2.imwrite(str(out_path), image)
        if not ok:
            raise RuntimeError(f"写入截图失败: {out_path}")
        print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
