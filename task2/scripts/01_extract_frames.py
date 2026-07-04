#!/usr/bin/env python3
"""从视频抽帧到指定目录。"""

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video frames for task2 baseline.")
    parser.add_argument("--video", required=True, help="输入视频路径，例如 path/to/video.mp4")
    parser.add_argument("--out_dir", required=True, help="输出帧目录，例如 task2/data/frames/demo")
    parser.add_argument("--fps", type=float, default=None, help="可选目标帧率；默认保留原视频所有帧")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="输出图片格式")
    parser.add_argument("--manifest", default=None, help="输出 frame manifest JSON；默认写到 out_dir/frame_manifest.json")
    parser.add_argument("--clean", action="store_true", help="抽帧前清理 out_dir 中已有图片和 manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "frame_manifest.json"

    if not video_path.is_file():
        raise FileNotFoundError(f"输入视频不存在: {video_path}")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps 必须为正数")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for item in out_dir.iterdir():
            if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".json"}:
                item.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not src_fps or src_fps <= 0:
        src_fps = 30.0

    frame_interval = 1
    if args.fps is not None:
        frame_interval = max(1, round(src_fps / args.fps))

    target_fps = float(args.fps) if args.fps is not None else float(src_fps)
    saved = 0
    frame_idx = 0
    frame_records = []
    progress = tqdm(total=total_frames if total_frames > 0 else None, desc="extract frames")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                out_path = out_dir / f"{saved:06d}.{args.ext}"
                success = cv2.imwrite(str(out_path), frame)
                if not success:
                    raise RuntimeError(f"写入帧失败: {out_path}")
                frame_records.append(
                    {
                        "frame_id": saved,
                        "file_name": out_path.name,
                        "source_frame_idx": frame_idx,
                        "timestamp_sec": float(frame_idx / src_fps),
                    }
                )
                saved += 1
            frame_idx += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()

    print(f"[OK] video: {video_path}")
    print(f"[OK] source fps: {src_fps:.3f}")
    print(f"[OK] saved frames: {saved}")
    print(f"[OK] out_dir: {out_dir}")
    manifest = {
        "schema_version": "task2_frame_manifest_v1",
        "source_video": str(video_path),
        "source_fps": float(src_fps),
        "target_fps": target_fps,
        "frame_interval": int(frame_interval),
        "source_total_frames": int(total_frames),
        "saved_frames": int(saved),
        "frames_dir": str(out_dir),
        "frames": frame_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
