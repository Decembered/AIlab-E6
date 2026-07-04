#!/usr/bin/env python3
"""Create a hand mask overlay video from existing frames and mask PNG files."""

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize existing hand masks over RGB frames.")
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--out_video", default="task2/outputs/overlays/hand_mask_overlay.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--color", default="0,255,0", help="BGR mask color, for example 0,255,0")
    return parser.parse_args()


def parse_color(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise ValueError("--color must be three comma-separated integers in [0,255]")
    return parts[0], parts[1], parts[2]


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.masks_dir)
    out_video = Path(args.out_video)
    color_bgr = parse_color(args.color)

    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames_dir does not exist: {frames_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks_dir does not exist: {masks_dir}")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be in [0,1]")

    frames = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not frames:
        raise RuntimeError(f"no frames found in {frames_dir}")
    out_video.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(str(out_video), fps=args.fps, codec="libx264", macro_block_size=1)
    try:
        for frame_path in tqdm(frames, desc="write mask overlay"):
            image = cv2.imread(str(frame_path))
            if image is None:
                raise RuntimeError(f"failed to read frame: {frame_path}")
            mask_path = masks_dir / frame_path.with_suffix(".png").name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            color = np.zeros_like(image)
            color[:, :] = color_bgr
            weight = (mask > 0).astype(np.float32)[:, :, None] * args.alpha
            overlay = image.astype(np.float32) * (1.0 - weight) + color.astype(np.float32) * weight
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            cv2.putText(overlay, frame_path.name, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.append_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    print(f"[OK] out_video: {out_video}")


if __name__ == "__main__":
    main()
