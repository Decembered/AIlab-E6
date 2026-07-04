#!/usr/bin/env python3
"""基于 MediaPipe bbox 或关键点凸包生成粗手部 mask baseline。"""

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate coarse hand masks from MediaPipe landmarks.")
    parser.add_argument("--frames_dir", required=True, help="输入帧目录")
    parser.add_argument("--landmarks_json", required=True, help="02_run_mediapipe_hands.py 输出 JSON")
    parser.add_argument("--out_masks_dir", default="task2/outputs/masks/demo", help="输出 mask 目录")
    parser.add_argument("--out_overlay_video", default="task2/outputs/overlays/hand_mask_overlay.mp4", help="输出 mask overlay 视频")
    parser.add_argument("--mode", choices=["hull", "bbox"], default="hull", help="粗 mask 生成方式")
    parser.add_argument("--padding", type=float, default=0.18, help="bbox padding 比例，仅 bbox 模式使用")
    parser.add_argument("--dilate", type=int, default=15, help="mask 膨胀核大小，0 表示不膨胀")
    parser.add_argument("--fps", type=float, default=30.0, help="overlay 视频帧率")
    return parser.parse_args()


def padded_bbox(bbox: list[float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * padding
    py = bh * padding
    x1 = int(max(0, np.floor(x1 - px)))
    y1 = int(max(0, np.floor(y1 - py)))
    x2 = int(min(width - 1, np.ceil(x2 + px)))
    y2 = int(min(height - 1, np.ceil(y2 + py)))
    return x1, y1, x2, y2


def mask_from_hand(hand: dict, width: int, height: int, mode: str, padding: float) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if mode == "bbox":
        bbox = hand.get("bbox_xyxy")
        if not bbox:
            return mask
        x1, y1, x2, y2 = padded_bbox(bbox, width, height, padding)
        mask[y1:y2 + 1, x1:x2 + 1] = 255
        return mask

    points = np.asarray(hand.get("landmarks_2d", []), dtype=np.float32)
    if points.shape != (21, 2):
        return mask
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(image)
    color[:, :, 1] = 255
    alpha = (mask > 0).astype(np.float32)[:, :, None] * 0.45
    blended = image.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    landmarks_json = Path(args.landmarks_json)
    out_masks_dir = Path(args.out_masks_dir)
    out_overlay_video = Path(args.out_overlay_video)

    if not frames_dir.is_dir():
        raise FileNotFoundError(f"输入帧目录不存在: {frames_dir}")
    if not landmarks_json.is_file():
        raise FileNotFoundError(f"关键点 JSON 不存在: {landmarks_json}")
    if args.padding < 0:
        raise ValueError("--padding 必须 >= 0")
    if args.dilate < 0:
        raise ValueError("--dilate 必须 >= 0")
    if args.fps <= 0:
        raise ValueError("--fps 必须为正数")

    data = json.loads(landmarks_json.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"JSON 中没有 frames: {landmarks_json}")

    out_masks_dir.mkdir(parents=True, exist_ok=True)
    out_overlay_video.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        for frame in tqdm(frames, desc="generate coarse masks"):
            frame_path = frames_dir / frame["file_name"]
            image = cv2.imread(str(frame_path))
            if image is None:
                raise RuntimeError(f"读取图片失败: {frame_path}")
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            for hand in frame.get("hands", []):
                mask = np.maximum(mask, mask_from_hand(hand, width, height, args.mode, args.padding))

            if args.dilate > 0 and mask.any():
                kernel_size = args.dilate if args.dilate % 2 == 1 else args.dilate + 1
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

            mask_name = Path(frame["file_name"]).with_suffix(".png").name
            cv2.imwrite(str(out_masks_dir / mask_name), mask)

            overlay = overlay_mask(image, mask)
            cv2.putText(
                overlay,
                f"frame {frame.get('frame_id', 0)} mask_area={(mask > 0).sum()}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if writer is None:
                writer = imageio.get_writer(str(out_overlay_video), fps=args.fps, codec="libx264", macro_block_size=1)
            writer.append_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    finally:
        if writer is not None:
            writer.close()

    print(f"[OK] masks: {out_masks_dir}")
    print(f"[OK] overlay video: {out_overlay_video}")
    print("[NOTE] 这是 coarse visible hand mask baseline，后续可替换为 SAM2 输出。")


if __name__ == "__main__":
    main()
