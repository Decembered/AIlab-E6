#!/usr/bin/env python3
"""合成 Task2 评分审查视频：关键点 overlay + mask overlay + 3D skeleton + 帧级指标。"""

import argparse
import csv
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make synchronized task2 scoring review video.")
    parser.add_argument("--keypoint_video", required=True, help="MediaPipe keypoint overlay mp4")
    parser.add_argument("--mask_video", required=True, help="hand mask overlay mp4")
    parser.add_argument("--skeleton_video", required=True, help="3D skeleton mp4")
    parser.add_argument("--frame_metrics_csv", required=True, help="frame_metrics.csv")
    parser.add_argument("--out_video", default="task2/outputs/videos/task2_scoring_review.mp4", help="输出审查视频")
    parser.add_argument("--fps", type=float, default=15.0)
    return parser.parse_args()


def load_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"frame metrics 不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_frame(cap: cv2.VideoCapture, name: str):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"读取视频帧失败: {name}")
    return frame


def draw_metric_panel(width: int, height: int, row: dict) -> any:
    panel = 255 * (cv2.UMat(height, width, cv2.CV_8UC3).get())
    risk = row.get("risk_level", "ok")
    color = (0, 140, 0) if risk == "ok" else (0, 170, 255) if risk == "warn" else (0, 0, 220)
    lines = [
        f"frame={row['frame_id']}  time={float(row['timestamp_sec']):.2f}s  risk={risk}",
        f"quality={float(row['quality_score']):.3f}  hand={row['handedness']}({float(row['handedness_score']):.3f})  valid={row['valid']}",
        f"temporal={float(row['temporal_jump_score']):.3f}  wrist_step={float(row['wrist_step']):.5f}  fingertip_max={float(row['fingertip_step_max']):.5f}",
        f"bone={float(row['bone_length_error']):.5f}  mask_area={float(row['mask_area']):.0f}  bbox_area={float(row['bbox_area']):.0f}",
        f"reasons={row['risk_reasons']}",
    ]
    y = 26
    for idx, text in enumerate(lines):
        cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62 if idx == 0 else 0.52, color if idx == 0 else (30, 30, 30), 2 if idx == 0 else 1, cv2.LINE_AA)
        y += 31
    return panel


def main() -> None:
    args = parse_args()
    metrics = load_metrics(Path(args.frame_metrics_csv))
    caps = [
        cv2.VideoCapture(args.keypoint_video),
        cv2.VideoCapture(args.mask_video),
        cv2.VideoCapture(args.skeleton_video),
    ]
    for cap, name in zip(caps, [args.keypoint_video, args.mask_video, args.skeleton_video]):
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {name}")

    out_path = Path(args.out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_w, canvas_h = 1280, 900
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频: {out_path}")

    try:
        for idx, row in enumerate(metrics):
            key = cv2.resize(read_frame(caps[0], args.keypoint_video), (640, 360))
            mask = cv2.resize(read_frame(caps[1], args.mask_video), (640, 360))
            skel = cv2.resize(read_frame(caps[2], args.skeleton_video), (360, 360))
            metric_panel = draw_metric_panel(920, 180, row)
            canvas = 255 * (cv2.UMat(canvas_h, canvas_w, cv2.CV_8UC3).get())
            canvas[0:360, 0:640] = key
            canvas[0:360, 640:1280] = mask
            canvas[390:750, 0:360] = skel
            canvas[390:570, 360:1280] = metric_panel
            cv2.putText(canvas, "keypoint overlay", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "mask overlay", (656, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "3D skeleton (non-metric)", (16, 424), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
        for cap in caps:
            cap.release()
    print(f"[OK] wrote scoring review video: {out_path}")


if __name__ == "__main__":
    main()
