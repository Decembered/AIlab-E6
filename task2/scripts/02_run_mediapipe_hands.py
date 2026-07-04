#!/usr/bin/env python3
"""使用 MediaPipe Hands 生成手部关键点 baseline。"""

import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "task2" / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.hand_schema import HAND_CONNECTIONS  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MediaPipe hand landmark baseline on frames.")
    parser.add_argument("--frames_dir", required=True, help="输入帧目录")
    parser.add_argument("--out_json", default="task2/outputs/trajectories/mediapipe_landmarks.json", help="输出 JSON 路径")
    parser.add_argument("--vis_dir", default=None, help="可选逐帧可视化输出目录")
    parser.add_argument("--model_asset_path", default="task2/models/mediapipe/hand_landmarker.task", help="MediaPipe Tasks HandLandmarker 模型路径；新版 mediapipe 需要")
    parser.add_argument("--fps", type=float, default=30.0, help="输入帧序列帧率，用于 MediaPipe Tasks 视频时间戳")
    parser.add_argument("--manifest", default=None, help="可选 frame_manifest.json，用于读取真实 frame_id/timestamp")
    parser.add_argument("--max_num_hands", type=int, default=2, help="最大检测手数量")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5, help="跟踪置信度阈值")
    return parser.parse_args()


def list_frames(frames_dir: Path) -> list[Path]:
    frames = [p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(frames)


def load_frame_records(frames_dir: Path, manifest_path: Path | None) -> list[dict]:
    if manifest_path is None:
        return [
            {"frame_id": idx, "file_name": frame.name, "path": frame}
            for idx, frame in enumerate(list_frames(frames_dir))
        ]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for item in manifest.get("frames", []):
        frame_path = frames_dir / item["file_name"]
        if not frame_path.is_file():
            raise FileNotFoundError(f"manifest 中的帧不存在: {frame_path}")
        records.append({**item, "path": frame_path})
    return records


def load_manifest_meta(manifest_path: Path | None) -> dict:
    if manifest_path is None:
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "source_video": manifest.get("source_video", ""),
        "source_fps": manifest.get("source_fps", 0.0),
        "target_fps": manifest.get("target_fps", 0.0),
        "source_total_frames": manifest.get("source_total_frames", 0),
        "saved_frames": manifest.get("saved_frames", 0),
        "frame_interval": manifest.get("frame_interval", 1),
    }


def normalized_to_pixel_landmarks(landmarks, width: int, height: int) -> list[list[float]]:
    return [[float(lm.x * width), float(lm.y * height)] for lm in landmarks]


def normalized_landmarks(landmarks) -> list[list[float]]:
    return [[float(lm.x), float(lm.y), float(lm.z)] for lm in landmarks]


def world_landmarks(world) -> list[list[float]]:
    if world is None:
        return []
    if hasattr(world, "landmark"):
        world = world.landmark
    return [[float(lm.x), float(lm.y), float(lm.z)] for lm in world]


def category_label_score(category) -> tuple[str, float]:
    label = getattr(category, "label", None) or getattr(category, "category_name", None) or "Unknown"
    score = getattr(category, "score", 0.0)
    return str(label), float(score)


def bbox_from_points(points_2d: list[list[float]], width: int, height: int) -> list[float]:
    xs = [p[0] for p in points_2d]
    ys = [p[1] for p in points_2d]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width - 1), max(xs))
    y2 = min(float(height - 1), max(ys))
    return [x1, y1, x2, y2]


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
    out_json = Path(args.out_json)
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    manifest_path = Path(args.manifest) if args.manifest else None

    if not frames_dir.is_dir():
        raise FileNotFoundError(f"输入帧目录不存在: {frames_dir}")
    frame_records = load_frame_records(frames_dir, manifest_path)
    if not frame_records:
        raise FileNotFoundError(f"输入帧目录没有图片: {frames_dir}")
    if args.fps <= 0:
        raise ValueError("--fps 必须为正数")

    try:
        import mediapipe as mp
    except Exception as exc:
        raise ImportError("缺少 mediapipe，请先安装任务2环境或运行 bash task2/scripts/check_task2_env.sh") from exc

    out_json.parent.mkdir(parents=True, exist_ok=True)
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    manifest_meta = load_manifest_meta(manifest_path)
    results = {
        "schema_version": "task2_mediapipe_hands_v1",
        "frames_dir": str(frames_dir),
        "manifest": str(manifest_path) if manifest_path else "",
        "source_video": manifest_meta.get("source_video", ""),
        "source_fps": manifest_meta.get("source_fps", args.fps),
        "target_fps": manifest_meta.get("target_fps", args.fps),
        "source_total_frames": manifest_meta.get("source_total_frames", 0),
        "num_frames": len(frame_records),
        "settings": {
            "max_num_hands": args.max_num_hands,
            "min_detection_confidence": args.min_detection_confidence,
            "min_tracking_confidence": args.min_tracking_confidence,
            "fps": args.fps,
        },
        "frames": [],
    }

    if hasattr(mp, "solutions"):
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=args.max_num_hands,
            model_complexity=1,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        ) as hands:
            for idx, record in enumerate(tqdm(frame_records, desc="mediapipe hands")):
                frame_path = record["path"]
                frame_id = int(record.get("frame_id", idx))
                image = cv2.imread(str(frame_path))
                if image is None:
                    raise RuntimeError(f"读取图片失败: {frame_path}")
                height, width = image.shape[:2]
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detected = hands.process(rgb)

                frame_record = {
                    "frame_id": frame_id,
                    "file_name": frame_path.name,
                    "source_frame_idx": int(record.get("source_frame_idx", frame_id)),
                    "timestamp_sec": float(record.get("timestamp_sec", frame_id / args.fps)),
                    "image_size": [height, width],
                    "hands": [],
                }

                if detected.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(detected.multi_hand_landmarks):
                        handedness_item = detected.multi_handedness[hand_idx].classification[0]
                        points_2d = normalized_to_pixel_landmarks(hand_landmarks.landmark, width, height)
                        points_norm = normalized_landmarks(hand_landmarks.landmark)
                        world = None
                        if detected.multi_hand_world_landmarks:
                            world = detected.multi_hand_world_landmarks[hand_idx]
                        label, score = category_label_score(handedness_item)
                        frame_record["hands"].append(
                            {
                                "hand_index": hand_idx,
                                "handedness": label,
                                "handedness_score": score,
                                "bbox_xyxy": bbox_from_points(points_2d, width, height),
                                "landmarks_2d": points_2d,
                                "landmarks_normalized": points_norm,
                                "world_landmarks": world_landmarks(world),
                            }
                        )
                        if vis_dir:
                            draw_hand(image, points_2d, f"{label} {score:.2f}")

                results["frames"].append(frame_record)
                if vis_dir:
                    out_frame = vis_dir / frame_path.name
                    if not cv2.imwrite(str(out_frame), image):
                        raise RuntimeError(f"写入可视化帧失败: {out_frame}")
    else:
        model_asset_path = Path(args.model_asset_path)
        if not model_asset_path.is_file():
            raise FileNotFoundError(
                "当前 mediapipe 是 Tasks-only 版本，需要 HandLandmarker .task 模型文件。"
                f"请将模型放到 {model_asset_path}，或用 --model_asset_path 指定。"
            )
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_asset_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=args.max_num_hands,
            min_hand_detection_confidence=args.min_detection_confidence,
            min_hand_presence_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        with HandLandmarker.create_from_options(options) as landmarker:
            for idx, record in enumerate(tqdm(frame_records, desc="mediapipe tasks hands")):
                frame_path = record["path"]
                frame_id = int(record.get("frame_id", idx))
                image = cv2.imread(str(frame_path))
                if image is None:
                    raise RuntimeError(f"读取图片失败: {frame_path}")
                height, width = image.shape[:2]
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_sec = float(record.get("timestamp_sec", frame_id / args.fps))
                detected = landmarker.detect_for_video(mp_image, int(timestamp_sec * 1000))

                frame_record = {
                    "frame_id": frame_id,
                    "file_name": frame_path.name,
                    "source_frame_idx": int(record.get("source_frame_idx", frame_id)),
                    "timestamp_sec": timestamp_sec,
                    "image_size": [height, width],
                    "hands": [],
                }

                for hand_idx, hand_landmarks in enumerate(detected.hand_landmarks):
                    category = detected.handedness[hand_idx][0] if detected.handedness and detected.handedness[hand_idx] else None
                    label, score = category_label_score(category) if category else ("Unknown", 0.0)
                    points_2d = normalized_to_pixel_landmarks(hand_landmarks, width, height)
                    points_norm = normalized_landmarks(hand_landmarks)
                    world = detected.hand_world_landmarks[hand_idx] if detected.hand_world_landmarks else None
                    frame_record["hands"].append(
                        {
                            "hand_index": hand_idx,
                            "handedness": label,
                            "handedness_score": score,
                            "bbox_xyxy": bbox_from_points(points_2d, width, height),
                            "landmarks_2d": points_2d,
                            "landmarks_normalized": points_norm,
                            "world_landmarks": world_landmarks(world),
                        }
                    )
                    if vis_dir:
                        draw_hand(image, points_2d, f"{label} {score:.2f}")

                results["frames"].append(frame_record)
                if vis_dir:
                    out_frame = vis_dir / frame_path.name
                    if not cv2.imwrite(str(out_frame), image):
                        raise RuntimeError(f"写入可视化帧失败: {out_frame}")

    valid_count = sum(1 for item in results["frames"] if item["hands"])
    results["valid_frames"] = valid_count
    results["valid_ratio"] = valid_count / len(frame_records)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] frames: {len(frame_records)}")
    print(f"[OK] valid frames: {valid_count} ({results['valid_ratio']:.2%})")
    print(f"[OK] out_json: {out_json}")
    if vis_dir:
        print(f"[OK] vis_dir: {vis_dir}")


if __name__ == "__main__":
    main()
