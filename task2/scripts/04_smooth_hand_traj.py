#!/usr/bin/env python3
"""对 MediaPipe 关键点轨迹做缺失帧插值和简单时间平滑。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "task2" / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.hand_schema import FINGERTIP_INDICES, HAND_CONNECTIONS, HAND_LANDMARK_COUNT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth MediaPipe hand trajectory and export npz.")
    parser.add_argument("--in_json", required=True, help="02_run_mediapipe_hands.py 输出 JSON")
    parser.add_argument("--out_npz", default="task2/outputs/trajectories/hand_traj_smooth.npz", help="输出平滑 npz")
    parser.add_argument("--quality_report", default="task2/reports/trajectory_quality.md", help="输出轨迹质量简报")
    parser.add_argument("--smooth_window", type=int, default=5, help="移动平均窗口，建议为奇数")
    parser.add_argument("--fps", type=float, default=30.0, help="轨迹帧率，默认 30；真实视频建议传入原始 fps")
    parser.add_argument("--sequence_id", default="unknown_sequence", help="序列名，写入 npz 元信息")
    return parser.parse_args()


def choose_primary_hand(hands: list[dict]) -> dict | None:
    if not hands:
        return None
    return max(hands, key=lambda item: item.get("handedness_score", 0.0))


def interpolate_nan(values: np.ndarray) -> np.ndarray:
    smoothed = values.copy()
    time = np.arange(values.shape[0])
    flat = smoothed.reshape(values.shape[0], -1)
    for col in range(flat.shape[1]):
        y = flat[:, col]
        valid = np.isfinite(y)
        if valid.sum() == 0:
            continue
        if valid.sum() == 1:
            y[~valid] = y[valid][0]
        else:
            y[~valid] = np.interp(time[~valid], time[valid], y[valid])
    return flat.reshape(values.shape)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, [(radius, radius)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    output = np.empty_like(values)
    for idx in range(values.shape[0]):
        output[idx] = np.nanmean(padded[idx: idx + window], axis=0)
    return output


def compute_bone_stats(keypoints3d: np.ndarray) -> tuple[float, float]:
    lengths = []
    for start, end in HAND_CONNECTIONS:
        segment = keypoints3d[:, end] - keypoints3d[:, start]
        length = np.linalg.norm(segment, axis=-1)
        lengths.append(length)
    stacked = np.stack(lengths, axis=1)
    return float(np.nanmean(stacked)), float(np.nanstd(stacked))


def parse_camera_id(sequence_id: str) -> str:
    for camera_id in ["camera_side_1", "camera_side_2", "camera_top"]:
        if sequence_id.endswith(camera_id):
            return camera_id
    return "unknown"


def rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    norm = np.linalg.norm(quat)
    return quat / norm if norm > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def estimate_palm_quat(points3d: np.ndarray) -> np.ndarray:
    quats = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (points3d.shape[0], 1))
    for idx, points in enumerate(points3d):
        wrist = points[0]
        index_mcp = points[5]
        middle_mcp = points[9]
        pinky_mcp = points[17]
        if not np.all(np.isfinite([wrist, index_mcp, middle_mcp, pinky_mcp])):
            continue
        x_axis = index_mcp - pinky_mcp
        y_axis = middle_mcp - wrist
        if np.linalg.norm(x_axis) < 1e-6 or np.linalg.norm(y_axis) < 1e-6:
            continue
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis - x_axis * float(np.dot(x_axis, y_axis))
        if np.linalg.norm(y_axis) < 1e-6:
            continue
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        if np.linalg.norm(z_axis) < 1e-6:
            continue
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        rot = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
        quats[idx] = rotmat_to_quat_xyzw(rot)
    return quats


def main() -> None:
    args = parse_args()
    in_json = Path(args.in_json)
    out_npz = Path(args.out_npz)
    quality_report = Path(args.quality_report)

    if not in_json.is_file():
        raise FileNotFoundError(f"输入 JSON 不存在: {in_json}")
    if args.smooth_window < 1:
        raise ValueError("--smooth_window 必须 >= 1")
    if args.fps <= 0:
        raise ValueError("--fps 必须为正数")

    data = json.loads(in_json.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"JSON 中没有 frames: {in_json}")

    num_frames = len(frames)
    frame_ids = np.array([int(frame.get("frame_id", idx)) for idx, frame in enumerate(frames)], dtype=np.int64)
    timestamps = np.array([float(frame.get("timestamp_sec", frame_ids[idx] / float(args.fps))) for idx, frame in enumerate(frames)], dtype=np.float32)
    keypoints2d = np.full((num_frames, HAND_LANDMARK_COUNT, 2), np.nan, dtype=np.float32)
    keypoints3d = np.full((num_frames, HAND_LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    confidence = np.zeros((num_frames, HAND_LANDMARK_COUNT), dtype=np.float32)
    valid = np.zeros((num_frames,), dtype=bool)
    handedness = np.array(["Unknown"] * num_frames, dtype="U16")
    image_size = np.array(frames[0].get("image_size", [0, 0]), dtype=np.int64)

    for idx, frame in enumerate(frames):
        hand = choose_primary_hand(frame.get("hands", []))
        if hand is None:
            continue
        landmarks_2d = np.asarray(hand.get("landmarks_2d", []), dtype=np.float32)
        world_landmarks = np.asarray(hand.get("world_landmarks", []), dtype=np.float32)
        if landmarks_2d.shape == (HAND_LANDMARK_COUNT, 2):
            keypoints2d[idx] = landmarks_2d
        if world_landmarks.shape == (HAND_LANDMARK_COUNT, 3):
            keypoints3d[idx] = world_landmarks
        valid[idx] = True
        handedness[idx] = hand.get("handedness", "Unknown")
        confidence[idx, :] = float(hand.get("handedness_score", 1.0))

    interp2d = interpolate_nan(keypoints2d)
    interp3d = interpolate_nan(keypoints3d)
    smooth2d = moving_average(interp2d, args.smooth_window)
    smooth3d = moving_average(interp3d, args.smooth_window)
    wrist_pos = smooth3d[:, 0, :]
    fingertips3d = smooth3d[:, FINGERTIP_INDICES, :]
    interpolated_flag = ~valid
    quality_score = confidence.mean(axis=1)
    palm_pos = np.nanmean(smooth3d[:, [0, 5, 9, 13, 17], :], axis=1).astype(np.float32)
    palm_rot = estimate_palm_quat(smooth3d)
    wrist_rot = palm_rot.copy()

    wrist_diff = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1) if num_frames > 1 else np.array([], dtype=np.float32)
    jump_threshold = float(np.nanmedian(wrist_diff) + 3.0 * np.nanstd(wrist_diff)) if wrist_diff.size else 0.0
    jump_frames = np.where(wrist_diff > jump_threshold)[0] + 1 if wrist_diff.size else np.array([], dtype=np.int64)
    bone_mean, bone_std = compute_bone_stats(smooth3d)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        schema_version=np.array("task2_hand_traj_v1"),
        sequence_id=np.array(args.sequence_id),
        fps=np.array(float(args.fps), dtype=np.float32),
        frame_ids=frame_ids,
        timestamps=timestamps,
        image_size=image_size,
        valid=valid,
        interpolated_flag=interpolated_flag,
        quality_score=quality_score,
        hand_landmarks_2d=smooth2d,
        hand_landmarks_3d=smooth3d,
        keypoints2d=smooth2d,
        keypoints2d_score=confidence,
        keypoints3d=smooth3d,
        keypoints3d_score=confidence,
        world_landmarks=smooth3d,
        confidence=confidence,
        handedness=handedness,
        handedness_score=confidence.mean(axis=1),
        wrist_pos=wrist_pos,
        wrist_rot=wrist_rot,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        fingertips3d=fingertips3d,
        fingertips_score=confidence[:, FINGERTIP_INDICES],
        coord_frame=np.array("mediapipe_world"),
        units=np.array("non_metric_mediapipe_world"),
        quat_order=np.array("xyzw"),
        keypoint_convention=np.array("mediapipe_21"),
        source=np.array("mediapipe baseline"),
        notes=np.array("MediaPipe world landmarks are baseline 3D skeletons, not guaranteed metric 3D. Missing frames are linearly interpolated and smoothed."),
        source_video=np.array(data.get("source_video", "")),
        frame_manifest=np.array(data.get("manifest", "")),
        dataset_name=np.array("HO-Tracker-Challenge"),
        primary_camera_id=np.array(parse_camera_id(args.sequence_id)),
        camera_ids=np.array([parse_camera_id(args.sequence_id)], dtype="U32"),
        metric_3d_valid=np.array(False),
        world_alignment_valid=np.array(False),
        camera_calib_valid=np.array(False),
        contact_valid=np.array(False),
        phase_valid=np.array(False),
        wrist_rot_valid=np.array(True),
        palm_rot_valid=np.array(True),
    )

    quality_report.parent.mkdir(parents=True, exist_ok=True)
    missing = frame_ids[~valid].tolist()
    report = [
        "# 轨迹质量简报",
        "",
        f"- 输入 JSON：`{in_json}`",
        f"- 输出 NPZ：`{out_npz}`",
        f"- 序列名：{args.sequence_id}",
        f"- FPS：{args.fps:.3f}",
        f"- 总帧数：{num_frames}",
        f"- 检测成功帧数：{int(valid.sum())}",
        f"- 检测成功比例：{valid.mean():.2%}",
        f"- 缺失帧：{missing}",
        f"- 平滑窗口：{args.smooth_window}",
        f"- wrist 跳变阈值：{jump_threshold:.6f}",
        f"- wrist 跳变帧：{jump_frames.tolist()}",
        f"- 3D 骨长均值：{bone_mean:.6f}",
        f"- 3D 骨长标准差：{bone_std:.6f}",
        "",
        "## 说明",
        "",
        "当前结果来自 MediaPipe baseline。3D world landmarks 不是严格真实尺度，遮挡和运动模糊时可能出现漏检或跳变。",
    ]
    quality_report.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"[OK] out_npz: {out_npz}")
    print(f"[OK] quality_report: {quality_report}")
    print(f"[OK] valid ratio: {valid.mean():.2%}")


if __name__ == "__main__":
    main()
