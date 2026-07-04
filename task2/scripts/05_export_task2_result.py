#!/usr/bin/env python3
"""整理任务2结果为统一 hand_traj.npz 接口。"""

import argparse
from pathlib import Path

import numpy as np


FINGERTIP_INDICES = [4, 8, 12, 16, 20]
RETARGET_INDICES = [0, 4, 8, 12, 16, 20, 5, 9, 13, 17]
RETARGET_NAMES = [
    "wrist",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
    "index_mcp",
    "middle_mcp",
    "ring_mcp",
    "pinky_mcp",
    "palm_center",
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export unified task2 hand_traj.npz.")
    parser.add_argument("--in_npz", required=True, help="04_smooth_hand_traj.py 输出 npz")
    parser.add_argument("--out_npz", default="task2/outputs/trajectories/hand_traj.npz", help="统一接口输出 npz")
    parser.add_argument("--sequence_id", default=None, help="可选覆盖 sequence_id")
    parser.add_argument("--fps", type=float, default=None, help="可选覆盖 fps")
    return parser.parse_args()


def make_pose_mats(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    mats = np.tile(np.eye(4, dtype=np.float32), (pos.shape[0], 1, 1))
    mats[:, :3, 3] = pos.astype(np.float32)
    quat = quat_xyzw.astype(np.float32)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = np.divide(quat, np.maximum(norm, 1e-6))
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    mats[:, 0, 0] = 1 - 2 * (y * y + z * z)
    mats[:, 0, 1] = 2 * (x * y - z * w)
    mats[:, 0, 2] = 2 * (x * z + y * w)
    mats[:, 1, 0] = 2 * (x * y + z * w)
    mats[:, 1, 1] = 1 - 2 * (x * x + z * z)
    mats[:, 1, 2] = 2 * (y * z - x * w)
    mats[:, 2, 0] = 2 * (x * z - y * w)
    mats[:, 2, 1] = 2 * (y * z + x * w)
    mats[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return mats


def velocity(values: np.ndarray, fps: float) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, 1.0 / fps, axis=0).astype(np.float32)


def temporal_jump_score(wrist_pos: np.ndarray) -> np.ndarray:
    score = np.zeros((wrist_pos.shape[0],), dtype=np.float32)
    if wrist_pos.shape[0] <= 1:
        return score
    diff = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)
    denom = np.nanpercentile(diff, 95) + 1e-6
    score[1:] = np.clip(diff / denom, 0.0, 10.0).astype(np.float32)
    return score


def bone_length_error(keypoints3d: np.ndarray) -> np.ndarray:
    lengths = []
    for start, end in HAND_CONNECTIONS:
        lengths.append(np.linalg.norm(keypoints3d[:, end] - keypoints3d[:, start], axis=1))
    stacked = np.stack(lengths, axis=1)
    median = np.nanmedian(stacked, axis=0, keepdims=True)
    return np.nanmean(np.abs(stacked - median), axis=1).astype(np.float32)


def failure_reasons(valid: np.ndarray, interpolated: np.ndarray, quality_score: np.ndarray, jump_score: np.ndarray) -> np.ndarray:
    reasons = np.array(["ok"] * valid.shape[0], dtype="U32")
    reasons[~valid] = "detector_lost"
    reasons[interpolated] = "interpolated"
    reasons[(valid) & (quality_score < 0.6)] = "low_confidence"
    reasons[(valid) & (jump_score > 1.5)] = "temporal_jump"
    return reasons


def main() -> None:
    args = parse_args()
    in_npz = Path(args.in_npz)
    out_npz = Path(args.out_npz)

    if not in_npz.is_file():
        raise FileNotFoundError(f"输入 NPZ 不存在: {in_npz}")

    data = np.load(in_npz, allow_pickle=True)
    required = ["frame_ids", "hand_landmarks_2d", "hand_landmarks_3d", "confidence", "handedness", "wrist_pos"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"输入 NPZ 缺少字段: {missing}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    frame_ids = data["frame_ids"]
    fps = np.array(args.fps if args.fps is not None else float(data["fps"]) if "fps" in data else 30.0, dtype=np.float32)
    sequence_id = np.array(args.sequence_id if args.sequence_id is not None else str(data["sequence_id"]) if "sequence_id" in data else "unknown_sequence")
    timestamps = data["timestamps"] if "timestamps" in data else frame_ids.astype(np.float32) / float(fps)
    confidence = data["confidence"]
    wrist_rot = data["wrist_rot"] if "wrist_rot" in data else np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (len(frame_ids), 1))
    palm_pos = data["palm_pos"] if "palm_pos" in data else np.nanmean(data["hand_landmarks_3d"][:, [0, 5, 9, 13, 17], :], axis=1)
    palm_rot = data["palm_rot"] if "palm_rot" in data else wrist_rot
    keypoints3d = data["keypoints3d"] if "keypoints3d" in data else data["hand_landmarks_3d"]
    keypoints2d = data["keypoints2d"] if "keypoints2d" in data else data["hand_landmarks_2d"]
    valid = data["valid"] if "valid" in data else np.ones_like(frame_ids, dtype=bool)
    interpolated_flag = data["interpolated_flag"] if "interpolated_flag" in data else np.zeros_like(frame_ids, dtype=bool)
    quality_score = data["quality_score"] if "quality_score" in data else confidence.mean(axis=1)
    fingertips3d = data["fingertips3d"] if "fingertips3d" in data else keypoints3d[:, FINGERTIP_INDICES, :]
    retarget_keypoints3d = np.concatenate([keypoints3d[:, RETARGET_INDICES, :], palm_pos[:, None, :]], axis=1).astype(np.float32)
    retarget_keypoints3d_palm = (retarget_keypoints3d - palm_pos[:, None, :]).astype(np.float32)
    fingertips3d_palm = (fingertips3d - palm_pos[:, None, :]).astype(np.float32)
    wrist_vel = velocity(data["wrist_pos"], float(fps))
    keypoints3d_vel = velocity(keypoints3d, float(fps))
    fingertips_vel = velocity(fingertips3d, float(fps))
    jump_score = temporal_jump_score(data["wrist_pos"])
    bone_error = bone_length_error(keypoints3d)
    failure_reason = failure_reasons(valid, interpolated_flag, quality_score, jump_score)
    retarget_weights = np.array([1.0, 2.0, 2.2, 2.0, 1.8, 1.6, 1.0, 1.0, 0.9, 0.9, 1.5], dtype=np.float32)
    np.savez_compressed(
        out_npz,
        schema_version=data["schema_version"] if "schema_version" in data else np.array("task2_hand_traj_v1"),
        sequence_id=sequence_id,
        fps=fps,
        frame_ids=frame_ids,
        timestamps=timestamps,
        hand_landmarks_2d=data["hand_landmarks_2d"],
        hand_landmarks_3d=data["hand_landmarks_3d"],
        keypoints2d=keypoints2d,
        keypoints2d_score=data["keypoints2d_score"] if "keypoints2d_score" in data else confidence,
        keypoints3d=keypoints3d,
        keypoints3d_score=data["keypoints3d_score"] if "keypoints3d_score" in data else confidence,
        world_landmarks=data["world_landmarks"] if "world_landmarks" in data else data["hand_landmarks_3d"],
        keypoints3d_raw=data["keypoints3d_raw"] if "keypoints3d_raw" in data else keypoints3d,
        keypoints3d_smooth=data["keypoints3d_smooth"] if "keypoints3d_smooth" in data else keypoints3d,
        confidence=confidence,
        handedness=data["handedness"],
        handedness_score=data["handedness_score"] if "handedness_score" in data else confidence.mean(axis=1),
        wrist_pos=data["wrist_pos"],
        wrist_pos_smooth=data["wrist_pos"],
        wrist_rot=wrist_rot,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        T_world_wrist=make_pose_mats(data["wrist_pos"], wrist_rot),
        T_world_palm=make_pose_mats(palm_pos, palm_rot),
        valid=valid,
        interpolated_flag=interpolated_flag,
        quality_score=quality_score,
        image_size=data["image_size"] if "image_size" in data else np.array([0, 0], dtype=np.int64),
        fingertips3d=fingertips3d,
        fingertips_score=data["fingertips_score"] if "fingertips_score" in data else confidence[:, [4, 8, 12, 16, 20]],
        fingertips3d_palm=fingertips3d_palm,
        wrist_vel=wrist_vel,
        keypoints3d_vel=keypoints3d_vel,
        fingertips_vel=fingertips_vel,
        temporal_jump_score=jump_score,
        bone_length_error=bone_error,
        failure_reason=failure_reason,
        retarget_landmark_names=np.array(RETARGET_NAMES, dtype="U32"),
        retarget_keypoints3d=retarget_keypoints3d,
        retarget_keypoints3d_palm=retarget_keypoints3d_palm,
        retarget_weights=retarget_weights,
        contact_likelihood=np.zeros((len(frame_ids), 5), dtype=np.float32),
        active_fingers=np.zeros((len(frame_ids), 5), dtype=bool),
        phase=np.array(["unknown"] * len(frame_ids), dtype="U16"),
        metric_3d_valid=np.array(False),
        world_alignment_valid=np.array(False),
        camera_calib_valid=np.array(False),
        contact_valid=np.array(False),
        phase_valid=np.array(False),
        wrist_rot_valid=np.array(True),
        palm_rot_valid=np.array(True),
        source_video=data["source_video"] if "source_video" in data else np.array(""),
        frame_manifest=data["frame_manifest"] if "frame_manifest" in data else np.array(""),
        dataset_name=data["dataset_name"] if "dataset_name" in data else np.array("unknown"),
        primary_camera_id=data["primary_camera_id"] if "primary_camera_id" in data else np.array("unknown"),
        camera_ids=data["camera_ids"] if "camera_ids" in data else np.array(["unknown"], dtype="U32"),
        coord_frame=data["coord_frame"] if "coord_frame" in data else np.array("mediapipe_world"),
        units=data["units"] if "units" in data else np.array("non_metric_mediapipe_world"),
        quat_order=data["quat_order"] if "quat_order" in data else np.array("xyzw"),
        keypoint_convention=data["keypoint_convention"] if "keypoint_convention" in data else np.array("mediapipe_21"),
        source=np.array("mediapipe baseline"),
        notes=np.array("This is a MediaPipe non-metric baseline export. wrist_rot/palm_rot are approximate palm-frame quaternions estimated from 21 landmarks, not metric world orientation. Missing frames are interpolated and marked by valid/interpolated_flag. Replace 3D with HaMeR/MANO or multi-view metric joints for high-quality retargeting."),
    )
    print(f"[OK] exported unified hand trajectory: {out_npz}")


if __name__ == "__main__":
    main()
