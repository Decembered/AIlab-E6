#!/usr/bin/env python3
from __future__ import annotations
"""Triangulate task2 MediaPipe 2D landmarks from multiple calibrated views.

This script is intentionally independent from the single-view baseline. It
produces a separate multiview hand trajectory file and does not overwrite the
existing `hand_traj.npz` unless the caller explicitly chooses that output path.
"""

import argparse
import json
from pathlib import Path

import numpy as np


CAMERAS = ("camera_side_1", "camera_side_2", "camera_top")
HAND_LANDMARK_COUNT = 21
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
    parser = argparse.ArgumentParser(description="Triangulate multiview task2 hand trajectory.")
    parser.add_argument("--sequence_id", required=True, help="Normalized sequence id, without camera suffix.")
    parser.add_argument("--by_view_dir", default="task2/outputs/by_view", help="task2 by_view output directory.")
    parser.add_argument("--calib_json", required=True, help="camera_calib.json exported by 09_export_camera_calib.py.")
    parser.add_argument("--out_npz", required=True, help="Output multiview hand trajectory npz.")
    parser.add_argument("--out_report", default=None, help="Optional markdown report path.")
    parser.add_argument("--primary_camera", default="camera_side_2", choices=CAMERAS, help="Single-view fields use this camera when available.")
    parser.add_argument("--target_handedness", default="auto", choices=["auto", "Left", "Right"], help="Prefer this MediaPipe handedness label.")
    parser.add_argument("--smooth_window", type=int, default=5, help="Odd moving-average window for 3D smoothing.")
    parser.add_argument("--min_support_joints", type=int, default=11, help="Frame is valid if at least this many joints have support >=2.")
    parser.add_argument("--reproj_sigma_px", type=float, default=10.0, help="Quality score reprojection scale.")
    return parser.parse_args()


def choose_hand(hands: list[dict], target_handedness: str) -> dict | None:
    if not hands:
        return None
    if target_handedness != "auto":
        candidates = [hand for hand in hands if hand.get("handedness") == target_handedness]
        if candidates:
            return max(candidates, key=lambda item: item.get("handedness_score", 0.0))
    return max(hands, key=lambda item: item.get("handedness_score", 0.0))


def load_view(path: Path, target_handedness: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"No frames in {path}")
    by_frame = {}
    for idx, frame in enumerate(frames):
        source_idx = int(frame.get("source_frame_idx", frame.get("frame_id", idx)))
        hand = choose_hand(frame.get("hands", []), target_handedness)
        image_size = np.asarray(frame.get("image_size", [0, 0]), dtype=np.int64)
        item = {
            "frame_id": int(frame.get("frame_id", source_idx)),
            "source_frame_idx": source_idx,
            "timestamp_sec": float(frame.get("timestamp_sec", source_idx / float(data.get("target_fps", 30.0)))),
            "image_size": image_size,
            "valid": False,
            "keypoints2d": np.full((HAND_LANDMARK_COUNT, 2), np.nan, dtype=np.float32),
            "world_landmarks": np.full((HAND_LANDMARK_COUNT, 3), np.nan, dtype=np.float32),
            "score": np.zeros((HAND_LANDMARK_COUNT,), dtype=np.float32),
            "handedness": "Unknown",
            "handedness_score": 0.0,
            "bbox_xyxy": np.full((4,), np.nan, dtype=np.float32),
        }
        if hand is not None:
            keypoints2d = np.asarray(hand.get("landmarks_2d", []), dtype=np.float32)
            world = np.asarray(hand.get("world_landmarks", []), dtype=np.float32)
            if keypoints2d.shape == (HAND_LANDMARK_COUNT, 2):
                item["keypoints2d"] = keypoints2d
                item["valid"] = True
            if world.shape == (HAND_LANDMARK_COUNT, 3):
                item["world_landmarks"] = world
            score = float(hand.get("handedness_score", 1.0))
            item["score"][:] = score
            item["handedness"] = hand.get("handedness", "Unknown")
            item["handedness_score"] = score
            bbox = np.asarray(hand.get("bbox_xyxy", []), dtype=np.float32)
            if bbox.shape == (4,):
                item["bbox_xyxy"] = bbox
        by_frame[source_idx] = item
    return {
        "json_path": str(path),
        "source_video": data.get("source_video", ""),
        "source_fps": float(data.get("source_fps", data.get("target_fps", 30.0))),
        "target_fps": float(data.get("target_fps", data.get("source_fps", 30.0))),
        "frames": by_frame,
    }


def load_calibration(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    cameras = {}
    for cam in data.get("cameras", []):
        camera_id = cam["camera_id"]
        cameras[camera_id] = {
            "K": np.asarray(cam["K"], dtype=np.float64),
            "extrinsic": np.asarray(cam["extrinsic"], dtype=np.float64),
        }
    missing = [camera for camera in CAMERAS if camera not in cameras]
    if missing:
        raise KeyError(f"Calibration missing cameras: {missing}")
    return cameras


def projection_matrix(camera: dict, mode: str) -> np.ndarray:
    extrinsic = camera["extrinsic"]
    if mode == "world_to_camera":
        ext = extrinsic[:3, :]
    elif mode == "camera_to_world":
        ext = np.linalg.inv(extrinsic)[:3, :]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return camera["K"] @ ext


def triangulate_point(projections: list[np.ndarray], points: list[np.ndarray]) -> np.ndarray:
    rows = []
    for proj, point in zip(projections, points):
        x, y = float(point[0]), float(point[1])
        rows.append(x * proj[2] - proj[0])
        rows.append(y * proj[2] - proj[1])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    xyz_h = vh[-1]
    if abs(float(xyz_h[3])) < 1e-12:
        return np.full((3,), np.nan, dtype=np.float32)
    return (xyz_h[:3] / xyz_h[3]).astype(np.float32)


def reproject(proj: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    homog = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    uvw = proj @ homog
    if abs(float(uvw[2])) < 1e-12:
        return np.full((2,), np.nan, dtype=np.float32)
    return (uvw[:2] / uvw[2]).astype(np.float32)


def interpolate_nan(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = values.copy()
    filled = np.zeros((values.shape[0],), dtype=bool)
    time = np.arange(values.shape[0])
    flat = output.reshape(values.shape[0], -1)
    for col in range(flat.shape[1]):
        y = flat[:, col]
        valid = np.isfinite(y)
        if valid.sum() == 0:
            y[:] = 0.0
            filled[:] = True
            continue
        if valid.sum() == 1:
            y[~valid] = y[valid][0]
        else:
            y[~valid] = np.interp(time[~valid], time[valid], y[valid])
        filled |= ~valid
    return flat.reshape(values.shape), filled


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, [(radius, radius)] + [(0, 0)] * (values.ndim - 1), mode="edge")
    output = np.empty_like(values)
    for idx in range(values.shape[0]):
        output[idx] = np.mean(padded[idx: idx + window], axis=0)
    return output.astype(np.float32)


def rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rot[2, 1] - rot[1, 2]) / scale
        qy = (rot[0, 2] - rot[2, 0]) / scale
        qz = (rot[1, 0] - rot[0, 1]) / scale
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rot[0, 1] + rot[1, 0]) / scale
            qz = (rot[0, 2] + rot[2, 0]) / scale
        elif idx == 1:
            scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / scale
            qx = (rot[0, 1] + rot[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rot[1, 2] + rot[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / scale
            qx = (rot[0, 2] + rot[2, 0]) / scale
            qy = (rot[1, 2] + rot[2, 1]) / scale
            qz = 0.25 * scale
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
        quats[idx] = rotmat_to_quat_xyzw(np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32))
    return quats


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
    lengths = [np.linalg.norm(keypoints3d[:, end] - keypoints3d[:, start], axis=1) for start, end in HAND_CONNECTIONS]
    stacked = np.stack(lengths, axis=1)
    median = np.nanmedian(stacked, axis=0, keepdims=True)
    return np.nanmean(np.abs(stacked - median), axis=1).astype(np.float32)


def failure_reasons(valid: np.ndarray, interpolated: np.ndarray, quality_score: np.ndarray, jump_score: np.ndarray) -> np.ndarray:
    reasons = np.array(["ok"] * valid.shape[0], dtype="U32")
    reasons[~valid] = "triangulation_low_support"
    reasons[interpolated] = "interpolated"
    reasons[(valid) & (quality_score < 0.35)] = "low_confidence"
    reasons[(valid) & (jump_score > 1.5)] = "temporal_jump"
    return reasons


def triangulate_mode(projections: dict[str, np.ndarray], keypoints2d: np.ndarray, view_valid: np.ndarray) -> dict:
    num_frames, num_views = keypoints2d.shape[:2]
    xyz = np.full((num_frames, HAND_LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    support = np.zeros((num_frames, HAND_LANDMARK_COUNT), dtype=np.int16)
    reproj = np.full((num_frames, num_views, HAND_LANDMARK_COUNT), np.nan, dtype=np.float32)
    errors = []
    for t in range(num_frames):
        for joint in range(HAND_LANDMARK_COUNT):
            projs = []
            pts = []
            view_indices = []
            for view_idx, camera_id in enumerate(CAMERAS):
                point = keypoints2d[t, view_idx, joint]
                if view_valid[t, view_idx] and np.all(np.isfinite(point)):
                    projs.append(projections[camera_id])
                    pts.append(point)
                    view_indices.append(view_idx)
            support[t, joint] = len(pts)
            if len(pts) < 2:
                continue
            point3d = triangulate_point(projs, pts)
            if not np.all(np.isfinite(point3d)):
                continue
            xyz[t, joint] = point3d
            for local_idx, view_idx in enumerate(view_indices):
                projected = reproject(projs[local_idx], point3d)
                if np.all(np.isfinite(projected)):
                    err = float(np.linalg.norm(projected - pts[local_idx]))
                    reproj[t, view_idx, joint] = err
                    errors.append(err)
    median = float(np.nanmedian(np.asarray(errors, dtype=np.float32))) if errors else float("inf")
    return {"xyz": xyz, "support": support, "reprojection_error": reproj, "median_reprojection_px": median}


def main() -> None:
    args = parse_args()
    by_view_dir = Path(args.by_view_dir)
    calib_json = Path(args.calib_json)
    out_npz = Path(args.out_npz)
    out_report = Path(args.out_report) if args.out_report else out_npz.with_suffix(".md")

    views = {}
    for camera_id in CAMERAS:
        path = by_view_dir / f"{args.sequence_id}_{camera_id}" / "trajectories" / "mediapipe_landmarks.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing view landmarks: {path}")
        views[camera_id] = load_view(path, args.target_handedness)

    common_frames = set.intersection(*(set(view["frames"].keys()) for view in views.values()))
    if not common_frames:
        raise ValueError("No common source_frame_idx across views")
    frame_keys = np.array(sorted(common_frames), dtype=np.int64)
    num_frames = len(frame_keys)
    num_views = len(CAMERAS)

    primary_camera = args.primary_camera if args.primary_camera in CAMERAS else CAMERAS[0]
    fps = float(views[primary_camera]["target_fps"])
    frame_ids = frame_keys.astype(np.int64)
    timestamps = np.array([views[primary_camera]["frames"][int(key)]["timestamp_sec"] for key in frame_keys], dtype=np.float32)
    image_sizes = np.stack([views[camera]["frames"][int(frame_keys[0])]["image_size"] for camera in CAMERAS], axis=0).astype(np.int64)
    primary_idx = CAMERAS.index(primary_camera)

    keypoints2d_mv = np.full((num_frames, num_views, HAND_LANDMARK_COUNT, 2), np.nan, dtype=np.float32)
    keypoints2d_score_mv = np.zeros((num_frames, num_views, HAND_LANDMARK_COUNT), dtype=np.float32)
    world_mv = np.full((num_frames, num_views, HAND_LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    view_valid = np.zeros((num_frames, num_views), dtype=bool)
    handedness_mv = np.empty((num_frames, num_views), dtype="U16")
    handedness_score_mv = np.zeros((num_frames, num_views), dtype=np.float32)
    bbox_mv = np.full((num_frames, num_views, 4), np.nan, dtype=np.float32)

    for view_idx, camera_id in enumerate(CAMERAS):
        frames = views[camera_id]["frames"]
        for t, key in enumerate(frame_keys):
            item = frames[int(key)]
            keypoints2d_mv[t, view_idx] = item["keypoints2d"]
            keypoints2d_score_mv[t, view_idx] = item["score"]
            world_mv[t, view_idx] = item["world_landmarks"]
            view_valid[t, view_idx] = bool(item["valid"])
            handedness_mv[t, view_idx] = item["handedness"]
            handedness_score_mv[t, view_idx] = float(item["handedness_score"])
            bbox_mv[t, view_idx] = item["bbox_xyxy"]

    calib = load_calibration(calib_json)
    candidates = {}
    projection_by_mode = {}
    for mode in ["world_to_camera", "camera_to_world"]:
        projections = {camera_id: projection_matrix(calib[camera_id], mode) for camera_id in CAMERAS}
        projection_by_mode[mode] = projections
        candidates[mode] = triangulate_mode(projections, keypoints2d_mv, view_valid)
    best_mode = min(candidates, key=lambda mode: candidates[mode]["median_reprojection_px"])
    best = candidates[best_mode]

    interp3d, interpolated_flag = interpolate_nan(best["xyz"])
    smooth3d = moving_average(interp3d, args.smooth_window)
    keypoints2d_primary, keypoints2d_interpolated = interpolate_nan(keypoints2d_mv[:, primary_idx])
    keypoints2d_primary = moving_average(keypoints2d_primary, 1)
    support = best["support"]
    support_score = np.clip(support.astype(np.float32) / float(num_views), 0.0, 1.0)
    frame_reproj = np.nanmedian(best["reprojection_error"], axis=(1, 2)).astype(np.float32)
    frame_reproj = np.where(np.isfinite(frame_reproj), frame_reproj, np.nanmax(frame_reproj[np.isfinite(frame_reproj)]) if np.isfinite(frame_reproj).any() else 999.0)
    reproj_score = np.exp(-frame_reproj / max(args.reproj_sigma_px, 1e-6)).astype(np.float32)
    support_frame_score = support_score.mean(axis=1)
    primary_score = keypoints2d_score_mv[:, primary_idx].mean(axis=1)
    quality_score = np.clip(0.5 * support_frame_score + 0.3 * reproj_score + 0.2 * primary_score, 0.0, 1.0).astype(np.float32)
    valid = ((support >= 2).sum(axis=1) >= args.min_support_joints)
    interpolated_flag = interpolated_flag | keypoints2d_interpolated | ~valid

    wrist_pos = smooth3d[:, 0]
    fingertips3d = smooth3d[:, FINGERTIP_INDICES]
    palm_pos = smooth3d[:, [0, 5, 9, 13, 17]].mean(axis=1).astype(np.float32)
    palm_rot = estimate_palm_quat(smooth3d)
    wrist_rot = palm_rot.copy()
    retarget_keypoints3d = np.concatenate([smooth3d[:, RETARGET_INDICES], palm_pos[:, None]], axis=1).astype(np.float32)
    retarget_keypoints3d_palm = (retarget_keypoints3d - palm_pos[:, None]).astype(np.float32)
    fingertips3d_palm = (fingertips3d - palm_pos[:, None]).astype(np.float32)
    jump_score = temporal_jump_score(wrist_pos)
    bone_error = bone_length_error(smooth3d)
    failure_reason = failure_reasons(valid, interpolated_flag, quality_score, jump_score)
    confidence = support_score.astype(np.float32)
    projection_matrices = np.stack([projection_by_mode[best_mode][camera_id] for camera_id in CAMERAS], axis=0).astype(np.float32)
    K = np.stack([calib[camera_id]["K"] for camera_id in CAMERAS], axis=0).astype(np.float32)
    extrinsics = np.stack([calib[camera_id]["extrinsic"] for camera_id in CAMERAS], axis=0).astype(np.float32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        schema_version=np.array("task2_hand_traj_v1"),
        sequence_id=np.array(args.sequence_id),
        fps=np.array(fps, dtype=np.float32),
        frame_ids=frame_ids,
        timestamps=timestamps,
        image_size=image_sizes[primary_idx],
        valid=valid,
        interpolated_flag=interpolated_flag,
        quality_score=quality_score,
        hand_landmarks_2d=keypoints2d_primary.astype(np.float32),
        hand_landmarks_3d=smooth3d.astype(np.float32),
        keypoints2d=keypoints2d_primary.astype(np.float32),
        keypoints2d_score=keypoints2d_score_mv[:, primary_idx].astype(np.float32),
        keypoints3d=smooth3d.astype(np.float32),
        keypoints3d_score=confidence,
        world_landmarks=smooth3d.astype(np.float32),
        keypoints3d_raw=best["xyz"].astype(np.float32),
        keypoints3d_smooth=smooth3d.astype(np.float32),
        confidence=confidence,
        handedness=handedness_mv[:, primary_idx],
        handedness_score=handedness_score_mv[:, primary_idx].astype(np.float32),
        wrist_pos=wrist_pos.astype(np.float32),
        wrist_pos_smooth=wrist_pos.astype(np.float32),
        wrist_rot=wrist_rot,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        T_world_wrist=make_pose_mats(wrist_pos, wrist_rot),
        T_world_palm=make_pose_mats(palm_pos, palm_rot),
        fingertips3d=fingertips3d.astype(np.float32),
        fingertips_score=confidence[:, FINGERTIP_INDICES],
        fingertips3d_palm=fingertips3d_palm,
        wrist_vel=velocity(wrist_pos, fps),
        keypoints3d_vel=velocity(smooth3d, fps),
        fingertips_vel=velocity(fingertips3d, fps),
        temporal_jump_score=jump_score,
        bone_length_error=bone_error,
        failure_reason=failure_reason,
        retarget_landmark_names=np.array(RETARGET_NAMES, dtype="U32"),
        retarget_keypoints3d=retarget_keypoints3d,
        retarget_keypoints3d_palm=retarget_keypoints3d_palm,
        retarget_weights=np.array([1.0, 2.0, 2.2, 2.0, 1.8, 1.6, 1.0, 1.0, 0.9, 0.9, 1.5], dtype=np.float32),
        contact_likelihood=np.zeros((num_frames, 5), dtype=np.float32),
        active_fingers=np.zeros((num_frames, 5), dtype=bool),
        phase=np.array(["unknown"] * num_frames, dtype="U16"),
        camera_ids=np.array(CAMERAS, dtype="U32"),
        primary_camera_id=np.array(primary_camera),
        multi_view_keypoints2d=keypoints2d_mv.astype(np.float32),
        multi_view_keypoints2d_score=keypoints2d_score_mv.astype(np.float32),
        multi_view_image_size=image_sizes,
        view_valid=view_valid,
        hand_bbox2d=bbox_mv.astype(np.float32),
        handedness_by_view=handedness_mv,
        handedness_score_by_view=handedness_score_mv.astype(np.float32),
        K=K,
        camera_extrinsics_raw=extrinsics,
        projection_matrices=projection_matrices,
        triangulation_support=support,
        reprojection_error=best["reprojection_error"].astype(np.float32),
        reprojection_error_frame_median=frame_reproj.astype(np.float32),
        triangulation_mode=np.array(best_mode),
        triangulation_median_reprojection_px=np.array(best["median_reprojection_px"], dtype=np.float32),
        metric_3d_valid=np.array(True),
        world_alignment_valid=np.array(False),
        camera_calib_valid=np.array(True),
        contact_valid=np.array(False),
        phase_valid=np.array(False),
        wrist_rot_valid=np.array(True),
        palm_rot_valid=np.array(True),
        source_video=np.array(views[primary_camera]["source_video"]),
        frame_manifest=np.array(""),
        dataset_name=np.array("HO-Tracker-Challenge"),
        coord_frame=np.array("hotracker_camera_calib_world"),
        units=np.array("meter"),
        quat_order=np.array("xyzw"),
        keypoint_convention=np.array("mediapipe_21"),
        source=np.array("mediapipe multiview triangulation"),
        notes=np.array("3D joints are DLT-triangulated from MediaPipe 2D landmarks and HO-Tracker camera calibration. Extrinsic convention is selected by median reprojection error. Coordinates are in the camera-calibration world frame; they are not yet aligned to IsaacGym/world task coordinates."),
    )

    out_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Multiview Hand Triangulation Report",
        "",
        f"- sequence_id: `{args.sequence_id}`",
        f"- output: `{out_npz}`",
        f"- calibration: `{calib_json}`",
        f"- cameras: {', '.join(CAMERAS)}",
        f"- primary_camera: `{primary_camera}`",
        f"- frames: {num_frames}",
        f"- fps: {fps:.3f}",
        f"- selected_extrinsic_mode: `{best_mode}`",
        f"- median_reprojection_px: {best['median_reprojection_px']:.4f}",
        f"- valid_frames: {int(valid.sum())}/{num_frames} ({valid.mean():.2%})",
        f"- mean_support_per_joint: {float(support.mean()):.3f}",
        "",
        "## Mode Comparison",
        "",
    ]
    for mode in ["world_to_camera", "camera_to_world"]:
        lines.append(f"- `{mode}`: median_reprojection_px={candidates[mode]['median_reprojection_px']:.4f}")
    lines.extend(["", "## View Valid Frames", ""])
    for view_idx, camera_id in enumerate(CAMERAS):
        lines.append(f"- `{camera_id}`: {int(view_valid[:, view_idx].sum())}/{num_frames} ({view_valid[:, view_idx].mean():.2%})")
    lines.extend([
        "",
        "## Notes",
        "",
        "- This file is a multiview enhancement and does not overwrite the existing single-view baseline.",
        "- `world_alignment_valid=False` means the trajectory is not yet aligned to the IsaacGym task/world frame.",
    ])
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_npz}")
    print(f"[OK] wrote {out_report}")
    print(f"[OK] selected mode: {best_mode}, median reprojection px: {best['median_reprojection_px']:.4f}")


if __name__ == "__main__":
    main()
