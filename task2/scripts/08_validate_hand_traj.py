#!/usr/bin/env python3
from __future__ import annotations
"""校验 task2 hand_traj.npz 是否满足标准接口。"""

import argparse
from pathlib import Path

import numpy as np


REQUIRED = {
    "schema_version": (),
    "sequence_id": (),
    "fps": (),
    "frame_ids": ("T",),
    "timestamps": ("T",),
    "image_size": None,
    "valid": ("T",),
    "quality_score": ("T",),
    "handedness": ("T",),
    "handedness_score": ("T",),
    "keypoints2d": ("T", 21, 2),
    "keypoints2d_score": ("T", 21),
    "keypoints3d": ("T", 21, 3),
    "keypoints3d_score": ("T", 21),
    "wrist_pos": ("T", 3),
    "wrist_rot": ("T", 4),
    "palm_pos": ("T", 3),
    "palm_rot": ("T", 4),
    "fingertips3d": ("T", 5, 3),
    "fingertips_score": ("T", 5),
    "coord_frame": (),
    "units": (),
    "keypoint_convention": (),
    "source": (),
    "notes": (),
}

RECOMMENDED = [
    "interpolated_flag",
    "failure_reason",
    "temporal_jump_score",
    "bone_length_error",
    "retarget_landmark_names",
    "retarget_keypoints3d",
    "retarget_keypoints3d_palm",
    "retarget_weights",
    "T_world_wrist",
    "T_world_palm",
    "metric_3d_valid",
    "world_alignment_valid",
    "camera_calib_valid",
    "contact_valid",
    "phase_valid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate task2 hand_traj.npz schema and basic quality.")
    parser.add_argument("--npz", default="task2/outputs/trajectories/hand_traj.npz", help="待校验 hand_traj.npz")
    parser.add_argument("--out_report", default="task2/reports/hand_traj_validation.md", help="输出校验报告")
    parser.add_argument("--min_valid_ratio", type=float, default=0.2, help="低于该有效帧比例时报错")
    return parser.parse_args()


def shape_ok(actual: tuple[int, ...], expected: tuple, t: int) -> bool:
    if expected is None:
        return True
    if len(actual) != len(expected):
        return False
    for got, exp in zip(actual, expected):
        if exp == "T":
            if got != t:
                return False
        elif got != exp:
            return False
    return True


def as_bool_scalar(value) -> bool:
    return bool(np.asarray(value).item())


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    out_report = Path(args.out_report)
    if not npz_path.is_file():
        raise FileNotFoundError(f"hand_traj.npz 不存在: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    errors = []
    warnings = []
    lines = ["# hand_traj.npz 校验报告", "", f"- 文件：`{npz_path}`"]

    if "frame_ids" not in data:
        raise KeyError("缺少 frame_ids，无法确定时间维度")
    t = int(data["frame_ids"].shape[0])

    for key, expected in REQUIRED.items():
        if key not in data:
            errors.append(f"缺少 required 字段：{key}")
            continue
        if expected is not None and not shape_ok(data[key].shape, expected, t):
            errors.append(f"字段 {key} shape 错误：got {data[key].shape}, expected {expected}")

    for key in RECOMMENDED:
        if key not in data:
            warnings.append(f"缺少 recommended 字段：{key}")

    fps = float(data["fps"]) if "fps" in data else 0.0
    if fps <= 0:
        errors.append("fps 必须为正数")

    valid_ratio = float(data["valid"].mean()) if "valid" in data else 0.0
    if valid_ratio < args.min_valid_ratio:
        errors.append(f"有效帧比例过低：{valid_ratio:.2%} < {args.min_valid_ratio:.2%}")

    if t <= 0:
        errors.append("T 必须大于 0")

    if "frame_ids" in data:
        frame_ids = data["frame_ids"]
        if np.any(np.diff(frame_ids) <= 0):
            errors.append("frame_ids 必须严格递增且无重复")
        if frame_ids[0] < 0:
            errors.append("frame_ids 不能为负")

    if "timestamps" in data:
        timestamps = data["timestamps"]
        if not np.all(np.isfinite(timestamps)):
            errors.append("timestamps 存在 NaN/Inf")
        if np.any(np.diff(timestamps) < -1e-6):
            errors.append("timestamps 非单调递增")
        if "frame_ids" in data and fps > 0:
            expected_ts = (data["frame_ids"] - data["frame_ids"][0]).astype(np.float32) / fps
            max_err = float(np.nanmax(np.abs(timestamps - expected_ts)))
            if max_err > 1e-3:
                warnings.append(f"timestamps 与 frame_ids/fps 最大偏差 {max_err:.6f}s")

    for key in ["keypoints2d", "keypoints3d", "wrist_pos", "palm_pos", "fingertips3d", "retarget_keypoints3d", "retarget_keypoints3d_palm", "T_world_wrist", "T_world_palm"]:
        if key in data and not np.all(np.isfinite(data[key])):
            errors.append(f"{key} 存在 NaN/Inf")

    for key in ["quality_score", "handedness_score", "keypoints2d_score", "keypoints3d_score", "fingertips_score"]:
        if key in data:
            value = data[key]
            if not np.all(np.isfinite(value)):
                errors.append(f"{key} 存在 NaN/Inf")
            if np.nanmin(value) < -1e-4 or np.nanmax(value) > 1.0001:
                warnings.append(f"{key} 超出 [0,1] 范围：min={np.nanmin(value):.4f}, max={np.nanmax(value):.4f}")

    if "wrist_rot" in data:
        norm = np.linalg.norm(data["wrist_rot"], axis=1)
        if not np.all(np.isfinite(data["wrist_rot"])):
            errors.append("wrist_rot 存在 NaN/Inf")
        if np.nanmin(norm) < 1e-6:
            errors.append("wrist_rot 存在接近零的四元数")
        if np.nanmax(np.abs(norm - 1.0)) > 1e-2:
            warnings.append("wrist_rot 四元数 norm 偏离 1")
        if np.allclose(data["wrist_rot"], np.array([0, 0, 0, 1], dtype=np.float32)):
            warnings.append("wrist_rot 全部为单位四元数，当前应视为占位")

    if "palm_rot" in data:
        norm = np.linalg.norm(data["palm_rot"], axis=1)
        if not np.all(np.isfinite(data["palm_rot"])):
            errors.append("palm_rot 存在 NaN/Inf")
        if np.nanmin(norm) < 1e-6:
            errors.append("palm_rot 存在接近零的四元数")
        if np.nanmax(np.abs(norm - 1.0)) > 1e-2:
            warnings.append("palm_rot 四元数 norm 偏离 1")
        if np.allclose(data["palm_rot"], np.array([0, 0, 0, 1], dtype=np.float32)):
            warnings.append("palm_rot 全部为单位四元数，当前应视为占位")

    if "image_size" in data and "keypoints2d" in data:
        image_size = data["image_size"]
        if image_size.shape != (2,):
            warnings.append(f"image_size 当前仅校验单视角 [2]，got {image_size.shape}")
        else:
            h, w = float(image_size[0]), float(image_size[1])
            if h <= 0 or w <= 0:
                errors.append("image_size 必须为正数")
            pts = data["keypoints2d"]
            out_ratio = float(((pts[..., 0] < -1e-3) | (pts[..., 0] > w + 1e-3) | (pts[..., 1] < -1e-3) | (pts[..., 1] > h + 1e-3)).mean())
            if out_ratio > 0:
                warnings.append(f"keypoints2d 出界比例 {out_ratio:.2%}")

    if "wrist_pos" in data and "keypoints3d" in data and not np.allclose(data["wrist_pos"], data["keypoints3d"][:, 0], atol=1e-5):
        warnings.append("wrist_pos 与 keypoints3d[:,0] 不一致")
    if "fingertips3d" in data and "keypoints3d" in data and not np.allclose(data["fingertips3d"], data["keypoints3d"][:, [4, 8, 12, 16, 20]], atol=1e-5):
        warnings.append("fingertips3d 与 keypoints3d fingertip indices 不一致")

    if {"retarget_landmark_names", "retarget_keypoints3d", "retarget_keypoints3d_palm", "retarget_weights"}.issubset(set(data.files)):
        k = data["retarget_keypoints3d"].shape[1]
        if len(data["retarget_landmark_names"]) != k:
            errors.append("retarget_landmark_names 长度与 retarget_keypoints3d K 不一致")
        if data["retarget_keypoints3d_palm"].shape != data["retarget_keypoints3d"].shape:
            errors.append("retarget_keypoints3d_palm shape 与 retarget_keypoints3d 不一致")
        if data["retarget_weights"].shape != (k,):
            errors.append("retarget_weights shape 与 K 不一致")
        if np.any(data["retarget_weights"] < 0):
            errors.append("retarget_weights 不能为负")
        expected_local = data["retarget_keypoints3d"] - data["palm_pos"][:, None, :]
        if not np.allclose(data["retarget_keypoints3d_palm"], expected_local, atol=1e-5):
            warnings.append("retarget_keypoints3d_palm 与 retarget_keypoints3d - palm_pos 不一致")

    for key, pos_key in [("T_world_wrist", "wrist_pos"), ("T_world_palm", "palm_pos")]:
        if key in data:
            mats = data[key]
            if mats.shape != (t, 4, 4):
                errors.append(f"{key} shape 应为 [T,4,4]")
                continue
            if not np.allclose(mats[:, 3, :], np.array([0, 0, 0, 1], dtype=np.float32), atol=1e-5):
                errors.append(f"{key} 最后一行不是 [0,0,0,1]")
            if pos_key in data and not np.allclose(mats[:, :3, 3], data[pos_key], atol=1e-5):
                warnings.append(f"{key} 平移与 {pos_key} 不一致")

    if "units" in data and str(data["units"]) != "meter":
        warnings.append(f"units={data['units']}，不能直接当作 IsaacGym/world metric 坐标")

    for key in ["metric_3d_valid", "world_alignment_valid", "camera_calib_valid", "contact_valid", "phase_valid"]:
        if key in data and not as_bool_scalar(data[key]):
            warnings.append(f"{key}=False")

    lines.extend([
        f"- T：{t}",
        f"- fps：{fps:.3f}",
        f"- valid_ratio：{valid_ratio:.2%}",
        f"- errors：{len(errors)}",
        f"- warnings：{len(warnings)}",
        "",
        "## 字段清单",
        "",
    ])
    for key in sorted(data.files):
        lines.append(f"- `{key}`: shape={data[key].shape}, dtype={data[key].dtype}")
    lines.extend(["", "## Errors", ""])
    lines.extend([f"- {item}" for item in errors] or ["- 无"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in warnings] or ["- 无"])

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote validation report: {out_report}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
