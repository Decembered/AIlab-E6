#!/usr/bin/env python3.8
"""Preflight checks for task 2-4 hand-object integration.

This script is intentionally IsaacGym-free. It validates the file-level contract
between Member B hand trajectories and Member C object assets before launching a
slow simulator rollout.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FILES = ("scan.urdf", "scan.obj", "left_obj.pkl", "left_hand.pkl")


def load_pickle(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open("rb") as f:
            return pickle.load(f), None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, f"failed_to_load_pickle: {exc}"


def as_array(data: dict[str, Any], key: str) -> np.ndarray | None:
    value = data.get(key)
    if value is None:
        return None
    try:
        return np.asarray(value)
    except Exception:
        return None


def is_placeholder_hand(hand_data: dict[str, Any], hand_pose: np.ndarray | None) -> bool:
    note = str(hand_data.get("note", "")).lower()
    if "placeholder" in note:
        return True
    if hand_pose is not None and hand_pose.size > 0 and np.allclose(hand_pose, 0.0):
        return True
    return False


def check_sequence(seq_dir: Path, allow_placeholder_hand: bool) -> dict[str, Any]:
    left_urdf = seq_dir / "left_urdf"
    result: dict[str, Any] = {
        "sequence": seq_dir.name,
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "files": {},
    }

    if not left_urdf.is_dir():
        result["status"] = "FAIL"
        result["errors"].append("missing_left_urdf_dir")
        return result

    for name in REQUIRED_FILES:
        exists = (left_urdf / name).exists()
        result["files"][name] = exists
        if not exists:
            result["errors"].append(f"missing_{name}")

    if result["errors"]:
        result["status"] = "FAIL"
        return result

    obj_data_raw, obj_error = load_pickle(left_urdf / "left_obj.pkl")
    hand_data_raw, hand_error = load_pickle(left_urdf / "left_hand.pkl")
    if obj_error:
        result["errors"].append(obj_error)
    if hand_error:
        result["errors"].append(hand_error)
    if result["errors"]:
        result["status"] = "FAIL"
        return result
    if not isinstance(obj_data_raw, dict):
        result["errors"].append("left_obj.pkl_not_dict")
    if not isinstance(hand_data_raw, dict):
        result["errors"].append("left_hand.pkl_not_dict")
    if result["errors"]:
        result["status"] = "FAIL"
        return result

    obj_data: dict[str, Any] = obj_data_raw
    hand_data: dict[str, Any] = hand_data_raw
    obj_pose = as_array(obj_data, "obj_pose")
    obj_timestamps = as_array(obj_data, "timestamps")
    hand_pose = as_array(hand_data, "hand_pose")
    hand_timestamps = as_array(hand_data, "timestamps")

    if obj_pose is None or obj_pose.ndim != 2 or obj_pose.shape[1] != 7:
        result["errors"].append("obj_pose_must_be_Tx7_xyz_quatxyzw")
    if hand_pose is None or hand_pose.ndim != 2:
        result["errors"].append("hand_pose_must_be_TxD")

    if obj_pose is not None:
        result["obj_pose_shape"] = list(obj_pose.shape)
        if obj_pose.shape[0] < 2:
            result["errors"].append("obj_pose_too_short")
        if not np.isfinite(obj_pose).all():
            result["errors"].append("obj_pose_has_nan_or_inf")
        quat_norm = np.linalg.norm(obj_pose[:, 3:7], axis=1) if obj_pose.ndim == 2 and obj_pose.shape[1] == 7 else np.array([])
        if quat_norm.size and np.nanmax(np.abs(quat_norm - 1.0)) > 0.05:
            result["warnings"].append("object_quaternion_norm_deviation_gt_0.05")
        if obj_timestamps is not None:
            result["obj_timestamps_shape"] = list(obj_timestamps.shape)
            if obj_timestamps.shape[0] != obj_pose.shape[0]:
                result["errors"].append("obj_timestamps_length_mismatch")

    if hand_pose is not None:
        result["hand_pose_shape"] = list(hand_pose.shape)
        if hand_pose.shape[0] < 2:
            result["errors"].append("hand_pose_too_short")
        if not np.isfinite(hand_pose).all():
            result["errors"].append("hand_pose_has_nan_or_inf")
        if hand_timestamps is not None:
            result["hand_timestamps_shape"] = list(hand_timestamps.shape)
            if hand_timestamps.shape[0] != hand_pose.shape[0]:
                result["errors"].append("hand_timestamps_length_mismatch")
        if is_placeholder_hand(hand_data, hand_pose):
            message = "left_hand.pkl_is_placeholder_or_all_zero"
            if allow_placeholder_hand:
                result["warnings"].append(message)
            else:
                result["errors"].append(message)

    if obj_pose is not None and hand_pose is not None and obj_pose.ndim == 2 and hand_pose.ndim == 2:
        if obj_pose.shape[0] != hand_pose.shape[0]:
            result["warnings"].append("object_and_hand_frame_counts_differ_resampling_required")
        if obj_timestamps is not None and hand_timestamps is not None:
            if obj_timestamps.shape == hand_timestamps.shape:
                max_dt = float(np.max(np.abs(obj_timestamps.astype(float) - hand_timestamps.astype(float))))
                result["max_timestamp_delta_s"] = max_dt
                if max_dt > 1e-3:
                    result["warnings"].append("object_and_hand_timestamps_differ")

    if result["errors"]:
        result["status"] = "FAIL"
    elif result["warnings"]:
        result["status"] = "WARN"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight task 2-4 hand-object integration inputs")
    parser.add_argument("--root", default="data/pipeline_assets", help="pipeline assets root")
    parser.add_argument("--seq", default=None, help="check one sequence only")
    parser.add_argument("--allow-placeholder-hand", action="store_true", help="allow zero placeholder hand_pose for object-only smoke tests")
    parser.add_argument("--json-out", default=None, help="optional path to write JSON summary")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = Path.cwd() / root
    if not root.is_dir():
        print(f"FAIL: root not found: {root}")
        return 2

    if args.seq:
        seq_dirs = [root / args.seq]
    else:
        seq_dirs = sorted(p for p in root.iterdir() if (p / "left_urdf").is_dir())
    results = [check_sequence(seq_dir, args.allow_placeholder_hand) for seq_dir in seq_dirs]
    fail_count = sum(1 for item in results if item["status"] == "FAIL")
    warn_count = sum(1 for item in results if item["status"] == "WARN")
    pass_count = sum(1 for item in results if item["status"] == "PASS")

    summary = {
        "root": str(root),
        "allow_placeholder_hand": args.allow_placeholder_hand,
        "total": len(results),
        "pass": pass_count,
        "warn": warn_count,
        "fail": fail_count,
        "results": results,
    }

    for item in results:
        print(f"{item['status']:4} {item['sequence']}")
        for error in item["errors"]:
            print(f"  ERROR: {error}")
        for warning in item["warnings"]:
            print(f"  WARN: {warning}")
        if "obj_pose_shape" in item or "hand_pose_shape" in item:
            print(f"  obj_pose={item.get('obj_pose_shape')} hand_pose={item.get('hand_pose_shape')}")

    print(f"\nSUMMARY: pass={pass_count} warn={warn_count} fail={fail_count} total={len(results)}")
    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")

    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
