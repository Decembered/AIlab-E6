#!/usr/bin/env python3
"""导出 HO-Tracker human_demo 相机标定为任务2 JSON/NPZ。"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


CAMERAS = ["camera_side_1", "camera_side_2", "camera_top"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HO-Tracker camera calibration for task2.")
    parser.add_argument("--sequence_dir", required=True, help="human_demo 下的序列目录")
    parser.add_argument("--out_json", required=True, help="输出标定 JSON")
    parser.add_argument("--out_npz", default=None, help="可选输出标定 NPZ")
    return parser.parse_args()


def load_pkl(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"标定文件不存在: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"序列目录不存在: {sequence_dir}")

    cameras = []
    k_list = []
    t_list = []
    for camera in CAMERAS:
        calib_dir = sequence_dir / "camera_calib" / camera
        intr = np.asarray(load_pkl(calib_dir / "cam_intr.pkl"), dtype=np.float32)
        extr = np.asarray(load_pkl(calib_dir / "cam_extr.pkl"), dtype=np.float64)
        if intr.shape != (3, 3):
            raise ValueError(f"内参 shape 错误: {calib_dir / 'cam_intr.pkl'} got {intr.shape}")
        if extr.shape != (4, 4):
            raise ValueError(f"外参 shape 错误: {calib_dir / 'cam_extr.pkl'} got {extr.shape}")
        cameras.append({"camera_id": camera, "K": intr.tolist(), "extrinsic": extr.tolist()})
        k_list.append(intr)
        t_list.append(extr)

    result = {
        "schema_version": "task2_camera_calib_v1",
        "sequence_dir": str(sequence_dir),
        "camera_ids": CAMERAS,
        "note": "cam_extr.pkl direction is not assumed here; verify projection direction before metric reprojection.",
        "cameras": cameras,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote camera json: {out_json}")

    if args.out_npz:
        out_npz = Path(args.out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            schema_version=np.array("task2_camera_calib_v1"),
            camera_ids=np.array(CAMERAS, dtype="U32"),
            K=np.stack(k_list, axis=0),
            T_raw_extrinsic=np.stack(t_list, axis=0),
            note=np.array("cam_extr.pkl direction must be verified before use."),
        )
        print(f"[OK] wrote camera npz: {out_npz}")


if __name__ == "__main__":
    main()
