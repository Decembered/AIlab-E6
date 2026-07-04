#!/usr/bin/env python3
"""汇总同一序列多个视角的 task2 baseline 质量。"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare task2 by-view baseline outputs.")
    parser.add_argument("--by_view_dir", default="task2/outputs/by_view", help="按视角归档目录")
    parser.add_argument("--sequence_prefix", required=True, help="视角 run_id 前缀，例如 weigh_drink_yykx_left_0052_53")
    parser.add_argument("--out_report", default="task2/reports/view_comparison.md", help="输出报告")
    return parser.parse_args()


def summarize(run_dir: Path) -> dict:
    json_path = run_dir / "trajectories" / "mediapipe_landmarks.json"
    npz_path = run_dir / "trajectories" / "hand_traj.npz"
    if not json_path.is_file() or not npz_path.is_file():
        return {"run_id": run_dir.name, "exists": False}
    data_json = json.loads(json_path.read_text(encoding="utf-8"))
    frames = data_json.get("frames", [])
    valid_json = np.array([bool(frame.get("hands")) for frame in frames], dtype=bool)
    data_npz = np.load(npz_path, allow_pickle=True)
    valid = data_npz["valid"] if "valid" in data_npz else valid_json
    quality = data_npz["quality_score"] if "quality_score" in data_npz else np.zeros_like(valid, dtype=np.float32)
    missing = np.where(~valid)[0].tolist()
    return {
        "run_id": run_dir.name,
        "exists": True,
        "num_frames": int(len(frames)),
        "valid_frames": int(valid.sum()),
        "valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "missing_frames": missing,
        "quality_mean": float(np.nanmean(quality)) if quality.size else 0.0,
        "traj_npz": str(npz_path),
        "overlay": str(run_dir / "videos" / "mediapipe_overlay.mp4"),
        "mask_overlay": str(run_dir / "overlays" / "hand_mask_overlay.mp4"),
    }


def main() -> None:
    args = parse_args()
    by_view_dir = Path(args.by_view_dir)
    if not by_view_dir.is_dir():
        raise FileNotFoundError(f"按视角归档目录不存在: {by_view_dir}")
    runs = sorted(p for p in by_view_dir.iterdir() if p.is_dir() and p.name.startswith(args.sequence_prefix))
    if not runs:
        raise FileNotFoundError(f"没有找到匹配视角结果: {args.sequence_prefix}")

    summaries = [summarize(run) for run in runs]
    best = max((s for s in summaries if s["exists"]), key=lambda item: item["valid_ratio"], default=None)
    lines = ["# 多视角 baseline 对比", "", f"- 序列前缀：`{args.sequence_prefix}`", ""]
    if best:
        lines.append(f"- 推荐优先视角：`{best['run_id']}`，检测成功率 {best['valid_ratio']:.2%}")
        lines.append("")
    lines.extend(["## 结果", "", "| run_id | frames | valid | valid_ratio | quality_mean |", "|---|---:|---:|---:|---:|"])
    for item in summaries:
        if not item["exists"]:
            lines.append(f"| {item['run_id']} | missing | missing | missing | missing |")
            continue
        lines.append(f"| {item['run_id']} | {item['num_frames']} | {item['valid_frames']} | {item['valid_ratio']:.2%} | {item['quality_mean']:.4f} |")
    lines.extend(["", "## 输出路径", ""])
    for item in summaries:
        if not item["exists"]:
            continue
        lines.extend([
            f"### {item['run_id']}",
            f"- hand_traj：`{item['traj_npz']}`",
            f"- keypoint overlay：`{item['overlay']}`",
            f"- mask overlay：`{item['mask_overlay']}`",
            f"- 缺失帧：{item['missing_frames']}",
            "",
        ])
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote view comparison: {out_report}")


if __name__ == "__main__":
    main()
