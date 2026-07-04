#!/usr/bin/env python3
"""批量运行 HO-Tracker human_demo 的 Task2 MediaPipe baseline。"""

import argparse
import csv
import re
import subprocess
from pathlib import Path

import numpy as np


CAMERAS = ["camera_side_1", "camera_side_2", "camera_top"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run task2 baseline for all HO-Tracker human_demo videos.")
    parser.add_argument("--human_demo_dir", default="task1/data/HO-Tracker-Challenge/human_demo", help="human_demo 根目录")
    parser.add_argument("--by_view_dir", default="task2/outputs/by_view", help="按视角输出目录")
    parser.add_argument("--by_sequence_dir", default="task2/outputs/by_sequence", help="按序列汇总输出目录")
    parser.add_argument("--summary_csv", default="task2/reports/all_human_demo_metrics.csv", help="全量指标 CSV")
    parser.add_argument("--summary_md", default="task2/reports/all_human_demo_summary.md", help="全量汇总 Markdown")
    parser.add_argument("--fps", type=float, default=15.0, help="抽帧/输出 FPS")
    parser.add_argument("--views", nargs="+", default=CAMERAS, choices=CAMERAS, help="要处理的相机视角")
    parser.add_argument("--limit_sequences", type=int, default=0, help="仅处理前 N 条序列，0 表示全部")
    parser.add_argument("--skip_existing", action="store_true", help="若 by_view 结果已存在则跳过")
    parser.add_argument("--dry_run", action="store_true", help="只打印将执行的命令")
    return parser.parse_args()


def safe_sequence_id(name: str) -> str:
    text = name.replace("__left__", "_left_").replace("__", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text


def run_cmd(cmd: list[str], dry_run: bool) -> int:
    print("[CMD] " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def load_npz_summary(run_dir: Path) -> dict:
    npz_path = run_dir / "trajectories" / "hand_traj.npz"
    if not npz_path.is_file():
        return {"exists": False}
    data = np.load(npz_path, allow_pickle=True)
    valid = data["valid"].astype(bool)
    quality = data["quality_score"].astype(np.float32)
    interpolated = data["interpolated_flag"].astype(bool) if "interpolated_flag" in data else ~valid
    temporal = data["temporal_jump_score"].astype(np.float32) if "temporal_jump_score" in data else np.zeros_like(quality)
    bone = data["bone_length_error"].astype(np.float32) if "bone_length_error" in data else np.zeros_like(quality)
    flags = data["quality_flags"].astype(bool) if "quality_flags" in data else np.zeros((len(valid), 0), dtype=bool)
    mask_area = data["mask_area"].astype(np.float32) if "mask_area" in data else np.zeros_like(quality)
    return {
        "exists": True,
        "frames": int(len(valid)),
        "valid": int(valid.sum()),
        "valid_ratio": float(valid.mean()) if len(valid) else 0.0,
        "interpolated": int(interpolated.sum()),
        "quality_mean": float(np.nanmean(quality)) if len(quality) else 0.0,
        "quality_min": float(np.nanmin(quality)) if len(quality) else 0.0,
        "temporal_max": float(np.nanmax(temporal)) if len(temporal) else 0.0,
        "bone_mean": float(np.nanmean(bone)) if len(bone) else 0.0,
        "risk_frames": int(flags.any(axis=1).sum()) if flags.size else 0,
        "mask_nonempty": int((mask_area > 0).sum()) if len(mask_area) else 0,
        "traj_npz": str(npz_path),
        "review_video": str(run_dir / "videos" / "task2_scoring_review.mp4"),
    }


def write_summary(rows: list[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id", "view_id", "run_id", "status", "frames", "valid", "valid_ratio",
        "interpolated", "quality_mean", "quality_min", "temporal_max", "bone_mean",
        "risk_frames", "mask_nonempty", "traj_npz", "review_video",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    lines = [
        "# 全量 human_demo Task2 Baseline 汇总",
        "",
        "本报告基于 MediaPipe pseudo-label baseline。所有 3D 仍为 non-metric MediaPipe world landmarks，不是 GT。",
        "",
        f"- 总视角数：{len(rows)}",
        f"- 成功视角数：{len(ok_rows)}",
        f"- CSV：`{csv_path}`",
        "",
        "## 每视角指标",
        "",
        "| sequence | view | status | frames | valid_ratio | quality_mean | risk_frames | mask_nonempty |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('sequence_id', '')}` | `{row.get('view_id', '')}` | {row.get('status', '')} | "
            f"{row.get('frames', '')} | {float(row.get('valid_ratio', 0.0)):.2%} | "
            f"{float(row.get('quality_mean', 0.0)):.4f} | {row.get('risk_frames', '')} | {row.get('mask_nonempty', '')} |"
        )

    lines.extend(["", "## 每序列推荐视角", "", "| sequence | best_run | valid_ratio | quality_mean |", "|---|---|---:|---:|"])
    for seq in sorted({r["sequence_id"] for r in ok_rows}):
        seq_rows = [r for r in ok_rows if r["sequence_id"] == seq]
        best = max(seq_rows, key=lambda r: (float(r.get("valid_ratio", 0.0)), float(r.get("quality_mean", 0.0))))
        lines.append(f"| `{seq}` | `{best['run_id']}` | {float(best['valid_ratio']):.2%} | {float(best['quality_mean']):.4f} |")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    human_demo_dir = Path(args.human_demo_dir)
    if not human_demo_dir.is_dir():
        raise FileNotFoundError(f"human_demo 目录不存在: {human_demo_dir}")

    seq_dirs = sorted(p for p in human_demo_dir.iterdir() if p.is_dir())
    if args.limit_sequences > 0:
        seq_dirs = seq_dirs[: args.limit_sequences]

    rows: list[dict] = []
    for seq_dir in seq_dirs:
        sequence_id = safe_sequence_id(seq_dir.name)
        seq_out = Path(args.by_sequence_dir) / sequence_id
        for view_id in args.views:
            video = seq_dir / "video" / f"{view_id}.mkv"
            run_id = f"{sequence_id}_{view_id}"
            run_dir = Path(args.by_view_dir) / run_id
            row = {"sequence_id": sequence_id, "view_id": view_id, "run_id": run_id, "status": "pending"}
            if not video.is_file():
                row["status"] = "missing_video"
                rows.append(row)
                continue
            if args.skip_existing and (run_dir / "trajectories" / "hand_traj.npz").is_file():
                row["status"] = "ok"
                row.update(load_npz_summary(run_dir))
                rows.append(row)
                continue
            code = run_cmd([
                "bash", "task2/scripts/20_run_single_view_baseline.sh",
                "--video", str(video),
                "--sequence_id", sequence_id,
                "--view_id", view_id,
                "--fps", str(args.fps),
            ], args.dry_run)
            row["status"] = "dry_run" if args.dry_run and code == 0 else "ok" if code == 0 else f"failed_{code}"
            if code == 0 and not args.dry_run:
                row.update(load_npz_summary(run_dir))
            rows.append(row)

        if not args.dry_run:
            run_cmd([
                "python", "task2/scripts/09_export_camera_calib.py",
                "--sequence_dir", str(seq_dir),
                "--out_json", str(seq_out / "camera_calib.json"),
                "--out_npz", str(seq_out / "camera_calib.npz"),
            ], False)
            run_cmd([
                "python", "task2/scripts/12_compare_views.py",
                "--by_view_dir", str(args.by_view_dir),
                "--sequence_prefix", sequence_id,
                "--out_report", str(seq_out / "view_comparison.md"),
            ], False)

    write_summary(rows, Path(args.summary_csv), Path(args.summary_md))
    print(f"[OK] wrote summary: {args.summary_md}")


if __name__ == "__main__":
    main()
