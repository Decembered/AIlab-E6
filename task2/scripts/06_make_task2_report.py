#!/usr/bin/env python3
"""自动生成任务2 baseline 报告。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "task2" / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.hand_schema import FINGERTIP_INDICES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate task2 baseline report.")
    parser.add_argument("--landmarks_json", default="task2/outputs/trajectories/mediapipe_landmarks.json", help="MediaPipe JSON 路径")
    parser.add_argument("--traj_npz", default="task2/outputs/trajectories/hand_traj.npz", help="统一 hand_traj.npz 路径")
    parser.add_argument("--out_report", default="task2/reports/task2_baseline_report.md", help="输出报告路径")
    parser.add_argument("--overlay_video", default="task2/outputs/videos/mediapipe_overlay.mp4", help="overlay 视频路径")
    parser.add_argument("--mask_overlay_video", default="task2/outputs/overlays/hand_mask_overlay.mp4", help="hand mask overlay 视频路径")
    parser.add_argument("--masks_dir", default="task2/outputs/masks", help="hand mask 输出目录")
    parser.add_argument("--skeleton_video", default="task2/outputs/videos/hand_3d_skeleton.mp4", help="3D skeleton 视频路径")
    parser.add_argument("--validation_report", default="task2/reports/hand_traj_validation.md", help="hand_traj 校验报告路径")
    parser.add_argument("--frame_quality_report", default=None, help="可选帧级质量审计报告")
    parser.add_argument("--frame_metrics_csv", default=None, help="可选帧级指标 CSV")
    parser.add_argument("--scoring_review_video", default=None, help="可选评分审查视频")
    parser.add_argument("--temporal_refiner_report", default=None, help="可选 temporal refiner 应用报告")
    return parser.parse_args()


def load_json_stats(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    valid = [bool(frame.get("hands")) for frame in frames]
    missing = [idx for idx, ok in enumerate(valid) if not ok]
    return {
        "exists": True,
        "frames_dir": data.get("frames_dir", ""),
        "num_frames": len(frames),
        "valid_frames": int(sum(valid)),
        "valid_ratio": float(sum(valid) / len(frames)) if frames else 0.0,
        "missing_frames": missing,
    }


def load_traj_stats(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False}
    data = np.load(path, allow_pickle=True)
    keypoints3d = data["hand_landmarks_3d"]
    valid = data["valid"] if "valid" in data else np.isfinite(keypoints3d[:, 0, 0])
    wrist = data["wrist_pos"]
    wrist_jump = np.linalg.norm(np.diff(wrist, axis=0), axis=1) if wrist.shape[0] > 1 else np.array([], dtype=np.float32)
    fingertip = keypoints3d[:, FINGERTIP_INDICES, :]
    fingertip_jump = np.linalg.norm(np.diff(fingertip, axis=0), axis=-1) if keypoints3d.shape[0] > 1 else np.array([], dtype=np.float32)
    return {
        "exists": True,
        "fields": list(data.files),
        "num_frames": int(keypoints3d.shape[0]),
        "valid_frames": int(valid.sum()),
        "valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "missing_frames": np.where(~valid)[0].tolist(),
        "wrist_jump_mean": float(np.nanmean(wrist_jump)) if wrist_jump.size else 0.0,
        "wrist_jump_max": float(np.nanmax(wrist_jump)) if wrist_jump.size else 0.0,
        "fingertip_jump_mean": float(np.nanmean(fingertip_jump)) if fingertip_jump.size else 0.0,
        "fingertip_jump_max": float(np.nanmax(fingertip_jump)) if fingertip_jump.size else 0.0,
    }


def load_mask_stats(path: Path) -> dict:
    if not path.is_dir():
        return {"exists": False}
    import cv2

    masks = sorted(path.glob("*.png"))
    areas = []
    for mask_path in masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            areas.append(int((mask > 0).sum()))
    areas = np.asarray(areas, dtype=np.float32)
    return {
        "exists": True,
        "count": len(masks),
        "non_empty": int((areas > 0).sum()) if areas.size else 0,
        "empty": int((areas == 0).sum()) if areas.size else 0,
        "area_mean": float(areas.mean()) if areas.size else 0.0,
        "area_max": float(areas.max()) if areas.size else 0.0,
    }


def main() -> None:
    args = parse_args()
    landmarks_json = Path(args.landmarks_json)
    traj_npz = Path(args.traj_npz)
    out_report = Path(args.out_report)
    overlay_video = Path(args.overlay_video)
    mask_overlay_video = Path(args.mask_overlay_video)
    masks_dir = Path(args.masks_dir)
    skeleton_video = Path(args.skeleton_video)
    validation_report = Path(args.validation_report)
    frame_quality_report = Path(args.frame_quality_report) if args.frame_quality_report else None
    frame_metrics_csv = Path(args.frame_metrics_csv) if args.frame_metrics_csv else None
    scoring_review_video = Path(args.scoring_review_video) if args.scoring_review_video else None
    temporal_refiner_report = Path(args.temporal_refiner_report) if args.temporal_refiner_report else None

    json_stats = load_json_stats(landmarks_json)
    traj_stats = load_traj_stats(traj_npz)
    mask_stats = load_mask_stats(masks_dir)

    report = [
        "# 任务2 Baseline 报告",
        "",
        "## 输入与输出",
        "",
        f"- MediaPipe JSON：`{landmarks_json}`，存在：{json_stats['exists']}",
        f"- hand_traj.npz：`{traj_npz}`，存在：{traj_stats['exists']}",
        f"- 关键点 overlay 视频：`{overlay_video}`，存在：{overlay_video.is_file()}",
        f"- hand mask overlay 视频：`{mask_overlay_video}`，存在：{mask_overlay_video.is_file()}",
        f"- hand mask 目录：`{masks_dir}`，存在：{masks_dir.is_dir()}",
        f"- 3D skeleton 视频：`{skeleton_video}`，存在：{skeleton_video.is_file()}",
        f"- hand_traj 校验报告：`{validation_report}`，存在：{validation_report.is_file()}",
        f"- 帧级质量审计：`{frame_quality_report}`，存在：{frame_quality_report.is_file() if frame_quality_report else False}",
        f"- 帧级指标 CSV：`{frame_metrics_csv}`，存在：{frame_metrics_csv.is_file() if frame_metrics_csv else False}",
        f"- 评分审查视频：`{scoring_review_video}`，存在：{scoring_review_video.is_file() if scoring_review_video else False}",
        f"- temporal refiner 应用报告：`{temporal_refiner_report}`，存在：{temporal_refiner_report.is_file() if temporal_refiner_report else False}",
        "",
        "## 检测统计",
        "",
    ]

    if json_stats["exists"]:
        report.extend(
            [
                f"- 输入帧目录：`{json_stats['frames_dir']}`",
                f"- 输入帧数：{json_stats['num_frames']}",
                f"- 检测成功帧数：{json_stats['valid_frames']}",
                f"- 检测成功帧比例：{json_stats['valid_ratio']:.2%}",
                f"- 缺失帧列表：{json_stats['missing_frames']}",
            ]
        )
    else:
        report.append("- 未找到 MediaPipe JSON，无法统计检测成功率。")

    report.extend(["", "## 轨迹跳变统计", ""])
    if traj_stats["exists"]:
        report.extend(
            [
                f"- 轨迹帧数：{traj_stats['num_frames']}",
                f"- 有效帧数：{traj_stats['valid_frames']}",
                f"- 有效帧比例：{traj_stats['valid_ratio']:.2%}",
                f"- 平滑前缺失帧列表：{traj_stats['missing_frames']}",
                f"- wrist 平均相邻位移：{traj_stats['wrist_jump_mean']:.6f}",
                f"- wrist 最大相邻位移：{traj_stats['wrist_jump_max']:.6f}",
                f"- fingertips 平均相邻位移：{traj_stats['fingertip_jump_mean']:.6f}",
                f"- fingertips 最大相邻位移：{traj_stats['fingertip_jump_max']:.6f}",
            ]
        )
    else:
        report.append("- 未找到 hand_traj.npz，无法统计轨迹跳变。")

    report.extend(["", "## Mask 统计", ""])
    if mask_stats["exists"]:
        report.extend(
            [
                f"- mask 文件数：{mask_stats['count']}",
                f"- 非空 mask：{mask_stats['non_empty']}",
                f"- 空 mask：{mask_stats['empty']}",
                f"- 平均 mask 面积：{mask_stats['area_mean']:.2f} px",
                f"- 最大 mask 面积：{mask_stats['area_max']:.2f} px",
            ]
        )
    else:
        report.append("- 未找到 mask 目录，无法统计 mask。")

    report.extend(["", "## hand_traj.npz 字段", ""])
    if traj_stats["exists"]:
        report.extend([f"- `{field}`" for field in sorted(traj_stats["fields"])])
    else:
        report.append("- 未找到 hand_traj.npz。")

    report.extend(
        [
            "",
            "## 可视化文件路径",
            "",
            f"- MediaPipe overlay：`{overlay_video}`",
            f"- hand mask overlay：`{mask_overlay_video}`，当前为 coarse visible hand mask baseline，后续可替换为 SAM2。",
            f"- hand mask 目录：`{masks_dir}`",
            f"- 3D skeleton 可视化：`{skeleton_video}`，当前为 MediaPipe world landmarks skeleton，非 MANO mesh。",
            f"- hand_traj 校验报告：`{validation_report}`",
            f"- 帧级质量审计：`{frame_quality_report}`" if frame_quality_report else "- 帧级质量审计：未传入",
            f"- 帧级指标 CSV：`{frame_metrics_csv}`" if frame_metrics_csv else "- 帧级指标 CSV：未传入",
            f"- 评分审查视频：`{scoring_review_video}`" if scoring_review_video else "- 评分审查视频：未传入",
            f"- temporal refiner 应用报告：`{temporal_refiner_report}`" if temporal_refiner_report else "- temporal refiner 应用报告：未传入",
            "",
            "## 当前局限性",
            "",
            "- MediaPipe world landmarks 不是严格 metric 3D，不能直接等同真实手部尺度。",
            "- 遮挡、手物接触和运动模糊时可能漏检。",
            "- 当前 baseline 没有真实 MANO mesh，3D 重建得分需要后续接入 HaMeR/MANO 提升。",
            "- 当前 mask 为 bbox/关键点凸包生成的 coarse visible hand mask baseline，不是高质量分割。",
            "- `T_world_wrist` / `T_world_palm` 当前不是 IsaacGym/world metric transform；`world_alignment_valid=False`。",
            "- `contact_likelihood`、`active_fingers`、`phase` 当前是占位字段，不能作为真实手物交互标签。",
            "- temporal refiner 使用 MediaPipe pseudo-label 训练，只作为辅助去噪/质量建模对照，不是 GT 或 metric 3D。",
            "",
            "## 下一步改进建议",
            "",
            "- 接入 SAM2，用 MediaPipe bbox 或关键点作为 prompt 生成 hand mask。",
            "- 接入 HaMeR 和 MANO，导出 mesh、MANO 参数和更稳定的 3D joints。",
            "- 做左右手 ID 维护，避免多手场景下 handedness 切换。",
            "- 加入骨长约束、OneEuro filter 或鲁棒优化，降低跳变。",
            "- 若有相机参数，补充坐标系说明和重投影误差统计。",
        ]
    )

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[OK] wrote report: {out_report}")


if __name__ == "__main__":
    main()
