#!/usr/bin/env python3
"""生成任务2帧级质量审计报告，并可把审计指标追加到 hand_traj.npz。"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FLAG_NAMES = [
    "low_confidence",
    "interpolated",
    "detector_lost",
    "temporal_jump",
    "bone_length_outlier",
    "wrist_step_outlier",
    "fingertip_step_outlier",
    "large_bbox_change",
    "small_or_empty_mask",
    "mask_bbox_mismatch",
    "out_of_frame",
    "handedness_switch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frame-level quality for task2 hand trajectory.")
    parser.add_argument("--traj_npz", required=True, help="输入 hand_traj.npz")
    parser.add_argument("--landmarks_json", required=True, help="MediaPipe JSON")
    parser.add_argument("--masks_dir", required=True, help="mask PNG 目录")
    parser.add_argument("--overlay_frames_dir", required=True, help="MediaPipe overlay 帧目录")
    parser.add_argument("--skeleton_frames_dir", required=True, help="3D skeleton 帧目录")
    parser.add_argument("--out_dir", required=True, help="审计输出目录")
    parser.add_argument("--out_npz", default=None, help="可选：写入追加指标后的 hand_traj.npz")
    parser.add_argument("--quality_warn", type=float, default=0.960)
    parser.add_argument("--quality_critical", type=float, default=0.955)
    parser.add_argument("--temporal_warn", type=float, default=0.800)
    parser.add_argument("--temporal_critical", type=float, default=1.200)
    parser.add_argument("--bone_warn", type=float, default=0.00373)
    parser.add_argument("--bone_critical", type=float, default=0.00400)
    parser.add_argument("--wrist_warn", type=float, default=0.00361)
    parser.add_argument("--wrist_critical", type=float, default=0.00426)
    parser.add_argument("--fingertip_warn", type=float, default=0.00577)
    parser.add_argument("--fingertip_critical", type=float, default=0.00675)
    parser.add_argument("--mask_bbox_ratio_min", type=float, default=0.50)
    parser.add_argument("--mask_bbox_ratio_max", type=float, default=1.25)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON 不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def primary_hand(frame: dict) -> dict | None:
    hands = frame.get("hands", [])
    if not hands:
        return None
    return max(hands, key=lambda item: item.get("handedness_score", 0.0))


def bbox_area_xyxy(bbox: np.ndarray) -> float:
    if not np.all(np.isfinite(bbox)):
        return 0.0
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def mask_stats(mask_path: Path, bbox: np.ndarray) -> tuple[float, float, float, float]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0.0, 0.0, 0.0, 0.0
    binary = mask > 0
    area = float(binary.sum())
    if area <= 0 or not np.all(np.isfinite(bbox)):
        return area, 0.0, 0.0, 0.0
    h, w = binary.shape
    x1 = int(np.clip(np.floor(bbox[0]), 0, w - 1))
    y1 = int(np.clip(np.floor(bbox[1]), 0, h - 1))
    x2 = int(np.clip(np.ceil(bbox[2]), 0, w - 1))
    y2 = int(np.clip(np.ceil(bbox[3]), 0, h - 1))
    inside = binary[y1:y2 + 1, x1:x2 + 1].sum()
    inside_frac = float(inside / max(area, 1.0))

    ys, xs = np.where(binary)
    mx1, mx2 = float(xs.min()), float(xs.max())
    my1, my2 = float(ys.min()), float(ys.max())
    inter_x1 = max(mx1, float(bbox[0]))
    inter_y1 = max(my1, float(bbox[1]))
    inter_x2 = min(mx2, float(bbox[2]))
    inter_y2 = min(my2, float(bbox[3]))
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    mask_bbox_area = max(0.0, mx2 - mx1) * max(0.0, my2 - my1)
    bbox_area = bbox_area_xyxy(bbox)
    union = mask_bbox_area + bbox_area - inter
    iou = float(inter / union) if union > 0 else 0.0
    fill = float(area / max(mask_bbox_area, 1.0))
    return area, inside_frac, iou, fill


def summarize(values: np.ndarray) -> dict:
    return {
        "min": float(np.nanmin(values)),
        "p05": float(np.nanpercentile(values, 5)),
        "median": float(np.nanmedian(values)),
        "p95": float(np.nanpercentile(values, 95)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
    }


def top_indices(values: np.ndarray, k: int = 10, largest: bool = True) -> list[int]:
    order = np.argsort(values)
    if largest:
        order = order[::-1]
    return [int(i) for i in order[: min(k, len(order))]]


def save_curve(out_path: Path, x: np.ndarray, series: list[tuple[str, np.ndarray]], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for label, values in series:
        ax.plot(x, values, label=label, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("frame_id")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_panel(frame_ids: list[int], overlay_dir: Path, skeleton_dir: Path, out_path: Path, rows: list[dict]) -> None:
    tiles = []
    for frame_id in frame_ids:
        overlay_path = overlay_dir / f"{frame_id:06d}.jpg"
        skeleton_path = skeleton_dir / f"{frame_id:06d}.png"
        overlay = cv2.imread(str(overlay_path))
        skeleton = cv2.imread(str(skeleton_path))
        if overlay is None or skeleton is None:
            continue
        overlay = cv2.resize(overlay, (360, 202))
        skeleton = cv2.resize(skeleton, (202, 202))
        canvas = np.zeros((260, 562, 3), dtype=np.uint8) + 255
        canvas[:202, :360] = overlay
        canvas[:202, 360:] = skeleton
        row = rows[frame_id]
        text = f"f={frame_id} t={row['timestamp_sec']:.2f}s q={row['quality_score']:.3f} jump={row['temporal_jump_score']:.3f}"
        cv2.putText(canvas, text, (8, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, row["risk_reasons"][:80], (8, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1, cv2.LINE_AA)
        tiles.append(canvas)
    if not tiles:
        return
    cols = 3
    tile_h, tile_w = tiles[0].shape[:2]
    rows_count = int(np.ceil(len(tiles) / cols))
    panel = np.zeros((rows_count * tile_h, cols * tile_w, 3), dtype=np.uint8) + 240
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        panel[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
    cv2.imwrite(str(out_path), panel)


def copy_npz_with_metrics(src: Path, dst: Path, metrics: dict) -> None:
    data = np.load(src, allow_pickle=True)
    payload = {key: data[key] for key in data.files}
    payload.update(metrics)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **payload)


def main() -> None:
    args = parse_args()
    traj_npz = Path(args.traj_npz)
    landmarks_json = Path(args.landmarks_json)
    masks_dir = Path(args.masks_dir)
    overlay_dir = Path(args.overlay_frames_dir)
    skeleton_dir = Path(args.skeleton_frames_dir)
    out_dir = Path(args.out_dir)
    if not traj_npz.is_file():
        raise FileNotFoundError(f"NPZ 不存在: {traj_npz}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"mask 目录不存在: {masks_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(traj_npz, allow_pickle=True)
    json_data = load_json(landmarks_json)
    frames = json_data.get("frames", [])
    frame_ids = data["frame_ids"].astype(np.int64)
    timestamps = data["timestamps"].astype(np.float32)
    t = len(frame_ids)
    if len(frames) != t:
        raise ValueError(f"JSON frames 与 NPZ T 不一致: {len(frames)} vs {t}")

    quality = data["quality_score"].astype(np.float32)
    handedness_score = data["handedness_score"].astype(np.float32)
    temporal = data["temporal_jump_score"].astype(np.float32)
    bone = data["bone_length_error"].astype(np.float32)
    valid = data["valid"].astype(bool)
    interpolated = data["interpolated_flag"].astype(bool)
    wrist_pos = data["wrist_pos"].astype(np.float32)
    fingertips = data["fingertips3d"].astype(np.float32)
    keypoints2d = data["keypoints2d"].astype(np.float32)
    failure_reason = data["failure_reason"].astype(str) if "failure_reason" in data else np.array(["ok"] * t)
    handedness = data["handedness"].astype(str)
    image_size = data["image_size"].astype(np.float32)
    height, width = float(image_size[0]), float(image_size[1])

    wrist_step = np.zeros((t,), dtype=np.float32)
    wrist_step[1:] = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)
    fingertip_step = np.zeros((t, 5), dtype=np.float32)
    fingertip_step[1:] = np.linalg.norm(np.diff(fingertips, axis=0), axis=2)
    fingertip_step_max = fingertip_step.max(axis=1)
    fingertip_step_mean = fingertip_step.mean(axis=1)

    hand_bbox2d = np.full((t, 4), np.nan, dtype=np.float32)
    bbox_area = np.zeros((t,), dtype=np.float32)
    mask_area = np.zeros((t,), dtype=np.float32)
    mask_inside_bbox = np.zeros((t,), dtype=np.float32)
    mask_bbox_iou = np.zeros((t,), dtype=np.float32)
    mask_bbox_fill = np.zeros((t,), dtype=np.float32)
    json_num_hands = np.zeros((t,), dtype=np.int32)
    out_of_frame = np.zeros((t,), dtype=bool)

    for idx, frame in enumerate(frames):
        hands = frame.get("hands", [])
        json_num_hands[idx] = len(hands)
        hand = primary_hand(frame)
        if hand is not None:
            hand_bbox2d[idx] = np.asarray(hand.get("bbox_xyxy", [np.nan] * 4), dtype=np.float32)
            bbox_area[idx] = bbox_area_xyxy(hand_bbox2d[idx])
        mask_path = masks_dir / f"{int(frame_ids[idx]):06d}.png"
        area, inside, iou, fill = mask_stats(mask_path, hand_bbox2d[idx])
        mask_area[idx] = area
        mask_inside_bbox[idx] = inside
        mask_bbox_iou[idx] = iou
        mask_bbox_fill[idx] = fill
        pts = keypoints2d[idx]
        out_of_frame[idx] = bool(np.any((pts[:, 0] < 0) | (pts[:, 0] > width) | (pts[:, 1] < 0) | (pts[:, 1] > height)))

    bbox_change = np.zeros((t,), dtype=np.float32)
    bbox_change[1:] = np.abs(np.diff(bbox_area)) / np.maximum((bbox_area[1:] + bbox_area[:-1]) * 0.5, 1.0)
    mask_change = np.zeros((t,), dtype=np.float32)
    mask_change[1:] = np.abs(np.diff(mask_area)) / np.maximum((mask_area[1:] + mask_area[:-1]) * 0.5, 1.0)
    handedness_switch = np.zeros((t,), dtype=bool)
    handedness_switch[1:] = handedness[1:] != handedness[:-1]

    flags = np.zeros((t, len(FLAG_NAMES)), dtype=bool)
    name_to_idx = {name: idx for idx, name in enumerate(FLAG_NAMES)}
    flags[:, name_to_idx["low_confidence"]] = quality < args.quality_warn
    flags[:, name_to_idx["interpolated"]] = interpolated
    flags[:, name_to_idx["detector_lost"]] = ~valid
    flags[:, name_to_idx["temporal_jump"]] = temporal > args.temporal_warn
    flags[:, name_to_idx["bone_length_outlier"]] = bone > args.bone_warn
    flags[:, name_to_idx["wrist_step_outlier"]] = wrist_step > args.wrist_warn
    flags[:, name_to_idx["fingertip_step_outlier"]] = fingertip_step_max > args.fingertip_warn
    flags[:, name_to_idx["large_bbox_change"]] = bbox_change > 0.5
    flags[:, name_to_idx["small_or_empty_mask"]] = mask_area <= 0
    ratio = np.divide(mask_area, np.maximum(bbox_area, 1.0))
    flags[:, name_to_idx["mask_bbox_mismatch"]] = (ratio < args.mask_bbox_ratio_min) | (ratio > args.mask_bbox_ratio_max) | (mask_inside_bbox < 0.75)
    flags[:, name_to_idx["out_of_frame"]] = out_of_frame
    flags[:, name_to_idx["handedness_switch"]] = handedness_switch

    rows = []
    for idx in range(t):
        reasons = [name for name, flag_idx in name_to_idx.items() if flags[idx, flag_idx]]
        if failure_reason[idx] != "ok" and failure_reason[idx] not in reasons:
            reasons.append(failure_reason[idx])
        if quality[idx] < args.quality_critical and "low_confidence" not in reasons:
            reasons.append("low_confidence")
        if temporal[idx] > args.temporal_critical and "temporal_jump" not in reasons:
            reasons.append("temporal_jump")
        if bone[idx] > args.bone_critical and "bone_length_outlier" not in reasons:
            reasons.append("bone_length_outlier")
        if wrist_step[idx] > args.wrist_critical and "wrist_step_outlier" not in reasons:
            reasons.append("wrist_step_outlier")
        if fingertip_step_max[idx] > args.fingertip_critical and "fingertip_step_outlier" not in reasons:
            reasons.append("fingertip_step_outlier")
        risk_level = "critical" if (failure_reason[idx] != "ok" or any(reason in reasons for reason in ["low_confidence", "temporal_jump", "bone_length_outlier", "wrist_step_outlier", "fingertip_step_outlier", "detector_lost"])) else "warn" if reasons else "ok"
        rows.append(
            {
                "frame_id": int(frame_ids[idx]),
                "timestamp_sec": float(timestamps[idx]),
                "file_name": frames[idx].get("file_name", f"{int(frame_ids[idx]):06d}.jpg"),
                "valid": bool(valid[idx]),
                "interpolated_flag": bool(interpolated[idx]),
                "quality_score": float(quality[idx]),
                "handedness": handedness[idx],
                "handedness_score": float(handedness_score[idx]),
                "temporal_jump_score": float(temporal[idx]),
                "bone_length_error": float(bone[idx]),
                "wrist_step": float(wrist_step[idx]),
                "fingertip_step_max": float(fingertip_step_max[idx]),
                "fingertip_step_mean": float(fingertip_step_mean[idx]),
                "bbox_area": float(bbox_area[idx]),
                "mask_area": float(mask_area[idx]),
                "mask_bbox_ratio": float(ratio[idx]),
                "mask_inside_bbox": float(mask_inside_bbox[idx]),
                "mask_bbox_iou": float(mask_bbox_iou[idx]),
                "mask_area_change": float(mask_change[idx]),
                "bbox_area_change": float(bbox_change[idx]),
                "failure_reason": failure_reason[idx],
                "risk_level": risk_level,
                "risk_reasons": ";".join(reasons) if reasons else "ok",
            }
        )

    csv_path = out_dir / "frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = frame_ids
    save_curve(figures_dir / "quality_scores.png", x, [("quality_score", quality), ("handedness_score", handedness_score)], "Quality Scores", "score")
    save_curve(figures_dir / "temporal_motion.png", x, [("temporal_jump_score", temporal), ("wrist_step", wrist_step), ("fingertip_step_max", fingertip_step_max)], "Temporal Motion Metrics", "non-metric units")
    save_curve(figures_dir / "bone_length_error.png", x, [("bone_length_error", bone)], "Bone Length Error", "error")
    save_curve(figures_dir / "mask_area.png", x, [("mask_area", mask_area), ("bbox_area", bbox_area)], "Mask/BBox Area", "px^2")
    save_curve(figures_dir / "mask_alignment.png", x, [("mask_bbox_ratio", ratio), ("mask_inside_bbox", mask_inside_bbox), ("mask_bbox_iou", mask_bbox_iou)], "Mask/BBox Alignment", "ratio")

    risk_frames = sorted(set(
        top_indices(quality, 5, largest=False)
        + top_indices(temporal, 5, largest=True)
        + top_indices(wrist_step, 5, largest=True)
        + top_indices(fingertip_step_max, 5, largest=True)
        + top_indices(bone, 5, largest=True)
        + top_indices(mask_change, 5, largest=True)
        + [0, t // 4, t // 2, (3 * t) // 4, t - 1]
    ))
    make_panel(risk_frames[:18], overlay_dir, skeleton_dir, figures_dir / "quality_keyframe_panel.jpg", {row["frame_id"]: row for row in rows})

    report_lines = [
        "# 帧级质量审计报告",
        "",
        f"- 输入轨迹：`{traj_npz}`",
        f"- 输入 JSON：`{landmarks_json}`",
        f"- mask 目录：`{masks_dir}`",
        f"- 总帧数：{t}",
        f"- 有效帧：{int(valid.sum())} / {t} ({valid.mean():.2%})",
        f"- 插值帧：{int(interpolated.sum())}",
        f"- CSV：`{csv_path}`",
        "",
        "## 序列级指标",
        "",
        f"- quality_score：{summarize(quality)}",
        f"- temporal_jump_score：{summarize(temporal)}",
        f"- bone_length_error：{summarize(bone)}",
        f"- wrist_step：{summarize(wrist_step)}",
        f"- fingertip_step_max：{summarize(fingertip_step_max)}",
        f"- mask_area：{summarize(mask_area)}",
        f"- bbox_area：{summarize(bbox_area)}",
        "",
        "## Top 风险帧",
        "",
        "| frame | time | risk | reasons | quality | temporal | wrist_step | fingertip_max | bone | mask_area |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    critical_rows = [row for row in rows if row["risk_level"] == "critical"]
    warn_rows = [row for row in rows if row["risk_level"] == "warn"]
    shown = critical_rows + warn_rows[: max(0, 20 - len(critical_rows))]
    for row in shown[:20]:
        report_lines.append(
            f"| {row['frame_id']} | {row['timestamp_sec']:.3f} | {row['risk_level']} | {row['risk_reasons']} | {row['quality_score']:.4f} | {row['temporal_jump_score']:.4f} | {row['wrist_step']:.6f} | {row['fingertip_step_max']:.6f} | {row['bone_length_error']:.6f} | {row['mask_area']:.1f} |"
        )
    report_lines.extend([
        "",
        "## 图表",
        "",
        f"- quality scores：`{figures_dir / 'quality_scores.png'}`",
        f"- temporal motion：`{figures_dir / 'temporal_motion.png'}`",
        f"- bone length error：`{figures_dir / 'bone_length_error.png'}`",
        f"- mask area：`{figures_dir / 'mask_area.png'}`",
        f"- mask alignment：`{figures_dir / 'mask_alignment.png'}`",
        f"- keyframe panel：`{figures_dir / 'quality_keyframe_panel.jpg'}`",
        "",
        "## 说明",
        "",
        "这些 step / velocity 指标来自 MediaPipe non-metric world landmarks，只用于同序列内部相对质量审计，不能解释为真实米制位移。",
    ])
    report_path = out_dir / "frame_quality_audit.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    metrics = {
        "hand_bbox2d": hand_bbox2d.astype(np.float32),
        "bbox_area": bbox_area.astype(np.float32),
        "mask_area": mask_area.astype(np.float32),
        "mask_bbox_ratio": ratio.astype(np.float32),
        "mask_inside_bbox": mask_inside_bbox.astype(np.float32),
        "mask_bbox_iou": mask_bbox_iou.astype(np.float32),
        "wrist_step": wrist_step.astype(np.float32),
        "fingertip_step": fingertip_step.astype(np.float32),
        "fingertip_step_max": fingertip_step_max.astype(np.float32),
        "quality_flags": flags,
        "quality_flag_names": np.array(FLAG_NAMES, dtype="U32"),
    }
    if args.out_npz:
        copy_npz_with_metrics(traj_npz, Path(args.out_npz), metrics)
        print(f"[OK] wrote enriched npz: {args.out_npz}")
    print(f"[OK] wrote frame metrics csv: {csv_path}")
    print(f"[OK] wrote frame audit report: {report_path}")


if __name__ == "__main__":
    main()
