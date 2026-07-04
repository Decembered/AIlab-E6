#!/usr/bin/env python3
"""应用 temporal refiner，生成训练迭代后的对照 NPZ 和报告。"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


FINGERTIP_INDICES = [4, 8, 12, 16, 20]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


class TemporalRefiner(nn.Module):
    def __init__(self, dim: int = 63, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim + 2, hidden, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.GELU(),
        )
        self.delta = nn.Conv1d(hidden, dim, kernel_size=1)
        self.quality = nn.Sequential(nn.Conv1d(hidden, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, quality: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(torch.cat([x, quality, valid], dim=-1).transpose(1, 2))
        return x + self.delta(feat).transpose(1, 2), self.quality(feat).transpose(1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply task2 temporal refiner.")
    parser.add_argument("--checkpoint", default="task2/outputs/training/temporal_refiner/temporal_refiner_best.pt", help="训练 checkpoint")
    parser.add_argument("--in_npz", default="task2/outputs/trajectories/hand_traj.npz", help="输入 hand_traj.npz")
    parser.add_argument("--out_npz", default="task2/outputs/trajectories/hand_traj_temporal_refined.npz", help="输出对照 NPZ")
    parser.add_argument("--out_report", default="task2/reports/temporal_refiner_apply_report.md", help="输出报告")
    parser.add_argument("--blend", type=float, default=0.35, help="低质量帧最大融合比例")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    return parser.parse_args()


def temporal_steps(points: np.ndarray) -> tuple[float, float]:
    wrist = points[:, 0]
    tips = points[:, FINGERTIP_INDICES]
    wrist_step = np.zeros((points.shape[0],), dtype=np.float32)
    tip_step = np.zeros((points.shape[0],), dtype=np.float32)
    if points.shape[0] > 1:
        wrist_step[1:] = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
        tip_step[1:] = np.linalg.norm(np.diff(tips, axis=0), axis=2).max(axis=1)
    return float(np.nanmax(wrist_step)), float(np.nanmax(tip_step))


def bone_error(points: np.ndarray) -> float:
    lengths = [np.linalg.norm(points[:, b] - points[:, a], axis=1) for a, b in HAND_CONNECTIONS]
    stacked = np.stack(lengths, axis=1)
    med = np.nanmedian(stacked, axis=0, keepdims=True)
    return float(np.nanmean(np.abs(stacked - med)))


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    in_npz = Path(args.in_npz)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
    if not in_npz.is_file():
        raise FileNotFoundError(f"输入 NPZ 不存在: {in_npz}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden = int(ckpt.get("args", {}).get("hidden", 256))
    model = TemporalRefiner(hidden=hidden).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean = torch.as_tensor(ckpt["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(ckpt["std"], dtype=torch.float32, device=device)

    data = np.load(in_npz, allow_pickle=True)
    keypoints = data["keypoints3d"].astype(np.float32)
    quality = data["quality_score"].astype(np.float32)
    valid = data["valid"].astype(np.float32)
    x = torch.as_tensor(keypoints.reshape(1, keypoints.shape[0], -1), dtype=torch.float32, device=device)
    x_norm = (x - mean) / std
    q = torch.as_tensor(quality.reshape(1, -1, 1), dtype=torch.float32, device=device)
    v = torch.as_tensor(valid.reshape(1, -1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_norm, pred_quality = model(x_norm, q, v)
    pred = (pred_norm * std + mean).cpu().numpy().reshape(keypoints.shape)
    pred_quality_np = pred_quality.cpu().numpy().reshape(-1).astype(np.float32)

    low_quality = np.clip((0.98 - quality) / 0.08, 0.0, 1.0).astype(np.float32)
    alpha = (args.blend * low_quality)[:, None, None]
    refined = (1.0 - alpha) * keypoints + alpha * pred

    payload = {key: data[key] for key in data.files}
    payload.update({
        "temporal_refiner_keypoints3d": refined.astype(np.float32),
        "temporal_refiner_raw_prediction3d": pred.astype(np.float32),
        "temporal_refiner_quality": pred_quality_np,
        "temporal_refiner_blend_alpha": alpha[:, 0, 0].astype(np.float32),
        "temporal_refiner_source": np.array(str(ckpt_path)),
        "temporal_refiner_note": np.array("Pseudo-label temporal denoising output. Core keypoints3d is preserved; refined output is provided as an auxiliary trained baseline."),
    })
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)

    before_wrist, before_tip = temporal_steps(keypoints)
    after_wrist, after_tip = temporal_steps(refined)
    before_bone = bone_error(keypoints)
    after_bone = bone_error(refined)
    report = [
        "# Temporal Refiner 应用报告",
        "",
        "该输出是训练迭代对照结果，训练目标为 MediaPipe pseudo-label 去噪，不是人工 GT、MANO 或 metric 3D。",
        "",
        f"- 输入：`{in_npz}`",
        f"- checkpoint：`{ckpt_path}`",
        f"- 输出：`{out_npz}`",
        f"- device：{device}",
        f"- max blend alpha：{float(alpha.max()):.4f}",
        "",
        "## 指标对比",
        "",
        "| metric | before | refined_aux |",
        "|---|---:|---:|",
        f"| wrist_step_max | {before_wrist:.6f} | {after_wrist:.6f} |",
        f"| fingertip_step_max | {before_tip:.6f} | {after_tip:.6f} |",
        f"| bone_length_error_mean | {before_bone:.6f} | {after_bone:.6f} |",
        "",
        "核心字段 `keypoints3d` 未被替换，训练结果保存在 `temporal_refiner_keypoints3d` 中，便于人工和下游单独评估。",
    ]
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[OK] wrote refined npz: {out_npz}")


if __name__ == "__main__":
    main()
