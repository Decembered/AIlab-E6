#!/usr/bin/env python3
"""用 MediaPipe pseudo-label 训练轻量时序去噪/质量模型。

该脚本不使用人工 GT，不声称训练目标是真实 3D。训练目标是从加噪/遮挡的
MediaPipe non-metric 3D skeleton 中恢复高质量 pseudo-label，并预测帧级质量。
"""

import argparse
import json
import re
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal refiner on task2 pseudo-label trajectories.")
    parser.add_argument("--by_view_dir", default="task2/outputs/by_view", help="按视角结果目录")
    parser.add_argument("--out_dir", default="task2/outputs/training/temporal_refiner", help="训练输出目录")
    parser.add_argument("--views", nargs="+", default=None, choices=["camera_side_1", "camera_side_2", "camera_top"], help="可选：仅使用指定视角")
    parser.add_argument("--include_run_regex", default=None, help="可选：仅使用匹配该正则的 run_id")
    parser.add_argument("--window", type=int, default=32, help="训练窗口长度")
    parser.add_argument("--stride", type=int, default=8, help="窗口步长")
    parser.add_argument("--epochs", type=int, default=30, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--hidden", type=int, default=256, help="TCN hidden size")
    parser.add_argument("--lr", type=float, default=2e-3, help="学习率")
    parser.add_argument("--noise_std", type=float, default=0.01, help="输入加噪标准差")
    parser.add_argument("--drop_prob", type=float, default=0.12, help="随机丢帧概率")
    parser.add_argument("--seed", type=int, default=20260704, help="随机种子")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    return parser.parse_args()


def sequence_prefix(run_id: str) -> str:
    for suffix in ["_camera_side_1", "_camera_side_2", "_camera_top"]:
        if run_id.endswith(suffix):
            return run_id[: -len(suffix)]
    return run_id


def run_view_id(run_id: str) -> str:
    for view_id in ["camera_side_1", "camera_side_2", "camera_top"]:
        if run_id.endswith("_" + view_id):
            return view_id
    return "unknown"


def load_runs(by_view_dir: Path, views: set[str] | None, include_run_regex: str | None) -> list[dict]:
    runs = []
    run_re = re.compile(include_run_regex) if include_run_regex else None
    for npz_path in sorted(by_view_dir.glob("*/trajectories/hand_traj.npz")):
        run_id = npz_path.parents[1].name
        if run_re is not None and not run_re.fullmatch(run_id):
            continue
        view_id = run_view_id(run_id)
        if views is not None and view_id not in views:
            continue
        data = np.load(npz_path, allow_pickle=True)
        keypoints = data["keypoints3d"].astype(np.float32)
        quality = data["quality_score"].astype(np.float32)
        valid = data["valid"].astype(bool)
        flags = data["quality_flags"].astype(bool) if "quality_flags" in data else np.zeros((len(valid), 0), dtype=bool)
        risk = flags.any(axis=1) if flags.size else np.zeros(len(valid), dtype=bool)
        weights = np.clip(quality, 0.0, 1.0).astype(np.float32)
        weights[~valid] *= 0.25
        weights[risk] *= 0.55
        runs.append({
            "run_id": run_id,
            "sequence_id": sequence_prefix(run_id),
            "view_id": view_id,
            "path": str(npz_path),
            "keypoints": keypoints,
            "quality": quality,
            "valid": valid.astype(np.float32),
            "weights": weights,
        })
    if not runs:
        raise FileNotFoundError(f"未找到 hand_traj.npz: {by_view_dir}")
    return runs


def make_split(runs: list[dict], seed: int) -> tuple[set[str], set[str]]:
    seqs = sorted({run["sequence_id"] for run in runs})
    rng = random.Random(seed)
    rng.shuffle(seqs)
    val_count = max(1, int(round(len(seqs) * 0.2))) if len(seqs) > 1 else 1
    val = set(seqs[:val_count])
    train = set(seqs[val_count:]) or set(seqs)
    return train, val


class WindowDataset(Dataset):
    def __init__(self, runs: list[dict], split_seqs: set[str], window: int, stride: int, mean: np.ndarray, std: np.ndarray):
        self.items = []
        self.runs = runs
        self.window = window
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        for run_idx, run in enumerate(runs):
            if run["sequence_id"] not in split_seqs:
                continue
            t = run["keypoints"].shape[0]
            if t < window:
                continue
            for start in range(0, t - window + 1, stride):
                self.items.append((run_idx, start))
        if not self.items:
            raise ValueError("没有可训练窗口，请检查 split/window/stride")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        run_idx, start = self.items[idx]
        run = self.runs[run_idx]
        sl = slice(start, start + self.window)
        y = run["keypoints"][sl].reshape(self.window, -1)
        y = (y - self.mean) / self.std
        quality = run["quality"][sl, None]
        valid = run["valid"][sl, None]
        weights = run["weights"][sl, None]
        return {
            "target": torch.from_numpy(y.astype(np.float32)),
            "quality": torch.from_numpy(quality.astype(np.float32)),
            "valid": torch.from_numpy(valid.astype(np.float32)),
            "weights": torch.from_numpy(weights.astype(np.float32)),
        }


class TemporalRefiner(nn.Module):
    def __init__(self, dim: int = 63, hidden: int = 256):
        super().__init__()
        in_dim = dim + 2
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.delta = nn.Conv1d(hidden, dim, kernel_size=1)
        self.quality = nn.Sequential(nn.Conv1d(hidden, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, quality: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x, quality, valid], dim=-1).transpose(1, 2)
        feat = self.net(inp)
        delta = self.delta(feat).transpose(1, 2)
        pred_q = self.quality(feat).transpose(1, 2)
        return x + delta, pred_q


def corrupt(target: torch.Tensor, valid: torch.Tensor, noise_std: float, drop_prob: float) -> torch.Tensor:
    x = target + torch.randn_like(target) * noise_std
    if drop_prob > 0:
        drop = (torch.rand(target.shape[:2], device=target.device) < drop_prob).unsqueeze(-1)
        x = torch.where(drop, torch.zeros_like(x), x)
    x = torch.where(valid > 0.5, x, torch.zeros_like(x))
    return x


def train_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device, args: argparse.Namespace) -> dict:
    model.train()
    totals = {"loss": 0.0, "coord": 0.0, "quality": 0.0, "smooth": 0.0, "n": 0}
    for batch in loader:
        target = batch["target"].to(device)
        quality = batch["quality"].to(device)
        valid = batch["valid"].to(device)
        weights = batch["weights"].to(device)
        x = corrupt(target, valid, args.noise_std, args.drop_prob)
        pred, pred_quality = model(x, quality, valid)
        coord = ((pred - target).abs() * weights).sum() / weights.sum().clamp_min(1.0) / target.shape[-1]
        qloss = ((pred_quality - quality).pow(2) * valid).sum() / valid.sum().clamp_min(1.0)
        vel = pred[:, 1:] - pred[:, :-1]
        smooth = vel.pow(2).mean()
        loss = coord + 0.5 * qloss + 0.02 * smooth
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        bsz = target.shape[0]
        totals["loss"] += float(loss.item()) * bsz
        totals["coord"] += float(coord.item()) * bsz
        totals["quality"] += float(qloss.item()) * bsz
        totals["smooth"] += float(smooth.item()) * bsz
        totals["n"] += bsz
    return {k: v / max(totals["n"], 1) for k, v in totals.items() if k != "n"}


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict:
    model.eval()
    totals = {"loss": 0.0, "coord": 0.0, "quality": 0.0, "n": 0}
    for batch in loader:
        target = batch["target"].to(device)
        quality = batch["quality"].to(device)
        valid = batch["valid"].to(device)
        weights = batch["weights"].to(device)
        x = corrupt(target, valid, args.noise_std, args.drop_prob)
        pred, pred_quality = model(x, quality, valid)
        coord = ((pred - target).abs() * weights).sum() / weights.sum().clamp_min(1.0) / target.shape[-1]
        qloss = ((pred_quality - quality).pow(2) * valid).sum() / valid.sum().clamp_min(1.0)
        loss = coord + 0.5 * qloss
        bsz = target.shape[0]
        totals["loss"] += float(loss.item()) * bsz
        totals["coord"] += float(coord.item()) * bsz
        totals["quality"] += float(qloss.item()) * bsz
        totals["n"] += bsz
    return {k: v / max(totals["n"], 1) for k, v in totals.items() if k != "n"}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(Path(args.by_view_dir), set(args.views) if args.views else None, args.include_run_regex)
    train_seqs, val_seqs = make_split(runs, args.seed)
    train_points = [run["keypoints"].reshape(-1, 63) for run in runs if run["sequence_id"] in train_seqs]
    stacked = np.concatenate(train_points, axis=0)
    mean = np.nanmean(stacked, axis=0).astype(np.float32)
    std = np.nanstd(stacked, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4)

    train_ds = WindowDataset(runs, train_seqs, args.window, args.stride, mean, std)
    val_ds = WindowDataset(runs, val_seqs, args.window, args.stride, mean, std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TemporalRefiner(hidden=args.hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best = float("inf")
    best_path = out_dir / "temporal_refiner_best.pt"
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optim, device, args)
        va = eval_epoch(model, val_loader, device, args)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        print(f"[EPOCH {epoch:03d}] train_loss={tr['loss']:.6f} val_loss={va['loss']:.6f} val_coord={va['coord']:.6f}")
        if va["loss"] < best:
            best = va["loss"]
            torch.save({
                "model": model.state_dict(),
                "mean": mean,
                "std": std,
                "args": vars(args),
                "train_sequences": sorted(train_seqs),
                "val_sequences": sorted(val_seqs),
                "runs": [{"run_id": r["run_id"], "sequence_id": r["sequence_id"], "path": r["path"]} for r in runs],
                "views": args.views,
                "include_run_regex": args.include_run_regex,
            }, best_path)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    report = [
        "# Temporal Refiner 训练报告",
        "",
        "本模型使用 MediaPipe non-metric skeleton pseudo-label 训练，不使用人工 GT，不输出 MANO/mesh。",
        "",
        f"- 训练视角数：{len(runs)}",
        f"- 使用视角：{args.views if args.views else 'all'}",
        f"- run_id 过滤：{args.include_run_regex if args.include_run_regex else 'none'}",
        f"- train sequences：{sorted(train_seqs)}",
        f"- val sequences：{sorted(val_seqs)}",
        f"- window/stride：{args.window}/{args.stride}",
        f"- device：{device}",
        f"- best val loss：{best:.6f}",
        f"- checkpoint：`{best_path}`",
    ]
    (out_dir / "training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [h["train_loss"] for h in history], label="train")
    plt.plot(epochs, [h["val_loss"] for h in history], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    print(f"[OK] wrote training outputs: {out_dir}")


if __name__ == "__main__":
    main()
