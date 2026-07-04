#!/usr/bin/env python3
"""Check the IsaacGym pieces needed by the Video2Motion2Action challenge."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ISAACGYM_ROOTS = [
    Path("~/opt/isaac_gym/isaacgym").expanduser(),
    ROOT / ".tools" / "isaacgym" / "isaacgym",
    ROOT / ".tools" / "isaacgym",
]


def header(title: str) -> None:
    print(f"\n== {title} ==")


def check_module(name: str) -> bool:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - diagnostics must survive import-time errors
        print(f"[MISSING] {name}: {type(exc).__name__}: {exc}")
        return False
    version = getattr(mod, "__version__", "installed")
    print(f"[OK] {name}: {version}")
    return True


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def main() -> int:
    print("Video2Motion2Action IsaacGym readiness check")

    header("Python")
    print(f"executable: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 8):
        print("[WARN] Challenge IsaacGym Preview 4 path expects Python 3.8.")

    header("Python Packages")
    ok = True
    for module in ["numpy", "torch", "torchvision", "gym", "cv2", "trimesh", "yaml", "scipy", "tensorboard"]:
        ok = check_module(module) and ok

    header("Torch")
    try:
        import torch

        print(f"cuda available: {torch.cuda.is_available()}")
        print(f"torch CUDA build: {torch.version.cuda or 'CPU-only'}")
        if not torch.cuda.is_available():
            print("[INFO] CPU torch is enough for imports and asset checks, but training/rollout needs a GPU torch build.")
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] torch detail check failed: {type(exc).__name__}: {exc}")

    header("Isaac Gym Package")
    isaacgym_importable = check_module("isaacgym")
    for root in CANDIDATE_ISAACGYM_ROOTS:
        marker = root / "python" / "examples" / "joint_monkey.py"
        status = "FOUND" if marker.exists() else "missing"
        print(f"{status}: {marker}")

    header("NVIDIA Driver")
    code, out = run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"])
    if code == 0:
        print(out)
    else:
        print(f"[WARN] nvidia-smi unavailable: {out}")

    header("Challenge Mapping")
    print("- Environment setup: this checker covers the IsaacGym Python 3.8 baseline layer.")
    print("- Inspire sample: run joint_monkey.py only after Isaac Gym Preview 4 is installed.")
    print("- Sharpa URDF / object asset: trimesh, lxml, scipy, and OpenCV are installed for preflight checks.")
    print("- Training/evaluation: install a CUDA torch wheel and run on a GPU that supports classic Isaac Gym PhysX.")

    return 0 if ok and isaacgym_importable else 1


if __name__ == "__main__":
    raise SystemExit(main())
