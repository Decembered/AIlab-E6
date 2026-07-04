#!/usr/bin/env python3
"""Diagnose local Isaac Gym / Isaac Sim readiness without large downloads."""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_PATHS = {
    "isaacgym-py38": ROOT / ".venvs" / "isaacgym-py38" / "bin" / "python",
    "isaaclab-py310": ROOT / ".venvs" / "isaaclab-py310" / "bin" / "python",
}


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def header(title: str) -> None:
    print(f"\n== {title} ==")


def parse_compute_capability_from_nvidia_smi() -> list[str]:
    code, out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def module_status(python: Path, module: str) -> str:
    code, out = run(
        [
            str(python),
            "-c",
            (
                "import importlib, sys; "
                f"m=importlib.import_module('{module}'); "
                "print(getattr(m, '__version__', 'installed'))"
            ),
        ]
    )
    if code == 0:
        return f"OK {out}"
    return "missing"


def check_current_python() -> None:
    header("Current Python")
    print(f"executable: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"glibc: {platform.libc_ver()[1] or 'unknown'}")
    print(f"cwd: {ROOT}")


def check_gpu() -> None:
    header("GPU / Driver")
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print("[MISSING] nvidia-smi")
        return
    code, out = run([nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if code == 0:
        print(out)
    else:
        print(out)

    caps = parse_compute_capability_from_nvidia_smi()
    if caps:
        print("compute capability:")
        for cap in caps:
            print(f"  {cap}")
        if any(re.search(r",\s*(8\.9|12\.0)\b", cap) for cap in caps):
            print("[WARN] Classic Isaac Gym Preview 4 GPU PhysX is likely incompatible with Ada/Blackwell GPUs.")
            print("       Prefer Isaac Sim + Isaac Lab for this machine.")


def check_envs() -> None:
    header("Project Environments")
    for name, py in ENV_PATHS.items():
        print(f"{name}: {py}")
        if not py.exists():
            print("  [MISSING] create with scripts/setup_isaac_stack.sh prepare")
            continue
        code, out = run([str(py), "--version"])
        print(f"  python: {out if code == 0 else 'broken'}")
        for module in ("pip", "torch", "isaacgym", "isaacsim"):
            print(f"  {module}: {module_status(py, module)}")


def check_archives() -> None:
    header("Local Isaac Packages")
    patterns = [
        "*IsaacGym*Preview*4*.tar*",
        "*IsaacGym*Preview*4*.zip",
        "*isaacgym*preview*4*.tar*",
        "*isaacgym*preview*4*.zip",
        "isaacgym",
    ]
    candidates: list[Path] = []
    for base in (ROOT, ROOT / ".tools", Path.home() / "Downloads"):
        if not base.exists():
            continue
        for pattern in patterns:
            candidates.extend(base.glob(pattern))
    if candidates:
        for path in sorted(set(candidates)):
            print(path)
    else:
        print("[MISSING] No Isaac Gym Preview 4 archive/extracted directory found.")
        print("          Download requires accepting NVIDIA's Isaac Gym license manually.")


def main() -> int:
    print("Isaac stack diagnostic")
    check_current_python()
    check_gpu()
    check_envs()
    check_archives()

    header("Recommendation")
    print("- For this RTX 4060 Laptop GPU, use Isaac Sim 4.5 + Isaac Lab as the main hackathon path.")
    print("- Use classic Isaac Gym Preview 4 only for reading/running old code on compatible GPUs, or CPU-only smoke tests.")
    print("- Do not claim raw VLA or learned actions are directly safe on real robots without a safety filter/controller layer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
