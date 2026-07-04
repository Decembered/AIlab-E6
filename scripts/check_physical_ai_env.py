#!/usr/bin/env python3
"""Lightweight Physical AI environment diagnostics.

This script only imports packages and prints actionable status. It does not
install anything, download assets, or fail when optional packages are missing.
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class PackageCheck:
    import_name: str
    display_name: str
    install_hint: str
    version_attr: str = "__version__"


CHECKS = [
    PackageCheck("torch", "PyTorch", "Install from https://pytorch.org/get-started/locally/"),
    PackageCheck("bitsandbytes", "bitsandbytes", "pip install bitsandbytes"),
    PackageCheck("cv2", "OpenCV", "pip install opencv-python"),
    PackageCheck("open3d", "Open3D", "pip install open3d"),
    PackageCheck("transformers", "transformers", "pip install transformers"),
    PackageCheck("accelerate", "accelerate", "pip install accelerate"),
]


def module_version(module: object, version_attr: str = "__version__") -> str:
    return str(getattr(module, version_attr, "unknown"))


def try_import(check: PackageCheck) -> tuple[str, Optional[object], Optional[str]]:
    try:
        module = importlib.import_module(check.import_name)
        return "installed", module, None
    except Exception as exc:  # noqa: BLE001 - diagnostics should catch import-time errors
        return "missing_or_broken", None, f"{type(exc).__name__}: {exc}"


def print_header(title: str) -> None:
    print(f"\n== {title} ==")


def check_python() -> None:
    print_header("Python")
    print(f"version: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.platform()}")


def check_packages() -> Optional[object]:
    print_header("Packages")
    torch_module: Optional[object] = None

    for check in CHECKS:
        status, module, error = try_import(check)
        if status == "installed" and module is not None:
            version = module_version(module, check.version_attr)
            print(f"[OK] {check.display_name}: {version}")
            if check.import_name == "torch":
                torch_module = module
        else:
            print(f"[MISSING] {check.display_name}")
            print(f"  error: {error}")
            print(f"  suggestion: {check.install_hint}")

    return torch_module


def check_cuda(torch_module: Optional[object]) -> None:
    print_header("CUDA / GPU")
    if torch_module is None:
        print("[SKIP] PyTorch is unavailable, so CUDA cannot be checked through torch.")
        return

    cuda = getattr(torch_module, "cuda", None)
    version = getattr(torch_module, "version", None)
    torch_cuda_version = getattr(version, "cuda", None) if version is not None else None
    print(f"torch CUDA build: {torch_cuda_version or 'unknown'}")

    try:
        available = bool(cuda.is_available())
        print(f"cuda available: {available}")
        if not available:
            print("suggestion: verify NVIDIA driver, CUDA_VISIBLE_DEVICES, and PyTorch CUDA build.")
            return

        count = int(cuda.device_count())
        print(f"gpu count: {count}")
        for index in range(count):
            name = cuda.get_device_name(index)
            capability = cuda.get_device_capability(index)
            print(f"gpu {index}: {name} (capability {capability[0]}.{capability[1]})")
    except Exception as exc:  # noqa: BLE001 - diagnostics should not crash
        print(f"[BROKEN] CUDA check failed: {type(exc).__name__}: {exc}")
        print("suggestion: compare nvidia-smi, PyTorch version, and driver compatibility.")


def main() -> int:
    print("Physical AI environment check")
    check_python()
    torch_module = check_packages()
    check_cuda(torch_module)

    print_header("Notes")
    print("- Missing optional packages are not fatal for this workspace.")
    print("- Do not install heavy simulator stacks or large model dependencies without a specific demo plan.")
    print("- For simulator issues, record this output in the experiment logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

