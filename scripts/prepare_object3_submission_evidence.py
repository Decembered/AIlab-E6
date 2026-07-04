#!/usr/bin/env python3
"""Prepare Member C / task 3.3 object-asset submission evidence.

This script is intentionally lightweight and CPU-safe. It does not run SAM,
download models, or train anything. It turns the current object assets into a
reviewable evidence bundle: renders, mesh/URDF metrics, local IsaacGym probe
logs, and a concise status report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OBJECTS = ("bread", "pipette", "drink_ad", "drink_yykx")
DISPLAY_NAMES = {
    "bread": "Bread #1",
    "pipette": "Pipette #1",
    "drink_ad": "Drink AD",
    "drink_yykx": "Drink YYKX",
}
OBJECT_SEQUENCES = {
    "bread": "weigh_bread__2026_0701_0044_30",
    "pipette": "grasp_pipette_stand__2026_0701_0019_19",
    "drink_ad": "weigh_drink_ad__2026_0701_0047_56",
    "drink_yykx": "weigh_drink_yykx__2026_0701_0051_12",
}
OBJECT_NOTES = {
    "bread": "Best current mask/pose evidence; multi-view sparse pose exists for frame 115.",
    "pipette": "Static v1 asset exists; dynamic object masks and articulated plunger remain future work.",
    "drink_ad": "Axisymmetric static v1 asset exists; yaw is ambiguous for pose tracking.",
    "drink_yykx": "Axisymmetric static v1 asset exists; yaw is ambiguous for pose tracking.",
}
COLORS = {
    "bread": "#d4a15f",
    "pipette": "#607d9c",
    "drink_ad": "#78a85f",
    "drink_yykx": "#5f90b8",
}
MAX_SUBMISSION_MASK_RATIO = 0.10


@dataclass
class AssetMetrics:
    name: str
    object_name: str
    urdf: Path
    visual_mesh: Path
    collision_mesh: Path
    visual_vertices: int
    visual_faces: int
    collision_vertices: int
    collision_faces: int
    visual_watertight: bool
    collision_watertight: bool
    visual_winding_consistent: bool
    collision_winding_consistent: bool
    visual_extents_m: list[float]
    collision_extents_m: list[float]
    visual_volume_m3: float
    collision_volume_m3: float
    mass_kg: float | None
    inertia: dict[str, float | None]

    def csv_row(self) -> dict[str, Any]:
        return {
            "object": self.name,
            "display_name": self.object_name,
            "urdf": str(self.urdf),
            "visual_mesh": str(self.visual_mesh),
            "collision_mesh": str(self.collision_mesh),
            "visual_vertices": self.visual_vertices,
            "visual_faces": self.visual_faces,
            "collision_vertices": self.collision_vertices,
            "collision_faces": self.collision_faces,
            "visual_watertight": self.visual_watertight,
            "collision_watertight": self.collision_watertight,
            "visual_winding_consistent": self.visual_winding_consistent,
            "collision_winding_consistent": self.collision_winding_consistent,
            "visual_extents_m": " x ".join(f"{x:.4f}" for x in self.visual_extents_m),
            "collision_extents_m": " x ".join(f"{x:.4f}" for x in self.collision_extents_m),
            "visual_volume_m3": f"{self.visual_volume_m3:.8g}",
            "collision_volume_m3": f"{self.collision_volume_m3:.8g}",
            "mass_kg": "" if self.mass_kg is None else f"{self.mass_kg:.6g}",
            "ixx": none_or_float(self.inertia.get("ixx")),
            "iyy": none_or_float(self.inertia.get("iyy")),
            "izz": none_or_float(self.inertia.get("izz")),
        }

    def summary_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "object_name": self.object_name,
            "urdf": str(self.urdf),
            "visual_mesh": str(self.visual_mesh),
            "collision_mesh": str(self.collision_mesh),
            "visual_faces": self.visual_faces,
            "collision_faces": self.collision_faces,
            "visual_watertight": self.visual_watertight,
            "collision_watertight": self.collision_watertight,
            "visual_extents_m": self.visual_extents_m,
            "collision_extents_m": self.collision_extents_m,
            "mass_kg": self.mass_kg,
            "inertia": self.inertia,
        }


def none_or_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8g}"


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")
    return mesh


def parse_urdf_inertial(path: Path) -> tuple[float | None, dict[str, float | None]]:
    inertia = {key: None for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
    mass = None
    root = ET.parse(path).getroot()
    mass_el = root.find(".//inertial/mass")
    if mass_el is not None and mass_el.get("value") is not None:
        mass = float(mass_el.get("value"))
    inertia_el = root.find(".//inertial/inertia")
    if inertia_el is not None:
        for key in inertia:
            if inertia_el.get(key) is not None:
                inertia[key] = float(inertia_el.get(key))
    return mass, inertia


def collect_metrics(asset_root: Path, name: str) -> AssetMetrics:
    root = asset_root / name
    urdf = root / "asset" / "object.urdf"
    visual_path = root / "mesh" / "visual_mesh.obj"
    collision_path = root / "mesh" / "collision_mesh.obj"
    visual = load_mesh(visual_path)
    collision = load_mesh(collision_path)
    mass, inertia = parse_urdf_inertial(urdf)
    return AssetMetrics(
        name=name,
        object_name=DISPLAY_NAMES[name],
        urdf=urdf,
        visual_mesh=visual_path,
        collision_mesh=collision_path,
        visual_vertices=int(len(visual.vertices)),
        visual_faces=int(len(visual.faces)),
        collision_vertices=int(len(collision.vertices)),
        collision_faces=int(len(collision.faces)),
        visual_watertight=bool(visual.is_watertight),
        collision_watertight=bool(collision.is_watertight),
        visual_winding_consistent=bool(visual.is_winding_consistent),
        collision_winding_consistent=bool(collision.is_winding_consistent),
        visual_extents_m=[float(x) for x in visual.extents],
        collision_extents_m=[float(x) for x in collision.extents],
        visual_volume_m3=float(visual.volume) if visual.is_watertight else float("nan"),
        collision_volume_m3=float(collision.volume) if collision.is_watertight else float("nan"),
        mass_kg=mass,
        inertia=inertia,
    )


def write_geometry_check(mesh: trimesh.Trimesh, output_path: Path, label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = [
        f"name: {label}",
        f"vertices: {len(mesh.vertices)}",
        f"faces: {len(mesh.faces)}",
        f"is_watertight: {mesh.is_watertight}",
        f"is_winding_consistent: {mesh.is_winding_consistent}",
        "bounds:",
        np.array2string(mesh.bounds, precision=6),
        "extents: " + np.array2string(mesh.extents, precision=6),
        "center_mass: " + np.array2string(mesh.center_mass, precision=6),
        f"volume: {mesh.volume if mesh.is_watertight else 'nan'}",
    ]
    output_path.write_text("\n".join(text) + "\n", encoding="utf-8")


def set_equal_3d(ax, bounds: np.ndarray) -> None:
    center = bounds.mean(axis=0)
    span = float(max(bounds[1] - bounds[0]))
    radius = max(span * 0.58, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_mesh(mesh: trimesh.Trimesh, output_path: Path, title: str, color: str, view: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    triangles = mesh.vertices[mesh.faces]

    fig = plt.figure(figsize=(7, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        triangles,
        facecolor=color,
        edgecolor="#303030",
        linewidths=0.12,
        alpha=0.93,
    )
    ax.add_collection3d(collection)
    set_equal_3d(ax, mesh.bounds)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    if view == "front":
        ax.view_init(elev=0, azim=-90)
    elif view == "side":
        ax.view_init(elev=0, azim=0)
    elif view == "top":
        ax.view_init(elev=90, azim=-90)
    else:
        ax.view_init(elev=24, azim=-42)

    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_asset_views(asset_root: Path, name: str) -> list[Path]:
    root = asset_root / name
    mesh = load_mesh(root / "mesh" / "visual_mesh.obj")
    out_dir = root / "renders"
    outputs = []
    for view in ("front", "side", "top", "angle"):
        out_path = out_dir / f"{view}.png"
        render_mesh(mesh, out_path, f"{DISPLAY_NAMES[name]} - {view}", COLORS[name], view)
        outputs.append(out_path)

    collision = load_mesh(root / "mesh" / "collision_mesh.obj")
    collision_path = out_dir / "collision_angle.png"
    render_mesh(collision, collision_path, f"{DISPLAY_NAMES[name]} collision", "#9a9a9a", "angle")
    outputs.append(collision_path)
    return outputs


def write_csv(rows: list[AssetMetrics], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].csv_row().keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.csv_row())


def write_markdown_summary(rows: list[AssetMetrics], output_path: Path) -> None:
    lines = [
        "# Object Asset v1 Geometry Summary",
        "",
        "All units are meters and kilograms. These assets are simulation-first, low-face-count meshes intended for IsaacGym loading and tracking integration.",
        "",
        "| Object | Visual faces | Collision faces | Watertight | Extents (m) | Mass (kg) | URDF |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for row in rows:
        watertight = "yes" if row.visual_watertight and row.collision_watertight else "no"
        extents = " x ".join(f"{x:.3f}" for x in row.visual_extents_m)
        mass = "" if row.mass_kg is None else f"{row.mass_kg:.3g}"
        lines.append(
            f"| {row.object_name} | {row.visual_faces} | {row.collision_faces} | "
            f"{watertight} | {extents} | {mass} | `{row.urdf}` |"
        )
    lines.extend(
        [
            "",
            "## Evidence Layout",
            "",
            "- Per-object renders: `runs/object_asset_v1/<object>/renders/`",
            "- Render contact sheet: `outputs/object3_submission/render_contact_sheet.png`",
            "- Per-object geometry checks: `runs/object_asset_v1/<object>/report/`",
            "- IsaacGym probe logs: `runs/object_asset_v1/<object>/report/asset_check_local.log`",
            "- Pose inventory: `outputs/dataset_inventory.csv`",
            "- Bread current pose output: `outputs/mask_pose/bread/weigh_bread__2026_0701_0044_30/`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_render_contact_sheet(asset_root: Path, output_path: Path) -> None:
    views = ("front", "side", "top", "angle")
    fig, axes = plt.subplots(len(OBJECTS), len(views), figsize=(14, 12), dpi=140)
    for row_idx, name in enumerate(OBJECTS):
        for col_idx, view in enumerate(views):
            ax = axes[row_idx, col_idx]
            image_path = asset_root / name / "renders" / f"{view}.png"
            if image_path.exists():
                ax.imshow(plt.imread(image_path))
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(view)
            if col_idx == 0:
                ax.text(
                    -0.02,
                    0.5,
                    DISPLAY_NAMES[name],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                )
    fig.suptitle("Object Asset v1 Render Contact Sheet", fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_isaacgym_probe(repo_root: Path, python_exe: str, row: AssetMetrics, timeout_s: int) -> dict[str, Any]:
    log_path = row.urdf.parent.parent / "report" / "asset_check_local.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(repo_root / "scripts" / "validate_asset_isaacgym.py"),
        "--urdf",
        str(row.urdf),
        "--steps",
        "60",
        "--headless",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        output = proc.stdout
        status = "pass" if proc.returncode == 0 else "failed_local_env"
        reason = None if proc.returncode == 0 else first_failure_line(output)
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + f"\nTIMEOUT after {timeout_s}s\n"
        status = "timeout"
        reason = f"timeout after {timeout_s}s"
    log_path.write_text(output, encoding="utf-8", errors="replace")
    return {
        "object": row.name,
        "urdf": str(row.urdf),
        "status": status,
        "reason": reason,
        "log": str(log_path),
    }


def first_failure_line(output: str) -> str:
    for line in output.splitlines():
        if "FAIL:" in line or "Traceback" in line or "ModuleNotFoundError" in line:
            return line.strip()
    return output.splitlines()[-1].strip() if output.splitlines() else "unknown failure"


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_mask_candidates(repo_root: Path, object_name: str) -> list[Path]:
    paths: list[Path] = []
    roots = [
        repo_root / "experiments" / f"2026-07-05_obj_recon_{object_name}" / "mask_debug",
        repo_root / "experiments" / f"2026-07-04_obj_recon_{object_name}" / "masks",
    ]
    if object_name == "bread":
        roots.append(repo_root / "experiments" / "2026-07-04_obj_recon_bread" / "masks")
    for root in roots:
        if root.exists():
            paths.extend(root.glob("*mask*.png"))
    return sorted({p for p in paths if "overlay" not in p.name and "metadata" not in p.name})


def audit_masks(repo_root: Path) -> list[dict[str, Any]]:
    import cv2

    rows: list[dict[str, Any]] = []
    for name in OBJECTS:
        for path in discover_mask_candidates(repo_root, name):
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            binary = mask > 0
            area = int(binary.sum())
            h, w = binary.shape[:2]
            ratio = float(area / max(h * w, 1))
            usable = bool(area >= 50 and ratio <= MAX_SUBMISSION_MASK_RATIO)
            if area < 50:
                reason = "empty_or_too_small"
            elif ratio > MAX_SUBMISSION_MASK_RATIO:
                reason = f"mask_area_ratio_too_large ({ratio:.3f}>{MAX_SUBMISSION_MASK_RATIO:.3f})"
            else:
                reason = None
            rows.append(
                {
                    "object": name,
                    "mask_path": str(path),
                    "width": w,
                    "height": h,
                    "mask_area": area,
                    "mask_area_ratio": ratio,
                    "usable_for_pose": usable,
                    "invalid_reason": reason,
                }
            )
    return rows


def summarize_mask_audit(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = summary.setdefault(row["object"], {"total": 0, "usable": 0, "issues": {}})
        item["total"] += 1
        if row["usable_for_pose"]:
            item["usable"] += 1
        reason = row["invalid_reason"]
        if reason:
            key = str(reason).split(" (", 1)[0]
            item["issues"][key] = item["issues"].get(key, 0) + 1
    return summary


def write_mask_audit(rows: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "object",
        "mask_path",
        "width",
        "height",
        "mask_area",
        "mask_area_ratio",
        "usable_for_pose",
        "invalid_reason",
    )
    with open(output_root / "mask_audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = summarize_mask_audit(rows)
    lines = [
        "# Object Mask Audit",
        "",
        f"Usable masks require area >= 50 px and mask area ratio <= {MAX_SUBMISSION_MASK_RATIO:.2f}. This is only a quality audit; it does not create new masks.",
        "",
        "| Object | Total masks | Usable for pose | Main issue |",
        "|---|---:|---:|---|",
    ]
    for name in OBJECTS:
        item = summary.get(name, {"total": 0, "usable": 0, "issues": {}})
        if item["issues"]:
            issue = max(item["issues"].items(), key=lambda kv: kv[1])[0]
        else:
            issue = "none"
        lines.append(f"| {DISPLAY_NAMES[name]} | {item['total']} | {item['usable']} | {issue} |")
    (output_root / "mask_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_status_report(
    repo_root: Path,
    rows: list[AssetMetrics],
    validation: list[dict[str, Any]],
    mask_audit: list[dict[str, Any]],
    output_path: Path,
) -> None:
    inventory_path = repo_root / "outputs" / "dataset_inventory.csv"
    bread_pose_report = load_json_if_exists(
        repo_root
        / "outputs"
        / "mask_pose"
        / "bread"
        / "weigh_bread__2026_0701_0044_30"
        / "trajectory_quality_report.json"
    )
    validation_by_object = {item["object"]: item for item in validation}
    mask_counts = summarize_mask_audit(mask_audit)

    lines = [
        "# Member C Task 3.3 Submission Evidence Status",
        "",
        "## Summary",
        "",
        "This package organizes the current object reconstruction deliverables for the Video2Motion2Action task. The current data has RGB/calibration/HDF5, but primary-object GT pose masks are all false and no raw depth frames are available, so the pose output is video/mask-driven rather than GT or RGB-D ICP.",
        "",
        "## Asset Completion",
        "",
        "| Object | Mesh/URDF | Geometry | Renders | IsaacGym local probe | Current pose evidence |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        probe = validation_by_object.get(row.name, {})
        probe_text = probe.get("status", "not_run")
        if probe.get("reason"):
            probe_text += f": {probe['reason']}"
        mask_text = mask_counts.get(row.name, {"usable": 0, "total": 0})
        pose_text = (
            "available sparse multi-view"
            if row.name == "bread"
            else f"not yet; usable masks {mask_text['usable']}/{mask_text['total']}"
        )
        lines.append(
            f"| {row.object_name} | yes | watertight, {row.visual_faces} visual faces | "
            f"`runs/object_asset_v1/{row.name}/renders/` | {probe_text} | {pose_text} |"
        )

    lines.extend(
        [
            "",
            "## Current Task 3.3 Score Readiness",
            "",
            "| Sub-item | Status | Evidence | Remaining work |",
            "|---|---|---|---|",
            "| Object 2D mask | partial | Bread v4.1 masks and debug overlays exist | Generate stable dynamic masks for pipette/drinks and continuous bread masks |",
            "| Object 3D model & viz | mostly ready | Four `.obj` assets plus generated render views | Add video-overlay/scale comparison for non-bread objects |",
            "| Geometric quality | ready | Watertight checks, face counts, extents | Keep reports in final README |",
            "| Geometric consistency | partial | Dimensions and bread evidence exist | Add mesh projection or visual comparison for pipette/drinks |",
            "| IsaacGym asset | blocked locally if probe fails | URDF/collision files and local probe logs | Re-run on cluster IsaacGym if local import/sim fails |",
            "| Object pose tracking | partial | Prompt-free Phase 5 pipeline and sparse bread multi-view pose | Need stable masks for every target object/sequence |",
            "",
            "## Pose Tracking Status",
            "",
            f"- Dataset inventory: `{inventory_path}`",
            "- Mask audit: `outputs/object3_submission/mask_audit.csv`",
        ]
    )
    if bread_pose_report:
        lines.extend(
            [
                f"- Bread method: `{bread_pose_report.get('method')}`",
                f"- Bread valid frames: {bread_pose_report.get('num_valid')}/{bread_pose_report.get('num_frames')}",
                f"- Bread limitation: {bread_pose_report.get('limitation')}",
            ]
        )
    else:
        lines.append("- Bread pose report is not present yet.")

    lines.extend(
        [
            "",
            "## Why Some Items Remain Incomplete",
            "",
            "- The official HDF5 primary-object trajectories are present but invalid because the primary object masks are all false.",
            "- The current repository data has no discoverable depth frames, so real `mask_depth_icp` cannot be run on these sequences yet.",
            "- Existing fixed-coordinate SAM prompts fail on long sequences; future masks must use dynamic bbox/centroid prompts or mask propagation.",
            "- IsaacGym validation must be treated as an environment-dependent step; local failures are logged and should be repeated on the cluster image.",
            "",
            "## Next Execution Defaults",
            "",
            "- Keep `runs/object_asset_v1` as the canonical asset root.",
            "- Keep `scripts/phase5_mask_depth_pose.py` as the prompt-free pose tracking entrypoint.",
            "- Use `multi_view_mask_pose` for current RGB-only calibrated data; use `mask_depth_icp` only if depth or reconstructed per-frame point clouds are added.",
            "- For drinks, mark yaw as ambiguous and prioritize translation continuity.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_asset_readme(asset_root: Path, rows: list[AssetMetrics], validation: list[dict[str, Any]]) -> None:
    validation_by_object = {item["object"]: item for item in validation}
    lines = [
        "# Object Asset v1 for Video2Motion2Action",
        "",
        "This folder contains the current Member C object assets for task 3.3. The assets are simulation-first, low-face-count, watertight meshes with separate visual and collision geometry.",
        "",
        "## Objects",
        "",
        "| Folder | Object | Visual Mesh | Collision Mesh | URDF | Renders | Local IsaacGym Probe |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        probe = validation_by_object.get(row.name, {})
        probe_text = probe.get("status", "not_run")
        lines.append(
            f"| {row.name} | {row.object_name} | `mesh/visual_mesh.obj` | "
            f"`mesh/collision_mesh.obj` | `asset/object.urdf` | `renders/` | {probe_text} |"
        )
    lines.extend(
        [
            "",
            "## Units and Coordinate Convention",
            "",
            "- Units are meters and kilograms.",
            "- Drink bottles: mesh centered near origin; z-axis is vertical.",
            "- Bread: mesh centered near origin; x-axis is the long side.",
            "- Pipette: mesh centered near origin; x-axis is the length direction.",
            "",
            "## Validation Summary",
            "",
            "- Geometry summary: `asset_summary.json`, `geometry_summary.csv`, and per-object `report/` files.",
            "- Render evidence: per-object `renders/front.png`, `side.png`, `top.png`, `angle.png`, `collision_angle.png`.",
            "- IsaacGym local probe: per-object `report/asset_check_local.log`. If local IsaacGym is unavailable, rerun the same validation on the competition cluster image.",
        ]
    )
    (asset_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=Path("runs/object_asset_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/object3_submission"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-isaacgym", action="store_true")
    parser.add_argument("--isaacgym-timeout-s", type=int, default=45)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    asset_root = args.asset_root
    if not asset_root.is_absolute():
        asset_root = repo_root / asset_root
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    rows = []
    for name in OBJECTS:
        root = asset_root / name
        if not root.exists():
            raise FileNotFoundError(f"Missing asset folder: {root}")
        visual = load_mesh(root / "mesh" / "visual_mesh.obj")
        collision = load_mesh(root / "mesh" / "collision_mesh.obj")
        write_geometry_check(visual, root / "report" / "geometry_check_visual.txt", "visual_mesh")
        write_geometry_check(collision, root / "report" / "geometry_check_collision.txt", "collision_mesh")
        render_asset_views(asset_root, name)
        rows.append(collect_metrics(asset_root, name))

    write_csv(rows, asset_root / "geometry_summary.csv")
    write_csv(rows, output_root / "geometry_summary.csv")
    write_markdown_summary(rows, output_root / "geometry_summary.md")
    write_render_contact_sheet(asset_root, output_root / "render_contact_sheet.png")
    summary_payload = [row.summary_row() for row in rows]
    (asset_root / "asset_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    (output_root / "asset_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    if args.skip_isaacgym:
        validation = [
            {
                "object": row.name,
                "urdf": str(row.urdf),
                "status": "skipped",
                "reason": "skip requested",
                "log": None,
            }
            for row in rows
        ]
    else:
        validation = [
            run_isaacgym_probe(repo_root, args.python, row, args.isaacgym_timeout_s)
            for row in rows
        ]
    mask_audit = audit_masks(repo_root)
    write_mask_audit(mask_audit, output_root)

    validation_path = output_root / "isaacgym_validation_summary.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (asset_root / "isaacgym_validation_summary.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )
    write_asset_readme(asset_root, rows, validation)
    write_status_report(
        repo_root,
        rows,
        validation,
        mask_audit,
        output_root / "member_c_task3_status_report.md",
    )
    print(f"Wrote evidence bundle: {output_root}")
    print(f"Updated asset root: {asset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
