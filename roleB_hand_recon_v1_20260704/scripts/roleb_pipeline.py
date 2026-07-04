#!/usr/bin/env python3
"""Role B hand trajectory pipeline for HO-Tracker human_demo sessions.

The script intentionally writes only inside the roleB workspace. It reuses the
existing lightweight MediaPipe extractor and smoothing scripts, then adds QA
metrics, proxy masks, thumbnails, and tentative multi-view triangulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import pickle
import subprocess
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import savgol_filter

try:
    import h5py
except ModuleNotFoundError:  # Keep the MediaPipe-only venv usable.
    h5py = None


CAMERAS = ("camera_side_1", "camera_side_2", "camera_top")
HAND_BONES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


@dataclass
class Paths:
    roleb: pathlib.Path
    source_scripts: pathlib.Path

    @property
    def data(self) -> pathlib.Path:
        return self.roleb / "data"

    @property
    def outputs(self) -> pathlib.Path:
        return self.roleb / "outputs"

    @property
    def log(self) -> pathlib.Path:
        return self.roleb / "ROLE_B_LOG.md"

    @property
    def summary_csv(self) -> pathlib.Path:
        return self.roleb / "SESSION_METRICS.csv"


def sh(cmd: list[str], *, cwd: pathlib.Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def append_log(paths: Paths, title: str, lines: list[str]) -> None:
    paths.log.parent.mkdir(parents=True, exist_ok=True)
    with paths.log.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_iso()} - {title}\n\n")
        for line in lines:
            f.write(f"- {line}\n")


def now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def session_path(session: str) -> str:
    return f"human_demo/{session}"


def hf_resolve_url(rel: str) -> str:
    return "https://hf-mirror.com/datasets/kelvin34501/HO-Tracker-Challenge/resolve/main/" + rel


def download_session(paths: Paths, session: str) -> None:
    rels: list[str] = []
    for cam in CAMERAS:
        rels.append(f"{session_path(session)}/video/{cam}.mkv")
        rels.append(f"{session_path(session)}/camera_calib/{cam}/cam_intr.pkl")
        rels.append(f"{session_path(session)}/camera_calib/{cam}/cam_extr.pkl")
    rels.append(f"{session_path(session)}/pose_3d.hdf5")

    append_log(
        paths,
        f"Download Session {session}",
        [
            "Downloading only this human_demo session via hf-mirror direct resolve URLs.",
            "Using resume and retry; files are stored under roleB/data, not the main HO-Tracker repo.",
        ],
    )

    for rel in rels:
        out = paths.data / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        is_tiny_payload = rel.endswith(".pkl") or rel.endswith(".hdf5")
        if out.exists() and (out.stat().st_size > 1024 or is_tiny_payload):
            print(f"skip existing {out}")
            continue
        cmd = ["curl", "-L", "--retry", "3", "--retry-delay", "3", "--connect-timeout", "20"]
        if not is_tiny_payload:
            cmd.extend(["-C", "-"])
        cmd.extend(["-o", str(out), hf_resolve_url(rel)])
        sh(cmd)


def inspect_session(paths: Paths, session: str) -> dict:
    base = paths.data / session_path(session)
    info: dict[str, object] = {"session": session, "cameras": {}}
    for cam in CAMERAS:
        video = base / "video" / f"{cam}.mkv"
        cap = cv2.VideoCapture(str(video))
        cam_info = {
            "opened": bool(cap.isOpened()),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
        cap.release()
        info["cameras"][cam] = cam_info

    pose_path = base / "pose_3d.hdf5"
    if h5py is not None and pose_path.exists() and pose_path.stat().st_size > 1024:
        with h5py.File(pose_path, "r") as f:
            info["task_name"] = read_scalar(f["meta/task_name"])
            info["side"] = read_scalar(f["meta/side"])
            if "obj/obj_id" in f:
                info["obj_id"] = read_scalar(f["obj/obj_id"])
            if "obj/pose/obj_transf" in f:
                info["obj_pose_shape"] = list(f["obj/pose/obj_transf"].shape)
    elif pose_path.exists():
        info["pose_3d_hdf5"] = str(pose_path)
        info["pose_3d_note"] = "h5py unavailable in this venv; skipped HDF5 metadata parsing"
    return info


def read_scalar(ds) -> str:
    value = ds[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def run_mediapipe(paths: Paths, session: str, max_frames: int, resize_width: int) -> None:
    out_dir = paths.outputs / session
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        video = paths.data / session_path(session) / "video" / f"{cam}.mkv"
        raw = out_dir / f"{cam}_hand_traj_raw.npz"
        overlay = out_dir / f"{cam}_hand_overlay.mp4"
        smooth = out_dir / f"{cam}_hand_traj_smooth.npz"
        if not raw.exists():
            sh(
                [
                    sys.executable,
                    str(paths.source_scripts / "extract_hand_traj.py"),
                    "--video",
                    str(video),
                    "--out",
                    str(raw),
                    "--overlay",
                    str(overlay),
                    "--max-frames",
                    str(max_frames),
                    "--resize-width",
                    str(resize_width),
                    "--min-detection-confidence",
                    "0.35",
                    "--min-tracking-confidence",
                    "0.35",
                ]
            )
        if not smooth.exists():
            sh(
                [
                    sys.executable,
                    str(paths.source_scripts / "smooth_hand_traj.py"),
                    "--infile",
                    str(raw),
                    "--out",
                    str(smooth),
                    "--scale",
                    "2.5",
                ]
            )


def interp_nan(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    t = np.arange(out.shape[0])
    for j in range(out.shape[1]):
        for c in range(out.shape[2]):
            y = out[:, j, c]
            good = np.isfinite(y)
            if good.sum() == 0:
                continue
            if good.sum() == 1:
                y[~good] = y[good][0]
            else:
                y[~good] = np.interp(t[~good], t[good], y[good])
            out[:, j, c] = y
    return out


def smooth_xyz(values: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    if values.shape[0] < 5:
        return values
    win = min(window, values.shape[0] if values.shape[0] % 2 else values.shape[0] - 1)
    if win <= polyorder:
        return values
    return savgol_filter(values, window_length=win, polyorder=polyorder, axis=0).astype(np.float32)


def trajectory_metrics(out_dir: pathlib.Path, cam: str) -> dict:
    data = np.load(out_dir / f"{cam}_hand_traj_raw.npz", allow_pickle=True)
    image = data["image_joints"].astype(np.float32)
    world = data["world_joints"].astype(np.float32)
    valid = data["valid"].astype(bool)
    n = len(valid)
    valid_ratio = float(valid.sum() / max(n, 1))

    uv = image[:, :, :2]
    wrist = uv[:, 0, :]
    good_wrist = valid & np.isfinite(wrist).all(axis=1)
    if good_wrist.sum() > 2:
        diffs = np.linalg.norm(np.diff(wrist[good_wrist], axis=0), axis=1)
        wrist_jitter = float(np.nanpercentile(diffs, 90))
    else:
        wrist_jitter = float("nan")

    areas = []
    for t in np.where(valid)[0]:
        pts = uv[t]
        finite = np.isfinite(pts).all(axis=1)
        if finite.sum() >= 3:
            mins = pts[finite].min(axis=0)
            maxs = pts[finite].max(axis=0)
            areas.append(float(np.prod(maxs - mins)))
    median_bbox_area = float(np.median(areas)) if areas else float("nan")

    lengths = []
    for a, b in HAND_BONES[:20]:
        d = np.linalg.norm(world[:, a, :] - world[:, b, :], axis=1)
        d = d[valid & np.isfinite(d)]
        if len(d) > 3 and np.nanmean(d) > 1e-6:
            lengths.append(float(np.nanstd(d) / np.nanmean(d)))
    bone_cv = float(np.nanmedian(lengths)) if lengths else float("nan")
    score = valid_ratio
    if np.isfinite(wrist_jitter):
        score -= 0.35 * min(wrist_jitter, 0.2)
    if np.isfinite(bone_cv):
        score -= 0.40 * min(bone_cv, 0.5)
    return {
        "valid_frames": int(valid.sum()),
        "total_frames": int(n),
        "valid_ratio": valid_ratio,
        "wrist_jitter_p90_uv": wrist_jitter,
        "median_bbox_area_uv": median_bbox_area,
        "bone_length_cv_median": bone_cv,
        "view_score": float(score),
    }


def make_thumbnails(out_dir: pathlib.Path) -> None:
    thumb_dir = out_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = (0, 80, 160, 234)
    for mp4 in sorted(out_dir.glob("*_hand_overlay.mp4")):
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for idx in sample_indices:
            if idx >= n:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(thumb_dir / f"{mp4.stem}_{idx:04d}.jpg"), frame)
        cap.release()


def make_proxy_masks(paths: Paths, session: str, cams: tuple[str, ...]) -> None:
    out_dir = paths.outputs / session
    base = paths.data / session_path(session)
    mask_dir = out_dir / "mask_proxy"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for cam in cams:
        raw = np.load(out_dir / f"{cam}_hand_traj_raw.npz", allow_pickle=True)
        image = raw["image_joints"].astype(np.float32)
        valid = raw["valid"].astype(bool)
        cap = cv2.VideoCapture(str(base / "video" / f"{cam}.mkv"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        writer = None
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame_idx >= len(valid):
                break
            if frame.shape[1] > 640:
                scale = 640 / frame.shape[1]
                frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
            h, w = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_dir / f"{cam}_hand_mask_proxy_overlay.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
            mask = np.zeros((h, w), np.uint8)
            if valid[frame_idx]:
                pts = np.stack([image[frame_idx, :, 0] * w, image[frame_idx, :, 1] * h], axis=1)
                finite = np.isfinite(pts).all(axis=1)
                if finite.sum() >= 3:
                    hull = cv2.convexHull(pts[finite].astype(np.int32))
                    cv2.fillConvexPoly(mask, hull, 255)
                    k = max(9, int(0.035 * w) // 2 * 2 + 1)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
            overlay = frame.copy()
            color = np.zeros_like(frame)
            color[:, :, 1] = 180
            color[:, :, 2] = 255
            overlay = np.where(mask[..., None] > 0, (0.55 * overlay + 0.45 * color).astype(np.uint8), overlay)
            if frame_idx in {0, 80, 160, 234}:
                cv2.imwrite(str(mask_dir / f"{cam}_mask_{frame_idx:04d}.png"), mask)
                cv2.imwrite(str(mask_dir / f"{cam}_mask_overlay_{frame_idx:04d}.jpg"), overlay)
            writer.write(overlay)
            frame_idx += 1
        cap.release()
        if writer is not None:
            writer.release()


def load_camera(base: pathlib.Path, cam: str, mode: str) -> np.ndarray:
    with open(base / "camera_calib" / cam / "cam_intr.pkl", "rb") as f:
        k = pickle.load(f).astype(np.float64)
    with open(base / "camera_calib" / cam / "cam_extr.pkl", "rb") as f:
        e = pickle.load(f).astype(np.float64)
    if mode == "world_to_camera":
        ext = e[:3, :]
    elif mode == "camera_to_world":
        ext = np.linalg.inv(e)[:3, :]
    else:
        raise ValueError(mode)
    return k @ ext


def triangulate_point(projections: list[np.ndarray], points: list[np.ndarray]) -> np.ndarray:
    a = []
    for p, uv in zip(projections, points):
        x, y = uv
        a.append(x * p[2] - p[0])
        a.append(y * p[2] - p[1])
    _, _, vh = np.linalg.svd(np.asarray(a))
    xh = vh[-1]
    if abs(xh[3]) < 1e-9:
        return np.full(3, np.nan)
    return xh[:3] / xh[3]


def reproject(p: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    h = p @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
    if abs(h[2]) < 1e-9:
        return np.array([np.nan, np.nan])
    return h[:2] / h[2]


def triangulate_session(paths: Paths, session: str) -> dict:
    out_dir = paths.outputs / session
    base = paths.data / session_path(session)
    raws = {cam: np.load(out_dir / f"{cam}_hand_traj_raw.npz", allow_pickle=True) for cam in CAMERAS}
    n = min(len(raws[cam]["valid"]) for cam in CAMERAS)
    dims = {}
    for cam in CAMERAS:
        cap = cv2.VideoCapture(str(base / "video" / f"{cam}.mkv"))
        dims[cam] = (
            float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280),
            float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720),
        )
        cap.release()

    best = None
    for mode in ("world_to_camera", "camera_to_world"):
        projections = {cam: load_camera(base, cam, mode) for cam in CAMERAS}
        xyz = np.full((n, 21, 3), np.nan, np.float32)
        reproj_errors = []
        support = np.zeros((n, 21), np.int16)
        for t in range(n):
            for j in range(21):
                ps = []
                uvs = []
                for cam in CAMERAS:
                    if not bool(raws[cam]["valid"][t]):
                        continue
                    uvn = raws[cam]["image_joints"][t, j, :2].astype(np.float64)
                    if not np.isfinite(uvn).all():
                        continue
                    w, h = dims[cam]
                    uv = np.array([uvn[0] * w, uvn[1] * h], np.float64)
                    ps.append(projections[cam])
                    uvs.append(uv)
                if len(ps) >= 2:
                    point = triangulate_point(ps, uvs)
                    xyz[t, j] = point.astype(np.float32)
                    support[t, j] = len(ps)
                    for p, uv in zip(ps, uvs):
                        reproj_errors.append(float(np.linalg.norm(reproject(p, point) - uv)))
        filled = interp_nan(xyz)
        smoothed = smooth_xyz(filled)
        err = float(np.nanmedian(reproj_errors)) if reproj_errors else float("inf")
        valid_points = int(np.isfinite(xyz[:, :, 0]).sum())
        result = {
            "mode": mode,
            "median_reprojection_px": err,
            "valid_points": valid_points,
            "xyz": xyz,
            "smoothed": smoothed,
            "support": support,
        }
        if best is None or result["median_reprojection_px"] < best["median_reprojection_px"]:
            best = result

    assert best is not None
    np.savez_compressed(
        out_dir / "multiview_hand_traj.npz",
        world_joints=best["xyz"],
        world_joints_smooth=best["smoothed"],
        support=best["support"],
        mode=np.array(best["mode"]),
        median_reprojection_px=np.float32(best["median_reprojection_px"]),
    )
    return {k: v for k, v in best.items() if k not in {"xyz", "smoothed", "support"}}


def update_summary(paths: Paths, row: dict) -> None:
    paths.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if paths.summary_csv.exists():
        with paths.summary_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("session") != row.get("session")]
    rows.append({k: "" if v is None else v for k, v in row.items()})
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with paths.summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_session(paths: Paths, session: str, max_frames: int, resize_width: int) -> dict:
    info = inspect_session(paths, session)
    run_mediapipe(paths, session, max_frames, resize_width)
    out_dir = paths.outputs / session
    metrics = {cam: trajectory_metrics(out_dir, cam) for cam in CAMERAS}
    best_cam = max(metrics, key=lambda c: metrics[c]["view_score"])
    make_thumbnails(out_dir)
    make_proxy_masks(paths, session, (best_cam, "camera_top") if best_cam != "camera_top" else ("camera_top",))
    tri = triangulate_session(paths, session)

    summary = {
        "session": session,
        "task_name": info.get("task_name"),
        "side": info.get("side"),
        "obj_id": info.get("obj_id"),
        "best_cam": best_cam,
        "triangulation_mode": tri["mode"],
        "triangulation_median_reprojection_px": tri["median_reprojection_px"],
        "triangulated_valid_points": tri["valid_points"],
    }
    for cam, m in metrics.items():
        for k, v in m.items():
            summary[f"{cam}_{k}"] = v
    (out_dir / "metrics.json").write_text(json.dumps({"info": info, "metrics": metrics, "best_cam": best_cam, "triangulation": tri}, indent=2), encoding="utf-8")
    update_summary(paths, summary)
    append_log(
        paths,
        f"Processed Session {session}",
        [
            f"task={summary.get('task_name')} side={summary.get('side')} obj={summary.get('obj_id')}",
            f"best_cam={best_cam}; valid frames: "
            + ", ".join(f"{cam} {metrics[cam]['valid_frames']}/{metrics[cam]['total_frames']}" for cam in CAMERAS),
            f"triangulation={tri['mode']} median_reprojection_px={tri['median_reprojection_px']:.2f} valid_points={tri['valid_points']}",
            f"outputs={out_dir}",
        ],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roleb", default="/data/autovla/ho_tracker_challenge/roleB_hand_recon")
    parser.add_argument("--source-scripts", default="/data/autovla/aILAB_hand_motion/scripts")
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-frames", type=int, default=235)
    parser.add_argument("--resize-width", type=int, default=640)
    args = parser.parse_args()

    paths = Paths(pathlib.Path(args.roleb), pathlib.Path(args.source_scripts))
    append_log(
        paths,
        "Role B Pipeline Run",
        [
            f"sessions={', '.join(args.sessions)}",
            f"download={args.download}",
            f"max_frames={args.max_frames}, resize_width={args.resize_width}",
        ],
    )
    results = []
    for session in args.sessions:
        if args.download:
            download_session(paths, session)
        results.append(process_session(paths, session, args.max_frames, args.resize_width))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
