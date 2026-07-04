#!/usr/bin/env python3
"""Phase 5: mask/depth driven object pose tracking.

Examples:
  python scripts/phase5_mask_depth_pose.py --inventory-only \
    --data-root data/human_demo --output-root outputs

  python scripts/phase5_mask_depth_pose.py --sequence weigh_bread__2026_0701_0044_30 \
    --camera camera_top --data-root data/human_demo --output-root outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.object_recon.pose_tracking import (  # noqa: E402
    TrackingConfig,
    extract_gt_trajectory,
    inventory_dataset,
    track_multi_view_mask_sequence,
    track_image_plane_sequence,
    track_mask_depth_icp_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt-free object pose tracking from GT, mask+depth, or mask-only data."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/human_demo"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--camera", default="camera_top")
    parser.add_argument("--sequence", help="Process only one sequence directory name.")
    parser.add_argument("--object-name", help="Override object name for the selected sequence.")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--force-mode",
        choices=(
            "use_gt_pose",
            "mask_depth_icp",
            "mask_depth_pose",
            "multi_view_mask_pose",
            "image_plane_pose",
            "need_segmentation_or_manual",
        ),
        help="Override the inventory-recommended mode.",
    )
    parser.add_argument("--mesh-path", type=Path, help="Canonical mesh/point-cloud path for ICP.")
    parser.add_argument("--no-icp", action="store_true", help="Disable ICP even if a mesh is available.")
    parser.add_argument(
        "--world-pose",
        action="store_true",
        help="Export T_world_object for mask_depth_icp using camera extrinsics.",
    )
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--min-mask-area", type=int, default=50)
    parser.add_argument("--min-depth-valid-ratio", type=float, default=0.2)
    parser.add_argument("--max-centroid-jump-m", type=float, default=0.12)
    parser.add_argument("--max-theta-jump-deg", type=float, default=45.0)
    parser.add_argument("--max-pose-jump-m", type=float, default=0.18)
    parser.add_argument("--min-point-count", type=int, default=64)
    parser.add_argument("--min-icp-fitness", type=float, default=0.15)
    parser.add_argument("--max-icp-rmse", type=float, default=0.04)
    parser.add_argument("--max-image-mask-area-ratio", type=float, default=0.10)
    parser.add_argument("--min-multiview-views", type=int, default=2)
    parser.add_argument("--plane-axis", type=int, default=1)
    parser.add_argument("--plane-value", type=float, default=0.02)
    parser.add_argument("--yaw-search-degrees", type=float, default=180.0)
    parser.add_argument("--yaw-search-steps", type=int, default=37)
    parser.add_argument("--min-silhouette-score", type=float, default=0.02)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrackingConfig:
    return TrackingConfig(
        camera=args.camera,
        depth_scale=args.depth_scale,
        min_mask_area=args.min_mask_area,
        min_depth_valid_ratio=args.min_depth_valid_ratio,
        max_centroid_jump_m=args.max_centroid_jump_m,
        max_theta_jump_deg=args.max_theta_jump_deg,
        max_pose_jump_m=args.max_pose_jump_m,
        min_point_count=args.min_point_count,
        min_icp_fitness=args.min_icp_fitness,
        max_icp_rmse=args.max_icp_rmse,
        max_image_mask_area_ratio=args.max_image_mask_area_ratio,
        min_multiview_views=args.min_multiview_views,
        plane_axis=args.plane_axis,
        plane_value=args.plane_value,
        yaw_search_degrees=args.yaw_search_degrees,
        yaw_search_steps=args.yaw_search_steps,
        min_silhouette_score=args.min_silhouette_score,
        output_world_pose=args.world_pose,
        enable_icp=not args.no_icp,
    )


def main() -> int:
    args = parse_args()
    inventory_csv = args.output_root / "dataset_inventory.csv"
    rows = inventory_dataset(
        args.data_root,
        output_csv=inventory_csv,
        experiments_root=args.experiments_root,
        camera=args.camera,
    )
    print(f"[inventory] wrote {inventory_csv} ({len(rows)} sequences)")

    if args.inventory_only:
        for row in rows:
            print(
                f"  {row.object_name:10s} {row.sequence_name:45s} "
                f"frames={row.num_frames:4d} mode={row.recommended_tracking_mode}"
            )
        return 0

    selected = rows
    if args.sequence:
        selected = [row for row in rows if row.sequence_name == args.sequence]
        if not selected:
            print(f"ERROR: sequence not found under {args.data_root}: {args.sequence}", file=sys.stderr)
            return 2

    config = build_config(args)
    failures = 0
    for row in selected:
        object_name = args.object_name or row.object_name
        mode = args.force_mode or row.recommended_tracking_mode
        if mode == "mask_depth_pose":
            mode = "mask_depth_icp"
        out_dir = args.output_root / "mask_pose" / object_name / row.sequence_name
        print(f"[track] {row.sequence_name} object={object_name} mode={mode}")
        try:
            if mode == "use_gt_pose":
                result = extract_gt_trajectory(row.sequence_dir, object_name, out_dir)
            elif mode == "mask_depth_icp":
                result = track_mask_depth_icp_sequence(
                    sequence_dir=row.sequence_dir,
                    output_dir=out_dir,
                    object_name=object_name,
                    camera=args.camera,
                    mask_paths=row.mask_paths,
                    depth_paths=row.depth_paths,
                    mesh_path=args.mesh_path,
                    config=config,
                )
            elif mode == "multi_view_mask_pose":
                result = track_multi_view_mask_sequence(
                    sequence_dir=row.sequence_dir,
                    output_dir=out_dir,
                    object_name=object_name,
                    mesh_path=args.mesh_path,
                    config=config,
                    experiments_root=args.experiments_root,
                )
            elif mode == "image_plane_pose":
                result = track_image_plane_sequence(
                    sequence_dir=row.sequence_dir,
                    output_dir=out_dir,
                    object_name=object_name,
                    camera=args.camera,
                    mask_paths=row.mask_paths,
                    config=config,
                )
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "trajectory_quality_report.json").write_text(
                    "{\n"
                    f'  "object_name": "{object_name}",\n'
                    f'  "sequence_name": "{row.sequence_name}",\n'
                    '  "method": "need_segmentation_or_manual",\n'
                    '  "limitation": "No valid GT pose, mask/depth pair, or mask-only track was discovered."\n'
                    "}\n"
                )
                print(f"  skipped: need segmentation/manual masks -> {out_dir}")
                continue
            print(
                f"  wrote {result['json_path']} "
                f"valid={result['num_valid']}/{result['num_frames']}"
            )
        except Exception as exc:
            failures += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
