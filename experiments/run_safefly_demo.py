#!/usr/bin/env python3
"""
Run SafeFly experiment — 4 conditions × 3 scenes × N episodes.

This is the main experiment runner for collecting hackathon metrics.
Run after the minimal demo is working:
    python demos/safe_fly_minimal.py --no-show    # verify first
    python experiments/run_safefly_demo.py          # full experiment

Output structure:
    experiments/YYYY-MM-DD_safefly_run/
    ├── README.md
    ├── config.yaml
    ├── command.sh
    ├── metrics.json
    ├── logs.txt
    ├── figures/
    │   ├── trajectory_comparison.png
    │   ├── bar_chart_collision.png
    │   └── bar_chart_success.png
    └── outputs/
        ├── results.json
        └── result_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demos.safe_fly_minimal import (
    DemoConfig,
    EpisodeResult,
    GridWorld,
    MockVLA2D,
    run_episode,
    visualize_results,
)


# ============================================================================
# Experiment configuration
# ============================================================================


def get_experiment_config() -> dict:
    """Define the full experiment matrix."""
    return {
        "scenes": [
            {
                "name": "open_hall",
                "start": (2.0, 4.0),
                "goal": (17.5, 7.0),
                "instruction": "fly to the exit on the far side",
                "description": "Open hall with a central wall — direct path blocked",
            },
            {
                "name": "narrow_gap",
                "start": (2.0, 2.0),
                "goal": (17.0, 12.0),
                "instruction": "fly through the gap to the upper right",
                "description": "Must navigate through narrow gap between wall and boundary",
            },
            {
                "name": "scattered_pillars",
                "start": (1.5, 7.5),
                "goal": (18.0, 7.5),
                "instruction": "fly straight across to the far marker",
                "description": "Scattered pillars near the direct path",
            },
        ],
        "conditions": ["vla_only", "safefly"],
        "episodes_per_condition": 10,
        "noise_levels": [0.3, 0.6, 1.0],  # Test different VLA noise levels
    }


# ============================================================================
# Main experiment
# ============================================================================


def run_full_experiment(
    output_dir: Path,
    config: DemoConfig,
    save_figures: bool = True,
) -> dict:
    """Run the full experiment matrix and save results."""
    exp_config = get_experiment_config()
    all_results = {}

    start_time = time.time()

    for scene in exp_config["scenes"]:
        scene_name = scene["name"]
        print(f"\n{'='*60}")
        print(f"  Scene: {scene_name} — {scene['description']}")
        print(f"  Start: {scene['start']} → Goal: {scene['goal']}")
        print(f"{'='*60}")

        # Update config for this scene
        config.start_pos = scene["start"]
        goal_pos = scene["goal"]

        # Build world for this scene
        world = GridWorld(config)

        for noise_lvl in exp_config["noise_levels"]:
            config.vla_noise_std = noise_lvl
            vla = MockVLA2D(config, goal_pos)

            for condition in exp_config["conditions"]:
                key = f"{scene_name}/noise_{noise_lvl}/{condition}"
                results_for_condition = []

                print(f"\n  [{key}] Running {exp_config['episodes_per_condition']} episodes...")

                for ep in range(exp_config["episodes_per_condition"]):
                    result = run_episode(
                        condition=condition,
                        world=world,
                        vla=vla,
                        config=config,
                        instruction=scene["instruction"],
                        goal_pos=goal_pos,
                    )
                    results_for_condition.append(result)

                    status = "✓" if result.success else "✗"
                    print(
                        f"    ep {ep+1:2d}/{exp_config['episodes_per_condition']} "
                        f"{status} "
                        f"collision={'YES' if result.collision else 'NO'} "
                        f"steps={result.steps:3d} "
                        f"interventions={len(result.interventions):2d}"
                    )

                # Aggregate
                successes = sum(1 for r in results_for_condition if r.success)
                collisions = sum(1 for r in results_for_condition if r.collision)
                avg_steps = float(np.mean([r.steps for r in results_for_condition]))
                avg_path = float(
                    np.mean([r.path_length for r in results_for_condition])
                )
                avg_min_dist = float(
                    np.mean(
                        [r.min_obstacle_distance for r in results_for_condition]
                    )
                )
                avg_interventions = float(
                    np.mean(
                        [len(r.interventions) for r in results_for_condition]
                    )
                )

                all_results[key] = {
                    "scene": scene_name,
                    "noise_level": noise_lvl,
                    "condition": condition,
                    "num_episodes": int(exp_config["episodes_per_condition"]),
                    "success_rate": float(successes / exp_config["episodes_per_condition"]),
                    "collision_rate": float(collisions / exp_config["episodes_per_condition"]),
                    "avg_steps": float(avg_steps),
                    "avg_path_length_m": float(avg_path),
                    "avg_min_obstacle_distance_m": float(avg_min_dist),
                    "avg_interventions_per_episode": float(avg_interventions),
                    "results": [
                        {
                            "success": bool(r.success),
                            "collision": bool(r.collision),
                            "steps": int(r.steps),
                            "path_length": float(r.path_length),
                            "min_obstacle_dist": float(r.min_obstacle_distance),
                            "interventions": int(len(r.interventions)),
                        }
                        for r in results_for_condition
                    ],
                }

                print(
                    f"    → success_rate={successes}/{exp_config['episodes_per_condition']} "
                    f"({100*successes/exp_config['episodes_per_condition']:.0f}%) "
                    f"collision_rate={collisions}/{exp_config['episodes_per_condition']} "
                    f"({100*collisions/exp_config['episodes_per_condition']:.0f}%)"
                )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Experiment complete in {elapsed:.1f}s")
    print(f"{'='*60}")

    # Save results
    results_path = output_dir / "outputs" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")

    # Save summary CSV
    csv_path = output_dir / "outputs" / "result_summary.csv"
    with open(csv_path, "w") as f:
        headers = [
            "scene",
            "noise_level",
            "condition",
            "num_episodes",
            "success_rate",
            "collision_rate",
            "avg_steps",
            "avg_path_length_m",
            "avg_min_obstacle_distance_m",
            "avg_interventions",
        ]
        f.write(",".join(headers) + "\n")
        for key, data in all_results.items():
            row = [str(data.get(h, "")) for h in headers]
            f.write(",".join(row) + "\n")
    print(f"✓ Summary CSV saved to: {csv_path}")

    # Save metrics.json (hackathon format)
    metrics_path = output_dir / "metrics.json"
    metrics = {
        "status": "success",
        "runtime_seconds": elapsed,
        "seed": 42,
        "metrics": {
            "total_scenes": len(exp_config["scenes"]),
            "total_conditions": len(exp_config["conditions"]),
            "total_noise_levels": len(exp_config["noise_levels"]),
            "episodes_per_condition": exp_config["episodes_per_condition"],
            "total_episodes": len(all_results)
            * exp_config["episodes_per_condition"],
            "overall_best_success_rate": max(
                d["success_rate"] for d in all_results.values()
            ),
            "overall_worst_collision_rate": max(
                d["collision_rate"] for d in all_results.values()
            ),
        },
        "artifacts": [
            str(results_path),
            str(csv_path),
        ],
        "failure_reason": None,
        "next_step": "Generate figures and prepare答辩 slides",
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")

    # Generate figures
    if save_figures:
        print("\nGenerating figures...")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Bar chart: success rate by condition and noise
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            for idx, metric in enumerate(["success_rate", "collision_rate"]):
                ax = axes[idx]
                scenes = exp_config["scenes"]
                conditions = exp_config["conditions"]
                x = np.arange(len(scenes))
                width = 0.2

                for ci, cond in enumerate(conditions):
                    values = []
                    for scene in scenes:
                        key = f"{scene['name']}/noise_0.6/{cond}"
                        values.append(
                            all_results.get(key, {}).get(metric, 0.0) * 100
                        )
                    ax.bar(
                        x + ci * width,
                        values,
                        width,
                        label=cond.replace("_", " ").title(),
                    )

                ax.set_xticks(x + width / 2)
                ax.set_xticklabels([s["name"].replace("_", " ") for s in scenes])
                ax.set_ylabel(f"{'Success' if metric == 'success_rate' else 'Collision'} Rate (%)")
                ax.set_title(
                    f"{'Success' if metric == 'success_rate' else 'Collision'} Rate by Scene"
                )
                ax.legend()
                ax.grid(axis="y", alpha=0.3)

            fig_path = output_dir / "figures" / "bar_charts.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✓ Bar chart saved to: {fig_path}")

        except Exception as e:
            print(f"⚠ Figure generation failed: {e}")

    return all_results


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SafeFly Experiment Runner — 4 conditions × 3 scenes"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/YYYY-MM-DD_safefly_run/)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 1 scene × 3 episodes (for smoke testing)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        output_dir = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / f"{timestamp}_safefly_run"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "outputs").mkdir(exist_ok=True)

    # Write config
    config = DemoConfig()
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(f"# SafeFly Experiment Config\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"task: safe_navigation\n")
        f.write(f"model_or_algorithm: directional_sampling_safety_wrapper\n")
        f.write(f"simulator: 2d_grid_world\n")
        f.write(f"dataset_or_asset: synthetic_obstacles\n")
        f.write(f"seed: 42\n")
        f.write(f"hardware: cpu\n")
        f.write(f"notes: Full experiment with 3 scenes × 3 noise levels × 2 conditions\n")

    # Write command.sh
    cmd_path = output_dir / "command.sh"
    with open(cmd_path, "w") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"# SafeFly Experiment Command\n")
        f.write(f"cd {Path(__file__).resolve().parents[1]}\n")
        f.write(f"python experiments/run_safefly_demo.py {'--quick' if args.quick else ''}\n")

    # Write README
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"# {output_dir.name}\n\n")
        f.write(f"## Goal\n")
        f.write(f"Evaluate SafeFly safety wrapper across 3 scenes with varying VLA noise levels.\n\n")
        f.write(f"## Setup\n")
        f.write(f"- Time: {datetime.now().isoformat()}\n")
        f.write(f"- Python: {sys.version.split()[0]}\n")
        f.write(f"- Git commit: N/A\n")
        f.write(f"- Hardware: CPU\n")
        f.write(f"- Seed: 42\n\n")

    # Run experiment
    if args.quick:
        print("=== QUICK MODE: 1 scene × 2 episodes ===")
        exp_config = get_experiment_config()
        exp_config["scenes"] = exp_config["scenes"][:1]
        exp_config["episodes_per_condition"] = 2
        exp_config["noise_levels"] = [0.6]
    else:
        exp_config = get_experiment_config()

    results = run_full_experiment(
        output_dir=output_dir,
        config=config,
        save_figures=not args.no_figures,
    )

    # Update README with results
    with open(readme_path, "a") as f:
        f.write(f"## Results\n")
        f.write(f"- Status: success\n")
        f.write(f"- Runtime: see metrics.json\n")
        f.write(f"- Output files: outputs/results.json, outputs/result_summary.csv\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
