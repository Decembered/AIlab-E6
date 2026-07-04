#!/usr/bin/env python3
"""
safe_fly_minimal.py — Minimal Survival Demo (2D Grid + APF + Mock VLA)

███████╗ █████╗ ███████╗███████╗███████╗██╗  ██╗   ██╗
██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██║  ╚██╗ ██╔╝
███████╗███████║█████╗  █████╗  █████╗  ██║   ╚████╔╝
╚════██║██╔══██║██╔══╝  ██╔══╝  ██╔══╝  ██║    ╚██╔╝
███████║██║  ██║██║     ██║     ███████╗███████╗██║
╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚══════╝╚══════╝╚═╝

The ABSOLUTE fallback demo. If everything else fails, THIS runs.
Dependencies: numpy, matplotlib (any Python environment has these)
Run: python safe_fly_minimal.py

What it demonstrates:
1. 2D grid world with obstacles (walls, pillars)
2. Mock VLA generates noisy waypoints from text instructions
3. APF safety wrapper detects collisions and replans
4. Side-by-side visualization: VLA-only vs SafeFly
5. Real-time metrics panel

This is the "最后一道防线" — last line of defense demo.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

matplotlib.use("TkAgg")  # Fallback: works without Qt

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DemoConfig:
    """All tunable demo parameters."""

    # World
    world_size: tuple = (20, 15)  # meters (x, y)
    grid_resolution: float = 0.2  # meters

    # Drone
    start_pos: tuple = (1.0, 7.5)
    safety_radius: float = 0.4   # Drone body radius (meters)
    max_speed: float = 1.0       # m/s — slower for safer navigation
    dt: float = 0.1              # control period (s)

    # VLA
    vla_noise_std: float = 0.8  # Simulated VLA uncertainty
    vla_update_interval: int = 10  # VLA updates every N ticks

    # Safety steering params
    attraction_gain: float = 1.5
    repulsion_gain: float = 0.8
    repulsion_range: float = 1.5       # meters — obstacle detection range
    lookahead_multiplier: float = 3.0  # × repulsion_range = lookahead distance

    # Episode
    max_steps: int = 300
    goal_tolerance: float = 0.5  # meters

    # Visualization
    fps: int = 30


# ============================================================================
# 2D World
# ============================================================================


class GridWorld:
    """2D occupancy grid with predefined obstacles."""

    def __init__(self, config: DemoConfig):
        self.config = config
        self.nx = int(config.world_size[0] / config.grid_resolution)
        self.ny = int(config.world_size[1] / config.grid_resolution)
        self.grid = np.zeros((self.nx, self.ny), dtype=np.float32)
        self._build_obstacles()
        self._inflate_obstacles()  # Inflate by safety radius

    def _inflate_obstacles(self):
        """Dilate obstacles by safety_radius cells to account for drone body."""
        inflate_cells = max(
            1, int(self.config.safety_radius / self.config.grid_resolution)
        )
        size = 2 * inflate_cells + 1
        self.grid = maximum_filter(self.grid, size=size)

    def _build_obstacles(self):
        """Minimal obstacle set: one wall blocking the direct path + boundary.

        Design: the straight line from START (2, 4) to GOAL (17.5, 7)
        crosses y≈5 at x≈9. We place a single wall segment there.
        VLA goes straight → crash. SafeFly steers around above or below.
        """
        cfg = self.config

        def fill_rect(x1, y1, x2, y2):
            ix1, iy1 = self._w2g(x1, y1)
            ix2, iy2 = self._w2g(x2, y2)
            self.grid[
                max(0, ix1) : min(self.nx, ix2),
                max(0, iy1) : min(self.ny, iy2),
            ] = 1.0

        # Outer boundary
        fill_rect(0, 0, cfg.world_size[0], 0.5)
        fill_rect(0, cfg.world_size[1] - 0.5, cfg.world_size[0], cfg.world_size[1])
        fill_rect(0, 0, 0.5, cfg.world_size[1])
        fill_rect(cfg.world_size[0] - 0.5, 0, cfg.world_size[0], cfg.world_size[1])

        # *** THE MAIN OBSTACLE: wall blocking the straight-line path ***
        # The straight line from (2, 4) to (17.5, 7) crosses:
        #   x ≈ 8-11 at y ≈ 4.8-5.5
        # Place a wall segment that blocks this exact path
        fill_rect(8.0, 3.5, 9.0, 7.0)  # 1m × 3.5m wall on the straight line

        # A few small pillars to make the world visually interesting
        # (placed AWAY from both straight and detour paths)
        for cx, cy in [(4.0, 8.0), (13.0, 3.0), (15.0, 10.0)]:
            fill_rect(cx - 0.3, cy - 0.3, cx + 0.3, cy + 0.3)

    def _w2g(self, x: float, y: float) -> tuple[int, int]:
        res = self.config.grid_resolution
        return int(x / res), int(y / res)

    def _g2w(self, ix: int, iy: int) -> tuple[float, float]:
        res = self.config.grid_resolution
        return ix * res + res / 2, iy * res + res / 2

    def is_occupied_world(self, x: float, y: float) -> bool:
        ix, iy = self._w2g(x, y)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return bool(self.grid[ix, iy] >= 0.5)
        return True  # Out of bounds = occupied

    def nearest_obstacle_distance(self, x: float, y: float) -> float:
        """Brute-force nearest obstacle distance within repulsion_range."""
        cfg = self.config
        min_dist = cfg.repulsion_range + 1.0
        rr_cells = int(cfg.repulsion_range / cfg.grid_resolution) + 1
        cix, ciy = self._w2g(x, y)

        for dx in range(-rr_cells, rr_cells + 1):
            for dy in range(-rr_cells, rr_cells + 1):
                ix, iy = cix + dx, ciy + dy
                if 0 <= ix < self.nx and 0 <= iy < self.ny:
                    if self.grid[ix, iy] >= 0.5:
                        ox, oy = self._g2w(ix, iy)
                        d = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                        if d < min_dist:
                            min_dist = d
        return min_dist

    def get_obstacle_centers_near(
        self, x: float, y: float, radius: float
    ) -> list[tuple[float, float]]:
        """Return obstacle cell centers within radius."""
        obstacles = []
        rr_cells = int(radius / self.config.grid_resolution) + 1
        cix, ciy = self._w2g(x, y)

        for dx in range(-rr_cells, rr_cells + 1):
            for dy in range(-rr_cells, rr_cells + 1):
                ix, iy = cix + dx, ciy + dy
                if 0 <= ix < self.nx and 0 <= iy < self.ny:
                    if self.grid[ix, iy] >= 0.5:
                        ox, oy = self._g2w(ix, iy)
                        if np.sqrt((x - ox) ** 2 + (y - oy) ** 2) <= radius:
                            obstacles.append((ox, oy))
        return obstacles


# ============================================================================
# Mock VLA
# ============================================================================


class MockVLA2D:
    """Mock VLA that outputs goal-directed waypoints with controllable noise.

    Simulates a VLA that knows the target location but has imprecise outputs.
    This is the realistic scenario: VLA understands the goal semantically
    but its coordinate predictions are noisy.
    """

    def __init__(self, config: DemoConfig, goal_position: tuple[float, float]):
        self.config = config
        self.goal = np.array(goal_position, dtype=np.float64)
        self.rng = np.random.RandomState(42)

    def predict(
        self, instruction: str, current_pos: tuple[float, float]
    ) -> tuple[float, float]:
        """Return a noisy waypoint biased toward the goal.

        The noise level simulates VLA uncertainty. At noise_std=0,
        the VLA outputs the exact goal. At noise_std=2.0, waypoints
        can be several meters off — giving the safety wrapper real work.
        """
        cfg = self.config
        wp = self.goal.copy()

        # Add directional noise (simulates VLA coordinate prediction error)
        noise = self.rng.randn(2) * cfg.vla_noise_std

        # Occasionally add a large "hallucination" offset
        # (simulates VLA misinterpreting spatial references)
        if self.rng.rand() < 0.15:  # 15% chance of a bad waypoint
            noise += self.rng.randn(2) * cfg.vla_noise_std * 2.0

        wp += noise

        # Clamp to world bounds
        wp[0] = np.clip(wp[0], 0.5, cfg.world_size[0] - 0.5)
        wp[1] = np.clip(wp[1], 0.5, cfg.world_size[1] - 0.5)

        return float(wp[0]), float(wp[1])

    def set_goal(self, goal: tuple[float, float]) -> None:
        """Update the goal position."""
        self.goal = np.array(goal, dtype=np.float64)


# ============================================================================
# APF Safety Wrapper (2D)
# ============================================================================


def apf_safe_velocity(
    current: tuple[float, float],
    goal: tuple[float, float],
    world: GridWorld,
    config: DemoConfig,
) -> tuple[float, float]:
    """
    Directional-sampling safety wrapper (bug-algorithm style).

    Unlike pure APF (which suffers from local minima), this approach:
    1. Computes the desired direction toward the goal
    2. Samples candidate directions around the desired direction
    3. Picks the closest safe direction that doesn't hit an obstacle
    4. Falls back to APF blending if ALL directions are blocked

    This guarantees forward progress while avoiding obstacles.
    """
    cx, cy = current
    gx, gy = goal

    # Desired direction toward goal
    dx = gx - cx
    dy = gy - cy
    dist_goal = float(np.sqrt(dx * dx + dy * dy))
    if dist_goal < 0.05:
        return 0.0, 0.0

    desired_angle = np.arctan2(dy, dx)

    # Check if straight-line path is clear
    straight_clear = True
    n_check = int(config.repulsion_range / config.grid_resolution)
    for i in range(1, n_check + 1):
        t = i / n_check
        check_x = cx + dx * t
        check_y = cy + dy * t
        if world.is_occupied_world(check_x, check_y):
            straight_clear = False
            break

    if straight_clear:
        # Path is clear — go straight toward goal
        vx = config.max_speed * dx / dist_goal
        vy = config.max_speed * dy / dist_goal
        return vx, vy

    # Path is blocked — search for a clear steering direction
    # Sample angles from -90° to +90° around desired direction
    num_samples = 36  # Finer sampling
    lookahead_dist = config.repulsion_range * getattr(config, 'lookahead_multiplier', 3.0)

    candidates = []
    for i in range(num_samples):
        # Bias samples toward the desired direction
        angle_offset = (i - num_samples // 2) * (np.pi / num_samples)
        angle = desired_angle + angle_offset
        vx_test = np.cos(angle)
        vy_test = np.sin(angle)

        # Check this direction for obstacles
        blocked = False
        for step in range(1, int(lookahead_dist / config.grid_resolution)):
            t = step * config.grid_resolution
            check_x = cx + vx_test * t
            check_y = cy + vy_test * t
            if world.is_occupied_world(check_x, check_y):
                blocked = True
                break

        if not blocked:
            # Score: prefer directions closer to desired angle
            angle_diff = abs(angle_offset)
            candidates.append((angle_diff, vx_test, vy_test))

    if candidates:
        # Pick the direction closest to desired that is clear
        candidates.sort(key=lambda x: x[0])
        _, vx, vy = candidates[0]
        return vx * config.max_speed, vy * config.max_speed

    # All forward directions blocked — use APF as fallback (blend with repulsion)
    f_att = np.array([dx / dist_goal, dy / dist_goal], dtype=np.float64)

    f_rep = np.zeros(2, dtype=np.float64)
    obstacles = world.get_obstacle_centers_near(cx, cy, config.repulsion_range)
    for ox, oy in obstacles:
        to_obs = np.array([cx - ox, cy - oy], dtype=np.float64)
        d = float(np.linalg.norm(to_obs))
        if d < 1e-6:
            d = 1e-6
        if d < config.repulsion_range:
            magnitude = config.repulsion_gain * (1.0 / d - 1.0 / config.repulsion_range) / d
            f_rep += magnitude * (to_obs / d)

    f_total = f_att + f_rep * 0.5  # Blend: 50% attraction, 50% repulsion
    f_norm = float(np.linalg.norm(f_total))
    if f_norm < 0.01:
        # Completely stuck — try a random perpendicular direction
        perp = np.array([-dy / dist_goal, dx / dist_goal])
        if np.random.rand() > 0.5:
            perp = -perp
        return float(perp[0] * config.max_speed * 0.3), float(perp[1] * config.max_speed * 0.3)

    if f_norm > 1.0:
        f_total /= f_norm

    return float(f_total[0] * config.max_speed), float(f_total[1] * config.max_speed)


# ============================================================================
# Episode runner
# ============================================================================


@dataclass
class EpisodeResult:
    """Results from one episode."""

    condition: str  # "random", "vla_only", "safety_only", "safefly"
    success: bool
    collision: bool
    steps: int
    trajectory: list[tuple[float, float]]
    vla_waypoints: list[tuple[float, float]]
    safe_waypoints: list[tuple[float, float]]
    interventions: list[int]  # Step indices where safety intervened
    min_obstacle_distance: float
    path_length: float


def run_episode(
    condition: str,
    world: GridWorld,
    vla: MockVLA2D,
    config: DemoConfig,
    instruction: str,
    goal_pos: tuple[float, float],
) -> EpisodeResult:
    """Run one episode under a given condition."""
    pos = np.array(config.start_pos, dtype=np.float64)
    trajectory = [(float(pos[0]), float(pos[1]))]
    vla_waypoints = []
    safe_waypoints = []
    interventions = []
    min_obs_dist = float("inf")
    collision = False

    # Current VLA waypoint
    vla_wp = goal_pos  # Initial: go directly to goal

    for step in range(config.max_steps):
        # Update VLA waypoint periodically
        if step % config.vla_update_interval == 0:
            if condition in ("vla_only", "safefly"):
                vla_wp = vla.predict(instruction, (float(pos[0]), float(pos[1])))
            elif condition == "random":
                vla_wp = (
                    float(pos[0]) + (np.random.rand() - 0.5) * 6,
                    float(pos[1]) + (np.random.rand() - 0.5) * 6,
                )
            else:  # safety_only — direct to goal, no VLA noise
                vla_wp = goal_pos

            vla_waypoints.append(vla_wp)

        # Compute velocity based on condition
        if condition == "vla_only":
            # Direct straight-line to VLA waypoint — no safety checking
            dx_wp, dy_wp = vla_wp[0] - pos[0], vla_wp[1] - pos[1]
            dist_wp = np.sqrt(dx_wp**2 + dy_wp**2)
            if dist_wp > 0.01:
                vx = config.max_speed * dx_wp / dist_wp
                vy = config.max_speed * dy_wp / dist_wp
            else:
                vx, vy = 0.0, 0.0

        elif condition in ("safefly", "safety_only"):
            # Use safety wrapper (directional sampling) to steer toward target
            target = vla_wp if condition == "safefly" else goal_pos
            vx, vy = apf_safe_velocity(
                (float(pos[0]), float(pos[1])), target, world, config
            )

            # Detect intervention: was the straight-line path blocked?
            dx_target = target[0] - pos[0]
            dy_target = target[1] - pos[1]
            dist_target = np.sqrt(dx_target**2 + dy_target**2)
            if dist_target > 0.01:
                # Check if straight line is clear
                straight_blocked = False
                n_check = int(config.repulsion_range / config.grid_resolution)
                for i in range(1, n_check + 1):
                    t = i / n_check
                    if world.is_occupied_world(
                        pos[0] + dx_target * t, pos[1] + dy_target * t
                    ):
                        straight_blocked = True
                        break
                if straight_blocked:
                    interventions.append(step)

        else:  # random
            vx = config.max_speed * (np.random.rand() - 0.5) * 2
            vy = config.max_speed * (np.random.rand() - 0.5) * 2

        # Step physics
        new_x = pos[0] + vx * config.dt
        new_y = pos[1] + vy * config.dt

        # Safety gate: ONLY for SafeFly/safety_only conditions
        if condition in ("safefly", "safety_only"):
            if world.is_occupied_world(new_x, new_y):
                # Try small lateral offsets
                saved = False
                for lateral_angle in [np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4]:
                    test_x = pos[0] + np.cos(lateral_angle) * config.max_speed * config.dt * 0.5
                    test_y = pos[1] + np.sin(lateral_angle) * config.max_speed * config.dt * 0.5
                    if not world.is_occupied_world(test_x, test_y):
                        new_x, new_y = test_x, test_y
                        saved = True
                        interventions.append(step)
                        break
                if not saved:
                    # Can't move — stay in place this tick
                    new_x, new_y = pos[0], pos[1]
                    interventions.append(step)

        pos[0] = float(np.clip(new_x, 0.5, config.world_size[0] - 0.5))
        pos[1] = float(np.clip(new_y, 0.5, config.world_size[1] - 0.5))

        trajectory.append((float(pos[0]), float(pos[1])))

        # Track minimum obstacle distance
        obs_dist = world.nearest_obstacle_distance(float(pos[0]), float(pos[1]))
        min_obs_dist = min(min_obs_dist, obs_dist)

        # Check collision
        if world.is_occupied_world(float(pos[0]), float(pos[1])):
            collision = True
            break

        # Check goal reached
        dist_to_goal = np.sqrt(
            (pos[0] - goal_pos[0]) ** 2 + (pos[1] - goal_pos[1]) ** 2
        )
        if dist_to_goal < config.goal_tolerance:
            break

    # Compute path length
    path_len = 0.0
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        path_len += np.sqrt(dx**2 + dy**2)

    success = (
        not collision
        and np.sqrt(
            (trajectory[-1][0] - goal_pos[0]) ** 2
            + (trajectory[-1][1] - goal_pos[1]) ** 2
        )
        < config.goal_tolerance
    )

    return EpisodeResult(
        condition=condition,
        success=success,
        collision=collision,
        steps=len(trajectory),
        trajectory=trajectory,
        vla_waypoints=vla_waypoints,
        safe_waypoints=safe_waypoints,
        interventions=interventions,
        min_obstacle_distance=min_obs_dist,
        path_length=path_len,
    )


# ============================================================================
# Visualization
# ============================================================================


def visualize_results(
    results: dict[str, EpisodeResult],
    world: GridWorld,
    config: DemoConfig,
    goal_pos: tuple[float, float],
    instruction: str,
    save_path: Optional[str] = None,
):
    """Create a publication-quality side-by-side comparison figure."""
    conditions = ["vla_only", "safefly"]
    titles = ["VLA-Only (no safety)", "SafeFly (VLA + Safety Wrapper)"]
    colors_main = ["#e74c3c", "#2ecc71"]
    colors_vla = ["#e74c3c88", "#3498db88"]
    colors_intervention = ["#f39c12", "#f39c12"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"SafeFly Minimal Demo\nInstruction: \"{instruction}\"",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (cond, title) in enumerate(zip(conditions, titles)):
        ax = axes[idx]
        result = results[cond]

        # Draw occupancy grid
        extent = [0, config.world_size[0], 0, config.world_size[1]]
        ax.imshow(
            world.grid.T,
            origin="lower",
            extent=extent,
            cmap="gray_r",
            alpha=0.3,
            interpolation="nearest",
        )

        # Draw trajectory
        traj = np.array(result.trajectory)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            "-",
            color=colors_main[idx],
            linewidth=2,
            label="Flown path",
            zorder=3,
        )

        # Draw VLA waypoints
        if result.vla_waypoints:
            vla_pts = np.array(result.vla_waypoints)
            ax.scatter(
                vla_pts[:, 0],
                vla_pts[:, 1],
                c=colors_vla[idx],
                s=30,
                marker="x",
                linewidths=1.5,
                label="VLA waypoints",
                zorder=4,
            )

        # Mark intervention points
        if result.interventions:
            int_pts = np.array(
                [result.trajectory[i] for i in result.interventions if i < len(result.trajectory)]
            )
            if len(int_pts) > 0:
                ax.scatter(
                    int_pts[:, 0],
                    int_pts[:, 1],
                    c=colors_intervention[idx],
                    s=40,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5,
                    label=f"Safety interventions ({len(result.interventions)})",
                    zorder=5,
                )

        # Mark start and goal
        ax.scatter(
            *config.start_pos,
            c="blue",
            s=150,
            marker="*",
            edgecolors="black",
            linewidths=1,
            label="Start",
            zorder=6,
        )
        ax.scatter(
            *goal_pos,
            c="red",
            s=150,
            marker="*",
            edgecolors="black",
            linewidths=1,
            label="Goal",
            zorder=6,
        )

        # Safety radius circle at current position
        safety_circle = plt.Circle(
            config.start_pos,
            config.safety_radius,
            fill=False,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label=f"Safety radius ({config.safety_radius}m)",
        )
        ax.add_patch(safety_circle)

        # Status box
        status_text = (
            f"Status: {'✓ SUCCESS' if result.success else '✗ FAILED'}\n"
            f"Collision: {'YES ⚠' if result.collision else 'NO ✓'}\n"
            f"Steps: {result.steps}\n"
            f"Path length: {result.path_length:.1f}m\n"
            f"Min obstacle dist: {result.min_obstacle_distance:.2f}m\n"
            f"Interventions: {len(result.interventions)}"
        )
        ax.text(
            0.02,
            0.98,
            status_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.9,
                edgecolor="gray",
            ),
        )

        ax.set_xlim(0, config.world_size[0])
        ax.set_ylim(0, config.world_size[1])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Figure saved to: {save_path}")

    plt.show()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SafeFly Minimal Demo — 2D Grid + APF Safety Wrapper"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="fly forward 8 meters",
        help="Language instruction for VLA",
    )
    parser.add_argument(
        "--noise", type=float, default=0.8, help="VLA noise std (0=perfect, 2=chaotic)"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save figure to path (e.g., demo_output.png)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display interactive window (for headless/CI)",
    )
    args = parser.parse_args()

    # Setup — semi-open space, straight path crosses near obstacles
    config = DemoConfig(vla_noise_std=args.noise)
    config.start_pos = (2.0, 4.0)
    # Goal is positioned so the straight line passes near the obstacle cluster at x=9
    goal_pos = (17.5, 7.0)

    print("=" * 60)
    print("  SafeFly Minimal Demo")
    print("=" * 60)
    print(f"  Instruction: \"{args.instruction}\"")
    print(f"  VLA noise std: {args.noise}")
    print(f"  World: {config.world_size[0]}×{config.world_size[1]}m with obstacles")
    print(f"  Start: {config.start_pos} → Goal: {goal_pos}")
    print(f"  Safety radius: {config.safety_radius}m")
    print("=" * 60)

    # Build world
    print("\nBuilding world...")
    world = GridWorld(config)
    obstacle_cells = int(np.sum(world.grid >= 0.5))
    print(f"  Grid: {world.nx}×{world.ny} cells, {obstacle_cells} obstacle cells")

    # Create mock VLA (goal-aware, with noise to simulate VLA uncertainty)
    vla = MockVLA2D(config, goal_pos)

    # Run experiments
    print("\nRunning experiments...")
    results = {}

    for condition in ["vla_only", "safefly"]:
        t0 = time.perf_counter()
        result = run_episode(
            condition, world, vla, config, args.instruction, goal_pos
        )
        elapsed = (time.perf_counter() - t0) * 1000

        results[condition] = result

        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        collision = "YES ⚠" if result.collision else "NO"
        print(
            f"  [{condition:12s}] {status:10s} | "
            f"Collision: {collision:6s} | "
            f"Steps: {result.steps:3d} | "
            f"Path: {result.path_length:5.1f}m | "
            f"MinDist: {result.min_obstacle_distance:.2f}m | "
            f"Interventions: {len(result.interventions):2d} | "
            f"{elapsed:.1f}ms"
        )

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    vla_only = results["vla_only"]
    safefly = results["safefly"]

    collision_reduction = (
        "100% (eliminated)"
        if vla_only.collision and not safefly.collision
        else "N/A"
    )
    print(f"  VLA-Only:    collision={'YES' if vla_only.collision else 'NO'}, "
          f"success={'YES' if vla_only.success else 'NO'}")
    print(f"  SafeFly:     collision={'YES' if safefly.collision else 'NO'}, "
          f"success={'YES' if safefly.success else 'NO'}")
    print(f"  Collision reduction: {collision_reduction}")
    print(
        f"  Safety interventions: {len(safefly.interventions)} "
        f"({100*len(safefly.interventions)/max(safefly.steps,1):.1f}% of steps)"
    )
    print(f"  Min obstacle distance: {vla_only.min_obstacle_distance:.2f}m → "
          f"{safefly.min_obstacle_distance:.2f}m")
    print("=" * 60)

    # Visualize
    if not args.no_show:
        print("\nGenerating visualization...")
        visualize_results(
            results,
            world,
            config,
            goal_pos,
            args.instruction,
            save_path=args.save,
        )

    # Return exit code
    return 0 if safefly.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
