"""
Safety Wrapper — the core contribution of SafeFly.

Given a VLA-proposed waypoint/velocity and the current occupancy grid,
this module checks for collisions and, if necessary, replans using
Artificial Potential Fields (APF).

Key design decisions:
- Stateless per tick: each call is independent; VLA is async
- O(M*D) complexity: M = nearby occupied cells, D = 3 (dimensions)
- Target <5ms per call on CPU at 50Hz control
- Every intervention is logged with geometric reason

Supports both 2D (mobile robot / top-down drone) and 3D modes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class SafetyState(Enum):
    """Safety wrapper finite state machine states."""

    NOMINAL = auto()  # VLA output is safe, passed through unchanged
    CORRECTED = auto()  # VLA output modified by safety wrapper
    HOVER = auto()  # No safe path found, hold position
    EMERGENCY = auto()  # Sensor fault or critical battery


@dataclass
class SafetyConfig:
    """All tunable parameters for the safety wrapper."""

    # Collision avoidance
    safety_radius_m: float = 0.5  # Drone safety sphere radius
    prediction_horizon_steps: int = 10  # Steps to check forward

    # Velocity limits
    max_linear_velocity_ms: float = 2.0
    max_angular_velocity_rads: float = 1.0

    # Workspace bounds (world frame)
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    z_min: float = 0.3  # Min flight altitude
    z_max: float = 5.0  # Max flight altitude

    # APF parameters
    attraction_gain: float = 1.0  # K_att — pull toward VLA waypoint
    repulsion_gain: float = 2.0  # K_rep — push away from obstacles
    repulsion_range_m: float = 1.0  # d0 — obstacles beyond this are ignored

    # Deadlock detection
    deadlock_velocity_threshold_ms: float = 0.1  # Below this = stuck
    deadlock_consecutive_ticks: int = 20  # Ticks before declaring deadlock

    # Recovery
    consecutive_safe_ticks_for_nominal: int = 5  # Safe ticks to return to NOMINAL

    # Grid
    grid_resolution_m: float = 0.2

    # Dimension mode
    dim_mode: int = 3  # 2 or 3


@dataclass
class SafetyIntervention:
    """Record of a single safety intervention."""

    timestamp: float
    state: SafetyState
    vla_input: np.ndarray  # Original VLA waypoint/velocity
    corrected_output: np.ndarray  # After safety processing
    reason: str  # Human-readable reason
    collision_detected: bool
    velocity_clamped: bool
    workspace_clamped: bool
    apf_active: bool
    min_obstacle_distance_m: float
    computation_time_ms: float


@dataclass
class SafetyResult:
    """Output of a single safety wrapper step."""

    safe_command: np.ndarray  # Final safe velocity/position command
    state: SafetyState
    intervention: Optional[SafetyIntervention] = None
    vla_accepted: bool = True  # Was the VLA output accepted as-is?
    debug: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Occupancy Grid (lightweight, dependency-free)
# ---------------------------------------------------------------------------


class OccupancyGrid:
    """
    Lightweight 3D occupancy grid. Pure numpy — no OctoMap dependency needed.

    For the hackathon demo, this provides the minimum viable occupancy
    representation. Can be swapped for real OctoMap in production.
    """

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.resolution = config.grid_resolution_m

        # Grid dimensions in cells
        self.nx = int((config.x_max - config.x_min) / self.resolution) + 1
        self.ny = int((config.y_max - config.y_min) / self.resolution) + 1
        self.nz = int((config.z_max - config.z_min) / self.resolution) + 1

        # Occupancy probability grid (log-odds style, but simplified to 0-1)
        self.grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.origin = np.array([config.x_min, config.y_min, config.z_min])

    def world_to_grid(self, point: np.ndarray) -> tuple[int, int, int]:
        """Convert world coordinates to grid indices."""
        idx = ((point - self.origin) / self.resolution).astype(int)
        return tuple(np.clip(idx, 0, [self.nx - 1, self.ny - 1, self.nz - 1]))

    def grid_to_world(self, idx: tuple[int, int, int]) -> np.ndarray:
        """Convert grid indices to world coordinates (cell center)."""
        return self.origin + np.array(idx) * self.resolution + self.resolution / 2

    def is_occupied(self, point: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if a world point is occupied."""
        ix, iy, iz = self.world_to_grid(point)
        return bool(self.grid[ix, iy, iz] >= threshold)

    def ray_cast(
        self, start: np.ndarray, end: np.ndarray, num_steps: int = 20
    ) -> tuple[bool, float]:
        """Cast a ray from start to end, return (collision, min_distance)."""
        min_dist = float("inf")
        for t in np.linspace(0, 1, num_steps):
            point = start + t * (end - start)
            if self.is_occupied(point):
                dist = float(np.linalg.norm(point - start))
                min_dist = min(min_dist, dist)
        if min_dist < float("inf"):
            return True, min_dist
        return False, float(np.linalg.norm(end - start))

    def mark_occupied(self, point: np.ndarray, prob: float = 1.0) -> None:
        """Mark a world point as occupied."""
        ix, iy, iz = self.world_to_grid(point)
        self.grid[ix, iy, iz] = max(self.grid[ix, iy, iz], prob)

    def mark_free(self, point: np.ndarray) -> None:
        """Mark a world point as free."""
        ix, iy, iz = self.world_to_grid(point)
        self.grid[ix, iy, iz] = 0.0

    def inflate_obstacles(self, inflation_cells: int) -> None:
        """Dilate occupied cells for safety margin. Simple box dilation."""
        from scipy.ndimage import maximum_filter

        if inflation_cells > 0:
            size = 2 * inflation_cells + 1
            self.grid = maximum_filter(self.grid, size=size)

    def get_nearby_obstacles(
        self, center: np.ndarray, radius_m: float
    ) -> list[np.ndarray]:
        """Return list of occupied grid cell centers within radius_m of center."""
        obstacles = []
        radius_cells = int(np.ceil(radius_m / self.resolution))
        cx, cy, cz = self.world_to_grid(center)

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    ix, iy, iz = cx + dx, cy + dy, cz + dz
                    if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
                        if self.grid[ix, iy, iz] >= 0.5:
                            pos = self.grid_to_world((ix, iy, iz))
                            if np.linalg.norm(pos - center) <= radius_m:
                                obstacles.append(pos)
        return obstacles


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------


def check_collision(
    start: np.ndarray,
    end: np.ndarray,
    grid: OccupancyGrid,
    config: SafetyConfig,
) -> tuple[bool, float]:
    """
    Check if the path from start to end collides with any obstacle.

    Returns (collision_detected, min_distance_to_obstacle).
    """
    # Quick check: is the end point itself in collision?
    if grid.is_occupied(end):
        return True, 0.0

    # Ray cast along the path
    collision, min_dist = grid.ray_cast(start, end)
    if collision and min_dist < config.safety_radius_m:
        return True, min_dist

    return False, min_dist


def check_velocity(
    velocity: np.ndarray, config: SafetyConfig
) -> tuple[bool, np.ndarray]:
    """Clamp velocity to safe bounds. Returns (was_clamped, safe_velocity)."""
    was_clamped = False
    safe_vel = velocity.copy()

    linear_speed = float(np.linalg.norm(velocity[:3]))
    if linear_speed > config.max_linear_velocity_ms:
        safe_vel[:3] = safe_vel[:3] / linear_speed * config.max_linear_velocity_ms
        was_clamped = True

    if config.dim_mode >= 3 and len(velocity) > 3:
        ang = abs(velocity[3])
        if ang > config.max_angular_velocity_rads:
            safe_vel[3] = np.sign(velocity[3]) * config.max_angular_velocity_rads
            was_clamped = True

    return was_clamped, safe_vel


def check_workspace(point: np.ndarray, config: SafetyConfig) -> tuple[bool, np.ndarray]:
    """Project point into workspace bounds. Returns (was_clamped, safe_point)."""
    was_clamped = False
    safe = point.copy()

    bounds = [
        (0, config.x_min, config.x_max),
        (1, config.y_min, config.y_max),
        (2, config.z_min, config.z_max),
    ]

    for axis, lo, hi in bounds:
        if axis < len(safe):
            if safe[axis] < lo:
                safe[axis] = lo
                was_clamped = True
            elif safe[axis] > hi:
                safe[axis] = hi
                was_clamped = True

    return was_clamped, safe


# ---------------------------------------------------------------------------
# APF local replanning
# ---------------------------------------------------------------------------


def potential_field_replan(
    current_pos: np.ndarray,
    goal_pos: np.ndarray,
    grid: OccupancyGrid,
    config: SafetyConfig,
    max_iterations: int = 100,
    step_size: float = 0.05,
) -> np.ndarray:
    """
    Artificial Potential Field local replanning.

    Iteratively steps from current_pos toward goal_pos while being
    repelled by nearby obstacles. Returns the first safe step direction
    (velocity vector).

    F_total = K_att * (goal - pos) + Σ K_rep * (1/d - 1/d0) * (1/d²) * (pos - obstacle)/d
    """
    pos = current_pos.copy().astype(np.float64)
    dim = min(config.dim_mode, len(pos))

    for _ in range(max_iterations):
        # Attraction toward goal
        to_goal = goal_pos[:dim] - pos[:dim]
        dist_to_goal = float(np.linalg.norm(to_goal))
        if dist_to_goal < 0.1:  # Reached goal vicinity
            break

        f_att = config.attraction_gain * to_goal / (dist_to_goal + 1e-8)

        # Repulsion from nearby obstacles
        f_rep = np.zeros(dim, dtype=np.float64)
        obstacles = grid.get_nearby_obstacles(pos, config.repulsion_range_m)

        for obs_pos in obstacles:
            to_obs = pos[:dim] - obs_pos[:dim]
            d = float(np.linalg.norm(to_obs))
            if d < 1e-8:
                d = 1e-8
            if d < config.repulsion_range_m:
                # Standard APF repulsion: F = K_rep * (1/d - 1/d0) * (1/d²)
                magnitude = (
                    config.repulsion_gain
                    * (1.0 / d - 1.0 / config.repulsion_range_m)
                    * (1.0 / (d * d))
                )
                f_rep += magnitude * (to_obs / d)

        # Total force
        f_total = f_att + f_rep

        # Limit step
        f_norm = float(np.linalg.norm(f_total))
        if f_norm > 1.0:
            f_total = f_total / f_norm

        # Step
        new_pos = pos.copy()
        new_pos[:dim] = pos[:dim] + step_size * f_total

        # Check if new position is safe
        if not grid.is_occupied(new_pos):
            pos = new_pos
        else:
            # Try lateral steps if forward is blocked
            for angle in [np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2]:
                lateral = pos.copy()
                if dim >= 2:
                    lateral[0] = pos[0] + step_size * np.cos(angle)
                    lateral[1] = pos[1] + step_size * np.sin(angle)
                if not grid.is_occupied(lateral):
                    pos = lateral
                    break
            else:
                # All directions blocked — stay in place
                break

    # Return velocity vector toward the computed safe position
    velocity = np.zeros_like(current_pos)
    velocity[:dim] = (pos[:dim] - current_pos[:dim]) / (step_size * max_iterations)
    velocity[:dim] = velocity[:dim] / (np.linalg.norm(velocity[:dim]) + 1e-8)

    # Scale to reasonable speed
    velocity[:dim] *= config.max_linear_velocity_ms * 0.5

    return velocity


# ---------------------------------------------------------------------------
# Main safety wrapper
# ---------------------------------------------------------------------------


class SafetyWrapper:
    """
    Main safety wrapper class.

    Usage:
        sw = SafetyWrapper(SafetyConfig(dim_mode=3))
        grid = OccupancyGrid(sw.config)
        # ... populate grid from sensor data ...
        result = sw.step(vla_waypoint, grid, current_pose, current_velocity)
        safe_command = result.safe_command  # Send this to the drone
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self._state: SafetyState = SafetyState.NOMINAL
        self._safe_tick_counter: int = 0
        self._deadlock_counter: int = 0
        self._last_position: Optional[np.ndarray] = None
        self._intervention_log: list[SafetyIntervention] = []

    @property
    def state(self) -> SafetyState:
        return self._state

    @property
    def intervention_history(self) -> list[SafetyIntervention]:
        return self._intervention_log

    def step(
        self,
        vla_waypoint: np.ndarray,
        occupancy_grid: OccupancyGrid,
        current_pose: np.ndarray,
        current_velocity: Optional[np.ndarray] = None,
    ) -> SafetyResult:
        """
        Main entry point. Call this at every control tick (e.g., 50Hz).

        Args:
            vla_waypoint: VLA-proposed 3D waypoint in world frame [x, y, z]
            occupancy_grid: Current occupancy grid
            current_pose: Current drone position [x, y, z, (roll, pitch, yaw)]
            current_velocity: Current drone velocity [vx, vy, vz, (vyaw)]

        Returns:
            SafetyResult with the safe velocity command and metadata.
        """
        t_start = time.perf_counter()
        dim = min(self.config.dim_mode, len(vla_waypoint))

        if current_velocity is None:
            current_velocity = np.zeros_like(vla_waypoint)

        # ---- Step 1: Workspace bounds check ----
        ws_clamped, waypoint_ws = check_workspace(vla_waypoint, self.config)

        # ---- Step 2: Collision check ----
        collision, min_dist = check_collision(
            current_pose[:dim], waypoint_ws[:dim], occupancy_grid, self.config
        )

        # ---- Step 3: Determine state and action ----
        needs_correction = collision or ws_clamped

        if needs_correction:
            # Try APF replanning
            safe_vel = potential_field_replan(
                current_pose, waypoint_ws, occupancy_grid, self.config
            )

            # Check if APF succeeded (non-zero velocity)
            apf_magnitude = float(np.linalg.norm(safe_vel[:dim]))
            if apf_magnitude < self.config.deadlock_velocity_threshold_ms:
                self._deadlock_counter += 1
                if self._deadlock_counter >= self.config.deadlock_consecutive_ticks:
                    self._state = SafetyState.HOVER
                    safe_vel = np.zeros_like(vla_waypoint)
            else:
                self._deadlock_counter = 0
                self._state = SafetyState.CORRECTED
        else:
            # VLA output is safe — use it directly
            direction = waypoint_ws[:dim] - current_pose[:dim]
            dist = float(np.linalg.norm(direction))
            if dist > 1e-8:
                safe_vel = np.zeros_like(vla_waypoint)
                safe_vel[:dim] = (
                    direction / dist * self.config.max_linear_velocity_ms * 0.5
                )
            else:
                safe_vel = np.zeros_like(vla_waypoint)

            self._safe_tick_counter += 1
            if (
                self._safe_tick_counter
                >= self.config.consecutive_safe_ticks_for_nominal
            ):
                self._state = SafetyState.NOMINAL
            self._deadlock_counter = 0

        # ---- Step 4: Velocity clamping ----
        vel_clamped, safe_vel = check_velocity(safe_vel, self.config)

        # Apply any additional clamping from workspace or velocity
        any_correction = needs_correction or vel_clamped

        # ---- Step 5: Log intervention ----
        intervention = None
        if any_correction:
            intervention = SafetyIntervention(
                timestamp=t_start,
                state=self._state,
                vla_input=vla_waypoint.copy(),
                corrected_output=safe_vel.copy(),
                reason=self._build_reason(
                    collision, ws_clamped, vel_clamped, min_dist
                ),
                collision_detected=collision,
                velocity_clamped=vel_clamped,
                workspace_clamped=ws_clamped,
                apf_active=collision,
                min_obstacle_distance_m=min_dist,
                computation_time_ms=(time.perf_counter() - t_start) * 1000,
            )
            self._intervention_log.append(intervention)

        # ---- Step 6: Deadlock recovery ----
        if self._state == SafetyState.HOVER:
            # Check if path has cleared
            _, new_min_dist = check_collision(
                current_pose[:dim], waypoint_ws[:dim], occupancy_grid, self.config
            )
            if not collision or new_min_dist > self.config.safety_radius_m:
                self._state = SafetyState.CORRECTED
                self._deadlock_counter = 0

        self._last_position = current_pose.copy()

        return SafetyResult(
            safe_command=safe_vel,
            state=self._state,
            intervention=intervention,
            vla_accepted=(not any_correction),
            debug={
                "collision_check": collision,
                "min_obstacle_distance": min_dist,
                "workspace_clamped": ws_clamped,
                "velocity_clamped": vel_clamped,
                "computation_time_ms": (time.perf_counter() - t_start) * 1000,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reason(
        self,
        collision: bool,
        ws_clamped: bool,
        vel_clamped: bool,
        min_dist: float,
    ) -> str:
        parts = []
        if collision:
            parts.append(f"collision risk (min_dist={min_dist:.2f}m)")
        if ws_clamped:
            parts.append("workspace bounds exceeded")
        if vel_clamped:
            parts.append("velocity exceeded limits")
        return "; ".join(parts) if parts else "unknown"

    def get_statistics(self) -> dict:
        """Return aggregate statistics over all interventions."""
        if not self._intervention_log:
            return {
                "total_interventions": 0,
                "intervention_rate": 0.0,
                "avg_computation_time_ms": 0.0,
                "collision_interventions": 0,
                "velocity_clamp_interventions": 0,
                "workspace_clamp_interventions": 0,
                "avg_min_obstacle_distance": float("inf"),
            }

        total = len(self._intervention_log)
        return {
            "total_interventions": total,
            "avg_computation_time_ms": float(
                np.mean([i.computation_time_ms for i in self._intervention_log])
            ),
            "collision_interventions": sum(
                1 for i in self._intervention_log if i.collision_detected
            ),
            "velocity_clamp_interventions": sum(
                1 for i in self._intervention_log if i.velocity_clamped
            ),
            "workspace_clamp_interventions": sum(
                1 for i in self._intervention_log if i.workspace_clamped
            ),
            "min_obstacle_distances": [
                i.min_obstacle_distance_m for i in self._intervention_log
            ],
        }

    def reset(self) -> None:
        """Reset state between episodes."""
        self._state = SafetyState.NOMINAL
        self._safe_tick_counter = 0
        self._deadlock_counter = 0
        self._last_position = None
        self._intervention_log.clear()
