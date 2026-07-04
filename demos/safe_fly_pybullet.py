#!/usr/bin/env python3
"""
SafeFly PyBullet 3D Demo

A 3D drone simulation demo using PyBullet. Shows:
1. Quadcopter in a 3D environment with obstacles
2. Mock VLA generating waypoints from instructions
3. Safety wrapper steering the drone around obstacles
4. Real-time 3D visualization (if GUI available)

Dependencies: numpy, pybullet
Install: pip install pybullet

This is the "upgrade path" from the 2D minimal demo. If PyBullet is installed,
this provides a much more impressive demo with a flying drone.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class PyBulletConfig:
    """PyBullet demo configuration."""

    # Simulation
    timestep: float = 1.0 / 240.0  # PyBullet physics timestep
    control_dt: float = 0.05  # Control loop period (s)
    max_steps: int = 2000

    # Drone
    start_pos: tuple = (0.0, 0.0, 1.0)  # (x, y, z)
    goal_pos: tuple = (8.0, 0.0, 1.5)
    max_speed: float = 1.5
    safety_radius: float = 0.5

    # VLA
    vla_noise_std: float = 0.3
    vla_update_interval: int = 50  # Control ticks between VLA updates

    # Obstacles
    obstacle_positions: list = None

    def __post_init__(self):
        if self.obstacle_positions is None:
            self.obstacle_positions = [
                # (x, y, z, half_extents_x, half_extents_y, half_extents_z)
                (4.0, 0.0, 1.0, 0.2, 1.5, 1.0),  # Wall blocking path
                (2.0, 1.5, 0.5, 0.3, 0.3, 0.5),  # Pillar
                (6.0, -1.5, 0.5, 0.3, 0.3, 0.5),  # Pillar
            ]


# ============================================================================
# Main demo
# ============================================================================


def run_pybullet_demo(config: Optional[PyBulletConfig] = None, gui: bool = True):
    """Run the 3D PyBullet demo.

    Returns True if the drone reached the goal without collision.
    """
    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        print("ERROR: pybullet not installed.")
        print("  Install: pip install pybullet")
        print("  Falling back to 2D minimal demo: python demos/safe_fly_minimal.py")
        return False

    if config is None:
        config = PyBulletConfig()

    # --- Setup PyBullet ---
    if gui:
        client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=10, cameraYaw=-45, cameraPitch=-30, cameraTargetPosition=[4, 0, 1]
        )
    else:
        client = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(config.timestep)

    # --- Load environment ---
    plane_id = p.loadURDF("plane.urdf")

    # Load drone (use a simple box or sphere as placeholder)
    # In a real demo, use a proper quadcopter URDF
    drone_start = list(config.start_pos)
    drone_id = p.loadURDF(
        "r2d2.urdf",  # Placeholder — replace with quadcopter URDF
        basePosition=drone_start,
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        globalScaling=0.3,
    )

    # Load obstacles
    obstacle_ids = []
    for ox, oy, oz, hx, hy, hz in config.obstacle_positions:
        col_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz]
        )
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            rgbaColor=[0.8, 0.3, 0.3, 1.0],
        )
        body_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[ox, oy, oz],
        )
        obstacle_ids.append(body_id)

    # --- Simple PID controller for drone ---
    # This is a simplified position controller — in production you'd use
    # PX4 SITL or a proper flight dynamics model.

    def compute_velocity_command(target_pos: np.ndarray, current_pos: np.ndarray):
        """Simple P controller to move drone toward target."""
        error = target_pos - current_pos
        speed = np.linalg.norm(error)
        if speed < 0.1:
            return np.zeros(3)

        # P gain
        vel = error * min(2.0, speed) / speed * config.max_speed
        return vel

    def check_collision(pos: np.ndarray) -> bool:
        """Check if position is inside any obstacle."""
        for ox, oy, oz, hx, hy, hz in config.obstacle_positions:
            if (
                abs(pos[0] - ox) < hx + config.safety_radius
                and abs(pos[1] - oy) < hy + config.safety_radius
                and abs(pos[2] - oz) < hz + config.safety_radius
            ):
                return True
        return False

    # --- Mock VLA ---
    vla_rng = np.random.RandomState(42)

    def mock_vla_waypoint(current_pos: np.ndarray) -> np.ndarray:
        """Generate noisy waypoint toward goal."""
        wp = np.array(config.goal_pos, dtype=np.float64)
        noise = vla_rng.randn(3) * config.vla_noise_std
        wp += noise
        wp[2] = max(0.5, wp[2])  # Min altitude
        return wp

    # --- Directional sampling safety wrapper (3D) ---
    def safe_velocity_3d(
        current: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """3D version of the directional sampling safety wrapper."""
        direction = target - current
        dist = np.linalg.norm(direction)
        if dist < 0.1:
            return np.zeros(3)

        desired = direction / dist

        # Check if straight path is clear
        lookahead = config.safety_radius * 6
        n_check = 20
        straight_clear = True
        for i in range(1, n_check + 1):
            t = i * lookahead / n_check
            check_pos = current + desired * t
            if check_collision(check_pos):
                straight_clear = False
                break

        if straight_clear:
            return desired * config.max_speed

        # Search for safe direction (simplified 3D sampling)
        num_samples = 36
        best_dir = desired
        best_score = -float("inf")

        for i in range(num_samples):
            # Sample on a cone around the desired direction
            phi = np.random.uniform(0, np.pi / 3)  # Max 60° deviation
            theta = np.random.uniform(0, 2 * np.pi)

            # Rotate desired direction by (phi, theta)
            # This is a simplified rotation — proper implementation would use
            # Rodrigues' rotation formula
            candidate = desired.copy()
            candidate[0] += phi * np.cos(theta) * 0.3
            candidate[1] += phi * np.sin(theta) * 0.3
            if np.random.rand() > 0.5:
                candidate[2] += phi * 0.2
            else:
                candidate[2] -= phi * 0.2

            candidate = candidate / np.linalg.norm(candidate)

            # Check if this direction is clear
            clear = True
            for j in range(1, n_check + 1):
                t = j * lookahead / n_check
                check_pos = current + candidate * t
                if check_collision(check_pos):
                    clear = False
                    break

            if clear:
                # Score: prefer directions closer to desired
                score = float(np.dot(candidate, desired))
                if score > best_score:
                    best_score = score
                    best_dir = candidate

        return best_dir * config.max_speed

    # --- Main simulation loop ---
    pos = np.array(config.start_pos, dtype=np.float64)
    vla_wp = np.array(config.goal_pos, dtype=np.float64)
    trajectory = [pos.copy()]
    interventions = 0
    collision = False
    success = False

    print(f"Starting PyBullet 3D demo...")
    print(f"  Start: {config.start_pos}")
    print(f"  Goal:  {config.goal_pos}")
    print(f"  Obstacles: {len(config.obstacle_positions)}")
    print()

    for step in range(config.max_steps):
        # Update VLA waypoint
        if step % config.vla_update_interval == 0:
            vla_wp = mock_vla_waypoint(pos)

        # Compute safe velocity
        vel = safe_velocity_3d(pos, vla_wp)

        # Check if safety intervened
        direct_dir = vla_wp - pos
        direct_dist = np.linalg.norm(direct_dir)
        if direct_dist > 0.1:
            direct_dir /= direct_dist
            if np.dot(vel, direct_dir) < np.linalg.norm(vel) * 0.9:
                interventions += 1

        # Step
        new_pos = pos + vel * config.control_dt

        # Safety gate
        if check_collision(new_pos):
            collision = True
            break

        pos = new_pos
        trajectory.append(pos.copy())

        # Update PyBullet
        p.resetBasePositionAndOrientation(
            drone_id,
            pos.tolist(),
            p.getQuaternionFromEuler([0, 0, 0]),  # Simplified — no attitude control
        )
        p.stepSimulation()

        if gui:
            time.sleep(config.control_dt * 0.1)  # Slow down for visualization

        # Check goal
        if np.linalg.norm(pos - config.goal_pos) < 0.5:
            success = True
            break

        if step % 200 == 0:
            print(
                f"  Step {step:4d}/{config.max_steps} | "
                f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | "
                f"Dist to goal: {np.linalg.norm(pos - config.goal_pos):.1f}m | "
                f"Interventions: {interventions}"
            )

    # --- Results ---
    print()
    print("=" * 50)
    print(f"  PyBullet Demo Complete")
    print("=" * 50)
    print(f"  Steps: {step+1}/{config.max_steps}")
    print(f"  Collision: {'YES' if collision else 'NO'}")
    print(f"  Success: {'YES' if success else 'NO'}")
    print(f"  Interventions: {interventions}")
    print(f"  Final position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
    print(f"  Distance to goal: {np.linalg.norm(pos - config.goal_pos):.1f}m")
    print("=" * 50)

    if gui:
        print("\nClose the PyBullet window to exit...")
        while p.isConnected():
            time.sleep(0.1)

    p.disconnect()

    return success and not collision


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SafeFly PyBullet 3D Demo"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (for CI/headless)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.3,
        help="VLA noise level",
    )
    args = parser.parse_args()

    config = PyBulletConfig(vla_noise_std=args.noise)
    success = run_pybullet_demo(config, gui=not args.headless)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
