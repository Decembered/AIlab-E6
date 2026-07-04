"""
VLA Interface — mock and real VLA model wrappers.

Provides a unified interface for:
1. MockVLA: Lightweight rule-based VLA for testing without a real model
2. OctoVLA: Real Octo model interface (requires `pip install octo`)
3. HeuristicPlanner: Fallback non-VLA planner for baseline comparison

All interfaces follow the same contract:
    Input:  rgb_image (H,W,3), instruction (str), current_pose
    Output: waypoint (x,y,z), confidence (0-1)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VLAOutput:
    """Standardized VLA output regardless of backend."""

    waypoint: np.ndarray  # 3D waypoint in world frame [x, y, z]
    confidence: float  # Model confidence [0, 1]
    raw_output: Optional[np.ndarray] = None  # Raw model output before mapping
    metadata: dict = None  # Extra model-specific info

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class VLABackend(ABC):
    """Abstract VLA backend. All implementations must satisfy this interface."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        current_pose: Optional[np.ndarray] = None,
    ) -> VLAOutput:
        """Given an RGB image and language instruction, predict a waypoint."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the backend is ready for inference."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...


# ---------------------------------------------------------------------------
# Mock VLA — rule-based, always available
# ---------------------------------------------------------------------------


class MockVLA(VLABackend):
    """
    Rule-based mock VLA for development and testing.

    Parses simple directional instructions and generates waypoints
    with controllable noise for testing the safety wrapper.

    Supported instructions (examples):
    - "fly forward 3 meters"
    - "go to the red marker"
    - "fly through the doorway"
    - "turn left and fly 2 meters"
    - "go up 1 meter"
    """

    def __init__(self, noise_std: float = 1.0, seed: int = 42):
        self._noise_std = noise_std
        self._rng = np.random.RandomState(seed)
        self._default_step = 3.0  # meters

    @property
    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "MockVLA"

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        current_pose: Optional[np.ndarray] = None,
    ) -> VLAOutput:
        """
        Parse instruction and generate waypoint relative to current pose.

        The mock VLA intentionally adds noise to simulate real VLA unreliability,
        giving the safety wrapper something to correct.
        """
        if current_pose is None:
            current_pose = np.zeros(3)

        # Parse instruction for directional cues
        direction = self._parse_direction(instruction)
        distance = self._parse_distance(instruction)

        # Compute waypoint
        waypoint = current_pose[:3].copy().astype(np.float64)
        waypoint[0] += direction[0] * distance
        waypoint[1] += direction[1] * distance
        waypoint[2] += direction[2] * distance

        # Add noise (simulates VLA uncertainty)
        noise = self._rng.randn(3) * self._noise_std
        waypoint += noise

        # Clamp z to reasonable altitude
        waypoint[2] = max(0.3, min(5.0, waypoint[2]))

        # Confidence drops with noise level
        confidence = max(0.3, 1.0 - self._noise_std / 5.0)

        return VLAOutput(
            waypoint=waypoint,
            confidence=confidence,
            raw_output=waypoint.copy(),
            metadata={
                "direction_parsed": direction,
                "distance_parsed": distance,
                "noise_added": noise,
            },
        )

    def _parse_direction(self, instruction: str) -> np.ndarray:
        """Basic direction parser. Returns unit vector."""
        instr = instruction.lower()
        direction = np.array([1.0, 0.0, 0.0])  # Default: forward (+x)

        if "left" in instr:
            direction = np.array([0.0, 1.0, 0.0])
        elif "right" in instr:
            direction = np.array([0.0, -1.0, 0.0])
        elif "back" in instr or "backward" in instr:
            direction = np.array([-1.0, 0.0, 0.0])
        elif "up" in instr:
            direction = np.array([0.0, 0.0, 1.0])
        elif "down" in instr:
            direction = np.array([0.0, 0.0, -1.0])
        elif "forward" in instr:
            direction = np.array([1.0, 0.0, 0.0])

        return direction

    def _parse_distance(self, instruction: str) -> float:
        """Extract distance from instruction string."""
        import re

        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:meter|m)", instruction.lower())
        if match:
            return float(match.group(1))
        return self._default_step

    def set_noise(self, std: float) -> None:
        """Adjust noise level for testing different VLA reliability."""
        self._noise_std = std


# ---------------------------------------------------------------------------
# Heuristic planner — non-VLA baseline
# ---------------------------------------------------------------------------


class HeuristicPlanner(VLABackend):
    """
    Simple heuristic planner for baseline comparison.

    Always proposes going straight toward a hardcoded goal position.
    No semantic understanding — purely geometric.
    """

    def __init__(self, goal_position: np.ndarray):
        self._goal = np.array(goal_position, dtype=np.float64)

    @property
    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "HeuristicPlanner"

    def predict(
        self,
        image: np.ndarray = None,
        instruction: str = "",
        current_pose: Optional[np.ndarray] = None,
    ) -> VLAOutput:
        if current_pose is None:
            current_pose = np.zeros(3)
        return VLAOutput(
            waypoint=self._goal.copy(),
            confidence=1.0,
            metadata={"type": "heuristic", "goal": self._goal.copy()},
        )


# ---------------------------------------------------------------------------
# Octo VLA — real model (optional, requires `pip install octo`)
# ---------------------------------------------------------------------------


class OctoVLA(VLABackend):
    """
    Real Octo VLA model interface.

    Requires: pip install octo
    Pre-download checkpoint: python -c "from octo import OctoModel; OctoModel.load('octo-small')"

    Octo outputs actions in the robot's end-effector frame. We repurpose this
    for drone navigation by interpreting the delta-position component as a
    waypoint offset in world frame.
    """

    def __init__(
        self,
        checkpoint: str = "octo-small",
        device: str = "cpu",
        image_size: tuple = (224, 224),
    ):
        self._checkpoint = checkpoint
        self._device = device
        self._image_size = image_size
        self._model = None
        self._available = False

        try:
            from octo import OctoModel

            self._model = OctoModel.load(checkpoint)
            self._model.to(device)
            self._available = True
        except ImportError:
            warnings.warn(
                "Octo not installed. Install with: pip install octo\n"
                "Using MockVLA as fallback."
            )
        except Exception as e:
            warnings.warn(f"Failed to load Octo model: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def name(self) -> str:
        return f"OctoVLA({self._checkpoint})"

    def predict(
        self,
        image: np.ndarray,
        instruction: str,
        current_pose: Optional[np.ndarray] = None,
    ) -> VLAOutput:
        if not self._available:
            raise RuntimeError("Octo model not available. Check installation.")

        if current_pose is None:
            current_pose = np.zeros(3)

        # Preprocess image
        import cv2

        img = cv2.resize(image, self._image_size)
        img = img.astype(np.float32) / 255.0

        # Run Octo inference
        # Note: This is simplified — actual Octo API may differ.
        # See https://github.com/octo-models/octo for exact usage.
        try:
            # Octo expects batched inputs
            import torch

            obs = {
                "image_primary": torch.from_numpy(img)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self._device),
                "timestep": torch.zeros(1, dtype=torch.int32, device=self._device),
            }
            task = self._model.create_tasks(texts=[instruction])

            with torch.no_grad():
                action = self._model.sample_actions(
                    obs, task, unnormalization_statistics=None
                )

            # Octo outputs delta EEF pose (7-DoF typically)
            # Map translation component to world-frame waypoint
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            delta = action[0, 0, :3]  # First batch, first timestep, xyz
            waypoint = current_pose[:3] + delta * 0.5  # Scale factor

            return VLAOutput(
                waypoint=waypoint,
                confidence=0.7,  # Placeholder
                raw_output=action,
                metadata={"delta": delta, "model": self._checkpoint},
            )
        except Exception as e:
            warnings.warn(f"Octo inference failed: {e}. Falling back to zero waypoint.")
            return VLAOutput(
                waypoint=current_pose[:3].copy(),
                confidence=0.0,
                metadata={"error": str(e)},
            )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vla_backend(
    backend: str = "mock",
    noise_std: float = 1.0,
    goal_position: Optional[np.ndarray] = None,
) -> VLABackend:
    """
    Factory to create the appropriate VLA backend.

    Args:
        backend: "mock", "heuristic", or "octo"
        noise_std: Noise level for MockVLA
        goal_position: Goal for HeuristicPlanner

    Returns:
        VLABackend instance
    """
    if backend == "mock":
        return MockVLA(noise_std=noise_std)
    elif backend == "heuristic":
        return HeuristicPlanner(
            goal_position=goal_position or np.array([5.0, 0.0, 1.5])
        )
    elif backend == "octo":
        return OctoVLA()
    else:
        raise ValueError(f"Unknown backend: {backend}")
