"""Domain Randomization Engine for RL Racing Agent.

Provides principled, reproducible sampling of environmental and physical parameters
across episodes for cross-difficulty generalization.

Supported parameters:
- Track geometry: width, min_radius, base_r, n_ctrl
- Vehicle dynamics: friction, acceleration, max_grip (when grip_limit=True)

Invariant parameters (not randomized to preserve contracts):
- Observation ray configuration (ray angles, max distance) -> SAC network input dim is fixed
- Action space bounds -> Box([-1, -1], [1, 1])
- Simulation dt (1/60s) -> numerical integration stability
- Car wheelbase (50px) -> geometric scale invariant
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


# Physical and architectural safety bounds
PARAM_SAFETY_BOUNDS = {
    "width": (15.0, 120.0),
    "min_radius": (40.0, 150.0),
    "base_r": (150.0, 320.0),
    "n_ctrl": (5, 16),
    "friction": (0.005, 0.07),
    "acceleration": (0.04, 0.16),
    "max_grip": (0.03, 0.25),
}

# Pre-configured domain randomization presets
DR_PRESETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Default preset reproduces the historical Jul 30/31 DR setup
    "default": {
        "width": (20.0, 70.0),
        "min_radius": (50.0, 90.0),
        "friction": (0.015, 0.05),
    },
    "width_only": {
        "width": (20.0, 70.0),
    },
    "friction_only": {
        "friction": (0.015, 0.05),
    },
    "geometry_only": {
        "width": (20.0, 70.0),
        "min_radius": (50.0, 90.0),
        "base_r": (200.0, 250.0),
    },
    "full": {
        "width": (20.0, 70.0),
        "min_radius": (50.0, 90.0),
        "base_r": (200.0, 250.0),
        "n_ctrl": (7, 11),
        "friction": (0.015, 0.05),
        "acceleration": (0.07, 0.09),
    },
    "calibrated": {
        "width": (25.0, 70.0),
        "min_radius": (65.0, 95.0),
        "base_r": (200.0, 260.0),
        "n_ctrl": (7, 11),
        "friction": (0.02, 0.045),
        "acceleration": (0.075, 0.095),
    },
    "grip": {
        "width": (20.0, 70.0),
        "min_radius": (50.0, 90.0),
        "friction": (0.015, 0.05),
        "max_grip": (0.06, 0.14),
    },
}


class DomainRandomizer:
    """Manages parameter randomization ranges, sampling, and reproducibility."""

    def __init__(
        self,
        enabled: bool = False,
        preset: Optional[str] = None,
        custom_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.enabled = bool(enabled)
        self.ranges: Dict[str, Tuple[float, float]] = {}

        if preset is not None:
            if preset not in DR_PRESETS:
                raise ValueError(
                    f"Unknown DR preset '{preset}'. Available: {list(DR_PRESETS.keys())}"
                )
            self.ranges.update(DR_PRESETS[preset])

        if custom_ranges:
            self.ranges.update(custom_ranges)

        if not self.ranges and self.enabled:
            # Fall back to default preset when enabled without explicit ranges
            self.ranges.update(DR_PRESETS["default"])

        self.validate()

    def validate(self) -> None:
        """Validates that all configured ranges fall within physical safety limits."""
        for param, bounds in self.ranges.items():
            if param not in PARAM_SAFETY_BOUNDS:
                raise ValueError(
                    f"Unsupported parameter '{param}' for domain randomization. "
                    f"Supported: {list(PARAM_SAFETY_BOUNDS.keys())}"
                )
            if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
                raise ValueError(
                    f"Range for '{param}' must be a (low, high) tuple/list. Got: {bounds}"
                )
            low, high = bounds
            if low > high:
                raise ValueError(
                    f"Invalid range for '{param}': low ({low}) > high ({high})"
                )
            safe_min, safe_max = PARAM_SAFETY_BOUNDS[param]
            if low < safe_min or high > safe_max:
                raise ValueError(
                    f"Range for '{param}' [{low}, {high}] exceeds safety bounds "
                    f"[{safe_min}, {safe_max}]"
                )

    def sample(
        self,
        np_random: np.random.Generator,
        base_track_kwargs: Optional[Dict[str, Any]] = None,
        base_car_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Samples parameters reproducibly using the provided Gymnasium RNG.

        Returns:
            sampled_track_kwargs: Dictionary of kwargs for Track(...)
            sampled_car_params: Dictionary of parameters for Car(...)
            dr_manifest: Dictionary documenting exactly which parameters were sampled
        """
        track_kwargs = dict(base_track_kwargs or {})
        car_params = dict(base_car_params or {})
        manifest: Dict[str, Any] = {"enabled": self.enabled}

        if not self.enabled:
            return track_kwargs, car_params, manifest

        # 1. Track parameters
        if "width" in self.ranges:
            low, high = self.ranges["width"]
            sampled_width = float(np_random.uniform(low, high))
            track_kwargs["width"] = sampled_width
            manifest["width"] = sampled_width

        if "min_radius" in self.ranges:
            low, high = self.ranges["min_radius"]
            sampled_min_radius = float(np_random.uniform(low, high))
            track_kwargs["min_radius"] = sampled_min_radius
            manifest["min_radius"] = sampled_min_radius

        if "base_r" in self.ranges:
            low, high = self.ranges["base_r"]
            sampled_base_r = float(np_random.uniform(low, high))
            track_kwargs["base_r"] = sampled_base_r
            manifest["base_r"] = sampled_base_r

        if "n_ctrl" in self.ranges:
            low, high = self.ranges["n_ctrl"]
            # Discrete integer sampling inclusive of bounds
            sampled_n_ctrl = int(np_random.integers(int(low), int(high) + 1))
            track_kwargs["n_ctrl"] = sampled_n_ctrl
            manifest["n_ctrl"] = sampled_n_ctrl

        # 2. Vehicle parameters
        if "friction" in self.ranges:
            low, high = self.ranges["friction"]
            sampled_friction = float(np_random.uniform(low, high))
            car_params["friction"] = sampled_friction
            manifest["friction"] = sampled_friction

        if "acceleration" in self.ranges:
            low, high = self.ranges["acceleration"]
            sampled_accel = float(np_random.uniform(low, high))
            car_params["acceleration"] = sampled_accel
            manifest["acceleration"] = sampled_accel

        if "max_grip" in self.ranges:
            low, high = self.ranges["max_grip"]
            sampled_grip = float(np_random.uniform(low, high))
            car_params["max_grip"] = sampled_grip
            manifest["max_grip"] = sampled_grip

        return track_kwargs, car_params, manifest

    def to_dict(self) -> Dict[str, Any]:
        """Serializes randomizer configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "ranges": {k: list(v) for k, v in self.ranges.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DomainRandomizer:
        """Constructs DomainRandomizer from serialized dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            custom_ranges={k: tuple(v) for k, v in data.get("ranges", {}).items()},
        )

    def save_json(self, filepath: str) -> None:
        """Saves configuration to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> DomainRandomizer:
        """Loads configuration from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
