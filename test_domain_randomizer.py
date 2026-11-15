"""Tests for Domain Randomization Engine."""

import pytest
import numpy as np
import gymnasium as gym

from env.environment import RacingEnv
from env.domain_randomizer import DomainRandomizer, PARAM_SAFETY_BOUNDS, DR_PRESETS


def test_disabled_by_default():
    """Verify that RacingEnv default has domain_randomize=False and baseline behavior is intact."""
    env = RacingEnv(track_kwargs={"width": 70, "min_radius": 80})
    obs, info = env.reset(seed=101)

    assert not env.domain_randomize
    assert env.track.width == 70
    assert env.car.friction == 0.03
    assert env.car.acceleration == 0.08
    assert "dr_params" in info
    assert info["dr_params"]["enabled"] is False

    next_obs, reward, terminated, truncated, step_info = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert "dr_params" in step_info
    assert step_info["dr_params"]["enabled"] is False


def test_bounds_validation():
    """Verify that invalid ranges and out-of-safety-bounds ranges raise ValueError."""
    # low > high
    with pytest.raises(ValueError, match="Invalid range"):
        DomainRandomizer(enabled=True, custom_ranges={"width": (70.0, 20.0)})

    # Out of safety bounds
    with pytest.raises(ValueError, match="exceeds safety bounds"):
        DomainRandomizer(enabled=True, custom_ranges={"width": (5.0, 70.0)})

    with pytest.raises(ValueError, match="exceeds safety bounds"):
        DomainRandomizer(enabled=True, custom_ranges={"friction": (0.01, 0.50)})

    # Unsupported parameter
    with pytest.raises(ValueError, match="Unsupported parameter"):
        DomainRandomizer(enabled=True, custom_ranges={"unsupported_param": (1.0, 2.0)})


def test_seed_reproducibility():
    """Verify that the same seed produces identical sampled parameters and trajectories."""
    dr = DomainRandomizer(enabled=True, preset="full")

    env1 = RacingEnv(domain_randomize=dr)
    obs1, info1 = env1.reset(seed=42)
    p1 = info1["dr_params"]

    env2 = RacingEnv(domain_randomize=dr)
    obs2, info2 = env2.reset(seed=42)
    p2 = info2["dr_params"]

    # Identical sampled parameters
    assert p1 == p2
    assert p1["width"] == p2["width"]
    assert p1["friction"] == p2["friction"]
    assert np.allclose(obs1, obs2)

    # Different seed produces different parameters
    obs3, info3 = env2.reset(seed=999)
    p3 = info3["dr_params"]
    assert p3["width"] != p1["width"] or p3["friction"] != p1["friction"]


def test_presets_and_application():
    """Verify preset loading and correct application to Track and Car objects."""
    dr = DomainRandomizer(enabled=True, preset="grip")
    env = RacingEnv(domain_randomize=dr, grip_limit=True)

    obs, info = env.reset(seed=123)
    params = info["dr_params"]

    assert params["enabled"] is True
    assert 20.0 <= params["width"] <= 70.0
    assert 50.0 <= params["min_radius"] <= 90.0
    assert 0.015 <= params["friction"] <= 0.05
    assert 0.06 <= params["max_grip"] <= 0.14

    # Applied to track and car
    assert env.track.width == params["width"]
    assert env.car.friction == params["friction"]
    assert env.car.max_grip == params["max_grip"]


def test_calibrated_preset():
    """Verify calibrated preset bounds, sampling, and physical sanity."""
    dr = DomainRandomizer(enabled=True, preset="calibrated")
    env = RacingEnv(domain_randomize=dr)

    obs, info = env.reset(seed=42)
    params = info["dr_params"]

    assert params["enabled"] is True
    assert 25.0 <= params["width"] <= 70.0
    assert 65.0 <= params["min_radius"] <= 95.0
    assert 200.0 <= params["base_r"] <= 260.0
    assert 7 <= params["n_ctrl"] <= 11
    assert 0.02 <= params["friction"] <= 0.045
    assert 0.075 <= params["acceleration"] <= 0.095

    # Check physical clearance: min acceleration strictly exceeds max friction
    assert params["acceleration"] > params["friction"]
    assert env.track.width == params["width"]
    assert env.car.friction == params["friction"]
    assert env.car.acceleration == params["acceleration"]


def test_serialization():
    """Verify dictionary serialization and reconstruction."""
    dr = DomainRandomizer(enabled=True, custom_ranges={"width": (30.0, 60.0), "friction": (0.02, 0.04)})
    data = dr.to_dict()

    dr_restored = DomainRandomizer.from_dict(data)
    assert dr_restored.enabled is True
    assert dr_restored.ranges["width"] == (30.0, 60.0)
    assert dr_restored.ranges["friction"] == (0.02, 0.04)


def test_smoke_episode_rollout():
    """Verify an agent can step through full episodes with domain randomization enabled."""
    env = RacingEnv(domain_randomize=True, max_steps=50)
    obs, info = env.reset(seed=77)

    steps = 0
    done = False
    while not done and steps < 50:
        action = np.array([0.1, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, step_info = env.step(action)
        assert "dr_params" in step_info
        assert step_info["dr_params"]["enabled"] is True
        done = terminated or truncated
        steps += 1

    assert steps > 0
