"""Test eval_hairpins loading, inspection, and execution."""
import os
import pytest
import torch

from analysis.eval_hairpins import (
    inspect_and_load_actor,
    run_episode,
    compute_aggregate,
    RAY_ANGLES_5,
    RAY_ANGLES_9,
)
from sac.actor import SACActor


def test_inspect_and_load_actor_5ray():
    actor_path = "runs/exp2_demos_s42_5ray_matched200k/best_actor.pth"
    if not os.path.exists(actor_path):
        pytest.skip(f"Checkpoint {actor_path} not found")

    actor, obs_dim, ray_angles, ray_max_dist = inspect_and_load_actor(actor_path)
    assert obs_dim == 9
    assert len(ray_angles) == 5
    assert ray_angles == RAY_ANGLES_5
    assert ray_max_dist == 200.0


def test_inspect_and_load_actor_9ray():
    actor_path = "runs/exp2_demos_s42_9ray/best_actor.pth"
    if not os.path.exists(actor_path):
        pytest.skip(f"Checkpoint {actor_path} not found")

    actor, obs_dim, ray_angles, ray_max_dist = inspect_and_load_actor(actor_path)
    assert obs_dim == 13
    assert len(ray_angles) == 9
    assert ray_angles == RAY_ANGLES_9
    assert ray_max_dist == 200.0


def test_compute_aggregate():
    episodes = [
        {"completed": True, "crashed": False, "timeout": False, "stationary": False, "lap_time": 400, "lap_progress": 1.0},
        {"completed": False, "crashed": True, "timeout": False, "stationary": False, "lap_time": None, "lap_progress": 0.2},
        {"completed": False, "crashed": False, "timeout": True, "stationary": True, "lap_time": None, "lap_progress": 0.0},
    ]
    agg = compute_aggregate(episodes)
    assert agg["total"] == 3
    assert agg["completed"] == 1
    assert agg["crashes"] == 1
    assert agg["timeouts"] == 1
    assert agg["stationary"] == 1
    assert agg["avg_lap_time"] == 400.0
    assert abs(agg["avg_progress"] - 0.4) < 1e-4
    assert abs(agg["median_progress"] - 0.2) < 1e-4


def test_run_episode_short():
    actor_path = "runs/exp2_demos_s42_5ray_matched200k/best_actor.pth"
    if not os.path.exists(actor_path):
        pytest.skip(f"Checkpoint {actor_path} not found")

    actor, obs_dim, ray_angles, ray_max_dist = inspect_and_load_actor(actor_path)
    res = run_episode(actor, seed=1, hairpin_factor=0.5, ray_angles=ray_angles, max_dist=ray_max_dist, max_steps=10)
    assert "outcome" in res
    assert "lap_progress" in res
    assert "stationary" in res
    assert "max_speed" in res
