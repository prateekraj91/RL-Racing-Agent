"""Unit and integration tests for policy saliency implementation.

Validates:
1. Frozen model weight integrity (SHA-256) and parameter freeze.
2. Observation feature ordering and ray sensor contract.
3. Correct action dimension mapping (steering and throttle/brake).
4. Gradient calculation via autograd without parameter gradient accumulation.
5. Finite-difference validation precision.
6. Deterministic reproducibility on identical observations.
7. Correct structure and validity of CSV and JSON deliverables.
"""

import csv
import json
import os
import numpy as np
import pytest
import torch

from analysis.policy_saliency import (
    CONFIG_PATH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_WEIGHTS_SHA256,
    FEATURE_NAMES,
    FROZEN_MODEL_PATH,
    RAY_ANGLES,
    RAY_FEATURE_NAMES,
    RAY_INDICES,
    SUMMARY_JSON_PATH,
    TIMESTEPS_CSV_PATH,
    compute_action_saliency,
    compute_sha256,
    load_frozen_actor,
    validate_gradients_finite_difference,
    verify_frozen_integrity,
)
from env.environment import RacingEnv
from sac.actor import SACActor


class TestFrozenModelIntegrityAndFreezeRule:
    """Validate that the model freeze rule is strictly observed."""

    def test_sha256_checksums(self):
        w_sha, c_sha = verify_frozen_integrity()
        assert w_sha == EXPECTED_WEIGHTS_SHA256
        assert c_sha == EXPECTED_CONFIG_SHA256

    def test_actor_loading_eval_and_requires_grad_false(self):
        actor = load_frozen_actor(FROZEN_MODEL_PATH)
        assert isinstance(actor, SACActor)
        assert not actor.training, "Actor must be in evaluation mode"
        for name, param in actor.named_parameters():
            assert not param.requires_grad, f"Parameter {name} must have requires_grad=False"


class TestObservationAndActionContracts:
    """Validate observation space ordering and action dimension mapping."""

    def test_observation_feature_ordering(self):
        assert len(FEATURE_NAMES) == 9
        assert FEATURE_NAMES == [
            "velocity",
            "heading_error",
            "signed_distance",
            "slip",
            "ray_minus_90",
            "ray_minus_45",
            "ray_0",
            "ray_plus_45",
            "ray_plus_90",
        ]
        assert RAY_INDICES == [4, 5, 6, 7, 8]
        assert RAY_ANGLES == [-90.0, -45.0, 0.0, 45.0, 90.0]

    def test_environment_observation_matches_contract(self):
        env = RacingEnv(
            obs_ray_angles=(-90.0, -45.0, 0.0, 45.0, 90.0),
            obs_ray_max_dist=200.0,
        )
        obs, _ = env.reset(options={"track_seed": 1001})
        assert obs.shape == (9,)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

        # Step and verify shape
        action = np.array([0.0, 1.0], dtype=np.float32)
        next_obs, reward, term, trunc, info = env.step(action)
        assert next_obs.shape == (9,)
        assert len(info["rays"]) == 5

    def test_action_dimension_mapping(self):
        env = RacingEnv()
        assert env.action_space.shape == (2,)
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)

        # Steering test: positive steering commands right, negative left
        env.reset(options={"track_seed": 1001})
        env.step(np.array([1.0, 0.0]))  # max right steering
        assert np.isclose(env.car.steering, 30.0)

        env.step(np.array([-1.0, 0.0]))  # max left steering
        assert np.isclose(env.car.steering, -30.0)


class TestGradientCalculationAndAutograd:
    """Validate gradient-based saliency computation and autograd behavior."""

    @pytest.fixture
    def actor(self):
        return load_frozen_actor(FROZEN_MODEL_PATH)

    def test_saliency_computation_shape_and_types(self, actor):
        obs = np.array([3.5, 12.0, -5.0, 0.0, 36.0, 52.0, 110.0, 65.0, 36.0], dtype=np.float32)
        result = compute_action_saliency(actor, obs, return_presquashed=True)

        assert "deployed_action" in result
        assert len(result["deployed_action"]) == 2
        steer_act, long_act = result["deployed_action"]
        assert -1.0 <= steer_act <= 1.0
        assert -1.0 <= long_act <= 1.0

        for key in ["grad_steer", "grad_long", "saliency_steer", "saliency_long"]:
            arr = result[key]
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (9,)
            assert not np.isnan(arr).any(), f"NaN detected in {key}"
            assert not np.isinf(arr).any(), f"Inf detected in {key}"

        # Saliency must equal absolute value of gradients
        assert np.allclose(result["saliency_steer"], np.abs(result["grad_steer"]))
        assert np.allclose(result["saliency_long"], np.abs(result["grad_long"]))

    def test_parameters_remain_unmodified_without_accumulated_gradients(self, actor):
        obs = np.array([2.0, -5.0, 3.0, 0.0, 40.0, 60.0, 150.0, 70.0, 38.0], dtype=np.float32)
        compute_action_saliency(actor, obs, return_presquashed=True)

        for name, param in actor.named_parameters():
            assert param.grad is None, f"Parameter {name} accumulated gradients!"


class TestFiniteDifferenceValidation:
    """Validate analytical autograd gradients against finite-difference perturbations."""

    @pytest.fixture
    def actor(self):
        return load_frozen_actor(FROZEN_MODEL_PATH)

    def test_finite_difference_accuracy_on_sample_states(self, actor):
        test_states = [
            np.array([3.5, 12.0, -5.0, 0.0, 36.0, 52.0, 110.0, 65.0, 36.0], dtype=np.float32),
            np.array([1.5, -20.0, 8.0, 0.5, 20.0, 30.0, 45.0, 35.0, 20.0], dtype=np.float32),
            np.array([4.0, 0.0, 0.0, 0.0, 35.0, 70.0, 200.0, 70.0, 35.0], dtype=np.float32),
        ]

        for s in test_states:
            val = validate_gradients_finite_difference(actor, s, eps=1e-3, tolerance=5e-3)
            assert val["passed"] is True
            assert val["max_overall_error"] < 5e-3
            assert val["steering"]["mean_error"] < 1e-3
            assert val["longitudinal"]["mean_error"] < 1e-3


class TestSaliencyReproducibility:
    """Validate bit-for-bit reproducibility of saliency on identical observations."""

    def test_identical_observation_reproducibility(self):
        actor = load_frozen_actor(FROZEN_MODEL_PATH)
        obs = np.array([3.2, 8.5, -2.1, 0.05, 36.0, 54.0, 118.0, 62.0, 36.0], dtype=np.float32)

        res1 = compute_action_saliency(actor, obs)
        res2 = compute_action_saliency(actor, obs)

        assert np.array_equal(res1["grad_steer"], res2["grad_steer"])
        assert np.array_equal(res1["grad_long"], res2["grad_long"])
        assert np.array_equal(res1["saliency_steer"], res2["saliency_steer"])
        assert np.array_equal(res1["saliency_long"], res2["saliency_long"])


class TestDeliverableIntegrity:
    """Validate that results CSV and JSON files exist, are well-formed, and valid."""

    def test_csv_file_exists_and_valid(self):
        assert os.path.isfile(TIMESTEPS_CSV_PATH), f"Missing CSV at {TIMESTEPS_CSV_PATH}"
        with open(TIMESTEPS_CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 10000, f"Expected > 10000 rows in dataset, got {len(rows)}"

        # Check required columns
        required_cols = [
            "track_seed", "benchmark", "benchmark_param", "episode_id", "step",
            "obs_velocity", "obs_heading_error", "obs_signed_distance", "obs_slip",
            "obs_ray_minus_90", "obs_ray_minus_45", "obs_ray_0", "obs_ray_plus_45", "obs_ray_plus_90",
            "action_steering", "action_longitudinal", "action_mode",
            "saliency_steer_ray_0", "saliency_long_ray_0",
            "track_section", "outcome"
        ]
        first_row = rows[0]
        for col in required_cols:
            assert col in first_row, f"Missing required CSV column: {col}"

    def test_summary_json_exists_and_valid(self):
        assert os.path.isfile(SUMMARY_JSON_PATH), f"Missing JSON at {SUMMARY_JSON_PATH}"
        with open(SUMMARY_JSON_PATH, "r") as f:
            summary = json.load(f)

        meta = summary["metadata"]
        assert meta["weights_unaltered"] is True
        assert meta["sha256_before_analysis"] == EXPECTED_WEIGHTS_SHA256
        assert meta["sha256_after_analysis"] == EXPECTED_WEIGHTS_SHA256

        answers = summary["findings_key_answers"]
        assert answers["braking_highest_sensitivity_ray"] == "ray_0"
        assert answers["steering_highest_sensitivity_ray_scaled"] in ["ray_minus_45", "ray_plus_45", "ray_0"]

        val = summary["analysis_results"]["finite_difference_validation"]
        assert val["all_passed"] is True
        assert val["pass_rate"] == 1.0
