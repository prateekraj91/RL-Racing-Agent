"""Tests for the ablation study infrastructure.

Verifies:
1. Environment accepts steer_penalty_coeff parameter
2. steer_penalty_coeff=0 disables the steering penalty
3. Ablation condition naming is consistent
4. Factorial effects computation is correct
5. All 8 conditions are distinct
"""

import math
import numpy as np
import pytest
import torch

from env.environment import RacingEnv
from env.car import Car
from sac.track_pools import POOL_CONFIG


class TestSteerPenaltyCoeff:
    """Verify the environment respects the steer_penalty_coeff parameter."""

    def test_default_coefficient(self):
        """Default steer_penalty_coeff should be 0.01."""
        env = RacingEnv(max_steps=100)
        assert env.steer_penalty_coeff == 0.01

    def test_zero_coefficient(self):
        """Setting steer_penalty_coeff=0 should create an env with zero penalty."""
        env = RacingEnv(max_steps=100, steer_penalty_coeff=0.0)
        assert env.steer_penalty_coeff == 0.0

    def test_custom_coefficient(self):
        """Arbitrary coefficient values should be accepted."""
        env = RacingEnv(max_steps=100, steer_penalty_coeff=0.05)
        assert env.steer_penalty_coeff == 0.05

    def test_penalty_affects_reward(self):
        """Reward should differ between steer_penalty_coeff=0 and 0.01
        when the agent applies a large steering change."""
        tk = POOL_CONFIG["track_kwargs"]

        env_pen = RacingEnv(max_steps=100, track_kwargs=tk, steer_penalty_coeff=0.01)
        env_no_pen = RacingEnv(max_steps=100, track_kwargs=tk, steer_penalty_coeff=0.0)

        obs_pen, _ = env_pen.reset(options={"track_seed": 42})
        obs_no, _ = env_no_pen.reset(options={"track_seed": 42})

        # First step: gentle forward
        action_fwd = np.array([0.0, 0.5], dtype=np.float32)
        env_pen.step(action_fwd)
        env_no_pen.step(action_fwd)

        # Second step: hard left turn (big steer change from 0 -> +1)
        action_turn = np.array([1.0, 0.5], dtype=np.float32)
        _, r_pen, _, _, _ = env_pen.step(action_turn)
        _, r_no_pen, _, _, _ = env_no_pen.step(action_turn)

        # The steering change is |1.0 - 0.0| = 1.0
        # Penalty contribution (before ×100 scaling) is 0.01 * 1.0 = 0.01
        # After ×100 scaling: 1.0 point difference
        # With penalty OFF, the reward should be higher
        assert r_no_pen > r_pen, (
            f"Steering penalty OFF reward ({r_no_pen:.4f}) should exceed "
            f"penalty ON reward ({r_pen:.4f})"
        )

    def test_no_penalty_identical_for_smooth_steering(self):
        """When steering doesn't change, both coefficients should give same reward."""
        tk = POOL_CONFIG["track_kwargs"]

        env_pen = RacingEnv(max_steps=100, track_kwargs=tk, steer_penalty_coeff=0.01)
        env_no_pen = RacingEnv(max_steps=100, track_kwargs=tk, steer_penalty_coeff=0.0)

        env_pen.reset(options={"track_seed": 42})
        env_no_pen.reset(options={"track_seed": 42})

        # Identical constant-steering actions
        action = np.array([0.0, 0.5], dtype=np.float32)
        _, r1, _, _, _ = env_pen.step(action)
        _, r2, _, _, _ = env_no_pen.step(action)

        # First step: steer change from prev_steering=0 to 0 is 0
        # So penalty term is 0 regardless of coefficient
        assert abs(r1 - r2) < 1e-6, (
            f"With zero steer change, rewards should match: {r1:.6f} vs {r2:.6f}"
        )


class TestAblationConditions:
    """Verify the ablation condition definitions are correct."""

    CONDITIONS = [
        ("A", True,  200, 0.01),
        ("B", True,  200, 0.00),
        ("C", True,  300, 0.01),
        ("D", True,  300, 0.00),
        ("E", False, 200, 0.01),
        ("F", False, 200, 0.00),
        ("G", False, 300, 0.01),
        ("H", False, 300, 0.00),
    ]

    def test_eight_distinct_conditions(self):
        """All 8 conditions should have unique factor combinations."""
        combos = set()
        for label, dr, look, steer in self.CONDITIONS:
            combo = (dr, look, steer)
            assert combo not in combos, f"Duplicate factor combination: {combo}"
            combos.add(combo)
        assert len(combos) == 8

    def test_labels_are_sequential(self):
        """Labels should be A through H."""
        labels = [c[0] for c in self.CONDITIONS]
        assert labels == list("ABCDEFGH")

    def test_full_factorial_coverage(self):
        """Every combination of (DR, Lookahead, SteerPenalty) should appear exactly once."""
        from itertools import product
        expected = set(product([True, False], [200, 300], [0.0, 0.01]))
        actual = {(c[1], c[2], c[3]) for c in self.CONDITIONS}
        assert actual == expected, f"Missing combinations: {expected - actual}"

    def test_run_name_generation(self):
        """Run names should be deterministic and distinct."""
        from analysis.eval_ablation import run_name_for_condition
        names = set()
        for label, dr, look, steer in self.CONDITIONS:
            name = run_name_for_condition(label, dr, look, steer)
            assert name not in names, f"Duplicate run name: {name}"
            assert label in name
            names.add(name)


class TestFactorialEffects:
    """Verify factorial effects computation with known values."""

    def test_main_effect_computation(self):
        """Test main effects with a known response pattern."""
        from analysis.eval_ablation import compute_factorial_effects

        # Construct a scenario where DR has a clear positive effect
        # DR=On conditions (A,B,C,D) get 80% completion
        # DR=Off conditions (E,F,G,H) get 40% completion
        # Main effect of DR should be +0.40
        condition_results = {}
        for label in "ABCD":
            condition_results[label] = {
                "standard_heldout": {"completion_rate": 0.80, "avg_progress": 0.9,
                                      "crash_rate": 0.1, "avg_lap_time": 400}
            }
        for label in "EFGH":
            condition_results[label] = {
                "standard_heldout": {"completion_rate": 0.40, "avg_progress": 0.5,
                                      "crash_rate": 0.3, "avg_lap_time": 450}
            }

        effects = compute_factorial_effects(condition_results)
        dr_effect = effects["completion_rate"]["main_effects"]["DR"]
        assert abs(dr_effect - 0.40) < 0.001, f"Expected DR main effect ≈ 0.40, got {dr_effect}"

    def test_no_interaction_when_additive(self):
        """If effects are purely additive, interactions should be zero."""
        from analysis.eval_ablation import compute_factorial_effects

        # Purely additive: DR adds 0.20, Lookahead adds 0.10, Steer adds 0.05
        base = 0.30
        responses = {
            "A": base + 0.20 + 0.00 + 0.05,  # DR=1, Look=0, Steer=1
            "B": base + 0.20 + 0.00 + 0.00,  # DR=1, Look=0, Steer=0
            "C": base + 0.20 + 0.10 + 0.05,  # DR=1, Look=1, Steer=1
            "D": base + 0.20 + 0.10 + 0.00,  # DR=1, Look=1, Steer=0
            "E": base + 0.00 + 0.00 + 0.05,  # DR=0, Look=0, Steer=1
            "F": base + 0.00 + 0.00 + 0.00,  # DR=0, Look=0, Steer=0
            "G": base + 0.00 + 0.10 + 0.05,  # DR=0, Look=1, Steer=1
            "H": base + 0.00 + 0.10 + 0.00,  # DR=0, Look=1, Steer=0
        }

        condition_results = {}
        for label, cr in responses.items():
            condition_results[label] = {
                "standard_heldout": {"completion_rate": cr, "avg_progress": cr,
                                      "crash_rate": 1 - cr, "avg_lap_time": 400}
            }

        effects = compute_factorial_effects(condition_results)
        eff = effects["completion_rate"]

        assert abs(eff["main_effects"]["DR"] - 0.20) < 0.001
        assert abs(eff["main_effects"]["Lookahead"] - 0.10) < 0.001
        assert abs(eff["main_effects"]["SteerPenalty"] - 0.05) < 0.001

        for pair, val in eff["two_factor_interactions"].items():
            assert abs(val) < 0.001, f"Interaction {pair} should be 0, got {val}"

        three = eff["three_factor_interaction"]["DR×Lookahead×SteerPenalty"]
        assert abs(three) < 0.001, f"Three-way interaction should be 0, got {three}"


class TestEnvironmentBackwardsCompatibility:
    """Ensure the steer_penalty_coeff change doesn't break existing usage."""

    def test_env_without_steer_penalty_arg(self):
        """RacingEnv() without steer_penalty_coeff should use default 0.01."""
        env = RacingEnv(max_steps=50)
        assert env.steer_penalty_coeff == 0.01

    def test_existing_training_script_default(self):
        """The training script parser default should be 0.01."""
        from sac.train_curriculum import build_parser
        parser = build_parser()
        args = parser.parse_args(["--name", "test_run"])
        assert args.steer_penalty == 0.01

    def test_training_script_zero_penalty(self):
        """--steer-penalty 0 should parse correctly."""
        from sac.train_curriculum import build_parser
        parser = build_parser()
        args = parser.parse_args(["--name", "test_run", "--steer-penalty", "0"])
        assert args.steer_penalty == 0.0
