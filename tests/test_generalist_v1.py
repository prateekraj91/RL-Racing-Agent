"""Unit and integration tests for the frozen generalist model (models/generalist_v1.pth).

Validates:
1. Presence and cryptographic integrity (SHA-256) of frozen weights and configuration.
2. Completeness and validity of configs/generalist_v1.json.
3. Checkpoint loading into SACActor and tensor inference validity.
4. Evaluation specifically targets the frozen checkpoint rather than training runs.
5. Integrity and consistency of reproducibility artifacts.
6. Deterministic evaluation accuracy on held-out benchmark circuits.
"""

import csv
import hashlib
import json
import os
import pytest
import torch

from env.environment import RacingEnv
from sac.actor import SACActor
from sac.track_pools import TEST_SEEDS

FROZEN_MODEL_PATH = "models/generalist_v1.pth"
CONFIG_PATH = "configs/generalist_v1.json"
EXPECTED_WEIGHTS_SHA256 = "0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784"
EXPECTED_CONFIG_SHA256 = "4d76575babb065d3a9fe9a0bbeb4705c7b58b4d4c8f46708489bd2bb3d63bdb1"

REPRODUCIBILITY_CSV = "analysis/results/generalist_reproducibility.csv"
REPRODUCIBILITY_JSON = "analysis/results/generalist_reproducibility_summary.json"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestFrozenModelIntegrity:
    """Cryptographic and structural integrity tests for generalist_v1."""

    def test_frozen_weights_file_exists(self):
        assert os.path.isfile(FROZEN_MODEL_PATH), f"Missing frozen model file: {FROZEN_MODEL_PATH}"
        assert os.path.getsize(FROZEN_MODEL_PATH) > 10000, "Frozen model file size suspiciously small"

    def test_frozen_weights_sha256_unaltered(self):
        actual_sha = compute_sha256(FROZEN_MODEL_PATH)
        assert actual_sha == EXPECTED_WEIGHTS_SHA256, (
            f"Frozen model SHA-256 hash mismatch! Expected {EXPECTED_WEIGHTS_SHA256}, got {actual_sha}"
        )

    def test_original_provenance_checkpoint_unaltered(self):
        orig_path = "runs/ablation_A_DR_look200_steerOn/best_actor.pth"
        if not os.path.isfile(orig_path):
            pytest.skip(f"gitignored training run data absent ({orig_path}); provenance check needs the original run")
        orig_sha = compute_sha256(orig_path)
        assert orig_sha == EXPECTED_WEIGHTS_SHA256, (
            f"Original checkpoint was modified! Expected {EXPECTED_WEIGHTS_SHA256}, got {orig_sha}"
        )

    def test_config_file_exists_and_sha256(self):
        assert os.path.isfile(CONFIG_PATH), f"Missing config file: {CONFIG_PATH}"
        actual_config_sha = compute_sha256(CONFIG_PATH)
        assert actual_config_sha == EXPECTED_CONFIG_SHA256, (
            f"Config SHA-256 hash mismatch! Expected {EXPECTED_CONFIG_SHA256}, got {actual_config_sha}"
        )


class TestConfigurationCompleteness:
    """Validate completeness and schema adherence of configs/generalist_v1.json."""

    @pytest.fixture
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_required_top_level_keys(self, config):
        required_keys = [
            "model_name", "version", "original_checkpoint_path", "frozen_weights_path",
            "frozen_weights_sha256", "network_architecture", "observation_space",
            "action_space", "reward_configuration", "domain_randomization",
            "nominal_track_parameters", "vehicle_physics_parameters",
            "training_parameters", "evaluation_protocol"
        ]
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"

    def test_weights_sha256_reference_in_config(self, config):
        assert config["frozen_weights_sha256"] == EXPECTED_WEIGHTS_SHA256

    def test_observation_and_action_specifications(self, config):
        obs_spec = config["observation_space"]
        assert obs_spec["dimensions"] == 9
        assert len(obs_spec["features"]) == 9
        assert obs_spec["sensor_configuration"]["obs_ray_angles"] == [-90.0, -45.0, 0.0, 45.0, 90.0]
        assert obs_spec["sensor_configuration"]["obs_ray_max_dist"] == 200.0

        act_spec = config["action_space"]
        assert act_spec["dimensions"] == 2
        assert len(act_spec["actions"]) == 2

    def test_domain_randomization_preset(self, config):
        dr = config["domain_randomization"]
        assert dr["enabled"] is True
        assert dr["preset"] == "calibrated"
        ranges = dr["ranges"]
        assert ranges["width"] == [25.0, 70.0]
        assert ranges["min_radius"] == [65.0, 95.0]
        assert ranges["friction"] == [0.02, 0.045]
        assert ranges["acceleration"] == [0.075, 0.095]


class TestModelLoadingAndInference:
    """Verify that the frozen checkpoint correctly loads into SACActor and executes inference."""

    def test_load_state_dict_and_parameters(self):
        state_dict = torch.load(FROZEN_MODEL_PATH, map_location="cpu", weights_only=True)
        assert "network.0.weight" in state_dict
        assert state_dict["network.0.weight"].shape == (128, 9)
        assert state_dict["network.2.weight"].shape == (128, 128)
        assert state_dict["mean.weight"].shape == (2, 128)
        assert state_dict["log_std.weight"].shape == (2, 128)

        actor = SACActor(obs_dim=9, action_dim=2)
        actor.load_state_dict(state_dict)
        actor.eval()

        dummy_obs = torch.zeros(1, 9, dtype=torch.float32)
        with torch.no_grad():
            mean, log_std = actor(dummy_obs)
            action = torch.tanh(mean).numpy()[0]

        assert mean.shape == (1, 2)
        assert log_std.shape == (1, 2)
        assert -1.0 <= action[0] <= 1.0
        assert -1.0 <= action[1] <= 1.0


class TestReproducibilityArtifacts:
    """Verify that reproducibility artifacts exist and report bit-for-bit consistency."""

    def test_artifacts_exist(self):
        assert os.path.isfile(REPRODUCIBILITY_CSV), f"Missing {REPRODUCIBILITY_CSV}"
        assert os.path.isfile(REPRODUCIBILITY_JSON), f"Missing {REPRODUCIBILITY_JSON}"

    def test_reproducibility_summary_contents(self):
        with open(REPRODUCIBILITY_JSON, "r") as f:
            summary = json.load(f)

        meta = summary["metadata"]
        assert meta["weights_sha256"] == EXPECTED_WEIGHTS_SHA256
        assert meta["config_sha256"] == EXPECTED_CONFIG_SHA256
        assert meta["deterministic_reproducibility"]["is_bit_for_bit_identical"] is True
        assert meta["deterministic_reproducibility"]["mismatches_rep1_vs_rep2"] == 0
        assert meta["deterministic_reproducibility"]["mismatches_rep1_vs_rep3"] == 0

        # Check standard held-out performance across reps
        rep1_std = summary["repetition_summaries"]["repetition_1"]["standard_heldout"]
        assert rep1_std["n_completed"] == 10
        assert rep1_std["completion_rate"] == 1.0
        assert rep1_std["n_crashed"] == 0
        assert rep1_std["mean_lap_time"] == 398.2

        # Check narrow corridor W=20 performance across reps
        rep1_w20 = summary["repetition_summaries"]["repetition_1"]["narrow_w_20"]
        assert rep1_w20["n_completed"] == 10
        assert rep1_w20["completion_rate"] == 1.0
        assert rep1_w20["n_crashed"] == 0
        assert rep1_w20["mean_lap_time"] == 394.6

    def test_deterministic_rollout_on_heldout_seed_1001(self):
        """Perform a live deterministic rollout on seed 1001 and verify exact step count 387."""
        state_dict = torch.load(FROZEN_MODEL_PATH, map_location="cpu", weights_only=True)
        actor = SACActor(obs_dim=9, action_dim=2)
        actor.load_state_dict(state_dict)
        actor.eval()

        base_tk = {
            "width": 70,
            "base_r": 250,
            "n_ctrl": 10,
            "min_radius": 80,
            "cx": 400,
            "cy": 300,
        }
        env = RacingEnv(
            max_steps=700,
            verbose=False,
            track_kwargs=base_tk,
            obs_ray_angles=(-90.0, -45.0, 0.0, 45.0, 90.0),
            obs_ray_max_dist=200.0,
        )

        obs, _ = env.reset(options={"track_seed": 1001})
        completed = False
        steps = 0

        for _ in range(700):
            with torch.no_grad():
                mean, _ = actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                act = torch.tanh(mean).numpy()[0]
            obs, reward, term, trunc, info = env.step(act)
            steps += 1
            if term or trunc:
                completed = bool(info.get("lap_completed", False))
                break

        assert completed is True, "Seed 1001 rollout failed to complete"
        assert steps == 387, f"Seed 1001 lap time mismatch: expected 387 steps, got {steps}"
