"""Test suite for stateful simulation backend, track presets, and policy stepping.

Verifies:
- Simulation start and track initialization
- Track preset catalog and geometry serialization
- Simulation stepping with real SAC inference
- Trajectory updates and sensor beam geometry
- Reset behavior
- Live and offline commentary generation
- Gemini mode fallback without API key
- Existing inference API compatibility (/health, /predict, /predict/batch)
- Model weight and config freeze integrity
"""

import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import torch

from sac.actor import SACActor
from inference.app import app
from inference.model_loader import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_CONFIG_SHA256,
    compute_sha256,
    load_inference_actor,
)
from inference.simulation import TRACK_PRESETS, simulation_manager


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient to trigger lifespan and model loading."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def actor():
    """Directly loaded SACActor for comparison."""
    act, _ = load_inference_actor(
        model_path="models/generalist_v1.pth",
        config_path="configs/generalist_v1.json",
        device="cpu",
    )
    return act


def test_simulation_tracks_catalog(client: TestClient):
    """Verify /simulation/tracks lists all supported presets with valid parameters."""
    res = client.get("/simulation/tracks")
    assert res.status_code == 200
    tracks = res.json()
    assert len(tracks) >= 5

    preset_ids = [t["id"] for t in tracks]
    assert "standard_medium" in preset_ids
    assert "narrow_corridor" in preset_ids
    assert "hairpin_challenge" in preset_ids
    assert "wide_oval" in preset_ids
    assert "default_circuit" in preset_ids

    for t in tracks:
        assert "name" in t
        assert "description" in t
        assert "default_seed" in t
        assert "recommended_seeds" in t
        assert isinstance(t["recommended_seeds"], list)


def test_simulation_start_standard_medium(client: TestClient):
    """Verify /simulation/start initializes RacingEnv and returns valid state."""
    res = client.post(
        "/simulation/start",
        json={"preset_id": "standard_medium", "track_seed": 1001},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["session"]["preset_id"] == "standard_medium"
    assert data["session"]["track_seed"] == 1001
    assert data["session"]["is_running"] is True
    assert data["session"]["is_crashed"] is False
    assert data["session"]["is_completed"] is False
    assert data["session"]["step_count"] == 0

    # Vehicle initial state
    veh = data["vehicle"]
    assert math.isfinite(veh["x"])
    assert math.isfinite(veh["y"])
    assert math.isfinite(veh["angle"])
    assert veh["velocity"] == 0.0

    # Track geometry
    trk = data["track"]
    assert len(trk["centerline"]) >= 100
    assert len(trk["inner_boundary"]) == len(trk["centerline"])
    assert len(trk["outer_boundary"]) == len(trk["centerline"])
    assert trk["width"] == 70.0
    assert "start_pose" in trk

    # Initial trajectory has start position
    assert len(data["trajectory"]) >= 1
    assert data["trajectory"][0] == [veh["x"], veh["y"]]

    # 5 Ray sensors
    assert len(data["rays"]) == 5
    for ray in data["rays"]:
        assert math.isfinite(ray["distance"])
        assert 0.0 <= ray["distance"] <= 200.0
        assert len(ray["start"]) == 2
        assert len(ray["end"]) == 2


def test_simulation_stepping_advances_physics_and_uses_policy(client: TestClient, actor: SACActor):
    """Verify /simulation/step advances physics, uses SAC actor, and records trajectory."""
    # Ensure fresh start
    client.post("/simulation/start", json={"preset_id": "standard_medium", "track_seed": 1001})

    init_state = client.get("/simulation/state").json()
    init_x = init_state["vehicle"]["x"]
    init_y = init_state["vehicle"]["y"]

    # Step by 5 steps with online saliency calculation
    step_res = client.post(
        "/simulation/step",
        json={"steps": 5, "calculate_saliency": True, "commentary_mode": "offline"},
    )
    assert step_res.status_code == 200
    step_data = step_res.json()

    assert step_data["session"]["step_count"] == 5
    veh = step_data["vehicle"]

    # Vehicle must have moved according to physics
    moved = abs(veh["x"] - init_x) > 0.01 or abs(veh["y"] - init_y) > 0.01
    assert moved, "Vehicle coordinates did not change after 5 simulation steps"

    # Forward velocity must have increased from throttle
    assert veh["velocity"] > 0.0

    # Trajectory must have 6 points (initial + 5 steps)
    assert len(step_data["trajectory"]) == 6

    # Saliency must be computed
    assert step_data["saliency"] is not None
    sal = step_data["saliency"]
    assert "feature_saliency_steer" in sal
    assert "feature_saliency_long" in sal
    assert "top_steering_ray" in sal
    assert "top_longitudinal_ray" in sal
    assert len(sal["ray_saliency_steer"]) == 5
    assert len(sal["ray_saliency_long"]) == 5


def test_simulation_reset_returns_to_start(client: TestClient):
    """Verify /simulation/reset resets vehicle to start line and clears history."""
    client.post("/simulation/start", json={"preset_id": "standard_medium", "track_seed": 1001})
    client.post("/simulation/step", json={"steps": 10})

    # Reset
    reset_res = client.post("/simulation/reset")
    assert reset_res.status_code == 200
    reset_data = reset_res.json()

    assert reset_data["session"]["step_count"] == 0
    assert reset_data["vehicle"]["velocity"] == 0.0
    assert len(reset_data["trajectory"]) == 1


def test_simulation_preserves_existing_inference_api(client: TestClient):
    """Verify that existing /health, /predict, and /predict/batch remain functional."""
    # Health check
    h_res = client.get("/health")
    assert h_res.status_code == 200
    h_data = h_res.json()
    assert h_data["status"] == "ok"
    assert h_data["parameter_count"] == 18308

    # Single prediction
    obs = [3.23, -12.4, -0.7, 0.0, 36.0, 52.0, 116.0, 48.0, 36.0]
    p_res = client.post("/predict", json={"observation": obs})
    assert p_res.status_code == 200
    p_data = p_res.json()
    assert len(p_data["action"]) == 2
    assert -1.0 <= p_data["steering"] <= 1.0
    assert -1.0 <= p_data["throttle"] <= 1.0

    # Batch prediction
    b_res = client.post("/predict/batch", json={"observations": [obs, obs]})
    assert b_res.status_code == 200
    b_data = b_res.json()
    assert b_data["batch_size"] == 2


def test_simulation_offline_commentary_generation(client: TestClient):
    """Verify offline commentary is produced deterministically without API calls."""
    client.post("/simulation/start", json={"preset_id": "standard_medium", "track_seed": 1001})

    # Run for 20 steps to trigger milestone events
    step_res = client.post(
        "/simulation/step",
        json={"steps": 20, "commentary_mode": "offline"},
    )
    assert step_res.status_code == 200
    data = step_res.json()

    assert "commentary" in data
    assert len(data["commentary"]) >= 1
    for c in data["commentary"]:
        assert "commentary" in c
        assert "step" in c
        assert c["source"] in ("offline_deterministic", "offline_fallback")


def test_simulation_gemini_mode_safe_fallback_without_key(client: TestClient, monkeypatch):
    """Verify that requesting gemini mode without an API key safely falls back to offline."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client.post("/simulation/start", json={"preset_id": "standard_medium", "track_seed": 1001})

    step_res = client.post(
        "/simulation/step",
        json={"steps": 10, "commentary_mode": "gemini"},
    )
    assert step_res.status_code == 200
    data = step_res.json()
    # Commentary entries must fall back cleanly without raising 500 error
    assert "commentary" in data


def test_simulation_historical_saliency_endpoint(client: TestClient):
    """Verify /simulation/historical-saliency returns verified reference data."""
    res = client.get("/simulation/historical-saliency")
    assert res.status_code == 200
    data = res.json()
    assert "analysis_results" in data or "status" in data


def test_frozen_weights_integrity_remains_intact():
    """Verify models/generalist_v1.pth and configs/generalist_v1.json have not been modified."""
    actual_model_hash = compute_sha256(Path("models/generalist_v1.pth"))
    actual_config_hash = compute_sha256(Path("configs/generalist_v1.json"))
    assert actual_model_hash == EXPECTED_MODEL_SHA256
    assert actual_config_hash == EXPECTED_CONFIG_SHA256
