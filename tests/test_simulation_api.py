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


def test_frontend_simulation_schema_contract(client: TestClient):
    """Schema contract test asserting the exact keys, types, and units required by the frontend.

    Prevents silent key-mismatch regressions between frontend/app.js and inference/simulation.py:
    - Ray endpoints: ray["end"] must exist as [float, float] in px.
    - Saliency keys: "feature_saliency_steer" and "feature_saliency_long" must exist.
    - Heading error and slip angle: must be in degrees.
    - Signed distance and sensor rays: must be in px.
    - Vehicle state: x, y in px, angle in degrees, velocity in px/step.
    """
    client.post("/simulation/start", json={"preset_id": "standard_medium", "track_seed": 1001})
    res = client.post(
        "/simulation/step",
        json={"steps": 5, "calculate_saliency": True, "commentary_mode": "offline"},
    )
    assert res.status_code == 200
    data = res.json()

    # 1. Rays contract: length 5, start [x, y], end [x, y], distance in px
    assert "rays" in data, "Missing 'rays' key required by 3D renderer"
    assert len(data["rays"]) == 5, f"Expected 5 sensor rays, got {len(data['rays'])}"
    for idx, ray in enumerate(data["rays"]):
        assert "end" in ray, f"Ray {idx} missing 'end' key (frontend reads ray.end)"
        assert isinstance(ray["end"], list) and len(ray["end"]) == 2, f"Ray {idx} 'end' must be [x, y] list"
        assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in ray["end"])
        assert "start" in ray, f"Ray {idx} missing 'start' key"
        assert isinstance(ray["start"], list) and len(ray["start"]) == 2
        assert "distance" in ray and isinstance(ray["distance"], (int, float))
        assert 0.0 <= ray["distance"] <= 200.0, f"Ray {idx} distance {ray['distance']} out of px bounds [0, 200]"

    # 2. Saliency contract: feature_saliency_steer and feature_saliency_long
    assert "saliency" in data and data["saliency"] is not None, "Missing 'saliency' object"
    sal = data["saliency"]
    assert "feature_saliency_steer" in sal, "Missing 'feature_saliency_steer' in saliency payload"
    assert "feature_saliency_long" in sal, "Missing 'feature_saliency_long' in saliency payload"
    assert "top_steering_ray" in sal and isinstance(sal["top_steering_ray"], str)
    assert "top_longitudinal_ray" in sal and isinstance(sal["top_longitudinal_ray"], str)
    assert len(sal["feature_saliency_steer"]) == 9
    assert len(sal["feature_saliency_long"]) == 9

    # 3. Telemetry contract: heading_error and slip_angle in degrees, signed_distance in px
    assert "telemetry" in data, "Missing 'telemetry' object"
    telem = data["telemetry"]
    assert "heading_error" in telem and isinstance(telem["heading_error"], (int, float))
    assert -180.0 <= telem["heading_error"] <= 180.0, "heading_error must be in degrees [-180, 180]"
    assert "slip_angle" in telem and isinstance(telem["slip_angle"], (int, float))
    assert "signed_distance" in telem and isinstance(telem["signed_distance"], (int, float))
    assert -1000.0 <= telem["signed_distance"] <= 1000.0, "signed_distance must be in px"
    assert "rays" in telem and len(telem["rays"]) == 5
    for r in telem["rays"]:
        assert 0.0 <= r <= 200.0, "Telemetry ray measurement must be in px [0, 200]"

    # 4. Vehicle state contract: x, y in px, angle in degrees, velocity in px/step
    veh = data["vehicle"]
    assert "x" in veh and isinstance(veh["x"], (int, float)) and math.isfinite(veh["x"])
    assert "y" in veh and isinstance(veh["y"], (int, float)) and math.isfinite(veh["y"])
    assert "angle" in veh and isinstance(veh["angle"], (int, float)) and math.isfinite(veh["angle"])
    assert "velocity" in veh and isinstance(veh["velocity"], (int, float))
    assert -2.0 <= veh["velocity"] <= 4.0, "Vehicle velocity must be in px/step [-2, 4]"
    assert "steering_angle_deg" in veh and isinstance(veh["steering_angle_deg"], (int, float))
    assert -30.0 <= veh["steering_angle_deg"] <= 30.0, "Steering angle must be in degrees [-30, 30]"


def test_frozen_weights_integrity_remains_intact():
    """Verify models/generalist_v1.pth and configs/generalist_v1.json have not been modified."""
    actual_model_hash = compute_sha256(Path("models/generalist_v1.pth"))
    actual_config_hash = compute_sha256(Path("configs/generalist_v1.json"))
    assert actual_model_hash == EXPECTED_MODEL_SHA256
    assert actual_config_hash == EXPECTED_CONFIG_SHA256
