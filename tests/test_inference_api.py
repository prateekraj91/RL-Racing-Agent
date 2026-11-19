"""Test suite for FastAPI inference service and frozen SAC generalist-v1.0 policy.

Covers requirements A through M:
- A: GET /health returns 200 with complete model metadata.
- B: Model loading and SHA-256 hash verification succeeds.
- C: POST /predict accepts 9 valid observations.
- D: Output contains exactly 2 normalized actions in [-1, 1].
- E: All outputs are finite floats (no NaN / Inf).
- F: Identical observations produce identical deterministic actions.
- G: API output matches direct SACActor tanh(mean) inference within 1e-5 tolerance.
- H: POST /predict/batch preserves ordering and matches individual predictions.
- I: Vectors with length != 9 are rejected with 422 Unprocessable Entity.
- J: NaNs and Infs are rejected with 422 Unprocessable Entity.
- K: Out-of-range observations (> 1000 or < -1000) are rejected with 422.
- L: Hash mismatch prevents startup or raises RuntimeError.
- M: Frozen model weights and configuration on disk remain strictly unchanged.
"""

import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import torch

from pydantic import ValidationError

from sac.actor import SACActor
from inference.app import app
from inference.schemas import ObservationRequest
from inference.model_loader import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_CONFIG_SHA256,
    compute_sha256,
    load_inference_actor,
    verify_file_integrity,
)


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient to trigger lifespan context manager."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def direct_actor():
    """Directly loaded SACActor for mathematical parity verification."""
    actor, _ = load_inference_actor(
        model_path="models/generalist_v1.pth",
        config_path="configs/generalist_v1.json",
        device="cpu",
    )
    return actor


# Sample realistic observation from racing telemetry:
# [velocity, err, dist, slip, ray_-90, ray_-45, ray_0, ray_45, ray_90]
SAMPLE_OBS_1 = [3.23, -12.4, -0.7, 0.0, 36.0, 52.0, 116.0, 48.0, 36.0]
SAMPLE_OBS_2 = [3.80, 0.0, 0.0, 0.0, 100.0, 150.0, 200.0, 150.0, 100.0]
SAMPLE_OBS_3 = [1.50, 45.0, 10.0, -0.2, 10.0, 20.0, 40.0, 90.0, 120.0]


def test_req_a_health_endpoint(client: TestClient):
    """Test A: GET /health returns 200 with complete model metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["weights_sha256"] == EXPECTED_MODEL_SHA256
    assert data["config_sha256"] == EXPECTED_CONFIG_SHA256
    assert data["obs_dim"] == 9
    assert data["action_dim"] == 2
    assert data["parameter_count"] == 18308
    assert data["device"] == "cpu"


def test_req_b_model_loading_and_hash_verification():
    """Test B: Model loading and hash verification succeeds."""
    model_path = Path("models/generalist_v1.pth")
    config_path = Path("configs/generalist_v1.json")

    # Hash verification passes
    model_hash = verify_file_integrity(model_path, EXPECTED_MODEL_SHA256, "model")
    config_hash = verify_file_integrity(config_path, EXPECTED_CONFIG_SHA256, "config")
    assert model_hash == EXPECTED_MODEL_SHA256
    assert config_hash == EXPECTED_CONFIG_SHA256

    # Model loader instantiates actor without errors
    actor, meta = load_inference_actor(model_path, config_path, device="cpu")
    assert isinstance(actor, SACActor)
    assert meta["parameter_count"] == 18308
    assert meta["weights_sha256"] == EXPECTED_MODEL_SHA256


def test_req_c_predict_valid_observation(client: TestClient):
    """Test C: POST /predict accepts 9 valid observations."""
    response = client.post("/predict", json={"observation": SAMPLE_OBS_1})
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert "steering" in data
    assert "throttle" in data
    assert "steering_angle_deg" in data
    assert "action_mode" in data
    assert "inference_latency_ms" in data


def test_req_d_normalized_action_range(client: TestClient):
    """Test D: Output contains exactly 2 normalized actions in [-1, 1]."""
    response = client.post("/predict", json={"observation": SAMPLE_OBS_1})
    assert response.status_code == 200
    data = response.json()
    action = data["action"]
    assert len(action) == 2
    assert -1.0 <= action[0] <= 1.0
    assert -1.0 <= action[1] <= 1.0
    assert -1.0 <= data["steering"] <= 1.0
    assert -1.0 <= data["throttle"] <= 1.0
    assert -30.0 <= data["steering_angle_deg"] <= 30.0
    assert data["action_mode"] in ("throttle", "deceleration", "coasting")


def test_req_e_finite_float_outputs(client: TestClient):
    """Test E: All outputs are finite floats (no NaN / Inf)."""
    response = client.post("/predict", json={"observation": SAMPLE_OBS_1})
    assert response.status_code == 200
    data = response.json()
    for val in data["action"]:
        assert isinstance(val, float)
        assert math.isfinite(val)
    assert math.isfinite(data["steering"])
    assert math.isfinite(data["throttle"])
    assert math.isfinite(data["steering_angle_deg"])
    if data["inference_latency_ms"] is not None:
        assert math.isfinite(data["inference_latency_ms"])


def test_req_f_deterministic_inference(client: TestClient):
    """Test F: Identical observations produce identical deterministic actions."""
    res1 = client.post("/predict", json={"observation": SAMPLE_OBS_1})
    res2 = client.post("/predict", json={"observation": SAMPLE_OBS_1})
    assert res1.status_code == 200
    assert res2.status_code == 200
    assert res1.json()["action"] == res2.json()["action"]


def test_req_g_parity_with_direct_actor(client: TestClient, direct_actor: SACActor):
    """Test G: API output matches direct SACActor tanh(mean) inference within 1e-5."""
    test_cases = [SAMPLE_OBS_1, SAMPLE_OBS_2, SAMPLE_OBS_3]
    for obs in test_cases:
        # API inference
        api_res = client.post("/predict", json={"observation": obs})
        assert api_res.status_code == 200
        api_action = api_res.json()["action"]

        # Direct mathematical inference
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            mean, _ = direct_actor(obs_tensor)
            expected_action = torch.tanh(mean).squeeze(0).tolist()

        # Verify within 1e-5 tolerance
        assert abs(api_action[0] - expected_action[0]) < 1e-5
        assert abs(api_action[1] - expected_action[1]) < 1e-5


def test_req_h_batch_prediction_ordering_and_parity(client: TestClient):
    """Test H: POST /predict/batch preserves ordering and matches individual predictions."""
    batch_obs = [SAMPLE_OBS_1, SAMPLE_OBS_2, SAMPLE_OBS_3]
    batch_res = client.post("/predict/batch", json={"observations": batch_obs})
    assert batch_res.status_code == 200
    batch_data = batch_res.json()

    assert batch_data["batch_size"] == len(batch_obs)
    assert len(batch_data["actions"]) == len(batch_obs)

    # Compare each batch item to individual prediction
    for i, obs in enumerate(batch_obs):
        single_res = client.post("/predict", json={"observation": obs})
        assert single_res.status_code == 200
        single_data = single_res.json()

        batch_action = batch_data["actions"][i]
        assert batch_action["action"] == single_data["action"]
        assert batch_action["steering"] == single_data["steering"]
        assert batch_action["throttle"] == single_data["throttle"]
        assert batch_action["steering_angle_deg"] == single_data["steering_angle_deg"]
        assert batch_action["action_mode"] == single_data["action_mode"]


def test_req_i_invalid_observation_length(client: TestClient):
    """Test I: Vectors with length != 9 are rejected with 422."""
    # Too few elements
    res_short = client.post("/predict", json={"observation": [1.0, 2.0, 3.0]})
    assert res_short.status_code == 422

    # Too many elements
    res_long = client.post("/predict", json={"observation": [0.0] * 10})
    assert res_long.status_code == 422

    # Empty batch item
    res_batch_short = client.post(
        "/predict/batch", json={"observations": [[1.0, 2.0], SAMPLE_OBS_1]}
    )
    assert res_batch_short.status_code == 422


def test_req_j_nan_and_inf_rejection(client: TestClient):
    """Test J: NaNs and Infs are rejected with 422."""
    # 1. Null / None in JSON body
    res_null = client.post(
        "/predict", json={"observation": [3.23, None, -0.7, 0.0, 36.0, 52.0, 116.0, 48.0, 36.0]}
    )
    assert res_null.status_code == 422

    # 2. Raw JSON NaN
    res_nan = client.post(
        "/predict",
        content='{"observation": [3.23, -12.4, NaN, 0.0, 36.0, 52.0, 116.0, 48.0, 36.0]}',
        headers={"Content-Type": "application/json"},
    )
    assert res_nan.status_code == 422

    # 3. Raw JSON Infinity
    res_inf = client.post(
        "/predict",
        content='{"observation": [Infinity, -12.4, -0.7, 0.0, 36.0, 52.0, 116.0, 48.0, 36.0]}',
        headers={"Content-Type": "application/json"},
    )
    assert res_inf.status_code == 422

    # 4. Direct Pydantic schema validation for Python float('nan') and float('inf')
    invalid_nan = list(SAMPLE_OBS_1)
    invalid_nan[2] = float("nan")
    with pytest.raises(ValidationError):
        ObservationRequest(observation=invalid_nan)

    invalid_inf = list(SAMPLE_OBS_1)
    invalid_inf[0] = float("inf")
    with pytest.raises(ValidationError):
        ObservationRequest(observation=invalid_inf)


def test_req_k_out_of_range_rejection(client: TestClient):
    """Test K: Out-of-range observations (> 1000 or < -1000) are rejected with 422."""
    # Positive out of range
    res_high = client.post(
        "/predict", json={"observation": [1000.5, 0.0, 0.0, 0.0, 50.0, 50.0, 50.0, 50.0, 50.0]}
    )
    assert res_high.status_code == 422

    # Negative out of range
    res_low = client.post(
        "/predict", json={"observation": [-1000.5, 0.0, 0.0, 0.0, 50.0, 50.0, 50.0, 50.0, 50.0]}
    )
    assert res_low.status_code == 422


def test_req_l_hash_mismatch_raises_error(tmp_path: Path):
    """Test L: Hash mismatch prevents startup or raises RuntimeError."""
    fake_weights = tmp_path / "corrupted_weights.pth"
    fake_weights.write_bytes(b"corrupted binary payload")

    with pytest.raises(RuntimeError) as exc_info:
        load_inference_actor(model_path=fake_weights)
    assert "Integrity check failed" in str(exc_info.value)


def test_req_m_model_weights_and_config_remain_unchanged():
    """Test M: Frozen model weights and configuration on disk remain unchanged."""
    actual_model_hash = compute_sha256("models/generalist_v1.pth")
    actual_config_hash = compute_sha256("configs/generalist_v1.json")

    assert (
        actual_model_hash == EXPECTED_MODEL_SHA256
    ), f"Model hash changed! Expected {EXPECTED_MODEL_SHA256}, got {actual_model_hash}"
    assert (
        actual_config_hash == EXPECTED_CONFIG_SHA256
    ), f"Config hash changed! Expected {EXPECTED_CONFIG_SHA256}, got {actual_config_hash}"
