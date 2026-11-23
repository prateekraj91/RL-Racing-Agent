"""Tests for the Gemini Telemetry Commentary Pipeline.

Validates:
1. Deterministic telemetry event detection and per-lap grouping.
2. Saliency trigger extraction and dominant beam attribution.
3. Structured prompt construction and schema adherence.
4. Grounding validator rigor (approving empirical facts, rejecting hallucinations).
5. Strict API call limits and counter enforcement.
6. Credential safety (verifying .env is git-ignored and .env.example contains placeholders).
7. Frozen generalist-v1.0 model checkpoint and config SHA-256 integrity.

ZERO live external API calls are made during these tests (mock_mode=True).
"""

import json
import os
import subprocess
from pathlib import Path
import pytest

from analysis.telemetry_events import (
    detect_episode_telemetry_events,
    group_telemetry_by_episode,
    load_commentary_config,
    load_telemetry_dataset,
    select_key_lap_events,
)
from analysis.telemetry_commentary import (
    CONFIG_PATH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_WEIGHTS_SHA256,
    FROZEN_MODEL_PATH,
    HISTORICAL_EXAMPLES_JSON_PATH,
    HISTORICAL_SUMMARY_JSON_PATH,
    REMEDIATED_EXAMPLES_JSON_PATH,
    REMEDIATED_SUMMARY_JSON_PATH,
    GeminiCommentaryClient,
    GroundingValidator,
    compute_sha256,
    verify_frozen_integrity,
)

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def telemetry_records():
    """Load the immutable empirical saliency dataset records."""
    records = load_telemetry_dataset()
    assert len(records) > 0
    return records


@pytest.fixture(scope="module")
def commentary_config():
    """Load the commentary configuration."""
    return load_commentary_config()


# ==============================================================================
# 1. Model and Configuration Freeze Verification
# ==============================================================================

def test_frozen_weights_integrity():
    """Verify that models/generalist_v1.pth exists and exactly matches the frozen SHA-256."""
    assert FROZEN_MODEL_PATH.is_file(), f"Model file not found at {FROZEN_MODEL_PATH}"
    actual_sha = compute_sha256(FROZEN_MODEL_PATH)
    assert actual_sha == EXPECTED_WEIGHTS_SHA256, (
        f"Frozen weights SHA-256 mismatch! Expected {EXPECTED_WEIGHTS_SHA256}, got {actual_sha}"
    )


def test_frozen_config_integrity():
    """Verify that configs/generalist_v1.json exists and exactly matches the frozen SHA-256."""
    assert CONFIG_PATH.is_file(), f"Config file not found at {CONFIG_PATH}"
    actual_sha = compute_sha256(CONFIG_PATH)
    assert actual_sha == EXPECTED_CONFIG_SHA256, (
        f"Frozen config SHA-256 mismatch! Expected {EXPECTED_CONFIG_SHA256}, got {actual_sha}"
    )


def test_verify_frozen_integrity_helper():
    """Ensure verify_frozen_integrity() helper succeeds without error."""
    w_sha, c_sha = verify_frozen_integrity()
    assert w_sha == EXPECTED_WEIGHTS_SHA256
    assert c_sha == EXPECTED_CONFIG_SHA256


# ==============================================================================
# 2. Credential Safety and Repository Sanitation
# ==============================================================================

def test_env_file_is_git_ignored():
    """Verify that .env is strictly ignored by git and cannot be accidentally tracked."""
    check_ignore = subprocess.run(
        ["git", "check-ignore", ".env"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
    )
    assert check_ignore.returncode == 0, ".env must be ignored by git!"
    assert ".env" in check_ignore.stdout.strip()


def test_env_example_has_no_secret():
    """Verify that .env.example contains only placeholders, never actual keys."""
    env_example_path = ROOT_DIR / ".env.example"
    assert env_example_path.is_file(), ".env.example is missing"
    content = env_example_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        if line.startswith("GEMINI_API_KEY="):
            val = line.split("=", 1)[1].strip()
            assert val == "" or val.startswith("your_"), (
                f".env.example contains non-placeholder key: {val}"
            )


# ==============================================================================
# 3. Telemetry Event Detection and Grouping
# ==============================================================================

def test_group_telemetry_by_episode(telemetry_records):
    """Verify that group_telemetry_by_episode groups records into episodes."""
    episodes = group_telemetry_by_episode(telemetry_records)
    assert len(episodes) >= 5, f"Expected at least 5 episodes, got {len(episodes)}"
    for ep_id, recs in episodes.items():
        assert len(recs) > 0
        steps = [int(r["step"]) for r in recs]
        assert steps == sorted(steps)


def test_detect_episode_telemetry_events(telemetry_records, commentary_config):
    """Verify that detect_episode_telemetry_events identifies valid driving events."""
    episodes = group_telemetry_by_episode(telemetry_records)
    ep_records = episodes.get("std_w70_s1001")
    assert ep_records is not None, "Episode std_w70_s1001 missing"

    events = detect_episode_telemetry_events(ep_records, commentary_config)
    assert len(events) > 0, "No events detected in std_w70_s1001"

    expected_types = {
        "braking_onset",
        "sustained_braking",
        "corner_approach",
        "corner_entry",
        "corner_apex",
        "corner_exit",
        "lateral_wall_proximity",
        "straight_cruising",
    }

    for ev in events:
        assert ev["event_type"] in expected_types
        assert "event_id" in ev
        assert "telemetry_summary" in ev
        assert "action_summary" in ev
        assert "sensor_summary" in ev
        assert "saliency_triggers" in ev
        assert "factual_bullet_points" in ev
        assert len(ev["factual_bullet_points"]) >= 3


def test_select_key_lap_events(telemetry_records, commentary_config):
    """Verify that select_key_lap_events limits events to target count while covering milestones."""
    episodes = group_telemetry_by_episode(telemetry_records)
    ep_records = episodes["hairpin_f0.35_s1007"]
    events = detect_episode_telemetry_events(ep_records, commentary_config)
    key_events = select_key_lap_events(events, target_count=5)

    assert len(key_events) <= 5
    event_types = {e["event_type"] for e in key_events}
    assert len(event_types) >= 2, "Expected diverse event milestones in selection"


# ==============================================================================
# 4. Prompt Construction and Saliency Trigger Attribution
# ==============================================================================

def test_prompt_construction(telemetry_records, commentary_config):
    """Verify that _build_lap_prompt constructs valid JSON payload."""
    client = GeminiCommentaryClient(mock_mode=True)
    episodes = group_telemetry_by_episode(telemetry_records)
    ep_records = episodes["std_w70_s1001"]
    events = detect_episode_telemetry_events(ep_records, commentary_config)
    key_events = select_key_lap_events(events, target_count=3)

    lap_meta = {
        "lap_id": "lap_1_std_1001",
        "track_seed": 1001,
        "benchmark": "std_w70",
        "description": "Standard width track",
    }

    prompt_str = client._build_lap_prompt(lap_meta, key_events)
    assert isinstance(prompt_str, str)
    parsed = json.loads(prompt_str)

    assert "circuit_metadata" in parsed
    assert "task" in parsed
    assert "events" in parsed
    assert len(parsed["events"]) == len(key_events)

    first_ev = parsed["events"][0]
    assert "saliency_triggers" in first_ev
    assert "ground_truth_facts" in first_ev
    assert "top_steering_ray" in first_ev["saliency_triggers"]
    assert "top_longitudinal_ray" in first_ev["saliency_triggers"]


# ==============================================================================
# 5. Grounding Validation Rigor
# ==============================================================================

def test_grounding_validator_approves_grounded_mock():
    """Verify that GroundingValidator approves authentic, grounded commentary."""
    client = GeminiCommentaryClient(mock_mode=True)
    lap_meta = {"lap_id": "lap_test", "track_seed": 1001, "benchmark": "std_w70"}

    gt_event = {
        "event_id": "ev_001",
        "event_type": "braking_onset",
        "step_range": [45, 55],
        "corner_id": "C1",
        "corner_phase": "entry",
        "telemetry_summary": {
            "mean_velocity": 3.25,
            "min_velocity": 2.80,
            "max_velocity": 3.65,
            "velocity_range": [2.80, 3.65],
            "mean_heading_error": 2.1,
            "mean_signed_distance": 2.1,
            "mean_corner_radius": 150.0,
            "min_corner_radius": 120.0,
        },
        "action_summary": {
            "mean_steering": 0.05,
            "steering_range": [-0.02, 0.1],
            "mean_longitudinal": -0.45,
            "longitudinal_range": [-0.6, -0.1],
            "action_mode": "braking",
        },
        "sensor_summary": {
            "mean_forward_ray_0": 95.0,
            "min_forward_ray_0": 78.5,
            "mean_left_diagonal": 110.0,
            "mean_right_diagonal": 105.0,
            "min_lateral_distance": 18.0,
        },
        "saliency_triggers": {
            "top_steering_ray": "ray_minus_45",
            "top_steering_sensitivity": 0.08,
            "top_longitudinal_ray": "ray_0",
            "top_longitudinal_sensitivity": 0.18,
        },
        "factual_bullet_points": [
            "Event braking_onset occurred from step 45 to 55.",
            "Vehicle velocity averaged 3.25 px/step.",
            "Longitudinal control initiated braking with regime: braking.",
            "Center forward beam (ray_0) averaged 95.0 px.",
            "Highest longitudinal sensitivity is on ray_0 (0.180).",
        ],
    }

    mock_resp = client._generate_mock_response(lap_meta, [gt_event])
    assert len(mock_resp) == 1
    item = mock_resp[0]

    val = GroundingValidator.validate_commentary_item(item, gt_event)
    assert val["is_valid"] is True
    assert val["verdict"] == "CORRECT_GROUNDED"
    assert val["grounding_score"] >= 0.70
    assert len(val["flagged_issues"]) == 0


def test_grounding_validator_rejects_hallucinations():
    """Verify that GroundingValidator flags and rejects ungrounded or forbidden claims."""
    gt_event = {
        "event_id": "ev_002",
        "event_type": "corner_apex",
        "telemetry_summary": {
            "mean_velocity": 3.2,
            "velocity_range": [2.8, 3.5],
            "mean_heading_error": 1.5,
            "mean_signed_distance": 1.5,
            "mean_corner_radius": 80.0,
            "min_corner_radius": 60.0,
        },
        "action_summary": {
            "mean_steering": 0.65,
            "steering_range": [0.5, 0.8],
            "mean_longitudinal": 0.1,
            "longitudinal_range": [0.0, 0.2],
            "action_mode": "throttle",
        },
        "sensor_summary": {
            "mean_forward_ray_0": 120.0,
            "min_forward_ray_0": 110.0,
            "mean_left_diagonal": 45.0,
            "mean_right_diagonal": 140.0,
            "min_lateral_distance": 15.0,
        },
        "saliency_triggers": {
            "top_steering_ray": "ray_minus_45",
            "top_steering_sensitivity": 0.45,
            "top_longitudinal_ray": "ray_0",
            "top_longitudinal_sensitivity": 0.12,
        },
        "factual_bullet_points": ["Apex reached at 3.2 px/step velocity."],
    }

    hallucinated_item = {
        "event_id": "ev_002",
        "event_type": "corner_apex",
        "commentary": "The car spun out and crashed into the barrier after overtaking another vehicle at 250 km/h.",
        "evidence": {
            "telemetry_facts": "Car crashed at 250 km/h in turn 4.",
            "action_facts": "Full throttle applied.",
            "saliency_facts": "Rear camera active.",
        },
        "confidence": 0.95,
        "interpretation_notes": "Driver error.",
        "model_id": "gemini-3.8-flash",
    }

    val = GroundingValidator.validate_commentary_item(hallucinated_item, gt_event)
    assert val["is_valid"] is False
    assert val["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("unsupported hallucinated concept" in issue for issue in val["flagged_issues"])


# ==============================================================================
# 6. Strict API Usage Limits Enforcement
# ==============================================================================

def test_api_calls_limit_enforcement():
    """Verify that GeminiCommentaryClient strictly raises RuntimeError when limit is reached."""
    client = GeminiCommentaryClient(mock_mode=True, max_calls_limit=2)
    lap_meta = {"lap_id": "lap_dummy", "track_seed": 1001, "benchmark": "std"}
    key_events = []

    # Call 1
    client.generate_lap_commentary(lap_meta, key_events)
    assert client.calls_made == 1

    # Call 2
    client.generate_lap_commentary(lap_meta, key_events)
    assert client.calls_made == 2

    # Call 3 should strictly raise RuntimeError
    with pytest.raises(RuntimeError, match="Strict API usage limit reached"):
        client.generate_lap_commentary(lap_meta, key_events)


def test_missing_api_key_raises_in_live_mode():
    """Verify that initializing live client without an API key raises ValueError."""
    with pytest.raises(ValueError, match="API key must be provided"):
        GeminiCommentaryClient(api_key=None, mock_mode=False)


# ==============================================================================
# 7. Mock Pipeline End-to-End Execution
# ==============================================================================

def test_end_to_end_mock_pipeline():
    """Verify that run_commentary_pipeline executes end-to-end in mock mode with 100% pass rate."""
    from analysis.telemetry_commentary import run_commentary_pipeline

    summary = run_commentary_pipeline(mock_mode=True, max_api_calls=5)
    assert summary["metadata"]["weights_unaltered"] is True
    assert summary["api_usage_metrics"]["total_requests"] == 5
    assert summary["grounding_validation_metrics"]["total_commentaries"] == 25
    assert summary["grounding_validation_metrics"]["passed_validation_count"] == 25
    assert summary["grounding_validation_metrics"]["pass_rate"] == 1.0


# ==============================================================================
# 8. Remediated Grounding Validator: Numerical & Semantic Rigor
# ==============================================================================

@pytest.fixture
def sample_corner_approach_gt():
    """Ground-truth telemetry fixture matching corner approach on held-out track."""
    return {
        "event_id": "ev_remediation_test",
        "event_type": "corner_approach",
        "benchmark": "standard_heldout",
        "track_seed": 1001,
        "start_step": 63,
        "end_step": 74,
        "corner_id": "corner_1",
        "corner_phase": "entry",
        "telemetry_summary": {
            "mean_velocity": 3.23,
            "velocity_range": [2.99, 3.46],
            "mean_heading_error": -0.18,
            "mean_signed_distance": -0.7,
            "mean_corner_radius": 218.9,
            "min_corner_radius": 168.1,
        },
        "action_summary": {
            "mean_steering": -0.457,
            "steering_range": [-0.674, -0.106],
            "mean_longitudinal": 0.911,
            "longitudinal_range": [0.894, 0.924],
            "action_mode": "throttle",
        },
        "sensor_summary": {
            "mean_forward_ray_0": 116.0,
            "min_forward_ray_0": 112.0,
            "mean_left_diagonal": 62.7,
            "mean_right_diagonal": 48.0,
            "min_lateral_distance": 36.0,
        },
        "saliency_triggers": {
            "top_steering_ray": "ray_minus_45",
            "top_steering_sensitivity": 3.371,
            "top_longitudinal_ray": "ray_plus_45",
            "top_longitudinal_sensitivity": 0.051,
        },
        "factual_bullet_points": [
            "Event corner_approach occurred from step 63 to 74.",
            "Vehicle velocity averaged 3.23 px/step.",
            "Steering action averaged -0.457 (wheel angle -13.7 deg).",
        ],
    }


def test_remediated_validator_rejects_impossible_velocities(sample_corner_approach_gt):
    """Verify that the remediated validator strictly rejects speeds exceeding physical limit (4.0 px/step)."""
    # Test 888.8 px/step
    item_888 = {
        "event_id": sample_corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Cruising at 888.8 px/step with lookahead ray_minus_45.",
        "evidence": {"telemetry_facts": ["Speed 888.8"]},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_888 = GroundingValidator.validate_commentary_item(item_888, sample_corner_approach_gt)
    assert val_888["is_valid"] is False
    assert val_888["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("Physically impossible velocity" in issue for issue in val_888["flagged_issues"])

    # Test 8.82 px/step
    item_882 = {
        "event_id": sample_corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Decelerating from 8.82 to 8.12 px/step approaching entry with ray_minus_45.",
        "evidence": {"telemetry_facts": ["Speed 8.82"]},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_882 = GroundingValidator.validate_commentary_item(item_882, sample_corner_approach_gt)
    assert val_882["is_valid"] is False
    assert val_882["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("Physically impossible velocity" in issue for issue in val_882["flagged_issues"])


def test_remediated_validator_rejects_evidence_leakage(sample_corner_approach_gt):
    """Verify that evidence dictionary contents cannot satisfy sensor citation when omitted from commentary text."""
    item_omitted = {
        "event_id": sample_corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Vehicle accelerates through turn with negative steering deflection.",
        "evidence": {
            "saliency_facts": ["Dominant steering sensor is ray_minus_45"],
        },
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val = GroundingValidator.validate_commentary_item(item_omitted, sample_corner_approach_gt)
    assert not any("referenced dominant steering sensor in commentary text" in f for f in val["verified_facts"])


def test_remediated_validator_rejects_fake_sensors(sample_corner_approach_gt):
    """Verify that nonexistent sensor hardware (e.g. rear_camera, lidar_360) is rejected."""
    item_fake_sensor = {
        "event_id": sample_corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Approaching corner guided by ray_minus_45 while rear_camera monitors track boundaries.",
        "evidence": {"saliency_facts": ["rear_camera active"]},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val = GroundingValidator.validate_commentary_item(item_fake_sensor, sample_corner_approach_gt)
    assert val["is_valid"] is False
    assert val["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("invalid or nonexistent sensor" in issue for issue in val["flagged_issues"])


def test_remediated_validator_rejects_unsupported_physics_and_gaze(sample_corner_approach_gt):
    """Verify that ungrounded concepts (tire grip, visual gaze) are flagged and rejected."""
    item_hallucinated = {
        "event_id": sample_corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": (
            "Initiating straight-line braking to maximize tire grip for turn-in, "
            "as the policy shifts its visual gaze to ray_minus_45."
        ),
        "evidence": {"telemetry_facts": ["Braking to maximize tire grip"]},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val = GroundingValidator.validate_commentary_item(item_hallucinated, sample_corner_approach_gt)
    assert val["is_valid"] is False
    assert val["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("tire grip" in issue for issue in val["flagged_issues"])
    assert any("visual gaze" in issue for issue in val["flagged_issues"])


def test_remediated_validator_rejects_contradictory_steering(sample_corner_approach_gt):
    """Verify that claiming 'steering centered' during an active corner approach with -13.7 deg lock fails."""
    contradictory_item = {
        "event_id": sample_corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Straightaway cruising at 3.23 px/step with steering centered, guided by ray_minus_45.",
        "evidence": {"action_facts": ["Steering action -0.457"]},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val = GroundingValidator.validate_commentary_item(contradictory_item, sample_corner_approach_gt)
    assert val["is_valid"] is False
    assert val["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("Contradictory steering claim" in issue for issue in val["flagged_issues"])


def test_mock_generator_corner_approach_remediation(sample_corner_approach_gt):
    """Verify that remediated mock generator produces grounded corner approach commentary."""
    client = GeminiCommentaryClient(mock_mode=True)
    lap_meta = {"lap_id": "lap_test", "track_seed": 1001, "benchmark": "std"}
    items = client._generate_mock_response(lap_meta, [sample_corner_approach_gt])
    assert len(items) == 1
    item = items[0]

    assert "Approaching corner entry" in item["commentary"]
    assert "steering angle -13.7 degrees" in item["commentary"]
    assert "ray_minus_45" in item["commentary"]
    assert "steering centered" not in item["commentary"]

    val = GroundingValidator.validate_commentary_item(item, sample_corner_approach_gt)
    assert val["is_valid"] is True
    assert val["verdict"] == "CORRECT_GROUNDED"
    assert len(val["flagged_issues"]) == 0


def test_remediated_deliverables_integrity():
    """Verify that remediated output files exist, contain 25 items, and preserve historical outputs."""
    assert REMEDIATED_EXAMPLES_JSON_PATH.is_file(), f"Missing {REMEDIATED_EXAMPLES_JSON_PATH}"
    assert REMEDIATED_SUMMARY_JSON_PATH.is_file(), f"Missing {REMEDIATED_SUMMARY_JSON_PATH}"
    assert HISTORICAL_EXAMPLES_JSON_PATH.is_file(), f"Missing historical {HISTORICAL_EXAMPLES_JSON_PATH}"
    assert HISTORICAL_SUMMARY_JSON_PATH.is_file(), f"Missing historical {HISTORICAL_SUMMARY_JSON_PATH}"

    with open(REMEDIATED_EXAMPLES_JSON_PATH, "r") as f:
        remediated_examples = json.load(f)
    assert len(remediated_examples) == 25, f"Expected 25 remediated items, got {len(remediated_examples)}"

    with open(REMEDIATED_SUMMARY_JSON_PATH, "r") as f:
        remediated_summary = json.load(f)
    assert remediated_summary["grounding_validation_metrics"]["pass_rate"] == 1.0
    assert remediated_summary["grounding_validation_metrics"]["passed_validation_count"] == 25
