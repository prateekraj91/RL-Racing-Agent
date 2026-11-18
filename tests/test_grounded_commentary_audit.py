"""Tests for the Grounded Commentary Integration Audit.

Specifically exposes and tests the architectural and semantic weaknesses of the
existing GroundingValidator and generated commentary outputs:
1. Numerical bypass: validator passes velocities exceeding physical limits (>4.0 px/step).
2. Dead-code numerical ranges: numbers parsed from text are never validated against valid_ranges.
3. Tautological evidence leak: sensor citation check passes even when text completely omits the sensor.
4. Semantic contradiction bypass: corner approach with -29 deg steering lock passes as 'steering centered'.
5. Unsupported physics bypass: statements claiming 'tire grip maximization' pass when grip_limit=False.
6. Anthropomorphic intent bypass: claims that policy 'shifts visual gaze' pass without detection.
7. Claim validation CSV deliverable integrity and schema completeness.
"""

import csv
import json
from pathlib import Path
import pytest

from analysis.telemetry_commentary import GroundingValidator
from env.car import Car

ROOT_DIR = Path(__file__).resolve().parent.parent
CLAIM_VALIDATION_CSV = ROOT_DIR / "analysis" / "results" / "grounded_commentary_claim_validation.csv"


@pytest.fixture
def base_gt_event():
    """Ground truth telemetry event slice matching a typical corner approach."""
    return {
        "event_id": "ev_test_corner_approach",
        "benchmark": "standard_heldout",
        "track_seed": 1001,
        "start_step": 63,
        "end_step": 74,
        "corner_id": "corner_1",
        "corner_phase": "entry",
        "telemetry_summary": {
            "mean_velocity": 3.231,
            "velocity_range": [2.993, 3.465],
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


# ==============================================================================
# 1. Physical Environment Velocity Bounds Verification
# ==============================================================================

def test_car_environment_physical_velocity_limits():
    """Verify that the simulated Car environment strictly caps forward velocity to 4.0 px/step."""
    car = Car()
    assert car.max_speed == 4.0, f"Expected car.max_speed == 4.0, got {car.max_speed}"
    assert car.grip_limit is False, "Expected grip_limit == False (disabled by default)"


# ==============================================================================
# 2. Validator Vulnerability: Numerical Range Bypass (Dead Code)
# ==============================================================================

def test_validator_fails_to_catch_impossible_velocity(base_gt_event):
    """VULNERABILITY PROOF:

    The existing GroundingValidator passes commentary citing 888.8 px/step or 8.82 px/step,
    because lines 172-179 define 'numbers' and 'valid_ranges' but never evaluate them.
    """
    hallucinated_item = {
        "event_id": base_gt_event["event_id"],
        "event_type": "corner_approach",
        "commentary": (
            "Cruising at 888.8 px/step (far beyond physical 4.0 limit) "
            "with forward lookahead ray_0 maintaining speed."
        ),
        "evidence": {
            "telemetry_facts": ["Speed is 888.8 px/step"],
            "action_facts": ["Throttle applied"],
            "saliency_facts": ["top sensor is ray_plus_45"],
        },
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }

    val = GroundingValidator.validate_commentary_item(hallucinated_item, base_gt_event)

    # DOCUMENT WEAKNESS: The validator erroneously flags NO issues and marks it CORRECT_GROUNDED!
    assert val["is_valid"] is True, "Validator should erroneously approve due to dead-code range checks"
    assert val["verdict"] == "CORRECT_GROUNDED"
    assert len(val["flagged_issues"]) == 0, (
        "Demonstrates vulnerability: validator completely ignored out-of-bounds speed 888.8 px/step!"
    )


# ==============================================================================
# 3. Validator Vulnerability: Tautological Evidence Lookup Loophole
# ==============================================================================

def test_validator_tautological_evidence_leak(base_gt_event):
    """VULNERABILITY PROOF:

    The check `top_l_ray in str(ev_data) or top_l_ray in text` passes even when the commentary
    text completely omits or contradicts the dominant sensor, because ev_data contains the ground truth!
    """
    omitted_sensor_item = {
        "event_id": base_gt_event["event_id"],
        "event_type": "corner_approach",
        "commentary": "Vehicle accelerates through turn without citing any sensor rays.",
        "evidence": {
            # ev_data contains the sensor name from telemetry
            "saliency_facts": ["Top longitudinal sensor is ray_plus_45"],
        },
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }

    val = GroundingValidator.validate_commentary_item(omitted_sensor_item, base_gt_event)

    # It claims to have verified the dominant sensor even though text never mentioned it
    assert any("Correctly referenced dominant longitudinal sensor" in f for f in val["verified_facts"]), (
        "Demonstrates loophole: validator claimed sensor was referenced when commentary text omitted it!"
    )


# ==============================================================================
# 4. Validator Vulnerability: Contradictory Steering State Bypass
# ==============================================================================

def test_validator_passes_centered_steering_claim_during_sharp_turn(base_gt_event):
    """VULNERABILITY PROOF:

    A corner approach event with -13.7 deg steering lock claiming 'steering centered'
    is passed by GroundingValidator as CORRECT_GROUNDED.
    """
    contradictory_item = {
        "event_id": base_gt_event["event_id"],
        "event_type": "corner_approach",
        "commentary": (
            "Flat-out straightaway cruising at 3.231 px/step with steering centered. "
            "Forward lookahead is open (116.0 px) maintaining top speed."
        ),
        "evidence": {
            "telemetry_facts": ["Velocity is 3.231 px/step"],
            "action_facts": ["Steering action is -0.457"],
            "saliency_facts": ["Top sensor is ray_plus_45"],
        },
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }

    val = GroundingValidator.validate_commentary_item(contradictory_item, base_gt_event)

    assert val["is_valid"] is True
    assert val["verdict"] == "CORRECT_GROUNDED"
    assert len(val["flagged_issues"]) == 0, (
        "Demonstrates vulnerability: validator approved 'steering centered' when actual lock was -13.7 deg!"
    )


# ==============================================================================
# 5. Validator Vulnerability: Unsupported Grip and Gaze Semantics
# ==============================================================================

def test_validator_fails_to_flag_unsupported_grip_and_gaze(base_gt_event):
    """VULNERABILITY PROOF:

    Validator has no checks for physical domain violations like 'maximizing tire grip'
    (grip_limit=False) or 'shifting visual gaze' (stateless 5-ray MLP).
    """
    semantic_hallucination_item = {
        "event_id": base_gt_event["event_id"],
        "event_type": "corner_approach",
        "commentary": (
            "The driver initiates straight-line braking to maximize tire grip for directional transition, "
            "while the policy shifts its visual gaze to immediate diagonal boundary rays."
        ),
        "evidence": {
            "telemetry_facts": ["Braking to maximize tire grip"],
            "action_facts": ["braking"],
            "saliency_facts": ["ray_plus_45"],
        },
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }

    val = GroundingValidator.validate_commentary_item(semantic_hallucination_item, base_gt_event)

    # Demonstrates that neither 'tire grip' nor 'visual gaze' is caught by FORBIDDEN_HALLUCINATIONS
    assert val["is_valid"] is True
    assert val["verdict"] == "CORRECT_GROUNDED"


# ==============================================================================
# 6. Claim Validation Deliverable Integrity
# ==============================================================================

def test_claim_validation_csv_exists_and_populated():
    """Verify that grounded_commentary_claim_validation.csv exists and contains valid audit records."""
    assert CLAIM_VALIDATION_CSV.is_file(), f"Missing claim validation CSV at {CLAIM_VALIDATION_CSV}"
    with open(CLAIM_VALIDATION_CSV, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) >= 100, f"Expected >= 100 audited claims, got {len(reader)}"

    required_columns = {
        "claim_id",
        "source",
        "lap_id",
        "event_id",
        "event_type",
        "claim_type",
        "statement_text",
        "referenced_telemetry_fields",
        "measured_values",
        "is_supported",
        "audit_verdict",
        "failure_reason",
    }
    assert required_columns.issubset(set(reader[0].keys())), "CSV columns mismatch"

    # Verify verdict distribution includes both supported and unsupported claims
    verdicts = {r["audit_verdict"] for r in reader}
    assert "SUPPORTED" in verdicts
    assert "CONTRADICTED" in verdicts
    assert "PHYSICALLY_IMPOSSIBLE" in verdicts
    assert "UNSUPPORTED_SEMANTICS" in verdicts
