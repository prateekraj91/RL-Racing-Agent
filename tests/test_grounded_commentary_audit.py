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
import re
from typing import Any, Dict
import pytest

from analysis.telemetry_commentary import GroundingValidator
from env.car import Car

ROOT_DIR = Path(__file__).resolve().parent.parent
CLAIM_VALIDATION_CSV = ROOT_DIR / "analysis" / "results" / "grounded_commentary_claim_validation.csv"


class LegacyVulnerableValidator:
    """Historical baseline validator preserved as a test fixture from commit 9a3db88.

    Used strictly for historical audit regression verification without requiring
    any legacy bypasses or stack inspection inside production code.
    Exposes the 4 baseline vulnerabilities:
    1. Dead-code numerical ranges (passes 888.8 and 8.82 px/step).
    2. Tautological evidence leak (validates sensor from ev_data string even if text omits it).
    3. Missing steering coherence checks (passes 'steering centered' at -13.7 deg lock).
    4. Missing checks for tire grip, visual gaze, downforce, and intentional apex clipping.
    """

    REQUIRED_KEYS = ["event_id", "event_type", "commentary", "evidence", "confidence", "model_id"]
    FORBIDDEN_HALLUCINATIONS = [
        "spun out", "spin out", "crash", "collision", "overtake", "overtaking",
        "tire puncture", "gear shift", "gearbox", "turbo", "pit stop"
    ]

    @classmethod
    def validate_commentary_item(
        cls,
        commentary_item: Dict[str, Any],
        ground_truth_event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Historical vulnerable baseline validation logic from commit 9a3db88."""
        verified_facts = []
        flagged_issues = []

        # 1. Schema Validation
        for k in cls.REQUIRED_KEYS:
            if k not in commentary_item:
                flagged_issues.append(f"Missing required key: {k}")

        if commentary_item.get("event_id") != ground_truth_event.get("event_id"):
            flagged_issues.append(
                f"event_id mismatch: got {commentary_item.get('event_id')}, expected {ground_truth_event.get('event_id')}"
            )

        conf = commentary_item.get("confidence", 0.0)
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            flagged_issues.append(f"Invalid confidence value: {conf}")

        text = commentary_item.get("commentary", "")
        ev_data = commentary_item.get("evidence", {})

        # 2. Hallucination Checks
        is_actual_crash = (
            ground_truth_event.get("benchmark", "") == "hairpin"
            and ground_truth_event.get("track_seed") == 1001
            and ground_truth_event.get("end_step", 0) >= 220
        )
        for term in cls.FORBIDDEN_HALLUCINATIONS:
            if term in ["crash", "collision"] and is_actual_crash:
                continue
            if re.search(r"\b" + re.escape(term) + r"\b", text.lower()):
                flagged_issues.append(f"Commentary contains unsupported hallucinated concept: '{term}'")

        # 3. Numerical Grounding Validation (Baseline with dead-code ranges)
        telem = ground_truth_event.get("telemetry_summary", {})
        sensors = ground_truth_event.get("sensor_summary", {})
        actions = ground_truth_event.get("action_summary", {})
        saliencies = ground_truth_event.get("saliency_triggers", {})

        # Verify action mode grounding
        act_mode = actions.get("action_mode", "")
        if act_mode == "braking":
            if any(w in text.lower() for w in ["brak", "decel", "slow"]):
                verified_facts.append("Correctly identified braking / deceleration control regime")
            else:
                flagged_issues.append("Failed to identify active negative longitudinal braking")
        elif act_mode == "throttle":
            if any(w in text.lower() for w in ["throttle", "accel", "speed", "cruis"]):
                verified_facts.append("Correctly identified throttle / cruising control regime")

        # Verify sensor citations (Vulnerable loophole: looks up in ev_data)
        top_s_ray = saliencies.get("top_steering_ray") or saliencies.get("dominant_steering_saliency_ray")
        top_l_ray = saliencies.get("top_longitudinal_ray") or saliencies.get("dominant_longitudinal_saliency_ray")
        if top_s_ray and (top_s_ray in str(ev_data) or top_s_ray in text):
            verified_facts.append(f"Correctly referenced dominant steering sensor: {top_s_ray}")
        if top_l_ray and (top_l_ray in str(ev_data) or top_l_ray in text):
            verified_facts.append(f"Correctly referenced dominant longitudinal sensor: {top_l_ray}")

        # Baseline dead-code ranges (defined but never checked against numbers)
        numbers = [float(x) for x in re.findall(r"\b\d+\.?\d*\b", text)]
        valid_ranges = [
            telem.get("velocity_range", [0, 4]),
            [sensors.get("min_forward_ray_0", 0) - 5.0, sensors.get("mean_forward_ray_0", 100) + 5.0],
            [actions.get("steering_range", [-1, 1])[0] * 30.0 - 5.0, actions.get("steering_range", [-1, 1])[1] * 30.0 + 5.0],
            [telem.get("min_corner_radius", 0) - 10.0, telem.get("mean_corner_radius", 100) + 10.0],
        ]

        grounding_score = max(0.0, 1.0 - (len(flagged_issues) * 0.25))

        if flagged_issues:
            verdict = "UNSUPPORTED_REJECTED"
            is_valid = False
        elif conf < 0.80 or "uncertain" in str(commentary_item.get("interpretation_notes", "")).lower():
            verdict = "UNCERTAIN_INTERPRETATION"
            is_valid = True
        else:
            verdict = "CORRECT_GROUNDED"
            is_valid = True

        return {
            "is_valid": is_valid,
            "verdict": verdict,
            "grounding_score": round(grounding_score, 2),
            "verified_facts": verified_facts,
            "flagged_issues": flagged_issues,
        }


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
    """VULNERABILITY PROOF & REMEDIATION VERIFICATION:

    Historical vulnerability: LegacyVulnerableValidator passes commentary citing 888.8 px/step
    due to dead-code range checks.
    Production remediation: GroundingValidator rejects impossible velocities > 4.0 px/step.
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

    # 1. Historical vulnerability proof: Legacy validator erroneously flags NO issues
    val_legacy = LegacyVulnerableValidator.validate_commentary_item(hallucinated_item, base_gt_event)
    assert val_legacy["is_valid"] is True, "Legacy validator should erroneously approve due to dead-code range checks"
    assert val_legacy["verdict"] == "CORRECT_GROUNDED"
    assert len(val_legacy["flagged_issues"]) == 0

    # 2. Production remediation: Strict GroundingValidator rejects impossible velocity
    val_prod = GroundingValidator.validate_commentary_item(hallucinated_item, base_gt_event)
    assert val_prod["is_valid"] is False, "Production validator must reject impossible velocity"
    assert val_prod["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("Physically impossible velocity cited" in issue for issue in val_prod["flagged_issues"])


# ==============================================================================
# 3. Validator Vulnerability: Tautological Evidence Lookup Loophole
# ==============================================================================

def test_validator_tautological_evidence_leak(base_gt_event):
    """VULNERABILITY PROOF & REMEDIATION VERIFICATION:

    Historical loophole: `top_l_ray in str(ev_data)` passes even when commentary
    text completely omits the sensor ray name.
    Production remediation: GroundingValidator requires sensor citations in commentary text.
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

    # 1. Historical loophole proof: Legacy validator falsely marks sensor as verified
    val_legacy = LegacyVulnerableValidator.validate_commentary_item(omitted_sensor_item, base_gt_event)
    assert any("Correctly referenced dominant longitudinal sensor" in f for f in val_legacy["verified_facts"]), (
        "Demonstrates loophole: legacy validator claimed sensor was referenced when commentary text omitted it!"
    )

    # 2. Production remediation: Strict GroundingValidator flags sensor omission in text
    val_prod = GroundingValidator.validate_commentary_item(omitted_sensor_item, base_gt_event)
    assert val_prod["is_valid"] is False
    assert val_prod["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("no sensor ray is cited in commentary text" in issue for issue in val_prod["flagged_issues"])


# ==============================================================================
# 4. Validator Vulnerability: Contradictory Steering State Bypass
# ==============================================================================

def test_validator_passes_centered_steering_claim_during_sharp_turn(base_gt_event):
    """VULNERABILITY PROOF & REMEDIATION VERIFICATION:

    Historical vulnerability: Corner approach event with -13.7 deg steering lock
    claiming 'steering centered' is passed as CORRECT_GROUNDED by legacy validator.
    Production remediation: GroundingValidator checks steering coherence against actual mean_steering.
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

    # 1. Historical vulnerability proof: Legacy validator approves centered steering claim during corner
    val_legacy = LegacyVulnerableValidator.validate_commentary_item(contradictory_item, base_gt_event)
    assert val_legacy["is_valid"] is True
    assert val_legacy["verdict"] == "CORRECT_GROUNDED"
    assert len(val_legacy["flagged_issues"]) == 0

    # 2. Production remediation: Strict GroundingValidator flags contradictory steering
    val_prod = GroundingValidator.validate_commentary_item(contradictory_item, base_gt_event)
    assert val_prod["is_valid"] is False
    assert val_prod["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("Contradictory steering claim" in issue for issue in val_prod["flagged_issues"])


# ==============================================================================
# 5. Validator Vulnerability: Unsupported Grip and Gaze Semantics
# ==============================================================================

def test_validator_fails_to_flag_unsupported_grip_and_gaze(base_gt_event):
    """VULNERABILITY PROOF & REMEDIATION VERIFICATION:

    Historical vulnerability: Legacy validator has no checks for physical domain violations
    ('maximizing tire grip' when grip_limit=False) or anthropomorphic claims ('visual gaze').
    Production remediation: GroundingValidator flags and rejects both terms.
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

    # 1. Historical vulnerability proof: Legacy validator passes grip and gaze claims
    val_legacy = LegacyVulnerableValidator.validate_commentary_item(semantic_hallucination_item, base_gt_event)
    assert val_legacy["is_valid"] is True
    assert val_legacy["verdict"] == "CORRECT_GROUNDED"

    # 2. Production remediation: Strict GroundingValidator rejects unsupported concepts
    val_prod = GroundingValidator.validate_commentary_item(semantic_hallucination_item, base_gt_event)
    assert val_prod["is_valid"] is False
    assert val_prod["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("tire grip" in issue for issue in val_prod["flagged_issues"])
    assert any("visual gaze" in issue for issue in val_prod["flagged_issues"])


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
