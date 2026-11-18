"""Adversarial Grounding Test Suite for RL Racing Agent Commentary Pipeline.

Directly tests the production GroundingValidator against 9 adversarial attack scenarios
mandated by the Independent Remediation Review:
1. Speed 8.82 px/step (exceeding simulation physical upper bound of 4.0 px/step).
2. Valid speed copied from wrong timestep / window outside event boundaries.
3. Valid values with wrong units (km/h, m/s, speed in deg, steering in px/step, radius in deg).
4. Real sensor name that does not support the claim (contradictory sensor attribution).
5. Sensor name present only in metadata/evidence, omitted from commentary claim text.
6. Real event with a fabricated corner label (e.g. Turn 4 on Turn 1, or Turn 2 on straight).
7. Steering lock described as straight-line cruising / centered steering.
8. Unsupported physical explanations using synonyms (tire traction, downforce, weight transfer, ocular gaze, rivals).
9. Claim whose evidence reference points to another lap (lap_id, track_seed, event_id mismatch).
10. Valid JSON schema bypass attempt (valid syntax & confidence, but rejected on grounding).
"""

from typing import Any, Dict
import pytest

from analysis.telemetry_commentary import GroundingValidator


@pytest.fixture
def corner_approach_gt() -> Dict[str, Any]:
    """Representative ground-truth event slice: Corner Approach (Turn 1)."""
    return {
        "event_id": "ev_std_w70_s1001_s063_corner_approach",
        "episode_id": "std_w70_s1001",
        "lap_id": "lap_std_w70_s1001",
        "track_seed": 1001,
        "benchmark": "standard_heldout",
        "event_type": "corner_approach",
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
    }


@pytest.fixture
def straight_cruising_gt() -> Dict[str, Any]:
    """Representative ground-truth event slice: Straight Cruising."""
    return {
        "event_id": "ev_std_w70_s1002_s100_straight_line_cruising",
        "episode_id": "std_w70_s1002",
        "lap_id": "lap_std_w70_s1002",
        "track_seed": 1002,
        "benchmark": "standard_heldout",
        "event_type": "straight_line_cruising",
        "start_step": 100,
        "end_step": 118,
        "corner_id": "none",
        "corner_phase": "straight",
        "telemetry_summary": {
            "mean_velocity": 4.0,
            "velocity_range": [3.98, 4.0],
            "mean_heading_error": 0.05,
            "mean_signed_distance": 0.1,
            "mean_corner_radius": 350.0,
            "min_corner_radius": 320.0,
        },
        "action_summary": {
            "mean_steering": -0.034,
            "steering_range": [-0.077, 0.067],
            "mean_longitudinal": 0.982,
            "longitudinal_range": [0.982, 0.982],
            "action_mode": "throttle",
        },
        "sensor_summary": {
            "mean_forward_ray_0": 200.0,
            "min_forward_ray_0": 195.0,
            "min_lateral_distance": 35.0,
        },
        "saliency_triggers": {
            "top_steering_ray": "ray_minus_45",
            "top_steering_sensitivity": 0.82,
            "top_longitudinal_ray": "ray_0",
            "top_longitudinal_sensitivity": 0.41,
        },
    }


# ==============================================================================
# Scenario 1: Speed 8.82 px/step (> 4.0 physical limit)
# ==============================================================================

def test_adversarial_impossible_velocity(corner_approach_gt):
    """Scenario 1: Reject velocity claims of 8.82 px/step or 888.8 px/step."""
    item_882 = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Approaching corner entry at 8.82 px/step guided by ray_minus_45.",
        "evidence": {"telemetry_facts": ["Speed 8.82 px/step"]},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val = GroundingValidator.validate_commentary_item(item_882, corner_approach_gt)
    assert val["is_valid"] is False
    assert val["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("Physically impossible velocity cited: 8.82" in issue for issue in val["flagged_issues"])

    item_888 = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Cruising along the straight at 888.8 px/step guided by ray_minus_45.",
        "evidence": {"telemetry_facts": ["Speed 888.8 px/step"]},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_888 = GroundingValidator.validate_commentary_item(item_888, corner_approach_gt)
    assert val_888["is_valid"] is False
    assert any("Physically impossible velocity cited: 888.8" in issue for issue in val_888["flagged_issues"])


# ==============================================================================
# Scenario 2: Valid speed copied from wrong timestep / window outside event
# ==============================================================================

def test_adversarial_speed_from_wrong_timestep(corner_approach_gt):
    """Scenario 2: Reject valid physical speeds copied from another event/step (e.g. 0.63 px/step from apex)."""
    # 0.63 px/step is a valid car speed, but outside the [2.993, 3.465] px/step window for this approach
    item_wrong_speed = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Entering corner at 0.63 px/step guided by ray_minus_45.",
        "evidence": {"telemetry_facts": ["Speed is 0.63 px/step"]},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val = GroundingValidator.validate_commentary_item(item_wrong_speed, corner_approach_gt)
    assert val["is_valid"] is False
    assert val["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("outside event measured velocity range" in issue for issue in val["flagged_issues"])

    # Step outside event window: event is [63, 74], citing step 189
    item_wrong_step = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Approaching corner entry at step 189 at 3.23 px/step guided by ray_minus_45.",
        "evidence": {"telemetry_facts": ["Step 189"]},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_step = GroundingValidator.validate_commentary_item(item_wrong_step, corner_approach_gt)
    assert val_step["is_valid"] is False
    assert any("outside event step window" in issue for issue in val_step["flagged_issues"])


# ==============================================================================
# Scenario 3: Valid values with wrong units or cross-unit swaps
# ==============================================================================

def test_adversarial_invalid_and_cross_units(corner_approach_gt):
    """Scenario 3: Reject real-world velocity units and cross-unit swaps."""
    # km/h unit
    item_kmh = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Entering corner at 3.23 km/h guided by ray_minus_45.",
        "evidence": {"telemetry_facts": ["3.23 km/h"]},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_kmh = GroundingValidator.validate_commentary_item(item_kmh, corner_approach_gt)
    assert val_kmh["is_valid"] is False
    assert any("Invalid unit: simulation velocity must be in 'px/step'" in issue for issue in val_kmh["flagged_issues"])

    # m/s unit
    item_ms = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Traveling at 3.23 m/s guided by ray_minus_45.",
        "evidence": {},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_ms = GroundingValidator.validate_commentary_item(item_ms, corner_approach_gt)
    assert val_ms["is_valid"] is False
    assert any("Invalid unit" in issue for issue in val_ms["flagged_issues"])

    # Speed cited in degrees (cross-unit)
    item_speed_deg = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Corner approach with velocity of 3.2 deg guided by ray_minus_45.",
        "evidence": {},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_speed_deg = GroundingValidator.validate_commentary_item(item_speed_deg, corner_approach_gt)
    assert val_speed_deg["is_valid"] is False
    assert any("Cross-unit error: velocity cannot be measured in degrees" in issue for issue in val_speed_deg["flagged_issues"])

    # Steering cited in px/step (cross-unit)
    item_steer_px = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Approaching corner with steering angle of -13.7 px/step guided by ray_minus_45.",
        "evidence": {},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_steer_px = GroundingValidator.validate_commentary_item(item_steer_px, corner_approach_gt)
    assert val_steer_px["is_valid"] is False
    assert any("Cross-unit error: steering cannot be measured in px/step" in issue for issue in val_steer_px["flagged_issues"])

    # Radius cited in degrees (cross-unit)
    item_radius_deg = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Corner approach with curvature radius of 218.9 deg guided by ray_minus_45.",
        "evidence": {},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_radius_deg = GroundingValidator.validate_commentary_item(item_radius_deg, corner_approach_gt)
    assert val_radius_deg["is_valid"] is False
    assert any("Cross-unit error: corner radius cannot be measured in degrees" in issue for issue in val_radius_deg["flagged_issues"])


# ==============================================================================
# Scenario 4: Real sensor name that contradicts ground truth attribution
# ==============================================================================

def test_adversarial_contradictory_sensor_attribution(corner_approach_gt):
    """Scenario 4: Reject claiming a real sensor ray (e.g. ray_plus_45) dictates steering when top is ray_minus_45."""
    item_contradictory_ray = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": (
            "Approaching corner entry at 3.23 px/step. "
            "Dominant steering sensor ray_plus_45 dictates lateral steering lock (-13.7 degrees)."
        ),
        "evidence": {
            "saliency_facts": ["Dominant steering sensor is ray_plus_45"],
        },
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val = GroundingValidator.validate_commentary_item(item_contradictory_ray, corner_approach_gt)
    assert val["is_valid"] is False
    assert val["verdict"] == "UNSUPPORTED_REJECTED"
    assert any(
        "Contradictory sensor attribution: commentary attributes steering to ray_plus_45" in issue
        or "Evidence falsely claims ray_plus_45 is top steering sensor" in issue
        for issue in val["flagged_issues"]
    )


# ==============================================================================
# Scenario 5: Sensor name present only in metadata/evidence, not in claim text
# ==============================================================================

def test_adversarial_sensor_in_metadata_only(corner_approach_gt):
    """Scenario 5: Reject when sensor is only cited in metadata/evidence while commentary text claims sensor guidance."""
    item_meta_only = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "The vehicle decelerates smoothly into the turn guided by sensor lookahead beam.",
        "evidence": {
            "saliency_facts": ["Top steering sensor is ray_minus_45"],
        },
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val = GroundingValidator.validate_commentary_item(item_meta_only, corner_approach_gt)
    assert val["is_valid"] is False
    assert val["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("no sensor ray is cited in commentary text" in issue for issue in val["flagged_issues"])


# ==============================================================================
# Scenario 6: Real event with fabricated corner label
# ==============================================================================

def test_adversarial_fabricated_corner_label(corner_approach_gt, straight_cruising_gt):
    """Scenario 6: Reject fabricated corner numbers (Turn 4 on Turn 1, or Turn 2 on straight)."""
    # Turn 4 cited when actual is corner_1
    item_wrong_turn = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Entering Turn 4 with track radius 218.9 px guided by ray_minus_45.",
        "evidence": {},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_turn = GroundingValidator.validate_commentary_item(item_wrong_turn, corner_approach_gt)
    assert val_turn["is_valid"] is False
    assert any("Fabricated corner label: commentary cites Turn 4, but ground truth event is corner_1" in issue for issue in val_turn["flagged_issues"])

    # Corner cited during a straight section with no corner
    item_straight_with_turn = {
        "event_id": straight_cruising_gt["event_id"],
        "event_type": "straight_line_cruising",
        "commentary": "Cruising along Turn 2 at 4.00 px/step with centered steering (-1.0 deg) guided by ray_minus_45.",
        "evidence": {},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_straight = GroundingValidator.validate_commentary_item(item_straight_with_turn, straight_cruising_gt)
    assert val_straight["is_valid"] is False
    assert any("during section with no corner" in issue for issue in val_straight["flagged_issues"])


# ==============================================================================
# Scenario 7: Steering lock described as straight-line cruising
# ==============================================================================

def test_adversarial_steering_lock_as_straight_cruising(corner_approach_gt):
    """Scenario 7: Reject claiming straight-line cruising / centered steering during corner with -13.7 deg lock."""
    item_lock_as_straight = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": (
            "Straight-line cruising at 3.23 px/step with steering centered, "
            "as beam ray_minus_45 monitors track corridor."
        ),
        "evidence": {"action_facts": ["mean_steering: -0.457"]},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val = GroundingValidator.validate_commentary_item(item_lock_as_straight, corner_approach_gt)
    assert val["is_valid"] is False
    assert val["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("Contradictory steering claim" in issue for issue in val["flagged_issues"])


# ==============================================================================
# Scenario 8: Unsupported physical explanations using synonyms
# ==============================================================================

@pytest.mark.parametrize("synonym_phrase", [
    "optimizing tire traction across the rubber contact patch",
    "increasing mechanical rubber friction to prevent slide",
    "relying on aerodynamic downforce and diffuser suction",
    "wing angle stabilizes aerodynamic drag reduction balance",
    "managing suspension weight transfer through chassis body roll",
    "the driver shifts ocular visual gaze to scan the horizon",
    "driver's eyeball tracking focuses on the boundary",
    "agent deliberately clips the apex curb",
    "driver intentionally aims for the inner kerb",
    "overtaking the competitor rival car",
    "drafting behind the opponent vehicle in the slipstream",
])
def test_adversarial_unsupported_physics_synonyms(corner_approach_gt, synonym_phrase):
    """Scenario 8: Reject domain violations disguised as synonyms (traction, downforce, weight transfer, ocular, rivals)."""
    item = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": f"Approaching corner entry at 3.23 px/step guided by ray_minus_45 while {synonym_phrase}.",
        "evidence": {},
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val = GroundingValidator.validate_commentary_item(item, corner_approach_gt)
    assert val["is_valid"] is False
    assert val["verdict"] == "UNSUPPORTED_REJECTED"
    assert any("Commentary contains unsupported" in issue for issue in val["flagged_issues"])


# ==============================================================================
# Scenario 9: Claim whose evidence reference points to another lap
# ==============================================================================

def test_adversarial_cross_lap_evidence_reference(corner_approach_gt):
    """Scenario 9: Reject claims whose evidence references a different lap_id, track_seed, or event_id."""
    # Lap ID mismatch
    item_wrong_lap = {
        "event_id": corner_approach_gt["event_id"],
        "lap_id": "lap_std_w70_s1001",
        "event_type": "corner_approach",
        "commentary": "Approaching corner entry at 3.23 px/step guided by ray_minus_45.",
        "evidence": {
            "lap_id": "lap_narrow_w20_s1005",  # Mismatched lap!
            "saliency_facts": ["ray_minus_45 active"],
        },
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_lap = GroundingValidator.validate_commentary_item(item_wrong_lap, corner_approach_gt)
    assert val_lap["is_valid"] is False
    assert any("evidence references lap 'lap_narrow_w20_s1005'" in issue for issue in val_lap["flagged_issues"])

    # Track seed mismatch
    item_wrong_seed = {
        "event_id": corner_approach_gt["event_id"],
        "lap_id": "lap_std_w70_s1001",
        "event_type": "corner_approach",
        "commentary": "Approaching corner entry at 3.23 px/step guided by ray_minus_45.",
        "evidence": {
            "track_seed": 1005,  # Expected 1001
        },
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_seed = GroundingValidator.validate_commentary_item(item_wrong_seed, corner_approach_gt)
    assert val_seed["is_valid"] is False
    assert any("evidence references track seed 1005" in issue for issue in val_seed["flagged_issues"])

    # Event ID mismatch in evidence
    item_wrong_eid = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": "Approaching corner entry at 3.23 px/step guided by ray_minus_45.",
        "evidence": {
            "event_id": "ev_std_w70_s1002_s187_corner_apex",  # Mismatched event!
        },
        "confidence": 0.95,
        "model_id": "gemini-3.8-flash",
    }
    val_eid = GroundingValidator.validate_commentary_item(item_wrong_eid, corner_approach_gt)
    assert val_eid["is_valid"] is False
    assert any("evidence references event 'ev_std_w70_s1002_s187_corner_apex'" in issue for issue in val_eid["flagged_issues"])


# ==============================================================================
# Scenario 10: Valid JSON Schema still rejected when claims are ungrounded
# ==============================================================================

def test_adversarial_valid_json_schema_still_rejected(corner_approach_gt):
    """Scenario 10: Demonstrate that 100% syntactically valid JSON with confidence=0.95 is strictly rejected."""
    valid_schema_item = {
        "event_id": corner_approach_gt["event_id"],
        "event_type": "corner_approach",
        "commentary": (
            "Approaching corner entry at 8.82 px/step while maximizing tire grip with visual gaze on ray_minus_45."
        ),
        "evidence": {
            "telemetry_facts": ["Speed 8.82 px/step"],
            "action_facts": ["Throttle active"],
            "saliency_facts": ["ray_minus_45 dominant"],
        },
        "confidence": 0.95,
        "interpretation_notes": "Syntactically flawless schema containing multiple physical violations.",
        "model_id": "gemini-3.8-flash",
    }

    # Verify schema keys are strictly present
    assert set(GroundingValidator.REQUIRED_KEYS).issubset(set(valid_schema_item.keys()))

    val = GroundingValidator.validate_commentary_item(valid_schema_item, corner_approach_gt)
    assert val["is_valid"] is False
    assert val["verdict"] == "UNSUPPORTED_REJECTED"
    assert val["grounding_score"] < 0.50
    assert len(val["flagged_issues"]) >= 2


# ==============================================================================
# Deliverable Integrity: Remediation Claim Validation CSV
# ==============================================================================

def test_remediation_claim_validation_deliverable_integrity():
    """Verify that grounded_commentary_remediation_claim_validation.csv exists and is complete."""
    import csv
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    rem_csv = root / "analysis" / "results" / "grounded_commentary_remediation_claim_validation.csv"
    hist_csv = root / "analysis" / "results" / "grounded_commentary_claim_validation.csv"

    # Verify historical CSV is preserved
    assert hist_csv.is_file(), f"Historical audit CSV must be preserved: {hist_csv}"
    with open(hist_csv, "r", encoding="utf-8") as f:
        hist_rows = list(csv.DictReader(f))
    assert len(hist_rows) == 109, f"Historical audit CSV row count changed: expected 109, got {len(hist_rows)}"

    # Verify remediation CSV
    assert rem_csv.is_file(), f"Remediation claim validation CSV missing: {rem_csv}"
    with open(rem_csv, "r", encoding="utf-8") as f:
        rem_rows = list(csv.DictReader(f))

    assert len(rem_rows) == 125, f"Expected exactly 125 audited claims, got {len(rem_rows)}"

    required_cols = {
        "claim_id",
        "source",
        "lap_id",
        "event_id",
        "event_type",
        "claim_type",
        "statement_text",
        "telemetry_source_file",
        "step_window",
        "referenced_telemetry_fields",
        "measured_values",
        "is_supported",
        "audit_verdict",
        "audit_explanation",
    }
    assert required_cols.issubset(set(rem_rows[0].keys())), "CSV schema missing required provenance columns"

    # All claims in remediated deliverable must be empirically supported
    for r in rem_rows:
        assert r["is_supported"] == "TRUE", f"Claim {r['claim_id']} failed support check"
        assert r["audit_verdict"] == "SUPPORTED", f"Claim {r['claim_id']} has non-supported verdict"
        assert r["telemetry_source_file"] == "analysis/results/corner_saliency_phases.csv"
        assert r["source"] == "telemetry_commentary_remediated_v1.json"
