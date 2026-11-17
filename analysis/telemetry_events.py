"""Deterministic Telemetry Event Detection for Racing Agent Telemetry.

Extracts meaningful driving events (e.g. corner approach, braking onset, sustained braking,
high steering demand, apex clipping, exit acceleration, lateral wall proximity, cruising)
from recorded vehicle telemetry and gradient-based saliency measurements.

Important: The event detector, not the LLM, decides whether an event occurred.
All event attributes are strictly grounded in empirical recorded values.
"""

import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repository root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "telemetry_commentary.json")
PHASES_CSV_PATH = os.path.join(ROOT_DIR, "analysis", "results", "corner_saliency_phases.csv")
TIMESTEPS_CSV_PATH = os.path.join(ROOT_DIR, "analysis", "results", "policy_saliency_timesteps.csv")
SUMMARY_JSON_PATH = os.path.join(ROOT_DIR, "analysis", "results", "policy_saliency_summary.json")

RAY_NAMES = ["ray_minus_90", "ray_minus_45", "ray_0", "ray_plus_45", "ray_plus_90"]


def load_commentary_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration for event detection and commentary generation."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config at: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def load_telemetry_dataset(
    phases_csv_path: str = PHASES_CSV_PATH,
    fallback_csv_path: str = TIMESTEPS_CSV_PATH,
) -> List[Dict[str, Any]]:
    """Load the richest available telemetry dataset with corner phases and saliencies."""
    path_to_use = phases_csv_path if os.path.isfile(phases_csv_path) else fallback_csv_path
    if not os.path.isfile(path_to_use):
        raise FileNotFoundError(f"No telemetry CSV found at {phases_csv_path} or {fallback_csv_path}")

    with open(path_to_use, "r") as f:
        rows = list(csv.DictReader(f))
    return rows


def group_telemetry_by_episode(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group flat timestep records by episode identifier."""
    episodes: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        ep_id = r["episode_id"]
        if ep_id not in episodes:
            episodes[ep_id] = []
        episodes[ep_id].append(r)

    # Sort timesteps by step index within each episode
    for ep_id in episodes:
        episodes[ep_id].sort(key=lambda x: int(x["step"]))
    return episodes


def extract_event_telemetry_slice(
    episode_records: List[Dict[str, Any]],
    start_step: int,
    end_step: int,
    event_type: str,
    corner_id: str = "none",
    corner_phase: str = "straight",
    context_window: int = 2,
) -> Dict[str, Any]:
    """Construct a compact, self-contained event record with factual summaries.
    
    Args:
        episode_records: Chronologically sorted list of all records for this episode.
        start_step: First timestep of detected event.
        end_step: Last timestep of detected event (inclusive).
        event_type: Category identifier of the driving event.
        corner_id: Associated corner identifier or 'none'.
        corner_phase: Associated corner phase ('entry', 'apex', 'exit', 'straight').
        context_window: Number of steps before and after included for contextual grounding.
        
    Returns:
        Structured event dictionary with exact metrics and factual grounding statements.
    """
    total_steps = len(episode_records)
    event_recs = [r for r in episode_records if start_step <= int(r["step"]) <= end_step]
    if not event_recs:
        raise ValueError(f"No records found for steps {start_step}..{end_step}")

    ep_id = event_recs[0]["episode_id"]
    track_seed = int(event_recs[0]["track_seed"])
    benchmark = event_recs[0].get("benchmark", "unknown")

    # Compute numerical summaries
    v_vals = [float(r["obs_velocity"]) for r in event_recs]
    steer_vals = [float(r["action_steering"]) for r in event_recs]
    long_vals = [float(r["action_longitudinal"]) for r in event_recs]
    he_vals = [float(r["obs_heading_error"]) for r in event_recs]
    dist_vals = [float(r["obs_signed_distance"]) for r in event_recs]
    r_vals = [float(r["corner_radius"]) for r in event_recs]
    ray0_vals = [float(r["obs_ray_0"]) for r in event_recs]
    ray_m45_vals = [float(r["obs_ray_minus_45"]) for r in event_recs]
    ray_p45_vals = [float(r["obs_ray_plus_45"]) for r in event_recs]
    min_lat_vals = [min(float(r["obs_ray_minus_90"]), float(r["obs_ray_plus_90"])) for r in event_recs]

    # Find top steering and longitudinal rays in this event slice
    steer_ray_means = {}
    long_ray_means = {}
    for ray in RAY_NAMES:
        key_steer = f"scaled_saliency_steer_{ray}" if f"scaled_saliency_steer_{ray}" in event_recs[0] else f"saliency_steer_{ray}"
        key_long = f"scaled_saliency_long_{ray}" if f"scaled_saliency_long_{ray}" in event_recs[0] else f"saliency_long_{ray}"
        steer_ray_means[ray] = float(np.mean([float(r[key_steer]) for r in event_recs]))
        long_ray_means[ray] = float(np.mean([float(r[key_long]) for r in event_recs]))

    top_steer_ray = max(steer_ray_means, key=steer_ray_means.get)
    top_long_ray = max(long_ray_means, key=long_ray_means.get)

    # Action mode
    has_brake = any(lv < 0 for lv in long_vals)
    has_throttle = any(lv >= 0 for lv in long_vals)
    if has_brake and not has_throttle:
        act_mode = "braking"
    elif has_throttle and not has_brake:
        act_mode = "throttle"
    else:
        act_mode = "transition"

    event_id = f"ev_{ep_id}_s{start_step:03d}_{event_type}"

    # Generate factual bullet statements strictly from numbers
    factual_bullets = [
        f"Event {event_type} occurred from step {start_step} to {end_step} (duration {end_step - start_step + 1} steps).",
        f"Vehicle velocity averaged {np.mean(v_vals):.2f} px/step (range {min(v_vals):.2f} to {max(v_vals):.2f} px/step).",
        f"Steering action averaged {np.mean(steer_vals):.3f} (approx front wheel angle {np.mean(steer_vals) * 30.0:.1f} deg).",
        f"Longitudinal action averaged {np.mean(long_vals):.3f} (control regime: {act_mode}).",
        f"Center forward beam (ray_0) averaged {np.mean(ray0_vals):.1f} px (range {min(ray0_vals):.1f} to {max(ray0_vals):.1f} px).",
        f"Heading error averaged {np.mean(he_vals):.1f} deg; centerline signed distance averaged {np.mean(dist_vals):.2f} px.",
        f"Track curvature radius averaged {np.mean(r_vals):.1f} px (corner phase: {corner_phase}, corner ID: {corner_id}).",
        f"Highest steering sensitivity is on {top_steer_ray} ({steer_ray_means[top_steer_ray]:.3f}); highest longitudinal sensitivity is on {top_long_ray} ({long_ray_means[top_long_ray]:.3f}).",
    ]

    return {
        "event_id": event_id,
        "episode_id": ep_id,
        "track_seed": track_seed,
        "benchmark": benchmark,
        "event_type": event_type,
        "start_step": start_step,
        "end_step": end_step,
        "duration_steps": end_step - start_step + 1,
        "corner_id": corner_id,
        "corner_phase": corner_phase,
        "telemetry_summary": {
            "mean_velocity": round(float(np.mean(v_vals)), 3),
            "velocity_range": [round(float(min(v_vals)), 3), round(float(max(v_vals)), 3)],
            "mean_heading_error": round(float(np.mean(he_vals)), 2),
            "mean_signed_distance": round(float(np.mean(dist_vals)), 2),
            "mean_corner_radius": round(float(np.mean(r_vals)), 1),
            "min_corner_radius": round(float(min(r_vals)), 1),
        },
        "action_summary": {
            "mean_steering": round(float(np.mean(steer_vals)), 3),
            "steering_range": [round(float(min(steer_vals)), 3), round(float(max(steer_vals)), 3)],
            "mean_longitudinal": round(float(np.mean(long_vals)), 3),
            "longitudinal_range": [round(float(min(long_vals)), 3), round(float(max(long_vals)), 3)],
            "action_mode": act_mode,
        },
        "sensor_summary": {
            "mean_forward_ray_0": round(float(np.mean(ray0_vals)), 1),
            "min_forward_ray_0": round(float(min(ray0_vals)), 1),
            "mean_left_diagonal": round(float(np.mean(ray_m45_vals)), 1),
            "mean_right_diagonal": round(float(np.mean(ray_p45_vals)), 1),
            "min_lateral_distance": round(float(min(min_lat_vals)), 1),
        },
        "saliency_triggers": {
            "top_steering_ray": top_steer_ray,
            "top_steering_sensitivity": round(steer_ray_means[top_steer_ray], 3),
            "top_longitudinal_ray": top_long_ray,
            "top_longitudinal_sensitivity": round(long_ray_means[top_long_ray], 3),
        },
        "factual_bullet_points": factual_bullets,
    }


def detect_episode_telemetry_events(
    episode_records: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Deterministically scan an episode trajectory and detect driving events.
    
    Identifies:
    - Braking onset & sustained braking
    - Corner approach & turn-in
    - Corner apex clipping
    - Corner exit acceleration
    - Lateral wall proximity (acute barrier clearance)
    - Straight-line high speed cruising
    """
    if thresholds is None:
        config = load_commentary_config()
        thresholds = config.get("event_detection_thresholds", {})

    events: List[Dict[str, Any]] = []
    n = len(episode_records)
    if n == 0:
        return events

    # Helper arrays
    steps = [int(r["step"]) for r in episode_records]
    v_arr = np.array([float(r["obs_velocity"]) for r in episode_records])
    steer_arr = np.array([float(r["action_steering"]) for r in episode_records])
    long_arr = np.array([float(r["action_longitudinal"]) for r in episode_records])
    radius_arr = np.array([float(r["corner_radius"]) for r in episode_records])
    min_lat_arr = np.array([min(float(r["obs_ray_minus_90"]), float(r["obs_ray_plus_90"])) for r in episode_records])
    phases = [r.get("corner_phase", "straight") for r in episode_records]
    corner_ids = [r.get("corner_id", "none") for r in episode_records]

    # 1. Detect Braking Events (Braking Onset & Sustained Braking)
    i = 0
    while i < n:
        if long_arr[i] < -0.01:
            start_i = i
            while i < n and long_arr[i] < -0.01:
                i += 1
            end_i = i - 1
            duration = end_i - start_i + 1
            e_type = "sustained_braking" if duration >= 2 else "braking_onset"
            ev = extract_event_telemetry_slice(
                episode_records,
                steps[start_i],
                steps[end_i],
                event_type=e_type,
                corner_id=corner_ids[start_i],
                corner_phase=phases[start_i],
            )
            events.append(ev)
        else:
            i += 1

    # 2. Detect Corner Phase Milestones (Entry, Apex, Exit)
    i = 0
    while i < n:
        cid = corner_ids[i]
        cph = phases[i]
        if cid != "none" and cph in ["entry", "apex", "exit"]:
            start_i = i
            while i < n and corner_ids[i] == cid and phases[i] == cph:
                i += 1
            end_i = i - 1
            e_type = f"corner_{cph}" if cph != "entry" else "corner_approach"
            ev = extract_event_telemetry_slice(
                episode_records,
                steps[start_i],
                steps[end_i],
                event_type=e_type,
                corner_id=cid,
                corner_phase=cph,
            )
            events.append(ev)
        else:
            i += 1

    # 3. Detect Acute Lateral Wall Proximity (< 9.0 px barrier clearance)
    i = 0
    while i < n:
        if min_lat_arr[i] <= 8.5:
            start_i = i
            while i < n and min_lat_arr[i] <= 8.5:
                i += 1
            end_i = i - 1
            ev = extract_event_telemetry_slice(
                episode_records,
                steps[start_i],
                steps[end_i],
                event_type="lateral_wall_proximity",
                corner_id=corner_ids[start_i],
                corner_phase=phases[start_i],
            )
            events.append(ev)
        else:
            i += 1

    # 4. Detect Straight-Line Top Speed Cruising
    i = 0
    while i < n:
        if v_arr[i] >= 3.85 and abs(steer_arr[i]) <= 0.08 and radius_arr[i] >= 200.0 and phases[i] == "straight":
            start_i = i
            while (
                i < n
                and v_arr[i] >= 3.85
                and abs(steer_arr[i]) <= 0.08
                and radius_arr[i] >= 200.0
                and phases[i] == "straight"
            ):
                i += 1
            end_i = i - 1
            if (end_i - start_i + 1) >= 12:  # Sustained flat-out run
                ev = extract_event_telemetry_slice(
                    episode_records,
                    steps[start_i],
                    steps[end_i],
                    event_type="straight_line_cruising",
                    corner_id="none",
                    corner_phase="straight",
                )
                events.append(ev)
        else:
            i += 1

    # Sort all detected events chronologically by start_step
    events.sort(key=lambda x: (x["start_step"], x["end_step"]))
    return events


def select_key_lap_events(
    events: List[Dict[str, Any]],
    target_count: int = 5,
) -> List[Dict[str, Any]]:
    """Select a diverse, compact subset of milestone events representing the full lap.
    
    Guarantees representation of:
    - Braking events (onset or sustained)
    - Corner entry / approach
    - Corner apex navigation
    - Corner exit acceleration
    - High-speed cruising or lateral wall proximity
    """
    if len(events) <= target_count:
        return events

    selected: List[Dict[str, Any]] = []

    # 1. Always prioritize braking events if present
    brakes = [e for e in events if "braking" in e["event_type"]]
    if brakes:
        selected.append(brakes[0])

    # 2. Pick corner apexes from distinct corners
    apexes = [e for e in events if e["event_type"] == "corner_apex"]
    seen_corners = set()
    for a in apexes:
        if a["corner_id"] not in seen_corners and len(selected) < target_count - 2:
            selected.append(a)
            seen_corners.add(a["corner_id"])

    # 3. Pick corner approach and exit
    approaches = [e for e in events if e["event_type"] == "corner_approach" and e["corner_id"] in seen_corners]
    if approaches and len(selected) < target_count - 1:
        selected.append(approaches[0])

    exits = [e for e in events if e["event_type"] == "corner_exit" and e["corner_id"] in seen_corners]
    if exits and len(selected) < target_count - 1:
        selected.append(exits[0])

    # 4. Pick wall proximity or straight cruise
    specials = [e for e in events if e["event_type"] in ["lateral_wall_proximity", "straight_line_cruising"]]
    if specials and len(selected) < target_count:
        selected.append(specials[0])

    # Fill remaining slots with chronologically distributed events
    if len(selected) < target_count:
        for e in events:
            if e["event_id"] not in {s["event_id"] for s in selected}:
                selected.append(e)
            if len(selected) >= target_count:
                break

    selected.sort(key=lambda x: x["start_step"])
    return selected
