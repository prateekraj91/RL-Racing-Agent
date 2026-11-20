"""Buddh International Circuit — Robustness & Perturbation Evaluation Suite.

Executes controlled trials across:
    - Deterministic reproducibility verification (repeat identical nominal trials).
    - Lateral starting offset perturbations (+/- 8px, +/- 15px).
    - Initial heading perturbations (+/- 5 deg, +/- 10 deg).
    - Combined lateral and heading perturbations.
    - Longitudinal start line offset perturbations.

Generates:
    - buddh/robustness_manifest.json (fixed evaluation manifest)
    - analysis/buddh/robustness_summary.json
    - analysis/buddh/robustness_summary.csv
"""

import csv
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from buddh.eval_buddh import evaluate_policy_on_buddh


ROBUSTNESS_CONFIGS: List[Dict[str, Any]] = [
    {
        "trial_id": "trial_01_nominal_rep1",
        "description": "Nominal baseline start pose (repetition 1)",
        "options": {},
    },
    {
        "trial_id": "trial_02_nominal_rep2",
        "description": "Nominal baseline start pose (repetition 2, bitwise check)",
        "options": {},
    },
    {
        "trial_id": "trial_03_lateral_left_15px",
        "description": "Lateral offset -15.0 px (-2.57m towards left boundary)",
        "options": {"lateral_offset_px": -15.0},
    },
    {
        "trial_id": "trial_04_lateral_right_15px",
        "description": "Lateral offset +15.0 px (+2.57m towards right boundary)",
        "options": {"lateral_offset_px": 15.0},
    },
    {
        "trial_id": "trial_05_lateral_left_8px",
        "description": "Lateral offset -8.0 px (-1.37m left offset)",
        "options": {"lateral_offset_px": -8.0},
    },
    {
        "trial_id": "trial_06_lateral_right_8px",
        "description": "Lateral offset +8.0 px (+1.37m right offset)",
        "options": {"lateral_offset_px": 8.0},
    },
    {
        "trial_id": "trial_07_heading_left_10deg",
        "description": "Initial heading perturbation -10.0 deg (angled left)",
        "options": {"heading_offset_deg": -10.0},
    },
    {
        "trial_id": "trial_08_heading_right_10deg",
        "description": "Initial heading perturbation +10.0 deg (angled right)",
        "options": {"heading_offset_deg": 10.0},
    },
    {
        "trial_id": "trial_09_heading_left_5deg",
        "description": "Initial heading perturbation -5.0 deg (angled left)",
        "options": {"heading_offset_deg": -5.0},
    },
    {
        "trial_id": "trial_10_heading_right_5deg",
        "description": "Initial heading perturbation +5.0 deg (angled right)",
        "options": {"heading_offset_deg": 5.0},
    },
    {
        "trial_id": "trial_11_combined_offset",
        "description": "Combined perturbation: Lateral +10.0 px, Heading -5.0 deg",
        "options": {"lateral_offset_px": 10.0, "heading_offset_deg": -5.0},
    },
    {
        "trial_id": "trial_12_longitudinal_10m",
        "description": "Longitudinal start line offset +10.0m (+58.33 px forward)",
        "options": {"longitudinal_offset_px": 58.33},
    },
]


def run_robustness_suite(
    manifest_output_path: str = "buddh/robustness_manifest.json",
    summary_output_json: str = "analysis/buddh/robustness_summary.json",
    summary_output_csv: str = "analysis/buddh/robustness_summary.csv",
) -> Dict[str, Any]:
    """Execute all configured robustness trials and summarize results."""
    os.makedirs(os.path.dirname(manifest_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_output_json), exist_ok=True)

    # Save fixed evaluation manifest
    with open(manifest_output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "suite_name": "Buddh Circuit Robustness Evaluation Suite",
                "model": "models/generalist_v1.pth",
                "max_steps": 15000,
                "trials": ROBUSTNESS_CONFIGS,
            },
            f,
            indent=2,
        )

    trial_summaries: List[Dict[str, Any]] = []

    print("=" * 70)
    print("STARTING BUDDH CIRCUIT ROBUSTNESS SUITE (12 TRIALS)")
    print("=" * 70)

    for i, cfg in enumerate(ROBUSTNESS_CONFIGS):
        tid = cfg["trial_id"]
        desc = cfg["description"]
        opts = cfg["options"]
        print(f"\n[{i+1:02d}/12] Executing {tid}: {desc}...")

        csv_path = f"analysis/buddh/rollouts/{tid}.csv"
        json_path = f"analysis/buddh/rollouts/{tid}.json"

        t0 = time.perf_counter()
        res = evaluate_policy_on_buddh(
            max_steps=15000,
            options=opts,
            output_csv_path=csv_path,
            output_json_path=json_path,
            verbose_interval=0,
        )
        elapsed = time.perf_counter() - t0
        m = res["metrics"]

        summary_entry = {
            "trial_id": tid,
            "description": desc,
            "completed": m["1_lap_completion"],
            "steps": m["2_total_simulation_steps"],
            "lap_time_s": m["3_lap_time_s"],
            "avg_speed_kmh": m["4_average_speed"]["km_per_h"],
            "max_speed_kmh": m["5_maximum_speed"]["km_per_h"],
            "min_steady_speed_kmh": m["6_minimum_speed"]["steady_minimum_km_per_h"],
            "rms_dist_m": m["7_distance_from_centerline_m"]["rms"],
            "max_dist_m": m["7_distance_from_centerline_m"]["max"],
            "rms_dist_px": m["7_distance_from_centerline_px"]["rms"],
            "max_dist_px": m["7_distance_from_centerline_px"]["max"],
            "off_track_steps": m["8_total_off_track_steps"],
            "collisions": m["9_collisions"],
            "mean_steer": m["10_steering_magnitude"]["mean"],
            "max_steer": m["10_steering_magnitude"]["max"],
            "steer_diff_mean": m["11_steering_variation_consecutive_steps"]["mean"],
            "rms_heading_err_deg": m["12_heading_error_deg"]["rms"],
            "max_heading_err_deg": m["12_heading_error_deg"]["max"],
            "termination_reason": m["13_episode_termination_reason"],
            "wall_clock_s": round(elapsed, 2),
        }
        trial_summaries.append(summary_entry)

        status_str = "COMPLETED" if m["1_lap_completion"] else "FAILED"
        print(
            f"       --> {status_str} in {m['2_total_simulation_steps']} steps "
            f"({m['3_lap_time_s']}s) | RMS dev: {m['7_distance_from_centerline_m']['rms']}m | "
            f"Collisions: {m['9_collisions']} | Off-track: {m['8_total_off_track_steps']}"
        )

    # Deterministic check between trial 1 and trial 2
    t1 = trial_summaries[0]
    t2 = trial_summaries[1]
    reproducible = (
        t1["completed"] == t2["completed"]
        and t1["steps"] == t2["steps"]
        and t1["lap_time_s"] == t2["lap_time_s"]
        and abs(t1["rms_dist_px"] - t2["rms_dist_px"]) < 1e-6
    )

    completed_trials = sum(1 for t in trial_summaries if t["completed"])
    success_rate_pct = round(completed_trials / len(trial_summaries) * 100.0, 1)

    suite_result = {
        "suite_name": "Buddh Circuit Robustness Evaluation Suite",
        "total_trials": len(trial_summaries),
        "completed_trials": completed_trials,
        "success_rate_pct": success_rate_pct,
        "deterministic_reproducibility": reproducible,
        "trials": trial_summaries,
    }

    # Save summary JSON
    with open(summary_output_json, "w", encoding="utf-8") as f:
        json.dump(suite_result, f, indent=2)

    # Save summary CSV
    if trial_summaries:
        keys = list(trial_summaries[0].keys())
        with open(summary_output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(trial_summaries)

    print("\n" + "=" * 70)
    print("ROBUSTNESS SUITE SUMMARY")
    print("=" * 70)
    print(f"Total Trials:                 {len(trial_summaries)}")
    print(f"Completed Laps:               {completed_trials}/{len(trial_summaries)} ({success_rate_pct}%)")
    print(f"Deterministic Reproducibility: {'[VERIFIED EXACT MATCH]' if reproducible else '[REPRODUCIBILITY MISMATCH]'}")
    print("=" * 70)

    return suite_result


if __name__ == "__main__":
    run_robustness_suite()
