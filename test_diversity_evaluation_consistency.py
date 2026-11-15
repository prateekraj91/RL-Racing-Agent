"""Test to verify consistency between diversity scaling CSV, JSON summary, and evaluation metrics."""

import csv
import json
import os
import numpy as np
import pytest

CSV_PATH = "analysis/results/diversity_scaling_episodes.csv"
JSON_PATH = "analysis/results/diversity_scaling_summary.json"
PLOTS = [
    "analysis/results/diversity_heldout_completion.png",
    "analysis/results/diversity_heldout_laptime.png",
    "analysis/results/diversity_narrow_completion.png",
    "analysis/results/diversity_hairpin_progress.png",
]

EXPECTED_MODELS = [
    "Single-Track v2 (No DR)",
    "Pool 5 (Calibrated DR)",
    "Pool 10 (Calibrated DR)",
    "Pool 20 (Calibrated DR)",
    "Pool 25 (Calibrated DR)",
]


def test_artifacts_exist():
    assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
    assert os.path.exists(JSON_PATH), f"Missing {JSON_PATH}"
    for p in PLOTS:
        assert os.path.exists(p), f"Missing plot {p}"
        assert os.path.getsize(p) > 1000, f"Plot {p} is empty or corrupted"


def test_episode_data_integrity():
    with open(CSV_PATH, "r") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 675, f"Expected 675 episodes (5 models * 135 runs), found {len(rows)}"

    models_in_csv = sorted(set(r["model"] for r in rows))
    assert models_in_csv == sorted(EXPECTED_MODELS)

    for m in EXPECTED_MODELS:
        m_rows = [r for r in rows if r["model"] == m]
        assert len(m_rows) == 135
        # 35 track pool (10 test + 25 train)
        assert len([r for r in m_rows if r["benchmark"] == "track_pool"]) == 35
        # 70 width sweep (7 widths * 10 test seeds)
        assert len([r for r in m_rows if r["benchmark"] == "width_sweep"]) == 70
        # 30 hairpin (3 pinch factors * 10 test seeds)
        assert len([r for r in m_rows if r["benchmark"] == "hairpin_heldout"]) == 30


def test_csv_and_json_exact_match():
    with open(CSV_PATH, "r") as f:
        rows = list(csv.DictReader(f))
    with open(JSON_PATH, "r") as f:
        summary_json = json.load(f)

    def compute_agg(ep_list):
        total = len(ep_list)
        completed = sum(1 for e in ep_list if e["completed"].lower() == "true")
        crashes = sum(1 for e in ep_list if e["crashed"].lower() == "true")
        timeouts = sum(1 for e in ep_list if e["timeout"].lower() == "true")
        stationary = sum(1 for e in ep_list if e["stationary"].lower() == "true")
        lap_times = [float(e["lap_time"]) for e in ep_list if e["completed"].lower() == "true" and e["lap_time"]]
        avg_lap_time = round(float(np.mean(lap_times)), 2) if lap_times else None
        std_lap_time = round(float(np.std(lap_times)), 2) if lap_times else None
        min_lap_time = int(np.min(lap_times)) if lap_times else None
        max_lap_time = int(np.max(lap_times)) if lap_times else None
        progs = [float(e["lap_progress"]) for e in ep_list]
        avg_progress = round(float(np.mean(progs)), 4)
        median_progress = round(float(np.median(progs)), 4)
        return {
            "total": total, "completed": completed, "crashes": crashes, "timeouts": timeouts,
            "stationary": stationary, "completion_rate": round(completed / total, 4),
            "crash_rate": round(crashes / total, 4), "timeout_rate": round(timeouts / total, 4),
            "stationary_rate": round(stationary / total, 4), "avg_lap_time": avg_lap_time,
            "std_lap_time": std_lap_time, "min_lap_time": min_lap_time, "max_lap_time": max_lap_time,
            "avg_progress": avg_progress, "median_progress": median_progress,
        }

    for m in EXPECTED_MODELS:
        # Standard heldout
        eps_test = [r for r in rows if r["model"] == m and r["benchmark"] == "track_pool" and r["suite"] == "TEST_SEEDS (1001..1010)"]
        calc = compute_agg(eps_test)
        json_stat = summary_json["track_pool_summary"][m]["TEST_SEEDS (1001..1010)"]
        for k in calc:
            assert calc[k] == json_stat[k], f"Mismatch in {m} track_pool TEST_SEEDS for {k}: {calc[k]} != {json_stat[k]}"

        # Width sweep
        for w in [70, 40, 36, 32, 28, 25, 20]:
            eps_w = [r for r in rows if r["model"] == m and r["benchmark"] == "width_sweep" and r["suite"] == f"width_{w}"]
            calc_w = compute_agg(eps_w)
            json_w = summary_json["width_sweep_summary"][m][f"width_{w}"]
            for k in calc_w:
                assert calc_w[k] == json_w[k], f"Mismatch in {m} width_{w} for {k}: {calc_w[k]} != {json_w[k]}"

        # Hairpin
        for f_val in [0.5, 0.35, 0.2]:
            eps_h = [r for r in rows if r["model"] == m and r["benchmark"] == "hairpin_heldout" and r["suite"] == f"pinch_{f_val}"]
            calc_h = compute_agg(eps_h)
            json_h = summary_json["hairpin_summary"][m][f"pinch_{f_val}"]
            for k in calc_h:
                assert calc_h[k] == json_h[k], f"Mismatch in {m} pinch_{f_val} for {k}: {calc_h[k]} != {json_h[k]}"


def test_hairpin_specific_findings():
    with open(JSON_PATH, "r") as f:
        summary_json = json.load(f)

    hs = summary_json["hairpin_summary"]

    # Pinch 0.50
    assert hs["Single-Track v2 (No DR)"]["pinch_0.5"]["completion_rate"] == 0.0
    assert hs["Pool 5 (Calibrated DR)"]["pinch_0.5"]["completion_rate"] == 0.0
    assert hs["Pool 10 (Calibrated DR)"]["pinch_0.5"]["completion_rate"] == 0.20  # 2/10
    assert hs["Pool 20 (Calibrated DR)"]["pinch_0.5"]["completion_rate"] == 0.0
    assert hs["Pool 25 (Calibrated DR)"]["pinch_0.5"]["completion_rate"] == 0.50  # 5/10

    # Pinch 0.35
    assert hs["Single-Track v2 (No DR)"]["pinch_0.35"]["completion_rate"] == 0.0
    assert hs["Pool 5 (Calibrated DR)"]["pinch_0.35"]["completion_rate"] == 0.0
    assert hs["Pool 10 (Calibrated DR)"]["pinch_0.35"]["completion_rate"] == 0.0
    assert hs["Pool 20 (Calibrated DR)"]["pinch_0.35"]["completion_rate"] == 0.0
    assert hs["Pool 25 (Calibrated DR)"]["pinch_0.35"]["completion_rate"] == 0.20  # 2/10

    # Pinch 0.20
    for m in EXPECTED_MODELS:
        assert hs[m]["pinch_0.2"]["completion_rate"] == 0.0
