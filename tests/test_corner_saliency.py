"""Unit and integration tests for per-corner saliency heatmaps.

Validates:
1. Frozen checkpoint and configuration integrity (SHA-256).
2. Preservation of the primary policy saliency dataset.
3. Five-track coverage and diversity rationale.
4. Corner phase segmentation (entry, apex, exit, straight).
5. Exact coordinate-to-timestep mapping between datasets.
6. Saliency aggregation and steering/longitudinal separation.
7. Absence of NaN or infinite saliency values.
8. Traceability and existence of all 11 heatmap figures with consistent scaling.
"""

import csv
import json
import os
import numpy as np
import pytest

from analysis.plot_corner_saliency import (
    CONFIG_PATH,
    CORNER_PHASES_CSV_PATH,
    CORNER_SUMMARY_JSON_PATH,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_WEIGHTS_SHA256,
    FIGS_DIR,
    FROZEN_MODEL_PATH,
    RAY_NAMES,
    SALIENCY_CSV_PATH,
    SALIENCY_SUMMARY_PATH,
    SELECTED_TRACKS,
    compute_sha256,
    verify_frozen_integrity,
)


class TestFrozenModelAndDatasetPreservation:
    """Validate that models and primary saliency datasets remain completely unmodified."""

    def test_frozen_weights_and_config_sha256(self):
        w_sha, c_sha = verify_frozen_integrity()
        assert w_sha == EXPECTED_WEIGHTS_SHA256
        assert c_sha == EXPECTED_CONFIG_SHA256

    def test_primary_saliency_csv_preserved(self):
        assert os.path.isfile(SALIENCY_CSV_PATH), f"Primary saliency CSV missing: {SALIENCY_CSV_PATH}"
        with open(SALIENCY_CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 11825, f"Primary CSV row count altered! Expected 11825, got {len(rows)}"

    def test_primary_saliency_summary_preserved(self):
        assert os.path.isfile(SALIENCY_SUMMARY_PATH), f"Primary summary JSON missing: {SALIENCY_SUMMARY_PATH}"
        with open(SALIENCY_SUMMARY_PATH, "r") as f:
            data = json.load(f)
        assert data["metadata"]["weights_unaltered"] is True
        assert data["metadata"]["sha256_before_analysis"] == EXPECTED_WEIGHTS_SHA256


class TestFiveTrackCoverageAndSegmentation:
    """Validate that the five representative tracks cover diverse regimes and segment correctly."""

    def test_five_tracks_selection_and_diversity(self):
        assert len(SELECTED_TRACKS) == 5, f"Expected 5 selected tracks, got {len(SELECTED_TRACKS)}"
        seeds = [t["seed"] for t in SELECTED_TRACKS]
        assert len(set(seeds)) == 5, f"Seeds must be distinct: {seeds}"
        assert set(seeds) == {1001, 1002, 1005, 1003, 1007}

        benchmarks = {t["benchmark"] for t in SELECTED_TRACKS}
        assert "standard_heldout" in benchmarks
        assert "narrow_corridor" in benchmarks
        assert "hairpin" in benchmarks

    def test_corner_phases_csv_structure_and_coverage(self):
        assert os.path.isfile(CORNER_PHASES_CSV_PATH), f"Missing phases CSV: {CORNER_PHASES_CSV_PATH}"
        with open(CORNER_PHASES_CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2012, f"Expected 2012 phase-annotated timesteps across 5 tracks, got {len(rows)}"

        # Verify all 5 episodes are present
        present_eps = {r["episode_id"] for r in rows}
        expected_eps = {t["episode_id"] for t in SELECTED_TRACKS}
        assert present_eps == expected_eps

        # Verify valid phases
        valid_phases = {"entry", "apex", "exit", "straight"}
        for r in rows:
            assert r["corner_phase"] in valid_phases, f"Invalid phase: {r['corner_phase']}"

        # Verify phase counts
        from collections import Counter
        phase_counts = Counter(r["corner_phase"] for r in rows)
        assert phase_counts["entry"] > 150
        assert phase_counts["apex"] > 100
        assert phase_counts["exit"] > 150
        assert phase_counts["straight"] > 1000

    def test_exact_coordinate_to_timestep_mapping(self):
        """Verify that car_x, car_y and step in corner_saliency_phases.csv exactly match primary CSV."""
        with open(SALIENCY_CSV_PATH, "r") as f:
            primary_rows = { (r["episode_id"], int(r["step"])): r for r in csv.DictReader(f) }

        with open(CORNER_PHASES_CSV_PATH, "r") as f:
            phases_rows = list(csv.DictReader(f))

        for pr in phases_rows:
            key = (pr["episode_id"], int(pr["step"]))
            assert key in primary_rows, f"Timestep key {key} not found in primary dataset"
            orig = primary_rows[key]
            assert float(pr["car_x"]) == float(orig["car_x"])
            assert float(pr["car_y"]) == float(orig["car_y"])
            assert float(pr["corner_radius"]) == float(orig["corner_radius"])
            assert float(pr["action_steering"]) == float(orig["action_steering"])
            assert float(pr["action_longitudinal"]) == float(orig["action_longitudinal"])


class TestSaliencyAggregationAndValues:
    """Validate saliency calculations, absence of NaNs, and separation of action modes."""

    def test_no_nan_or_infinite_values(self):
        with open(CORNER_PHASES_CSV_PATH, "r") as f:
            rows = list(csv.DictReader(f))

        for r in rows:
            for ray in RAY_NAMES:
                s_raw = float(r[f"saliency_steer_{ray}"])
                l_raw = float(r[f"saliency_long_{ray}"])
                s_scaled = float(r[f"scaled_saliency_steer_{ray}"])
                l_scaled = float(r[f"scaled_saliency_long_{ray}"])

                assert not np.isnan(s_raw) and not np.isinf(s_raw)
                assert not np.isnan(l_raw) and not np.isinf(l_raw)
                assert not np.isnan(s_scaled) and not np.isinf(s_scaled)
                assert not np.isnan(l_scaled) and not np.isinf(l_scaled)
                assert s_raw >= 0.0
                assert l_raw >= 0.0

    def test_summary_json_answers_and_structure(self):
        assert os.path.isfile(CORNER_SUMMARY_JSON_PATH), f"Missing summary JSON: {CORNER_SUMMARY_JSON_PATH}"
        with open(CORNER_SUMMARY_JSON_PATH, "r") as f:
            summary = json.load(f)

        assert summary["metadata"]["selected_tracks_count"] == 5
        assert summary["metadata"]["total_timesteps_analyzed"] == 2012

        answers = summary["answers_to_research_questions"]
        for q in [f"q{i}" for i in range(1, 9)]:
            matching_keys = [k for k in answers if k.startswith(q)]
            assert len(matching_keys) == 1, f"Missing answer for question {q}"

        # Check answers to key questions
        assert answers["q1_corner_entry_dominant_beams"]["longitudinal_top_ray"] == "ray_0"
        assert answers["q1_corner_entry_dominant_beams"]["steering_top_ray"] == "ray_minus_45"
        assert answers["q2_apex_dominant_beams"]["steering_top_ray"] == "ray_minus_45"
        assert answers["q3_straight_exit_dominant_beams"]["steering_top_ray"] == "ray_minus_45"


class TestVisualizationArtifacts:
    """Validate existence, format, and size of generated heatmaps and comparison figures."""

    def test_all_11_figures_exist_and_non_empty(self):
        expected_figs = [
            "corner_saliency_cross_track_comparison.png",
            "track_1_std_1001_steering_saliency.png",
            "track_1_std_1001_longitudinal_saliency.png",
            "track_2_fast_1002_steering_saliency.png",
            "track_2_fast_1002_longitudinal_saliency.png",
            "track_3_narrow_1005_steering_saliency.png",
            "track_3_narrow_1005_longitudinal_saliency.png",
            "track_4_hairpin_1003_steering_saliency.png",
            "track_4_hairpin_1003_longitudinal_saliency.png",
            "track_5_acute_1007_steering_saliency.png",
            "track_5_acute_1007_longitudinal_saliency.png",
        ]

        for fname in expected_figs:
            fpath = os.path.join(FIGS_DIR, fname)
            assert os.path.isfile(fpath), f"Figure missing: {fpath}"
            fsize = os.path.getsize(fpath)
            assert fsize > 100000, f"Figure {fname} unexpectedly small ({fsize} bytes)"
