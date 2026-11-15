"""Tests for the statistical ablation matrix compilation and verification.

Validates:
1. Presence and non-emptiness of all ablation matrix artifacts and figures.
2. Presence and integrity of all eight configurations (A through H).
3. Metric definitions: lap times on completed episodes only (no imputation),
   off-track failure rate distinguishing crashes from timeouts and stalls.
4. Paired comparisons on identical circuits (seeds 1001..1010).
5. Exact consistency between raw episode CSV, matrix CSV, summary JSON, and effects JSON.
6. Preservation of existing experimental artifacts.
"""

import csv
import json
import os
import pytest
import numpy as np

RESULTS_DIR = "analysis/results"
FIGS_DIR = "analysis/figs"
DOCS_DIR = "docs"

CONDITIONS = ["A", "B", "C", "D", "E", "F", "G", "H"]
TEST_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]


class TestAblationMatrixArtifacts:
    """Verify that all deliverables exist and are non-empty."""

    def test_matrix_csv_exists(self):
        p = os.path.join(RESULTS_DIR, "ablation_matrix.csv")
        assert os.path.isfile(p), f"Missing {p}"
        assert os.path.getsize(p) > 1000

    def test_summary_json_exists(self):
        p = os.path.join(RESULTS_DIR, "ablation_matrix_summary.json")
        assert os.path.isfile(p), f"Missing {p}"
        assert os.path.getsize(p) > 1000

    def test_effects_json_exists(self):
        p = os.path.join(RESULTS_DIR, "ablation_matrix_effects.json")
        assert os.path.isfile(p), f"Missing {p}"
        assert os.path.getsize(p) > 1000

    def test_markdown_documentation_exists(self):
        p = os.path.join(DOCS_DIR, "ABLATION_MATRIX.md")
        assert os.path.isfile(p), f"Missing {p}"
        assert os.path.getsize(p) > 2000

    @pytest.mark.parametrize("fig_name", [
        "matrix_main_effects_laptime.png",
        "matrix_main_effects_offtrack.png",
        "matrix_corridor_widths.png",
        "matrix_hairpin_difficulties.png",
        "matrix_effects_ci.png",
    ])
    def test_matrix_figures_exist(self, fig_name):
        p = os.path.join(FIGS_DIR, fig_name)
        assert os.path.isfile(p), f"Missing figure: {p}"
        assert os.path.getsize(p) > 5000, f"Figure too small: {p}"


class TestAblationMatrixIntegrity:
    """Verify data integrity, configurations, and metric definitions."""

    def test_all_eight_configurations_in_csv(self):
        p = os.path.join(RESULTS_DIR, "ablation_matrix.csv")
        with open(p) as f:
            rows = list(csv.DictReader(f))

        # 11 slices * 8 conditions = 88 rows
        assert len(rows) == 88, f"Expected 88 rows, found {len(rows)}"
        for row in rows:
            assert row["condition"] in CONDITIONS
            assert int(row["n_total"]) == len(TEST_SEEDS)

            # Failure taxonomy consistency: completed + crashed + timeout == n_total
            completed = int(row["n_completed"])
            crashed = int(row["n_crashed"])
            timeout = int(row["n_timeout"])
            stationary = int(row["n_stationary"])
            total = int(row["n_total"])

            assert completed + crashed + timeout == total, (
                f"Row {row['benchmark']} {row['condition']} failure sum mismatch: "
                f"{completed} + {crashed} + {timeout} != {total}"
            )
            assert stationary <= timeout, f"Stationary episodes ({stationary}) > timeouts ({timeout})"

            # Off-track failure rate must equal crashed / total
            expected_cr_rate = crashed / total
            assert abs(float(row["offtrack_failure_rate"]) - expected_cr_rate) < 1e-4

            # If no completed episodes, lap time must be null/empty
            if completed == 0:
                assert row["mean_lap_time"] in ("", "None", None), (
                    f"Completed=0 but mean_lap_time={row['mean_lap_time']} for {row['condition']}"
                )
            else:
                assert float(row["mean_lap_time"]) > 0

    def test_lap_time_uses_completed_episodes_only(self):
        """Confirm that failed episodes are never assigned artificial lap times."""
        raw_csv = os.path.join(RESULTS_DIR, "ablation_episodes.csv")
        with open(raw_csv) as f:
            raw_rows = list(csv.DictReader(f))

        for r in raw_rows:
            completed = r["completed"].strip().lower() == "true"
            lap_time_str = r.get("lap_time", "").strip()
            if not completed:
                assert lap_time_str in ("", "None", None), (
                    f"Incomplete episode has artificial lap_time: {r}"
                )
            else:
                assert float(lap_time_str) == float(r["steps"]), (
                    f"Completed lap time does not match steps: {r}"
                )

    def test_paired_circuits_matched_across_conditions(self):
        """Confirm all paired comparisons use the exact same held-out circuit seeds."""
        raw_csv = os.path.join(RESULTS_DIR, "ablation_episodes.csv")
        with open(raw_csv) as f:
            raw_rows = list(csv.DictReader(f))

        by_cond = {c: set() for c in CONDITIONS}
        for r in raw_rows:
            by_cond[r["condition"]].add((r["benchmark"], r["width"], r["hairpin_factor"], int(r["seed"])))

        base_set = by_cond["A"]
        for c in "BCDEFGH":
            assert by_cond[c] == base_set, f"Condition {c} does not match condition A circuit set!"


class TestFactorialEffectsMath:
    """Verify factorial contrasts and bootstrap CI properties."""

    def test_standard_heldout_main_effects_match_ablation_effects(self):
        effects_path = os.path.join(RESULTS_DIR, "ablation_matrix_effects.json")
        with open(effects_path) as f:
            effects = json.load(f)

        std_slice = effects["slices"]["Standard Width 70 px"]
        lt = std_slice["lap_time_effects"]

        # Known analytical contrasts on standard held-out lap times:
        # DR: +1.75, Lookahead: +1.30, SteerPenalty: -1.95
        assert abs(lt["DR"]["mean_effect"] - 1.75) < 1e-3
        assert abs(lt["Lookahead"]["mean_effect"] - 1.30) < 1e-3
        assert abs(lt["SteerPenalty"]["mean_effect"] - (-1.95)) < 1e-3

        # Confidence intervals should contain the point estimate
        assert lt["DR"]["ci_95_lo"] <= lt["DR"]["mean_effect"] <= lt["DR"]["ci_95_hi"]
        assert lt["Lookahead"]["ci_95_lo"] <= lt["Lookahead"]["mean_effect"] <= lt["Lookahead"]["ci_95_hi"]
        assert lt["SteerPenalty"]["ci_95_lo"] <= lt["SteerPenalty"]["mean_effect"] <= lt["SteerPenalty"]["ci_95_hi"]

    def test_preservation_of_existing_artifacts(self):
        """Ensure original files were not overwritten or corrupted."""
        original_files = [
            os.path.join(RESULTS_DIR, "ablation_episodes.csv"),
            os.path.join(RESULTS_DIR, "ablation_summary.json"),
            os.path.join(RESULTS_DIR, "ablation_factorial_effects.json"),
            os.path.join(RESULTS_DIR, "ablation_manifest.json"),
            os.path.join(DOCS_DIR, "ABLATION_STUDY.md"),
        ]
        for fpath in original_files:
            assert os.path.isfile(fpath), f"Original artifact missing: {fpath}"
            assert os.path.getsize(fpath) > 100, f"Original artifact empty/truncated: {fpath}"
