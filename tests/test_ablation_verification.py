"""Automated verification tests for the 2×2×2 Factorial Ablation Study.

Validates:
1. Presence and integrity of all 8 condition run directories and 200k-step checkpoints.
2. Configuration correctness (DR on/off, Lookahead 200/300, SteerPenalty 0.01/0.0).
3. Metric aggregation consistency between per-episode CSV and summary JSON.
4. Factorial effects integrity (main effects, 2-way interactions, 3-way interaction).
5. All 4 publication figures exist and are valid image files.
"""

import json
import os
import csv
import pytest

from analysis.eval_ablation import CONDITIONS, run_name_for_condition, WIDTH_LEVELS, HAIRPIN_FACTORS
from sac.track_pools import TEST_SEEDS

RUNS_DIR = "runs"
RESULTS_DIR = "analysis/results"
FIGS_DIR = "analysis/figs"


class TestAblationRunIntegrity:
    """Verify that all 8 training runs finished with 200k steps and valid checkpoints."""

    @pytest.mark.parametrize("label,dr,look,steer", CONDITIONS)
    def test_run_directory_and_checkpoints_exist(self, label, dr, look, steer):
        run_name = run_name_for_condition(label, dr, look, steer)
        rdir = os.path.join(RUNS_DIR, run_name)
        if not os.path.isdir(rdir):
            pytest.skip(f"gitignored training run data absent ({rdir}); regenerate via the ablation scripts")

        actor_path = os.path.join(rdir, "best_actor.pth")
        assert os.path.isfile(actor_path), f"Missing best_actor.pth in {rdir}"
        assert os.path.getsize(actor_path) > 10000, f"Checkpoint file too small: {actor_path}"

        cfg_path = os.path.join(rdir, "run_config.json")
        assert os.path.isfile(cfg_path), f"Missing run_config.json in {rdir}"
        with open(cfg_path) as f:
            cfg = json.load(f)

        assert cfg.get("total_steps") == 200000, f"Condition {label} total_steps != 200000: {cfg.get('total_steps')}"
        assert cfg.get("seed") == 42, f"Condition {label} seed != 42"
        assert cfg.get("train_pool") == 25, f"Condition {label} train_pool != 25"
        assert cfg.get("obs_ray_max_dist") == look, f"Condition {label} lookahead mismatch: {cfg.get('obs_ray_max_dist')} vs {look}"
        assert cfg.get("steer_penalty") == steer, f"Condition {label} steer_penalty mismatch: {cfg.get('steer_penalty')} vs {steer}"

        dr_cfg = cfg.get("domain_randomize", {})
        dr_enabled = dr_cfg.get("enabled", False) if isinstance(dr_cfg, dict) else bool(dr_cfg)
        assert dr_enabled == dr, f"Condition {label} DR mismatch: {dr_enabled} vs {dr}"


class TestAblationEvaluationDataIntegrity:
    """Verify evaluation CSV, JSON summary, and consistency."""

    def test_artifacts_exist(self):
        csv_path = os.path.join(RESULTS_DIR, "ablation_episodes.csv")
        json_path = os.path.join(RESULTS_DIR, "ablation_summary.json")
        effects_path = os.path.join(RESULTS_DIR, "ablation_factorial_effects.json")

        assert os.path.isfile(csv_path), f"Missing {csv_path}"
        assert os.path.isfile(json_path), f"Missing {json_path}"
        assert os.path.isfile(effects_path), f"Missing {effects_path}"

    def test_episode_count_and_completeness(self):
        csv_path = os.path.join(RESULTS_DIR, "ablation_episodes.csv")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Expected: 8 conditions * (10 std + 10*len(WIDTH_LEVELS) + 10*len(HAIRPIN_FACTORS))
        # 10 + 70 + 30 = 110 episodes per condition -> 880 total
        expected_per_cond = len(TEST_SEEDS) * (1 + len(WIDTH_LEVELS) + len(HAIRPIN_FACTORS))
        expected_total = 8 * expected_per_cond
        assert len(rows) == expected_total, f"Expected {expected_total} rows, found {len(rows)}"

        # Check all condition labels exist
        conditions_found = set(r["condition"] for r in rows)
        assert conditions_found == set("ABCDEFGH"), f"Missing conditions: {set('ABCDEFGH') - conditions_found}"

    def test_csv_json_consistency(self):
        csv_path = os.path.join(RESULTS_DIR, "ablation_episodes.csv")
        json_path = os.path.join(RESULTS_DIR, "ablation_summary.json")

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        with open(json_path) as f:
            summary = json.load(f)

        cr = summary["condition_results"]
        for label in "ABCDEFGH":
            assert label in cr, f"Missing condition {label} in summary JSON"
            std_summary = cr[label]["standard_heldout"]

            # Filter rows for standard heldout
            std_rows = [r for r in rows if r["condition"] == label and r["benchmark"] == "standard_heldout"]
            assert len(std_rows) == len(TEST_SEEDS)

            completed = sum(1 for r in std_rows if r["completed"] == "True")
            completion_rate = round(completed / len(std_rows), 4)
            assert std_summary["completed"] == completed
            assert abs(std_summary["completion_rate"] - completion_rate) < 1e-4

    def test_factorial_effects_structure(self):
        effects_path = os.path.join(RESULTS_DIR, "ablation_factorial_effects.json")
        with open(effects_path) as f:
            effects = json.load(f)

        assert "completion_rate" in effects
        cr_eff = effects["completion_rate"]
        assert "main_effects" in cr_eff
        assert set(cr_eff["main_effects"].keys()) == {"DR", "Lookahead", "SteerPenalty"}

        assert "two_factor_interactions" in cr_eff
        assert set(cr_eff["two_factor_interactions"].keys()) == {"DR×Lookahead", "DR×SteerPenalty", "Lookahead×SteerPenalty"}

        assert "three_factor_interaction" in cr_eff
        assert "DR×Lookahead×SteerPenalty" in cr_eff["three_factor_interaction"]


class TestAblationFiguresExist:
    """Verify that all generated figures exist and are non-empty."""

    @pytest.mark.parametrize("fig_name", [
        "ablation_condition_bars.png",
        "ablation_factorial_effects.png",
        "ablation_narrow_heatmap.png",
        "ablation_hairpin_bars.png",
    ])
    def test_figure_exists(self, fig_name):
        fpath = os.path.join(FIGS_DIR, fig_name)
        assert os.path.isfile(fpath), f"Missing figure: {fpath}"
        assert os.path.getsize(fpath) > 5000, f"Figure file too small: {fpath}"
