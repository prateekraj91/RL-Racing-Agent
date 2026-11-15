"""Compile rigorous statistical ablation matrix from episode-level data.

Computes:
1. Condition-level metrics for lap time, completion rate, off-track failure rate,
   timeouts, and stationary failures across all benchmark difficulties.
2. Paired factorial main effects, 2-way interactions, and 3-way interactions.
3. 95% bootstrap confidence intervals, paired t-tests, and exact Wilcoxon signed-rank tests.
4. Deliverables:
   - analysis/results/ablation_matrix.csv
   - analysis/results/ablation_matrix_summary.json
   - analysis/results/ablation_matrix_effects.json
   - analysis/figs/matrix_main_effects_laptime.png
   - analysis/figs/matrix_main_effects_offtrack.png
   - analysis/figs/matrix_corridor_widths.png
   - analysis/figs/matrix_hairpin_difficulties.png
   - analysis/figs/matrix_effects_ci.png
"""

import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, "analysis", "results")
FIGS_DIR = os.path.join(ROOT_DIR, "analysis", "figs")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")

# Ablation conditions: (label, dr, lookahead, steer_penalty)
CONDITIONS = [
    ("A", True, 200, 0.01),
    ("B", True, 200, 0.00),
    ("C", True, 300, 0.01),
    ("D", True, 300, 0.00),
    ("E", False, 200, 0.01),
    ("F", False, 200, 0.00),
    ("G", False, 300, 0.01),
    ("H", False, 300, 0.00),
]

COND_MAP = {c[0]: {"dr": c[1], "lookahead": c[2], "steer_penalty": c[3]} for c in CONDITIONS}

# Contrast vectors (+1 / -1) for conditions A through H:
# Factor levels:
# DR: High (+1) = A, B, C, D; Low (-1) = E, F, G, H
# Lookahead: High (+1) = C, D, G, H (300px); Low (-1) = A, B, E, F (200px)
# SteerPenalty: High (+1) = A, C, E, G (0.01); Low (-1) = B, D, F, H (0.00)
X_DR = np.array([+1, +1, +1, +1, -1, -1, -1, -1], dtype=float)
X_LOOK = np.array([-1, -1, +1, +1, -1, -1, +1, +1], dtype=float)
X_STEER = np.array([+1, -1, +1, -1, +1, -1, +1, -1], dtype=float)

X_DR_LOOK = X_DR * X_LOOK
X_DR_STEER = X_DR * X_STEER
X_LOOK_STEER = X_LOOK * X_STEER
X_3WAY = X_DR * X_LOOK * X_STEER

BENCHMARK_SLICES = [
    {"benchmark": "standard_heldout", "difficulty_type": "nominal_width", "difficulty_value": 70, "label": "Standard Width 70 px"},
    {"benchmark": "narrow_corridor", "difficulty_type": "width", "difficulty_value": 70, "label": "Narrow Corridor W=70 px"},
    {"benchmark": "narrow_corridor", "difficulty_type": "width", "difficulty_value": 40, "label": "Narrow Corridor W=40 px"},
    {"benchmark": "narrow_corridor", "difficulty_type": "width", "difficulty_value": 36, "label": "Narrow Corridor W=36 px"},
    {"benchmark": "narrow_corridor", "difficulty_type": "width", "difficulty_value": 32, "label": "Narrow Corridor W=32 px"},
    {"benchmark": "narrow_corridor", "difficulty_type": "width", "difficulty_value": 28, "label": "Narrow Corridor W=28 px"},
    {"benchmark": "narrow_corridor", "difficulty_type": "width", "difficulty_value": 25, "label": "Narrow Corridor W=25 px"},
    {"benchmark": "narrow_corridor", "difficulty_type": "width", "difficulty_value": 20, "label": "Narrow Corridor W=20 px"},
    {"benchmark": "hairpin", "difficulty_type": "pinch_factor", "difficulty_value": 0.50, "label": "Hairpin Pinch f=0.50"},
    {"benchmark": "hairpin", "difficulty_type": "pinch_factor", "difficulty_value": 0.35, "label": "Hairpin Pinch f=0.35"},
    {"benchmark": "hairpin", "difficulty_type": "pinch_factor", "difficulty_value": 0.20, "label": "Hairpin Pinch f=0.20"},
]


def load_episodes(csv_path: str) -> List[Dict[str, Any]]:
    """Load and type-cast raw evaluation episodes."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    episodes = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cond = r["condition"].strip()
            bench = r["benchmark"].strip()
            w = int(r["width"]) if r["width"] else None
            hf_str = r.get("hairpin_factor", "").strip()
            hf = float(hf_str) if hf_str and hf_str != "nan" else None
            seed = int(r["seed"])

            completed = r["completed"].strip().lower() == "true"
            crashed = r["crashed"].strip().lower() == "true"
            timeout = r["timeout"].strip().lower() == "true"
            steps = int(r["steps"])
            lap_time = float(r["lap_time"]) if r.get("lap_time") and r["lap_time"].strip() != "" and r["lap_time"].strip() != "None" else None
            lap_progress = float(r["lap_progress"])
            max_speed = float(r["max_speed"])

            # Reliable stationary stall detection:
            # Car timed out and made virtually zero progress or speed
            stationary = timeout and (max_speed < 0.05 or abs(lap_progress) < 0.005)

            outcome = r["outcome"].strip()
            if completed:
                clean_outcome = "completed"
            elif crashed:
                clean_outcome = "crashed"
            elif stationary:
                clean_outcome = "stationary"
            elif timeout:
                clean_outcome = "timeout"
            else:
                clean_outcome = outcome

            episodes.append({
                "condition": cond,
                "run_name": r.get("run_name", ""),
                "dr": COND_MAP[cond]["dr"],
                "lookahead": COND_MAP[cond]["lookahead"],
                "steer_penalty": COND_MAP[cond]["steer_penalty"],
                "benchmark": bench,
                "width": w,
                "hairpin_factor": hf,
                "seed": seed,
                "outcome": clean_outcome,
                "completed": completed,
                "crashed": crashed,  # Off-track / wall boundary failure
                "timeout": timeout,
                "stationary": stationary,
                "steps": steps,
                "lap_time": lap_time if completed else None,
                "lap_progress": lap_progress,
                "max_speed": max_speed,
            })
    return episodes


def filter_slice(episodes: List[Dict[str, Any]], bench: str, diff_type: str, diff_val: float) -> List[Dict[str, Any]]:
    """Filter episodes for a specific benchmark difficulty slice."""
    matched = []
    for e in episodes:
        if e["benchmark"] != bench:
            continue
        if diff_type in ("nominal_width", "width") and e["width"] == int(diff_val):
            matched.append(e)
        elif diff_type == "pinch_factor" and e["hairpin_factor"] is not None and abs(e["hairpin_factor"] - float(diff_val)) < 1e-4:
            matched.append(e)
    return matched


def student_t_pvalue(t_stat: float, df: int) -> float:
    """Compute two-sided p-value for Student's t distribution with df degrees of freedom."""
    if df <= 0 or math.isnan(t_stat):
        return float("nan")
    t = abs(float(t_stat))
    if t == 0.0:
        return 1.0
    if df == 1:
        cdf = 0.5 + math.atan(t) / math.pi
        return 2.0 * max(0.0, 1.0 - cdf)

    log_c = math.lgamma((df + 1) / 2) - 0.5 * math.log(math.pi * df) - math.lgamma(df / 2)
    c_df = math.exp(log_c)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    u = 0.5 * t * (nodes + 1)
    w = 0.5 * t * weights
    integral = float(np.sum(w * (1.0 + (u ** 2) / df) ** (-(df + 1) / 2.0)))
    cdf = 0.5 + c_df * integral
    return float(min(1.0, max(0.0, 2.0 * (1.0 - cdf))))


def exact_wilcoxon_signed_rank(diffs: np.ndarray) -> Tuple[float, float]:
    """Compute exact Wilcoxon signed-rank test statistic and two-sided permutation p-value."""
    nonzero = np.array([d for d in diffs if abs(d) > 1e-9], dtype=float)
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0

    abs_d = np.abs(nonzero)
    order = np.argsort(abs_d)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)

    # Average ranks for ties
    unique_vals, counts = np.unique(abs_d, return_counts=True)
    for val, count in zip(unique_vals, counts):
        if count > 1:
            indices = np.where(np.abs(abs_d - val) < 1e-9)[0]
            ranks[indices] = float(np.mean(ranks[indices]))

    pos_rank_sum = float(np.sum(ranks[nonzero > 0]))
    neg_rank_sum = float(np.sum(ranks[nonzero < 0]))
    w_stat = min(pos_rank_sum, neg_rank_sum)

    if n <= 14:
        # Exact enumeration of all 2^n sign configurations
        possible_w = []
        for mask in range(1 << n):
            w_sample = sum(ranks[i] for i in range(n) if (mask & (1 << i)))
            possible_w.append(w_sample)
        max_w = sum(ranks)
        tail_count = sum(1 for w in possible_w if w <= w_stat or w >= (max_w - w_stat))
        p_val = min(1.0, tail_count / len(possible_w))
    else:
        # Normal approximation with continuity correction
        mean_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0
        # tie correction
        for count in counts:
            if count > 1:
                var_w -= (count ** 3 - count) / 48.0
        sigma_w = math.sqrt(max(var_w, 1e-9))
        z = (w_stat - mean_w + 0.5) / sigma_w
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        p_val = min(1.0, max(0.0, p_val))

    return float(w_stat), float(p_val)


def bootstrap_ci_paired(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    """Compute paired bootstrap 95% confidence interval using percentile method."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    # Resample circuit indices with replacement
    boot_indices = rng.choice(n, size=(n_boot, n), replace=True)
    boot_means = np.mean(diffs[boot_indices], axis=1)
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(diffs)), lo, hi


def compute_condition_stats(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for a set of episodes belonging to one condition."""
    n_total = len(episodes)
    if n_total == 0:
        return {}

    n_completed = sum(1 for e in episodes if e["completed"])
    n_crashed = sum(1 for e in episodes if e["crashed"])
    n_timeout = sum(1 for e in episodes if e["timeout"])
    n_stationary = sum(1 for e in episodes if e["stationary"])

    completion_rate = n_completed / n_total
    offtrack_failure_rate = n_crashed / n_total
    timeout_rate = n_timeout / n_total
    stationary_rate = n_stationary / n_total

    lap_times = [e["lap_time"] for e in episodes if e["completed"] and e["lap_time"] is not None]
    if len(lap_times) > 0:
        mean_lt = float(np.mean(lap_times))
        med_lt = float(np.median(lap_times))
        std_lt = float(np.std(lap_times, ddof=1)) if len(lap_times) > 1 else 0.0
        min_lt = float(np.min(lap_times))
        max_lt = float(np.max(lap_times))
    else:
        mean_lt, med_lt, std_lt, min_lt, max_lt = None, None, None, None, None

    progresses = [e["lap_progress"] for e in episodes]
    mean_prog = float(np.mean(progresses))
    med_prog = float(np.median(progresses))

    return {
        "n_total": n_total,
        "n_completed": n_completed,
        "completion_rate": round(completion_rate, 4),
        "n_crashed": n_crashed,
        "offtrack_failure_rate": round(offtrack_failure_rate, 4),
        "n_timeout": n_timeout,
        "timeout_rate": round(timeout_rate, 4),
        "n_stationary": n_stationary,
        "stationary_rate": round(stationary_rate, 4),
        "mean_lap_time": round(mean_lt, 2) if mean_lt is not None else None,
        "median_lap_time": round(med_lt, 2) if med_lt is not None else None,
        "std_lap_time": round(std_lt, 2) if std_lt is not None else None,
        "min_lap_time": int(min_lt) if min_lt is not None else None,
        "max_lap_time": int(max_lt) if max_lt is not None else None,
        "mean_progress": round(mean_prog, 4),
        "median_progress": round(med_prog, 4),
    }


def analyze_slice(slice_episodes: List[Dict[str, Any]], slice_info: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Perform condition-level analysis and paired factorial effect analysis on one slice."""
    by_cond = {c[0]: [] for c in CONDITIONS}
    for e in slice_episodes:
        by_cond[e["condition"]].append(e)

    cond_stats = {}
    for c in "ABCDEFGH":
        cond_stats[c] = compute_condition_stats(by_cond[c])

    # Build seed-level matrices for paired comparisons
    seeds = sorted(list(set(e["seed"] for e in slice_episodes)))
    n_seeds = len(seeds)

    # 1. Off-track failure indicator: crashed in {0, 1}
    crash_matrix = np.zeros((n_seeds, 8), dtype=float)
    completed_matrix = np.zeros((n_seeds, 8), dtype=float)
    lap_time_matrix = np.full((n_seeds, 8), np.nan, dtype=float)

    for i, s in enumerate(seeds):
        for j, c in enumerate("ABCDEFGH"):
            ep_list = [e for e in by_cond[c] if e["seed"] == s]
            if len(ep_list) > 0:
                ep = ep_list[0]
                crash_matrix[i, j] = 1.0 if ep["crashed"] else 0.0
                completed_matrix[i, j] = 1.0 if ep["completed"] else 0.0
                if ep["completed"] and ep["lap_time"] is not None:
                    lap_time_matrix[i, j] = ep["lap_time"]

    # --- Factorial Effects on Off-Track Failure Rate ---
    # Contrast direction: negative change means fewer failures (- percentage points)
    dr_crash_diffs = np.dot(crash_matrix, X_DR) / 4.0
    look_crash_diffs = np.dot(crash_matrix, X_LOOK) / 4.0
    steer_crash_diffs = np.dot(crash_matrix, X_STEER) / 4.0
    dr_look_crash = np.dot(crash_matrix, X_DR_LOOK) / 4.0
    dr_steer_crash = np.dot(crash_matrix, X_DR_STEER) / 4.0
    look_steer_crash = np.dot(crash_matrix, X_LOOK_STEER) / 4.0
    three_way_crash = np.dot(crash_matrix, X_3WAY) / 4.0

    crash_effects = {}
    for name, diffs in [
        ("DR", dr_crash_diffs),
        ("Lookahead", look_crash_diffs),
        ("SteerPenalty", steer_crash_diffs),
        ("DR×Lookahead", dr_look_crash),
        ("DR×SteerPenalty", dr_steer_crash),
        ("Lookahead×SteerPenalty", look_steer_crash),
        ("DR×Lookahead×SteerPenalty", three_way_crash),
    ]:
        mean_eff, ci_lo, ci_hi = bootstrap_ci_paired(diffs)
        std_eff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        se_eff = std_eff / math.sqrt(len(diffs))
        t_stat = mean_eff / se_eff if se_eff > 1e-9 else 0.0
        t_p = student_t_pvalue(t_stat, len(diffs) - 1) if se_eff > 1e-9 else (1.0 if abs(mean_eff) < 1e-9 else 0.0)
        w_stat, w_p = exact_wilcoxon_signed_rank(diffs)

        crash_effects[name] = {
            "mean_effect": round(mean_eff, 4),
            "percentage_points": round(mean_eff * 100.0, 2),
            "ci_95_lo": round(ci_lo, 4),
            "ci_95_hi": round(ci_hi, 4),
            "std_error": round(se_eff, 4),
            "t_statistic": round(t_stat, 4) if not math.isnan(t_stat) else None,
            "t_pvalue": round(t_p, 6) if not math.isnan(t_p) else None,
            "wilcoxon_stat": round(w_stat, 2),
            "wilcoxon_pvalue": round(w_p, 6),
            "n_circuits": len(diffs),
        }

    # --- Factorial Effects on Lap Time ---
    # Only evaluated on circuits where ALL 8 conditions completed!
    # Contrast direction: negative change means faster (- steps)
    valid_lt_rows = ~np.isnan(lap_time_matrix).any(axis=1)
    n_valid_circuits = int(np.sum(valid_lt_rows))

    lt_effects = {}
    if n_valid_circuits >= 2:
        sub_lt = lap_time_matrix[valid_lt_rows, :]
        dr_lt_diffs = np.dot(sub_lt, X_DR) / 4.0
        look_lt_diffs = np.dot(sub_lt, X_LOOK) / 4.0
        steer_lt_diffs = np.dot(sub_lt, X_STEER) / 4.0
        dr_look_lt = np.dot(sub_lt, X_DR_LOOK) / 4.0
        dr_steer_lt = np.dot(sub_lt, X_DR_STEER) / 4.0
        look_steer_lt = np.dot(sub_lt, X_LOOK_STEER) / 4.0
        three_way_lt = np.dot(sub_lt, X_3WAY) / 4.0

        for name, diffs in [
            ("DR", dr_lt_diffs),
            ("Lookahead", look_lt_diffs),
            ("SteerPenalty", steer_lt_diffs),
            ("DR×Lookahead", dr_look_lt),
            ("DR×SteerPenalty", dr_steer_lt),
            ("Lookahead×SteerPenalty", look_steer_lt),
            ("DR×Lookahead×SteerPenalty", three_way_lt),
        ]:
            mean_eff, ci_lo, ci_hi = bootstrap_ci_paired(diffs)
            std_eff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            se_eff = std_eff / math.sqrt(len(diffs))
            t_stat = mean_eff / se_eff if se_eff > 1e-9 else 0.0
            t_p = student_t_pvalue(t_stat, len(diffs) - 1) if se_eff > 1e-9 else (1.0 if abs(mean_eff) < 1e-9 else 0.0)
            w_stat, w_p = exact_wilcoxon_signed_rank(diffs)

            lt_effects[name] = {
                "mean_effect": round(mean_eff, 2),
                "ci_95_lo": round(ci_lo, 2),
                "ci_95_hi": round(ci_hi, 2),
                "std_error": round(se_eff, 2),
                "t_statistic": round(t_stat, 4) if not math.isnan(t_stat) else None,
                "t_pvalue": round(t_p, 6) if not math.isnan(t_p) else None,
                "wilcoxon_stat": round(w_stat, 2),
                "wilcoxon_pvalue": round(w_p, 6),
                "n_circuits": n_valid_circuits,
            }
    else:
        # Lap time contrasts undefined due to non-completion across conditions
        for name in ["DR", "Lookahead", "SteerPenalty", "DR×Lookahead", "DR×SteerPenalty", "Lookahead×SteerPenalty", "DR×Lookahead×SteerPenalty"]:
            lt_effects[name] = {
                "mean_effect": None,
                "ci_95_lo": None,
                "ci_95_hi": None,
                "std_error": None,
                "t_statistic": None,
                "t_pvalue": None,
                "wilcoxon_stat": None,
                "wilcoxon_pvalue": None,
                "n_circuits": n_valid_circuits,
                "note": "Insufficient paired completed circuits across all 8 configurations (zero imputation rule)",
            }

    slice_result = {
        "benchmark": slice_info["benchmark"],
        "difficulty_type": slice_info["difficulty_type"],
        "difficulty_value": slice_info["difficulty_value"],
        "label": slice_info["label"],
        "n_circuits_total": n_seeds,
        "n_circuits_lap_time_paired": n_valid_circuits,
        "condition_stats": cond_stats,
        "factorial_effects": {
            "offtrack_failure_rate": crash_effects,
            "lap_time": lt_effects,
        },
    }
    return cond_stats, slice_result


def export_csv_matrix(all_slice_results: List[Dict[str, Any]], out_path: str):
    """Write the complete ablation matrix CSV file."""
    fieldnames = [
        "benchmark",
        "difficulty_type",
        "difficulty_value",
        "benchmark_label",
        "condition",
        "dr",
        "lookahead",
        "steer_penalty",
        "n_total",
        "n_completed",
        "completion_rate",
        "n_crashed",
        "offtrack_failure_rate",
        "n_timeout",
        "timeout_rate",
        "n_stationary",
        "stationary_rate",
        "mean_lap_time",
        "median_lap_time",
        "std_lap_time",
        "min_lap_time",
        "max_lap_time",
        "mean_progress",
        "median_progress",
    ]

    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_slice_results:
            bench = s["benchmark"]
            dtype = s["difficulty_type"]
            dval = s["difficulty_value"]
            label = s["label"]
            for c in "ABCDEFGH":
                st = s["condition_stats"][c]
                row = {
                    "benchmark": bench,
                    "difficulty_type": dtype,
                    "difficulty_value": dval,
                    "benchmark_label": label,
                    "condition": c,
                    "dr": COND_MAP[c]["dr"],
                    "lookahead": COND_MAP[c]["lookahead"],
                    "steer_penalty": COND_MAP[c]["steer_penalty"],
                    "n_total": st["n_total"],
                    "n_completed": st["n_completed"],
                    "completion_rate": st["completion_rate"],
                    "n_crashed": st["n_crashed"],
                    "offtrack_failure_rate": st["offtrack_failure_rate"],
                    "n_timeout": st["n_timeout"],
                    "timeout_rate": st["timeout_rate"],
                    "n_stationary": st["n_stationary"],
                    "stationary_rate": st["stationary_rate"],
                    "mean_lap_time": st["mean_lap_time"],
                    "median_lap_time": st["median_lap_time"],
                    "std_lap_time": st["std_lap_time"],
                    "min_lap_time": st["min_lap_time"],
                    "max_lap_time": st["max_lap_time"],
                    "mean_progress": st["mean_progress"],
                    "median_progress": st["median_progress"],
                }
                writer.writerow(row)


def plot_main_effects_laptime(all_slice_results: List[Dict[str, Any]], out_path: str):
    """Plot main effects on lap time across corridor widths."""
    # Filter slices where lap time is defined
    valid_slices = [s for s in all_slice_results if s["n_circuits_lap_time_paired"] >= 2 and s["benchmark"] in ("standard_heldout", "narrow_corridor")]
    
    # Sort widths descending (70 down to 20)
    # Deduplicate: if standard_heldout (70) and narrow_corridor (70) exist, keep distinct or narrow
    slice_labels = [f"W={s['difficulty_value']}" for s in valid_slices]
    # Unique order
    labels = []
    dr_effects, look_effects, steer_effects = [], [], []
    dr_err_lo, dr_err_hi = [], []
    look_err_lo, look_err_hi = [], []
    steer_err_lo, steer_err_hi = [], []

    for s in valid_slices:
        name = f"W={s['difficulty_value']}" if s['benchmark'] == 'narrow_corridor' else "Nominal (W=70)"
        labels.append(name)
        eff = s["factorial_effects"]["lap_time"]
        
        # DR
        dr_m = eff["DR"]["mean_effect"]
        dr_effects.append(dr_m)
        dr_err_lo.append(dr_m - eff["DR"]["ci_95_lo"])
        dr_err_hi.append(eff["DR"]["ci_95_hi"] - dr_m)

        # Look
        look_m = eff["Lookahead"]["mean_effect"]
        look_effects.append(look_m)
        look_err_lo.append(look_m - eff["Lookahead"]["ci_95_lo"])
        look_err_hi.append(eff["Lookahead"]["ci_95_hi"] - look_m)

        # Steer
        steer_m = eff["SteerPenalty"]["mean_effect"]
        steer_effects.append(steer_m)
        steer_err_lo.append(steer_m - eff["SteerPenalty"]["ci_95_lo"])
        steer_err_hi.append(eff["SteerPenalty"]["ci_95_hi"] - steer_m)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    rects1 = ax.bar(x - width, dr_effects, width, yerr=[dr_err_lo, dr_err_hi], capsize=3, label="DR (On vs Off)", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x, look_effects, width, yerr=[look_err_lo, look_err_hi], capsize=3, label="Lookahead (300 vs 200 px)", color="#d95f02", alpha=0.9)
    rects3 = ax.bar(x + width, steer_effects, width, yerr=[steer_err_lo, steer_err_hi], capsize=3, label="Steering Penalty (On vs Off)", color="#2ca02c", alpha=0.9)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Main Effect on Lap Time (simulation steps)\n[Negative = Faster]", fontsize=11, fontweight="bold")
    ax.set_title("Factorial Main Effects on Lap Time Across Track Widths (Completed Laps Only)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_main_effects_offtrack(all_slice_results: List[Dict[str, Any]], out_path: str):
    """Plot main effects on off-track failure rate across benchmark difficulties."""
    # Slices where at least one condition has non-zero failure rate
    active_slices = [s for s in all_slice_results if (s["benchmark"] == "narrow_corridor" and s["difficulty_value"] <= 25) or s["benchmark"] == "hairpin"]

    labels = []
    dr_effects, look_effects, steer_effects = [], [], []
    dr_err_lo, dr_err_hi = [], []
    look_err_lo, look_err_hi = [], []
    steer_err_lo, steer_err_hi = [], []

    for s in active_slices:
        if s["benchmark"] == "narrow_corridor":
            labels.append(f"Narrow W={s['difficulty_value']}")
        else:
            labels.append(f"Hairpin f={s['difficulty_value']}")

        eff = s["factorial_effects"]["offtrack_failure_rate"]

        # DR
        dr_m = eff["DR"]["percentage_points"]
        dr_effects.append(dr_m)
        dr_err_lo.append(dr_m - eff["DR"]["ci_95_lo"] * 100.0)
        dr_err_hi.append(eff["DR"]["ci_95_hi"] * 100.0 - dr_m)

        # Look
        look_m = eff["Lookahead"]["percentage_points"]
        look_effects.append(look_m)
        look_err_lo.append(look_m - eff["Lookahead"]["ci_95_lo"] * 100.0)
        look_err_hi.append(eff["Lookahead"]["ci_95_hi"] * 100.0 - look_m)

        # Steer
        steer_m = eff["SteerPenalty"]["percentage_points"]
        steer_effects.append(steer_m)
        steer_err_lo.append(steer_m - eff["SteerPenalty"]["ci_95_lo"] * 100.0)
        steer_err_hi.append(eff["SteerPenalty"]["ci_95_hi"] * 100.0 - steer_m)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(x - width, dr_effects, width, yerr=[dr_err_lo, dr_err_hi], capsize=3, label="DR (On vs Off)", color="#2b5c8f", alpha=0.9)
    ax.bar(x, look_effects, width, yerr=[look_err_lo, look_err_hi], capsize=3, label="Lookahead (300 vs 200 px)", color="#d95f02", alpha=0.9)
    ax.bar(x + width, steer_effects, width, yerr=[steer_err_lo, steer_err_hi], capsize=3, label="Steering Penalty (On vs Off)", color="#2ca02c", alpha=0.9)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Main Effect on Off-Track Failure Rate (percentage points)\n[Negative = Fewer Failures]", fontsize=11, fontweight="bold")
    ax.set_title("Factorial Main Effects on Off-Track Failure Rate Across Stress Benchmarks", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_corridor_widths(all_slice_results: List[Dict[str, Any]], out_path: str):
    """Plot lap time and completion rate across corridor widths."""
    narrow_slices = [s for s in all_slice_results if s["benchmark"] == "narrow_corridor"]
    narrow_slices.sort(key=lambda s: s["difficulty_value"], reverse=True)

    widths = [s["difficulty_value"] for s in narrow_slices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Condition styles: DR on (solid blues/greens), DR off (dashed oranges/reds)
    colors = {
        "A": "#1f77b4", "B": "#aec7e8", "C": "#2ca02c", "D": "#98df8a",
        "E": "#d62728", "F": "#ff9896", "G": "#ff7f0e", "H": "#8c564b"
    }
    styles = {
        "A": "-", "B": "--", "C": "-.", "D": ":",
        "E": "-", "F": "--", "G": "-.", "H": ":"
    }

    # Left: Lap Times
    for c in "ABCDEFGH":
        lts = [s["condition_stats"][c]["mean_lap_time"] for s in narrow_slices]
        valid_w = [w for w, lt in zip(widths, lts) if lt is not None]
        valid_lt = [lt for lt in lts if lt is not None]
        label = f"{c}: {COND_MAP[c]['dr'] and 'DR' or 'noDR'}, L{COND_MAP[c]['lookahead']}, S{COND_MAP[c]['steer_penalty']}"
        ax1.plot(valid_w, valid_lt, marker="o", label=label, color=colors[c], linestyle=styles[c], linewidth=1.8, markersize=5)

    ax1.set_xlabel("Corridor Width (px)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Lap Time (steps) [Completed Laps Only]", fontsize=11, fontweight="bold")
    ax1.set_title("Lap Time Across Narrowing Corridors", fontsize=12, fontweight="bold")
    ax1.invert_xaxis()  # 70 down to 20
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8, loc="upper right")

    # Right: Off-Track Failure Rate + Timeout Rate
    for c in ["A", "D", "E", "H"]:  # Highlight representative subset for clarity
        crashes = [s["condition_stats"][c]["offtrack_failure_rate"] * 100.0 for s in narrow_slices]
        label = f"{c} Crash Rate"
        ax2.plot(widths, crashes, marker="s", label=label, color=colors[c], linestyle="-", linewidth=2.0)

    # Add Run H timeout rate to show dual failure mode
    h_timeouts = [s["condition_stats"]["H"]["timeout_rate"] * 100.0 for s in narrow_slices]
    ax2.plot(widths, h_timeouts, marker="^", label="Run H Timeout Rate", color="#8c564b", linestyle="--", linewidth=1.8)

    ax2.set_xlabel("Corridor Width (px)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Failure Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Off-Track Crash & Timeout Rates at Extreme Narrowing", fontsize=12, fontweight="bold")
    ax2.invert_xaxis()
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_hairpin_difficulties(all_slice_results: List[Dict[str, Any]], out_path: str):
    """Plot outcomes across hairpin difficulties exposing survivor and timeout bias."""
    hairpin_slices = [s for s in all_slice_results if s["benchmark"] == "hairpin"]
    hairpin_slices.sort(key=lambda s: s["difficulty_value"], reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300, sharey=True)

    factors = [s["difficulty_value"] for s in hairpin_slices]
    conditions = list("ABCDEFGH")
    x = np.arange(len(conditions))

    for idx, (s, factor) in enumerate(zip(hairpin_slices, factors)):
        ax = axes[idx]
        completed = [s["condition_stats"][c]["completion_rate"] * 100.0 for c in conditions]
        crashed = [s["condition_stats"][c]["offtrack_failure_rate"] * 100.0 for c in conditions]
        timeout = [s["condition_stats"][c]["timeout_rate"] * 100.0 for c in conditions]

        # Stacked bar plot
        p1 = ax.bar(x, completed, color="#2ca02c", label="Completed", width=0.6, alpha=0.85)
        p2 = ax.bar(x, crashed, bottom=completed, color="#d62728", label="Off-Track Crash", width=0.6, alpha=0.85)
        # bottom for timeout is completed + crashed
        bottom_to = [c + cr for c, cr in zip(completed, crashed)]
        p3 = ax.bar(x, timeout, bottom=bottom_to, color="#ff7f0e", label="Timeout / Stall", width=0.6, alpha=0.85)

        ax.set_title(f"Hairpin Factor f = {factor:.2f}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=10)
        ax.set_xlabel("Condition", fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Episode Outcome Distribution (%)", fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("Outcome Breakdown Across Hairpin Curves: Exposing Crash vs Timeout Censoring", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_effects_confidence_intervals(all_slice_results: List[Dict[str, Any]], out_path: str):
    """Forest plot of main effects with 95% bootstrap confidence intervals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. Lap Time Main Effects (W=70, W=40, W=36, W=32, W=28, W=25)
    lt_slices = [s for s in all_slice_results if s["benchmark"] == "narrow_corridor" and s["difficulty_value"] >= 25]
    lt_slices.sort(key=lambda s: s["difficulty_value"], reverse=True)

    y_ticks_lt = []
    y_labels_lt = []
    y_idx = 0

    for s in lt_slices:
        w_lbl = f"W={s['difficulty_value']} px"
        eff = s["factorial_effects"]["lap_time"]
        for factor, color in [("DR", "#2b5c8f"), ("Lookahead", "#d95f02"), ("SteerPenalty", "#2ca02c")]:
            m = eff[factor]["mean_effect"]
            lo = eff[factor]["ci_95_lo"]
            hi = eff[factor]["ci_95_hi"]
            ax1.errorbar(m, y_idx, xerr=[[m - lo], [hi - m]], fmt="o", color=color, ecolor=color, elinewidth=1.8, capsize=4, markersize=5)
            y_ticks_lt.append(y_idx)
            y_labels_lt.append(f"{w_lbl} - {factor}")
            y_idx += 1
        y_idx += 0.5

    ax1.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_yticks(y_ticks_lt)
    ax1.set_yticklabels(y_labels_lt, fontsize=8)
    ax1.set_xlabel("Lap Time Main Effect (steps) [Negative = Faster]", fontsize=10, fontweight="bold")
    ax1.set_title("95% Bootstrap CIs: Lap Time Main Effects", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Off-track Failure Rate Main Effects (W=20, Hairpin 0.50, 0.35, 0.20)
    crash_slices = [s for s in all_slice_results if (s["benchmark"] == "narrow_corridor" and s["difficulty_value"] == 20) or s["benchmark"] == "hairpin"]
    crash_slices.sort(key=lambda s: (s["benchmark"] != "narrow_corridor", -s["difficulty_value"]))

    y_ticks_cr = []
    y_labels_cr = []
    y_idx_cr = 0

    for s in crash_slices:
        slice_lbl = f"W=20 px" if s["benchmark"] == "narrow_corridor" else f"Hairpin f={s['difficulty_value']:.2f}"
        eff = s["factorial_effects"]["offtrack_failure_rate"]
        for factor, color in [("DR", "#2b5c8f"), ("Lookahead", "#d95f02"), ("SteerPenalty", "#2ca02c")]:
            m = eff[factor]["percentage_points"]
            lo = eff[factor]["ci_95_lo"] * 100.0
            hi = eff[factor]["ci_95_hi"] * 100.0
            ax2.errorbar(m, y_idx_cr, xerr=[[m - lo], [hi - m]], fmt="s", color=color, ecolor=color, elinewidth=1.8, capsize=4, markersize=5)
            y_ticks_cr.append(y_idx_cr)
            y_labels_cr.append(f"{slice_lbl} - {factor}")
            y_idx_cr += 1
        y_idx_cr += 0.5

    ax2.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_yticks(y_ticks_cr)
    ax2.set_yticklabels(y_labels_cr, fontsize=8)
    ax2.set_xlabel("Off-Track Failure Rate Main Effect (%-pts) [Negative = Fewer Failures]", fontsize=10, fontweight="bold")
    ax2.set_title("95% Bootstrap CIs: Off-Track Crash Effects", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    raw_csv = os.path.join(RESULTS_DIR, "ablation_episodes.csv")
    print(f"Loading raw episodes from {raw_csv}...")
    episodes = load_episodes(raw_csv)
    print(f"Loaded {len(episodes)} episodes.")

    # Sanity checks
    conditions_present = set(e["condition"] for e in episodes)
    assert conditions_present == set("ABCDEFGH"), f"Missing conditions: {set('ABCDEFGH') - conditions_present}"
    assert len(episodes) == 880, f"Expected 880 episodes, found {len(episodes)}"

    all_slice_results = []
    condition_stats_by_slice = {}

    for s_info in BENCHMARK_SLICES:
        slice_eps = filter_slice(episodes, s_info["benchmark"], s_info["difficulty_type"], s_info["difficulty_value"])
        cond_stats, s_res = analyze_slice(slice_eps, s_info)
        all_slice_results.append(s_res)
        condition_stats_by_slice[s_info["label"]] = cond_stats

    # Export CSV Matrix
    out_csv = os.path.join(RESULTS_DIR, "ablation_matrix.csv")
    print(f"Exporting ablation matrix CSV to {out_csv}...")
    export_csv_matrix(all_slice_results, out_csv)

    # Export Summary JSON
    summary_json_path = os.path.join(RESULTS_DIR, "ablation_matrix_summary.json")
    print(f"Exporting ablation matrix summary JSON to {summary_json_path}...")
    summary_payload = {
        "metadata": {
            "n_configurations": 8,
            "conditions": [
                {"label": c[0], "dr": c[1], "lookahead": c[2], "steer_penalty": c[3]}
                for c in CONDITIONS
            ],
            "evaluation_seeds": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
            "training_seed": 42,
            "metric_definitions": {
                "lap_time": "Discrete simulation steps for completed episodes only. Incomplete episodes are excluded (no artificial penalty).",
                "offtrack_failure_rate": "Fraction of episodes terminating due to track boundary crossing (crashed==True) out of total episodes.",
                "timeout_rate": "Fraction of episodes reaching max_steps (700) without crashing or completing.",
                "stationary_rate": "Fraction of episodes timing out with max_speed < 0.05 or abs(lap_progress) < 0.005.",
                "completion_rate": "Fraction of episodes reaching 100% lap progress.",
            },
            "contrast_directions": {
                "lap_time": "Negative effect indicates faster performance (fewer simulation steps).",
                "offtrack_failure_rate": "Negative effect indicates reduction in failures (fewer off-track crashes).",
            },
            "statistical_methods": {
                "paired_circuits": "Comparisons are matched on identical circuit seeds (1001..1010).",
                "confidence_intervals": "95% paired bootstrap percentile interval with B=10,000 resamples (seed 42).",
                "hypothesis_tests": "Paired Student's t-test and exact Wilcoxon signed-rank permutation test.",
                "critical_limitation": "Single training seed (42) per condition. Intervals reflect test-circuit variation, not RL training seed stochasticity.",
            },
        },
        "benchmark_slices": all_slice_results,
    }
    with open(summary_json_path, mode="w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Export Effects JSON
    effects_json_path = os.path.join(RESULTS_DIR, "ablation_matrix_effects.json")
    print(f"Exporting ablation matrix factorial effects JSON to {effects_json_path}...")
    effects_payload = {
        "description": "2×2×2 factorial effects, interactions, and paired bootstrap 95% confidence intervals across benchmark slices.",
        "slices": {
            s["label"]: {
                "benchmark": s["benchmark"],
                "difficulty_type": s["difficulty_type"],
                "difficulty_value": s["difficulty_value"],
                "n_circuits_total": s["n_circuits_total"],
                "n_circuits_lap_time_paired": s["n_circuits_lap_time_paired"],
                "lap_time_effects": s["factorial_effects"]["lap_time"],
                "offtrack_failure_effects": s["factorial_effects"]["offtrack_failure_rate"],
            }
            for s in all_slice_results
        },
    }
    with open(effects_json_path, mode="w", encoding="utf-8") as f:
        json.dump(effects_payload, f, indent=2)

    # Generate Figures
    os.makedirs(FIGS_DIR, exist_ok=True)

    fig1 = os.path.join(FIGS_DIR, "matrix_main_effects_laptime.png")
    print(f"Generating {fig1}...")
    plot_main_effects_laptime(all_slice_results, fig1)

    fig2 = os.path.join(FIGS_DIR, "matrix_main_effects_offtrack.png")
    print(f"Generating {fig2}...")
    plot_main_effects_offtrack(all_slice_results, fig2)

    fig3 = os.path.join(FIGS_DIR, "matrix_corridor_widths.png")
    print(f"Generating {fig3}...")
    plot_corridor_widths(all_slice_results, fig3)

    fig4 = os.path.join(FIGS_DIR, "matrix_hairpin_difficulties.png")
    print(f"Generating {fig4}...")
    plot_hairpin_difficulties(all_slice_results, fig4)

    fig5 = os.path.join(FIGS_DIR, "matrix_effects_ci.png")
    print(f"Generating {fig5}...")
    plot_effects_confidence_intervals(all_slice_results, fig5)

    print("All deliverables generated successfully!")


if __name__ == "__main__":
    main()
