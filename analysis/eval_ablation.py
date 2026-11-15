"""Evaluate all ablation study conditions on held-out test suites.

Evaluates each of the 8 ablation conditions on:
1. Standard Held-Out Tracks: TEST_SEEDS (1001..1010) at width=70
2. Narrow-Corridor Generalization: widths [70, 40, 36, 32, 28, 25, 20] on TEST_SEEDS
3. Hairpin Generalization: pinch factors [0.50, 0.35, 0.20] on TEST_SEEDS

Outputs:
  - analysis/results/ablation_episodes.csv
  - analysis/results/ablation_summary.json
  - analysis/results/ablation_factorial_effects.json
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch

from env.environment import RacingEnv
from sac.actor import SACActor
from sac.track_pools import POOL_CONFIG, TRAIN_SEEDS, TEST_SEEDS

# Ablation conditions: (label, dr, lookahead, steer_penalty)
CONDITIONS = [
    ("A", True,  200, 0.01),
    ("B", True,  200, 0.00),
    ("C", True,  300, 0.01),
    ("D", True,  300, 0.00),
    ("E", False, 200, 0.01),
    ("F", False, 200, 0.00),
    ("G", False, 300, 0.01),
    ("H", False, 300, 0.00),
]

WIDTH_LEVELS = [70, 40, 36, 32, 28, 25, 20]
HAIRPIN_FACTORS = [0.50, 0.35, 0.20]


def run_name_for_condition(label: str, dr_on: bool, lookahead: int, steer_pen: float) -> str:
    dr_label = "DR" if dr_on else "noDR"
    look_label = f"look{lookahead}"
    steer_label = "steerOn" if steer_pen > 0 else "steerOff"
    return f"ablation_{label}_{dr_label}_{look_label}_{steer_label}"


def inspect_and_load_actor(checkpoint_path: str) -> Tuple[SACActor, int, Tuple[float, ...], float]:
    """Inspect checkpoint and return (actor, obs_dim, ray_angles, ray_max_dist)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    obs_dim = state_dict["network.0.weight"].shape[1]

    ray_angles = (-90.0, -45.0, 0.0, 45.0, 90.0) if obs_dim == 9 else None
    ray_max_dist = 200.0

    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    cfg_path = os.path.join(ckpt_dir, "run_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            if "obs_ray_angles" in cfg:
                ray_angles = tuple(float(a) for a in cfg["obs_ray_angles"])
            if "obs_ray_max_dist" in cfg:
                ray_max_dist = float(cfg["obs_ray_max_dist"])
        except Exception:
            pass

    if ray_angles is None:
        raise ValueError(f"Could not determine ray angles for obs_dim={obs_dim}")

    actor = SACActor(obs_dim=obs_dim, action_dim=2)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor, obs_dim, ray_angles, ray_max_dist


def run_single_episode(
    actor: SACActor,
    seed: int,
    track_kwargs: Dict[str, Any],
    ray_angles: Tuple[float, ...],
    max_dist: float = 200.0,
    max_steps: int = 700,
) -> Dict[str, Any]:
    """Execute a single deterministic evaluation rollout.

    NOTE: All evaluations use the same evaluation environment settings (default steer_penalty_coeff=0.01)
    so that metrics are comparable across conditions. The steer_penalty_coeff difference only affects training.
    """
    env = RacingEnv(
        max_steps=max_steps,
        verbose=False,
        track_kwargs=track_kwargs,
        obs_ray_angles=ray_angles,
        obs_ray_max_dist=max_dist,
    )

    obs, _ = env.reset(options={"track_seed": seed})
    completed = False
    crashed = False
    steps = 0
    final_info: Dict[str, Any] = {}
    max_speed = 0.0

    for step in range(max_steps):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean, _ = actor(obs_tensor)
            action = torch.tanh(mean).numpy()[0]

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        final_info = info

        spd = float(env.car.velocity)
        if spd > max_speed:
            max_speed = spd

        if terminated or truncated:
            completed = bool(info.get("lap_completed", False))
            crashed = bool(info.get("crashed", False))
            break

    lap_progress = float(final_info.get("lap_progress", 0.0))
    stationary = bool(final_info.get("stationary", False))

    if completed:
        outcome = "completed"
    elif crashed:
        outcome = "crashed"
    elif stationary:
        outcome = "stationary"
    elif steps >= max_steps:
        outcome = "timeout"
    else:
        outcome = "terminated"

    return {
        "seed": seed,
        "outcome": outcome,
        "completed": completed,
        "crashed": crashed,
        "timeout": outcome == "timeout",
        "stationary": stationary,
        "steps": steps,
        "lap_time": steps if completed else None,
        "lap_progress": round(lap_progress, 4),
        "max_speed": round(max_speed, 4),
    }


def compute_aggregate(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics, reporting completed lap times separately from incomplete progress."""
    total = len(episodes)
    if total == 0:
        return {
            "total": 0, "completed": 0, "crashes": 0, "timeouts": 0, "stationary": 0,
            "completion_rate": 0.0, "crash_rate": 0.0, "timeout_rate": 0.0,
            "avg_lap_time": None, "std_lap_time": None,
            "min_lap_time": None, "max_lap_time": None,
            "avg_progress": 0.0, "median_progress": 0.0,
            "avg_incomplete_progress": None,
        }

    completed = sum(1 for e in episodes if e["completed"])
    crashes = sum(1 for e in episodes if e["crashed"])
    timeouts = sum(1 for e in episodes if e["timeout"])
    stationary = sum(1 for e in episodes if e["stationary"])

    # Completed-lap times in discrete simulation steps (reported separately from incomplete episodes)
    lap_times = [e["lap_time"] for e in episodes if e["completed"] and e["lap_time"] is not None]
    avg_lap_time = round(float(np.mean(lap_times)), 2) if lap_times else None
    std_lap_time = round(float(np.std(lap_times)), 2) if lap_times else None
    min_lap_time = int(np.min(lap_times)) if lap_times else None
    max_lap_time = int(np.max(lap_times)) if lap_times else None

    # Overall progress across all episodes
    progresses = [e["lap_progress"] for e in episodes]
    avg_progress = round(float(np.mean(progresses)), 4)
    median_progress = round(float(np.median(progresses)), 4)

    # Incomplete episode progress reported separately
    inc_progresses = [e["lap_progress"] for e in episodes if not e["completed"]]
    avg_incomplete_progress = round(float(np.mean(inc_progresses)), 4) if inc_progresses else None

    return {
        "total": total,
        "completed": completed,
        "crashes": crashes,
        "timeouts": timeouts,
        "stationary": stationary,
        "completion_rate": round(completed / total, 4),
        "crash_rate": round(crashes / total, 4),
        "timeout_rate": round(timeouts / total, 4),
        "avg_lap_time": avg_lap_time,
        "std_lap_time": std_lap_time,
        "min_lap_time": min_lap_time,
        "max_lap_time": max_lap_time,
        "avg_progress": avg_progress,
        "median_progress": median_progress,
        "avg_incomplete_progress": avg_incomplete_progress,
    }


def compute_factorial_effects(condition_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute main effects and interactions for the 2×2×2 factorial design.

    Each factor has two levels:
      - DR: Low (0) = Off, High (1) = On
      - Lookahead: Low (0) = 200 px, High (1) = 300 px
      - SteerPenalty: Low (0) = 0.00, High (1) = 0.01

    Main effect: Mean(High) - Mean(Low).
    Two-way interaction (X × Y): ((Mean(HH) - Mean(HL)) - (Mean(LH) - Mean(LL))) / 2.
    Three-way interaction: (1/4) * sum(contrast_sign * response).
    """
    factor_levels = {
        "A": {"DR": 1, "Lookahead": 0, "SteerPenalty": 1},
        "B": {"DR": 1, "Lookahead": 0, "SteerPenalty": 0},
        "C": {"DR": 1, "Lookahead": 1, "SteerPenalty": 1},
        "D": {"DR": 1, "Lookahead": 1, "SteerPenalty": 0},
        "E": {"DR": 0, "Lookahead": 0, "SteerPenalty": 1},
        "F": {"DR": 0, "Lookahead": 0, "SteerPenalty": 0},
        "G": {"DR": 0, "Lookahead": 1, "SteerPenalty": 1},
        "H": {"DR": 0, "Lookahead": 1, "SteerPenalty": 0},
    }

    metrics = ["completion_rate", "avg_progress", "crash_rate", "avg_lap_time"]
    effects = {}

    for metric in metrics:
        responses = {}
        for label, levels in factor_levels.items():
            if label in condition_results and "standard_heldout" in condition_results[label]:
                val = condition_results[label]["standard_heldout"].get(metric)
                if val is not None:
                    responses[label] = val

        if len(responses) < 8:
            effects[metric] = {"note": f"Incomplete data ({len(responses)}/8 conditions)"}
            continue

        # Main effects: average response at high level minus average at low level
        main_effects = {}
        for factor in ["DR", "Lookahead", "SteerPenalty"]:
            high = [responses[l] for l, fl in factor_levels.items() if fl[factor] == 1 and l in responses]
            low = [responses[l] for l, fl in factor_levels.items() if fl[factor] == 0 and l in responses]
            if high and low:
                main_effects[factor] = round(float(np.mean(high) - np.mean(low)), 4)

        # Two-factor interactions
        interactions = {}
        factors = ["DR", "Lookahead", "SteerPenalty"]
        for i, f1 in enumerate(factors):
            for f2 in factors[i+1:]:
                hh = [responses[l] for l, fl in factor_levels.items()
                       if fl[f1] == 1 and fl[f2] == 1 and l in responses]
                hl = [responses[l] for l, fl in factor_levels.items()
                       if fl[f1] == 1 and fl[f2] == 0 and l in responses]
                lh = [responses[l] for l, fl in factor_levels.items()
                       if fl[f1] == 0 and fl[f2] == 1 and l in responses]
                ll = [responses[l] for l, fl in factor_levels.items()
                       if fl[f1] == 0 and fl[f2] == 0 and l in responses]

                if hh and hl and lh and ll:
                    interaction = (np.mean(hh) - np.mean(hl) - np.mean(lh) + np.mean(ll)) / 2
                    interactions[f"{f1}×{f2}"] = round(float(interaction), 4)

        # Three-factor interaction
        three_way = None
        if len(responses) == 8:
            # Contrast signs: (+1 for high, -1 for low)
            signs = {
                l: (2 * fl["DR"] - 1) * (2 * fl["Lookahead"] - 1) * (2 * fl["SteerPenalty"] - 1)
                for l, fl in factor_levels.items()
            }
            three_way = round(float(sum(signs[l] * responses[l] for l in signs) / 4), 4)

        effects[metric] = {
            "main_effects": main_effects,
            "two_factor_interactions": interactions,
            "three_factor_interaction": {"DR×Lookahead×SteerPenalty": three_way},
            "responses": responses,
        }

    return effects


def main():
    parser = argparse.ArgumentParser(description="Evaluate ablation study conditions on held-out tracks")
    parser.add_argument("--out-dir", default="analysis/results", help="Output directory")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Subset of condition labels to evaluate (default: all)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    selected = set(args.conditions) if args.conditions else {c[0] for c in CONDITIONS}

    print("=" * 90)
    print("ABLATION STUDY EVALUATION")
    print(f"Test Seeds: {TEST_SEEDS}")
    print(f"Width Levels: {WIDTH_LEVELS}")
    print(f"Hairpin Factors: {HAIRPIN_FACTORS}")
    print("=" * 90)

    base_tk = dict(POOL_CONFIG["track_kwargs"])
    all_episodes: List[Dict[str, Any]] = []
    condition_results: Dict[str, Dict[str, Any]] = {}

    for label, dr_on, lookahead, steer_pen in CONDITIONS:
        if label not in selected:
            continue

        run_name = run_name_for_condition(label, dr_on, lookahead, steer_pen)
        ckpt_path = os.path.join("runs", run_name, "best_actor.pth")

        if not os.path.exists(ckpt_path):
            print(f"\n⚠️  Condition {label} ({run_name}): checkpoint not found ({ckpt_path}). Skipping.")
            continue

        try:
            actor, obs_dim, ray_angles, ray_max_dist = inspect_and_load_actor(ckpt_path)
        except Exception as e:
            print(f"\n⚠️  Condition {label}: failed to load checkpoint: {e}")
            continue

        print(f"\n{'─' * 90}")
        print(f"CONDITION {label}: DR={'On' if dr_on else 'Off'} | "
              f"Lookahead={lookahead}px | SteerPenalty={steer_pen}")
        print(f"  Checkpoint: {ckpt_path} (obs_dim={obs_dim})")

        condition_results[label] = {}

        # SUITE 1: Standard Held-Out (width=70)
        print(f"  [1/3] Standard Held-Out Tracks (width=70)...")
        eps = []
        for s in TEST_SEEDS:
            res = run_single_episode(actor, s, base_tk, ray_angles, ray_max_dist,
                                      max_steps=POOL_CONFIG["max_steps"])
            res.update({
                "condition": label, "run_name": run_name,
                "dr": dr_on, "lookahead": lookahead, "steer_penalty": steer_pen,
                "benchmark": "standard_heldout", "width": 70, "hairpin_factor": None,
            })
            eps.append(res)
            all_episodes.append(res)
        agg = compute_aggregate(eps)
        condition_results[label]["standard_heldout"] = agg
        lt_str = f"{agg['avg_lap_time']} steps" if agg['avg_lap_time'] else "N/A"
        print(f"    Completion: {agg['completion_rate']*100:.1f}% | "
              f"Avg Lap: {lt_str} | Progress: {agg['avg_progress']:.4f}")

        # SUITE 2: Narrow Corridors
        print(f"  [2/3] Narrow Corridors ({WIDTH_LEVELS} px)...")
        for w in WIDTH_LEVELS:
            tk = dict(base_tk, width=w)
            eps = []
            for s in TEST_SEEDS:
                res = run_single_episode(actor, s, tk, ray_angles, ray_max_dist,
                                          max_steps=POOL_CONFIG["max_steps"])
                res.update({
                    "condition": label, "run_name": run_name,
                    "dr": dr_on, "lookahead": lookahead, "steer_penalty": steer_pen,
                    "benchmark": "narrow_corridor", "width": w, "hairpin_factor": None,
                })
                eps.append(res)
                all_episodes.append(res)
            agg = compute_aggregate(eps)
            condition_results[label][f"width_{w}"] = agg
            print(f"    Width={w:2d}px: {agg['completion_rate']*100:5.1f}% completion, "
                  f"{agg['crash_rate']*100:5.1f}% crash, prog={agg['avg_progress']:.4f}")

        # SUITE 3: Hairpins
        print(f"  [3/3] Hairpin Benchmark ({HAIRPIN_FACTORS})...")
        for hf in HAIRPIN_FACTORS:
            tk = dict(base_tk, hairpin=True, hairpin_factor=hf)
            eps = []
            for s in TEST_SEEDS:
                res = run_single_episode(actor, s, tk, ray_angles, ray_max_dist,
                                          max_steps=POOL_CONFIG["max_steps"])
                res.update({
                    "condition": label, "run_name": run_name,
                    "dr": dr_on, "lookahead": lookahead, "steer_penalty": steer_pen,
                    "benchmark": "hairpin", "width": 70, "hairpin_factor": hf,
                })
                eps.append(res)
                all_episodes.append(res)
            agg = compute_aggregate(eps)
            condition_results[label][f"hairpin_{hf}"] = agg
            print(f"    Hairpin f={hf:.2f}: {agg['completion_rate']*100:5.1f}% completion, "
                  f"prog={agg['avg_progress']:.4f}")

    # -------------------------------------------------------------------------
    # Factorial effects analysis
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("FACTORIAL EFFECTS ANALYSIS")
    print("=" * 90)

    factorial_effects = compute_factorial_effects(condition_results)

    for metric, eff in factorial_effects.items():
        if "note" in eff:
            print(f"\n{metric}: {eff['note']}")
            continue
        print(f"\n{metric}:")
        print(f"  Main Effects:")
        for factor, value in eff["main_effects"].items():
            direction = "↑" if value > 0 else "↓" if value < 0 else "─"
            print(f"    {factor}: {value:+.4f} {direction}")
        print(f"  Two-Factor Interactions:")
        for pair, value in eff["two_factor_interactions"].items():
            print(f"    {pair}: {value:+.4f}")
        if eff.get("three_factor_interaction"):
            for triple, value in eff["three_factor_interaction"].items():
                if value is not None:
                    print(f"  Three-Factor Interaction: {triple}: {value:+.4f}")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    csv_out = os.path.join(args.out_dir, "ablation_episodes.csv")
    json_out = os.path.join(args.out_dir, "ablation_summary.json")
    effects_out = os.path.join(args.out_dir, "ablation_factorial_effects.json")

    fieldnames = [
        "condition", "run_name", "dr", "lookahead", "steer_penalty",
        "benchmark", "width", "hairpin_factor", "seed",
        "outcome", "completed", "crashed", "timeout", "stationary",
        "steps", "lap_time", "lap_progress", "max_speed",
    ]

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_episodes)

    with open(json_out, "w") as f:
        json.dump({
            "metadata": {
                "conditions": [
                    {"label": l, "dr": d, "lookahead": lo, "steer_penalty": sp}
                    for l, d, lo, sp in CONDITIONS if l in selected
                ],
                "test_seeds": TEST_SEEDS,
                "width_levels": WIDTH_LEVELS,
                "hairpin_factors": HAIRPIN_FACTORS,
            },
            "condition_results": condition_results,
        }, f, indent=2)

    with open(effects_out, "w") as f:
        json.dump(factorial_effects, f, indent=2)

    print(f"\nSaved:")
    print(f"  - Episodes CSV: {csv_out} ({len(all_episodes)} rows)")
    print(f"  - Summary JSON: {json_out}")
    print(f"  - Factorial Effects: {effects_out}")

    # Print summary table
    print(f"\n{'=' * 110}")
    print(f"{'Condition':<10} | {'DR':<5} | {'Look':<6} | {'Steer':<6} | "
          f"{'Std Comp':<9} | {'Std Lap':<9} | {'W=40':<7} | {'W=25':<7} | "
          f"{'H=0.50':<7} | {'H=0.35':<7} | {'H=0.20':<7}")
    print(f"{'-' * 110}")
    for label, dr_on, lookahead, steer_pen in CONDITIONS:
        if label not in condition_results:
            continue
        cr = condition_results[label]
        std = cr.get("standard_heldout", {})
        c_rate = f"{std.get('completion_rate', 0)*100:5.1f}%" if "completion_rate" in std else "N/A"
        lt = f"{std.get('avg_lap_time'):.0f}" if std.get('avg_lap_time') else "N/A"
        w40 = f"{cr.get('width_40', {}).get('completion_rate', 0)*100:4.0f}%" if 'width_40' in cr else "-"
        w25 = f"{cr.get('width_25', {}).get('completion_rate', 0)*100:4.0f}%" if 'width_25' in cr else "-"
        h50 = f"{cr.get('hairpin_0.5', {}).get('completion_rate', 0)*100:4.0f}%" if 'hairpin_0.5' in cr else "-"
        h35 = f"{cr.get('hairpin_0.35', {}).get('completion_rate', 0)*100:4.0f}%" if 'hairpin_0.35' in cr else "-"
        h20 = f"{cr.get('hairpin_0.2', {}).get('completion_rate', 0)*100:4.0f}%" if 'hairpin_0.2' in cr else "-"
        print(f"{label:<10} | {'On' if dr_on else 'Off':<5} | {lookahead:<6} | {steer_pen:<6} | "
              f"{c_rate:<9} | {lt:<9} | {w40:<7} | {w25:<7} | {h50:<7} | {h35:<7} | {h20:<7}")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
