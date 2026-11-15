"""Consolidated evaluation of Stage A screening models across unseen hairpins."""
import json
import csv
import os
import sys
from typing import Dict, Any, List

import numpy as np

from analysis.eval_hairpins import (
    inspect_and_load_actor,
    run_episode,
    compute_aggregate,
    DEFAULT_FACTORS,
    DEFAULT_SEEDS,
    DEFAULT_MAX_STEPS,
)

SCREENING_MODELS = [
    {
        "name": "5ray_d200 (Screening Control)",
        "ckpt": "runs/screen_5ray_d200/best_actor.pth",
    },
    {
        "name": "9ray_d100 (Short Lookahead)",
        "ckpt": "runs/screen_9ray_d100/best_actor.pth",
    },
    {
        "name": "9ray_d300 (Deep Lookahead)",
        "ckpt": "runs/screen_9ray_d300/best_actor.pth",
    },
    {
        "name": "7ray_fwd_d200 (Fwd-Focused 7-ray)",
        "ckpt": "runs/screen_7ray_fwd_d200/best_actor.pth",
    },
    {
        "name": "9ray_fwd_d200 (Fwd-Dense 9-ray)",
        "ckpt": "runs/screen_9ray_fwd_d200/best_actor.pth",
    },
]


def main():
    factors = DEFAULT_FACTORS
    seeds = DEFAULT_SEEDS
    max_steps = DEFAULT_MAX_STEPS

    all_results: List[Dict[str, Any]] = []
    summary_by_model: Dict[str, Dict[str, Any]] = {}

    print("=" * 100)
    print("STAGE A SCREENING BENCHMARK: EVALUATING CANDIDATES ON UNSEEN HAIRPINS")
    print("=" * 100)

    for item in SCREENING_MODELS:
        name = item["name"]
        ckpt = item["ckpt"]
        print(f"\nLoading {name} from {ckpt}...")
        actor, obs_dim, ray_angles, ray_max_dist = inspect_and_load_actor(ckpt)
        print(f"  -> obs_dim={obs_dim}, rays={len(ray_angles)} {ray_angles}, max_dist={ray_max_dist}")

        summary_by_model[name] = {}
        for factor in factors:
            factor_episodes = []
            for seed in seeds:
                res = run_episode(
                    actor=actor,
                    seed=seed,
                    hairpin_factor=factor,
                    ray_angles=ray_angles,
                    max_dist=ray_max_dist,
                    max_steps=max_steps,
                )
                res["model_name"] = name
                res["checkpoint"] = ckpt
                res["obs_dim"] = obs_dim
                res["ray_count"] = len(ray_angles)
                res["obs_ray_max_dist"] = ray_max_dist
                all_results.append(res)
                factor_episodes.append(res)

            summary_by_model[name][str(factor)] = compute_aggregate(factor_episodes)

    # Print summary table
    print("\n" + "=" * 105)
    print("STAGE A SCREENING AGGREGATE SUMMARY (Preliminary 25k Models)")
    print("=" * 105)
    header = (
        f"{'Model Configuration':<32} | {'Factor':<6} | {'Completions':<11} | {'Crashes':<7} | "
        f"{'Timeouts (Stat)':<15} | {'Compl. Rate':<11} | {'Mean Prog':<10} | {'Median Prog':<11}"
    )
    print(header)
    print("-" * 105)

    for name in summary_by_model:
        for factor in factors:
            agg = summary_by_model[name][str(factor)]
            compl_str = f"{agg['completed']}/{agg['total']}"
            crashes_str = str(agg['crashes'])
            to_str = f"{agg['timeouts']} ({agg.get('stationary', 0)})"
            rate_str = f"{agg['completion_rate'] * 100:5.1f}%"
            mean_p = f"{agg['avg_progress']:.4f}"
            med_p = f"{agg['median_progress']:.4f}"
            print(
                f"{name:<32} | {factor:<6.2f} | {compl_str:<11} | {crashes_str:<7} | "
                f"{to_str:<15} | {rate_str:<11} | {mean_p:<10} | {med_p:<11}"
            )
        print("-" * 105)

    # Save to CSV and JSON
    os.makedirs("analysis/results", exist_ok=True)
    csv_path = "analysis/results/screening_comparison.csv"
    json_path = "analysis/results/screening_comparison.json"

    fieldnames = [
        "model_name", "checkpoint", "obs_dim", "ray_count", "obs_ray_max_dist",
        "hairpin_factor", "seed", "outcome", "completed", "crashed", "timeout",
        "stationary", "max_speed", "steps", "lap_time", "lap_progress"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"\n[Saved] Screening CSV results -> {csv_path}")

    with open(json_path, "w") as f:
        json.dump({"summary_by_model": summary_by_model}, f, indent=2)
    print(f"[Saved] Screening JSON results -> {json_path}")


if __name__ == "__main__":
    main()
