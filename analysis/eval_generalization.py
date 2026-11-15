"""Comprehensive Held-Out Generalization Evaluation for RL Racing Agent.

Evaluates Baseline (Single-Track v2) vs DR Agent (dr_multitrack_s42_full) on:
1. Track Pool Benchmark:
   - TRAIN_SEEDS (1..20) vs TEST_SEEDS (1001..1010) under standard medium track geometry (width 70)
2. Width Sweep Benchmark:
   - Widths [70, 40, 36, 32, 28, 25, 20] on held-out test seeds (1001..1010) and historical seeds (1..10)
3. Hairpin Benchmark:
   - Pinch factors [0.5, 0.35, 0.2] on held-out test seeds (1001..1010)

Quantifies:
- Lap completion rate (%)
- Lap times for completed laps (mean, min, max, std)
- Crashes, timeouts, stationary/stalled episodes
- Mean and median track progress
- Generalization gap (train-like vs held-out test)
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from env.environment import RacingEnv
from sac.actor import SACActor
from sac.track_pools import POOL_CONFIG, TRAIN_SEEDS, TEST_SEEDS


DEFAULT_BASELINE_CKPT = "models/best_single_track_v2.pth"
DEFAULT_DR_CKPT = "runs/dr_multitrack_s42_full/best_actor.pth"

WIDTH_LEVELS = [70, 40, 36, 32, 28, 25, 20]
HAIRPIN_FACTORS = [0.5, 0.35, 0.2]


def inspect_and_load_actor(checkpoint_path: str) -> Tuple[SACActor, int, Tuple[float, ...], float]:
    """Inspect checkpoint input dimension and return (actor, obs_dim, ray_angles, ray_max_dist)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    obs_dim = state_dict["network.0.weight"].shape[1]
    n_rays = obs_dim - 4

    ray_angles = (-90.0, -45.0, 0.0, 45.0, 90.0) if obs_dim == 9 else None
    ray_max_dist = 200.0

    # Check for adjacent run_config.json
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
        raise ValueError(f"Could not determine ray angles for obs_dim={obs_dim} in {checkpoint_path}")

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
    """Execute a single deterministic evaluation rollout."""
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
        steps = step + 1
        final_info = info
        spd = float(obs[0])
        if spd > max_speed:
            max_speed = spd

        if info.get("lap_completed", False):
            completed = True
        if info.get("crashed", False):
            crashed = True

        if terminated or truncated:
            break

    env.close()

    if completed:
        outcome = "completed"
    elif crashed:
        outcome = "crash"
    else:
        outcome = "timeout"

    lap_progress = float(final_info.get("lap_progress", 0.0))
    stationary = bool(outcome == "timeout" and (max_speed < 0.05 or abs(lap_progress) < 0.005))

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
    """Compute summary statistics for a set of episodes."""
    total = len(episodes)
    if total == 0:
        return {
            "total": 0, "completed": 0, "crashes": 0, "timeouts": 0, "stationary": 0,
            "completion_rate": 0.0, "avg_lap_time": None, "std_lap_time": None,
            "min_lap_time": None, "max_lap_time": None,
            "avg_progress": 0.0, "median_progress": 0.0,
        }

    completed = sum(1 for e in episodes if e["completed"])
    crashes = sum(1 for e in episodes if e["crashed"])
    timeouts = sum(1 for e in episodes if e["timeout"])
    stationary = sum(1 for e in episodes if e["stationary"])

    lap_times = [e["lap_time"] for e in episodes if e["completed"] and e["lap_time"] is not None]
    avg_lap_time = round(float(np.mean(lap_times)), 2) if lap_times else None
    std_lap_time = round(float(np.std(lap_times)), 2) if lap_times else None
    min_lap_time = int(np.min(lap_times)) if lap_times else None
    max_lap_time = int(np.max(lap_times)) if lap_times else None

    progresses = [e["lap_progress"] for e in episodes]
    avg_progress = round(float(np.mean(progresses)), 4)
    median_progress = round(float(np.median(progresses)), 4)

    return {
        "total": total,
        "completed": completed,
        "crashes": crashes,
        "timeouts": timeouts,
        "stationary": stationary,
        "completion_rate": round(completed / total, 4),
        "avg_lap_time": avg_lap_time,
        "std_lap_time": std_lap_time,
        "min_lap_time": min_lap_time,
        "max_lap_time": max_lap_time,
        "avg_progress": avg_progress,
        "median_progress": median_progress,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs DR agent on held-out tracks.")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE_CKPT, help="Path to baseline actor")
    parser.add_argument("--dr-agent", default=DEFAULT_DR_CKPT, help="Path to DR actor")
    parser.add_argument("--out-dir", default="analysis/results", help="Directory for output files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("HELD-OUT GENERALIZATION BENCHMARK: BASELINE VS DR AGENT")
    print("=" * 80)

    # 1. Load models
    b_actor, b_dim, b_rays, b_dist = inspect_and_load_actor(args.baseline)
    dr_actor, dr_dim, dr_rays, dr_dist = inspect_and_load_actor(args.dr_agent)
    print(f"Loaded Baseline: {args.baseline} (obs_dim={b_dim})")
    print(f"Loaded DR Agent: {args.dr_agent} (obs_dim={dr_dim})")

    models = [
        ("Baseline (Single-Track v2)", b_actor, b_rays, b_dist, args.baseline),
        ("DR Full (Multi-Track)", dr_actor, dr_rays, dr_dist, args.dr_agent),
    ]

    all_episodes: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # SUITE 1: Standard Track Pools (width=70, min_radius=80, max_steps=700)
    train_pool_label = f"TRAIN_SEEDS ({min(TRAIN_SEEDS)}..{max(TRAIN_SEEDS)})"
    test_pool_label = f"TEST_SEEDS ({min(TEST_SEEDS)}..{max(TEST_SEEDS)})"
    print(f"\n[Suite 1/3] Running Track Pools: {train_pool_label} and {test_pool_label}...")
    base_tk = dict(POOL_CONFIG["track_kwargs"])
    pool_summary: Dict[str, Dict[str, Any]] = {}

    for model_name, actor, rays, dist, ckpt in models:
        pool_summary[model_name] = {}
        for pool_name, seeds in [(train_pool_label, TRAIN_SEEDS), (test_pool_label, TEST_SEEDS)]:
            ep_list = []
            for s in seeds:
                res = run_single_episode(actor, s, base_tk, rays, dist, max_steps=POOL_CONFIG["max_steps"])
                res.update({
                    "benchmark": "track_pool",
                    "suite": pool_name,
                    "model": model_name,
                    "checkpoint": ckpt,
                    "width": base_tk["width"],
                    "hairpin_factor": None,
                })
                ep_list.append(res)
                all_episodes.append(res)
            pool_summary[model_name][pool_name] = compute_aggregate(ep_list)

    # -------------------------------------------------------------------------
    # SUITE 2: Width Sweep (widths: 70, 40, 36, 32, 28, 25, 20)
    # Tested on held-out TEST_SEEDS (1001..1010) AND historical seeds (1..10)
    # -------------------------------------------------------------------------
    print("\n[Suite 2/3] Running Width Sweeps across widths [70, 40, 36, 32, 28, 25, 20]...")
    width_summary: Dict[str, Dict[str, Dict[int, Any]]] = {}

    seed_sets = [
        ("held_out_test (1001..1010)", TEST_SEEDS),
        ("historical_seeds (1..10)", list(range(1, 11))),
    ]

    for model_name, actor, rays, dist, ckpt in models:
        width_summary[model_name] = {}
        for set_label, seeds in seed_sets:
            width_summary[model_name][set_label] = {}
            for w in WIDTH_LEVELS:
                tk = dict(base_tk)
                tk["width"] = w
                ep_list = []
                for s in seeds:
                    res = run_single_episode(actor, s, tk, rays, dist, max_steps=500)
                    res.update({
                        "benchmark": "width_sweep",
                        "suite": set_label,
                        "model": model_name,
                        "checkpoint": ckpt,
                        "width": w,
                        "hairpin_factor": None,
                    })
                    ep_list.append(res)
                    all_episodes.append(res)
                width_summary[model_name][set_label][w] = compute_aggregate(ep_list)

    # -------------------------------------------------------------------------
    # SUITE 3: Hairpin Benchmark on Held-Out Test Seeds (1001..1010)
    # -------------------------------------------------------------------------
    print("\n[Suite 3/3] Running Hairpin Benchmark on held-out TEST_SEEDS (1001..1010)...")
    hairpin_heldout_summary: Dict[str, Dict[float, Any]] = {}

    for model_name, actor, rays, dist, ckpt in models:
        hairpin_heldout_summary[model_name] = {}
        for hf in HAIRPIN_FACTORS:
            tk = dict(base_tk)
            tk["hairpin"] = True
            tk["hairpin_factor"] = hf
            ep_list = []
            for s in TEST_SEEDS:
                res = run_single_episode(actor, s, tk, rays, dist, max_steps=500)
                res.update({
                    "benchmark": "hairpin_heldout",
                    "suite": "TEST_SEEDS (1001..1010)",
                    "model": model_name,
                    "checkpoint": ckpt,
                    "width": base_tk["width"],
                    "hairpin_factor": hf,
                })
                ep_list.append(res)
                all_episodes.append(res)
            hairpin_heldout_summary[model_name][hf] = compute_aggregate(ep_list)

    # -------------------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. TRACK POOL BENCHMARK SUMMARY (Fixed width=70, max_steps=700)")
    print("=" * 80)
    header = f"{'Model':<28} | {'Pool':<24} | {'Completed':<10} | {'Compl. Rate':<12} | {'Avg Lap':<10} | {'Mean Prog':<10}"
    print(header)
    print("-" * 80)
    for model_name in pool_summary:
        for pool_name, agg in pool_summary[model_name].items():
            compl_str = f"{agg['completed']}/{agg['total']}"
            rate_str = f"{agg['completion_rate']*100:5.1f}%"
            lap_str = f"{agg['avg_lap_time']:.1f}" if agg['avg_lap_time'] else "N/A"
            print(f"{model_name:<28} | {pool_name:<24} | {compl_str:<10} | {rate_str:<12} | {lap_str:<10} | {agg['avg_progress']:.4f}")
        print("-" * 80)

    print("\n" + "=" * 90)
    print("2. WIDTH SWEEP BENCHMARK (Held-Out Test Seeds: 1001..1010)")
    print("=" * 90)
    header_w = f"{'Model':<28} | {'Width':<6} | {'Completed':<10} | {'Crashes':<8} | {'Timeouts (Stat)':<16} | {'Avg Lap':<10} | {'Mean Prog':<10}"
    print(header_w)
    print("-" * 90)
    for model_name in width_summary:
        set_data = width_summary[model_name]["held_out_test (1001..1010)"]
        for w in WIDTH_LEVELS:
            agg = set_data[w]
            compl_str = f"{agg['completed']}/{agg['total']}"
            to_str = f"{agg['timeouts']} ({agg['stationary']})"
            lap_str = f"{agg['avg_lap_time']:.1f}" if agg['avg_lap_time'] else "N/A"
            print(f"{model_name:<28} | {w:<6d} | {compl_str:<10} | {agg['crashes']:<8d} | {to_str:<16} | {lap_str:<10} | {agg['avg_progress']:.4f}")
        print("-" * 90)

    print("\n" + "=" * 90)
    print("3. HAIRPIN BENCHMARK (Held-Out Test Seeds: 1001..1010)")
    print("=" * 90)
    header_h = f"{'Model':<28} | {'Factor':<6} | {'Completed':<10} | {'Crashes':<8} | {'Timeouts (Stat)':<16} | {'Avg Lap':<10} | {'Mean Prog':<10}"
    print(header_h)
    print("-" * 90)
    for model_name in hairpin_heldout_summary:
        for hf in HAIRPIN_FACTORS:
            agg = hairpin_heldout_summary[model_name][hf]
            compl_str = f"{agg['completed']}/{agg['total']}"
            to_str = f"{agg['timeouts']} ({agg['stationary']})"
            lap_str = f"{agg['avg_lap_time']:.1f}" if agg['avg_lap_time'] else "N/A"
            print(f"{model_name:<28} | {hf:<6.2f} | {compl_str:<10} | {agg['crashes']:<8d} | {to_str:<16} | {lap_str:<10} | {agg['avg_progress']:.4f}")
        print("-" * 90)

    # -------------------------------------------------------------------------
    # SAVE OUTPUTS
    # -------------------------------------------------------------------------
    csv_out = os.path.join(args.out_dir, "heldout_generalization_episodes.csv")
    json_out = os.path.join(args.out_dir, "heldout_generalization_summary.json")

    fieldnames = [
        "benchmark", "suite", "model", "checkpoint", "seed", "width", "hairpin_factor",
        "outcome", "completed", "crashed", "timeout", "stationary", "steps",
        "lap_time", "lap_progress", "max_speed"
    ]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in all_episodes:
            writer.writerow({k: ep.get(k) for k in fieldnames})
    print(f"\n[Saved] Detailed episodes CSV -> {csv_out}")

    # Compute explicit generalization gaps
    # Gap definition: Train Completion Rate - Test Completion Rate (smaller is better; 0 is ideal)
    gaps = {}
    for model_name in pool_summary:
        train_agg = pool_summary[model_name][train_pool_label]
        test_agg = pool_summary[model_name][test_pool_label]
        gaps[model_name] = {
            "completion_gap_pct": round((train_agg["completion_rate"] - test_agg["completion_rate"]) * 100, 2),
            "lap_time_gap_steps": round(train_agg["avg_lap_time"] - test_agg["avg_lap_time"], 2) if (train_agg["avg_lap_time"] and test_agg["avg_lap_time"]) else None,
            "progress_gap": round(train_agg["avg_progress"] - test_agg["avg_progress"], 4),
        }

    summary_payload = {
        "metadata": {
            "baseline_ckpt": args.baseline,
            "dr_ckpt": args.dr_agent,
            "train_seeds": TRAIN_SEEDS,
            "test_seeds": TEST_SEEDS,
            "width_levels": WIDTH_LEVELS,
            "hairpin_factors": HAIRPIN_FACTORS,
        },
        "generalization_gaps": gaps,
        "track_pool_summary": pool_summary,
        "width_sweep_summary": width_summary,
        "hairpin_heldout_summary": hairpin_heldout_summary,
    }

    with open(json_out, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[Saved] Aggregate JSON summary -> {json_out}")


if __name__ == "__main__":
    main()
