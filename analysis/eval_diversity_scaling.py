"""Comprehensive Held-Out Evaluation for Track Pool Diversity Scaling.

Evaluates policies trained with different training track pool sizes (5, 10, 20, 25)
plus single-track baseline against identical held-out test conditions:
1. Held-Out Track Pool: TEST_SEEDS (1001..1010) under standard medium geometry (width 70)
2. Narrow-Width Generalization Sweep: widths [70, 40, 36, 32, 28, 25, 20] on TEST_SEEDS
3. Hairpin Generalization Benchmark: pinch factors [0.5, 0.35, 0.2] on TEST_SEEDS

Outputs:
  - analysis/results/diversity_scaling_episodes.csv
  - analysis/results/diversity_scaling_summary.json
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

DEFAULT_MODELS = [
    {
        "pool_size": 1,
        "name": "Single-Track v2 (No DR)",
        "ckpt": "models/best_single_track_v2.pth",
    },
    {
        "pool_size": 5,
        "name": "Pool 5 (Calibrated DR)",
        "ckpt": "runs/dr_multitrack_s42_calibrated_5tracks/best_actor.pth",
    },
    {
        "pool_size": 10,
        "name": "Pool 10 (Calibrated DR)",
        "ckpt": "runs/dr_multitrack_s42_calibrated_10tracks/best_actor.pth",
    },
    {
        "pool_size": 20,
        "name": "Pool 20 (Calibrated DR)",
        "ckpt": "runs/dr_multitrack_s42_calibrated_20tracks/best_actor.pth",
    },
    {
        "pool_size": 25,
        "name": "Pool 25 (Calibrated DR)",
        "ckpt": "runs/dr_multitrack_s42_calibrated_25tracks/best_actor.pth",
    },
]

WIDTH_LEVELS = [70, 40, 36, 32, 28, 25, 20]
HAIRPIN_FACTORS = [0.5, 0.35, 0.2]


def inspect_and_load_actor(checkpoint_path: str) -> Tuple[SACActor, int, Tuple[float, ...], float]:
    """Inspect checkpoint input dimension and return (actor, obs_dim, ray_angles, ray_max_dist)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    obs_dim = state_dict["network.0.weight"].shape[1]

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
            "completion_rate": 0.0, "crash_rate": 0.0, "timeout_rate": 0.0, "stationary_rate": 0.0,
            "avg_lap_time": None, "std_lap_time": None, "min_lap_time": None, "max_lap_time": None,
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
        "crash_rate": round(crashes / total, 4),
        "timeout_rate": round(timeouts / total, 4),
        "stationary_rate": round(stationary / total, 4),
        "avg_lap_time": avg_lap_time,
        "std_lap_time": std_lap_time,
        "min_lap_time": min_lap_time,
        "max_lap_time": max_lap_time,
        "avg_progress": avg_progress,
        "median_progress": median_progress,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate diversity-scaled models on held-out tracks.")
    parser.add_argument("--out-dir", default="analysis/results", help="Directory for output files")
    parser.add_argument("--include-single-track", action="store_true", default=True,
                        help="Include single-track v2 as reference baseline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("DIVERSITY SCALING GENERALIZATION BENCHMARK")
    print(f"Target Held-Out Test Seeds: {TEST_SEEDS} (Disjoint from all training pools)")
    print("=" * 80)

    # 1. Load available models
    active_models = []
    for m in DEFAULT_MODELS:
        if not args.include_single_track and m["pool_size"] == 1:
            continue
        if os.path.exists(m["ckpt"]):
            actor, obs_dim, ray_angles, ray_max_dist = inspect_and_load_actor(m["ckpt"])
            print(f"Loaded Pool {m['pool_size']:2d} | {m['name']} from {m['ckpt']} (obs_dim={obs_dim})")
            active_models.append((m["pool_size"], m["name"], actor, ray_angles, ray_max_dist, m["ckpt"]))
        else:
            print(f"⚠️ Checkpoint missing for Pool {m['pool_size']}: {m['ckpt']} (skipping)")

    if not active_models:
        print("❌ No models found to evaluate!")
        sys.exit(1)

    all_episodes: List[Dict[str, Any]] = []
    base_tk = dict(POOL_CONFIG["track_kwargs"])

    # -------------------------------------------------------------------------
    # SUITE 1: Standard Held-Out Test Tracks (width=70, min_radius=80, max_steps=700)
    # -------------------------------------------------------------------------
    test_pool_label = f"TEST_SEEDS ({min(TEST_SEEDS)}..{max(TEST_SEEDS)})"
    train_pool_label = f"TRAIN_SEEDS (1..{len(TRAIN_SEEDS)})"
    print(f"\n[Suite 1/3] Standard Tracks: Held-Out {test_pool_label} vs {train_pool_label}...")

    pool_summary: Dict[str, Dict[str, Any]] = {}

    for pool_size, model_name, actor, rays, dist, ckpt in active_models:
        pool_summary[model_name] = {}
        for pool_name, seeds in [(test_pool_label, TEST_SEEDS), (train_pool_label, TRAIN_SEEDS)]:
            ep_list = []
            for s in seeds:
                res = run_single_episode(actor, s, base_tk, rays, dist, max_steps=POOL_CONFIG["max_steps"])
                res.update({
                    "pool_size": pool_size,
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
    # SUITE 2: Width Sweep Benchmark across widths [70, 40, 36, 32, 28, 25, 20]
    # Tested on held-out TEST_SEEDS (1001..1010)
    # -------------------------------------------------------------------------
    print(f"\n[Suite 2/3] Width Sweep on Held-Out Tracks across widths {WIDTH_LEVELS}...")
    width_summary: Dict[str, Dict[str, Any]] = {}

    for pool_size, model_name, actor, rays, dist, ckpt in active_models:
        width_summary[model_name] = {}
        for w in WIDTH_LEVELS:
            tk = dict(base_tk)
            tk["width"] = w
            ep_list = []
            for s in TEST_SEEDS:
                res = run_single_episode(actor, s, tk, rays, dist, max_steps=POOL_CONFIG["max_steps"])
                res.update({
                    "pool_size": pool_size,
                    "benchmark": "width_sweep",
                    "suite": f"width_{w}",
                    "model": model_name,
                    "checkpoint": ckpt,
                    "width": w,
                    "hairpin_factor": None,
                })
                ep_list.append(res)
                all_episodes.append(res)
            width_summary[model_name][f"width_{w}"] = compute_aggregate(ep_list)

    # -------------------------------------------------------------------------
    # SUITE 3: Hairpin Generalization Benchmark (pinch factors 0.5, 0.35, 0.2)
    # Tested on held-out TEST_SEEDS (1001..1010)
    # -------------------------------------------------------------------------
    print(f"\n[Suite 3/3] Hairpin Generalization on Held-Out Tracks across pinch factors {HAIRPIN_FACTORS}...")
    hairpin_summary: Dict[str, Dict[str, Any]] = {}

    for pool_size, model_name, actor, rays, dist, ckpt in active_models:
        hairpin_summary[model_name] = {}
        for f in HAIRPIN_FACTORS:
            tk = dict(base_tk)
            tk["hairpin"] = True
            tk["hairpin_factor"] = f
            ep_list = []
            for s in TEST_SEEDS:
                res = run_single_episode(actor, s, tk, rays, dist, max_steps=POOL_CONFIG["max_steps"])
                res.update({
                    "pool_size": pool_size,
                    "benchmark": "hairpin_heldout",
                    "suite": f"pinch_{f}",
                    "model": model_name,
                    "checkpoint": ckpt,
                    "width": base_tk["width"],
                    "hairpin_factor": f,
                })
                ep_list.append(res)
                all_episodes.append(res)
            hairpin_summary[model_name][f"pinch_{f}"] = compute_aggregate(ep_list)

    # -------------------------------------------------------------------------
    # Save Raw Episode Data (CSV) and Aggregate Summaries (JSON)
    # -------------------------------------------------------------------------
    csv_out = os.path.join(args.out_dir, "diversity_scaling_episodes.csv")
    json_out = os.path.join(args.out_dir, "diversity_scaling_summary.json")

    fieldnames = [
        "pool_size", "model", "checkpoint", "benchmark", "suite", "seed",
        "width", "hairpin_factor", "outcome", "completed", "crashed",
        "timeout", "stationary", "steps", "lap_time", "lap_progress", "max_speed"
    ]

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_episodes)

    summary_payload = {
        "metadata": {
            "evaluated_models": [
                {"pool_size": p, "name": n, "ckpt": c} for p, n, _, _, _, c in active_models
            ],
            "train_seeds": TRAIN_SEEDS,
            "test_seeds": TEST_SEEDS,
            "width_levels": WIDTH_LEVELS,
            "hairpin_factors": HAIRPIN_FACTORS,
        },
        "track_pool_summary": pool_summary,
        "width_sweep_summary": width_summary,
        "hairpin_summary": hairpin_summary,
    }

    with open(json_out, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n[Saved] Detailed episodes CSV -> {csv_out}")
    print(f"[Saved] Summary JSON -> {json_out}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<28} | {'Pool':<4} | {'Held-Out Test (W=70)':<22} | {'Narrow W=28':<14} | {'Hairpin P=0.35':<14}")
    print(f"{'':<28} | {'':<4} | {'Compl%':<8} {'LapTime':<12} | {'Compl%':<14} | {'Compl%':<14}")
    print("-" * 90)
    for p, name, _, _, _, _ in active_models:
        std_test = pool_summary[name][test_pool_label]
        w28_test = width_summary[name].get("width_28", {})
        h35_test = hairpin_summary[name].get("pinch_0.35", {})

        lt_str = f"{std_test['avg_lap_time']} steps" if std_test['avg_lap_time'] else "N/A"
        print(f"{name:<28} | {p:<4} | {std_test['completion_rate']*100:5.1f}%   {lt_str:<12} | {w28_test.get('completion_rate', 0.0)*100:5.1f}%       | {h35_test.get('completion_rate', 0.0)*100:5.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
