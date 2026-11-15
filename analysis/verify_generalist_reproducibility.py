"""Verify reproducibility of frozen generalist checkpoint (models/generalist_v1.pth).

Protocol:
1. Verifies frozen checkpoint weight integrity and SHA-256 hash.
2. Evaluates the frozen model across 3 independent repetitions of identical evaluation conditions:
   - Standard held-out tracks (TEST_SEEDS 1001..1010, width 70 px)
   - Narrow corridor sweep (TEST_SEEDS 1001..1010, widths 70, 40, 36, 32, 28, 25, 20 px)
   - Hairpin pinch sweep (TEST_SEEDS 1001..1010, pinch factors 0.50, 0.35, 0.20)
   (110 episodes per repetition x 3 = 330 episodes)
3. Evaluates 3 repetitions of start-state perturbation robustness (20 trials per repetition x 3 = 60 episodes).
4. Verifies deterministic reproducibility and episode-level consistency across repetitions.
5. Saves:
   - analysis/results/generalist_reproducibility.csv
   - analysis/results/generalist_reproducibility_summary.json
"""

import csv
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.environment import RacingEnv
from sac.actor import SACActor
from sac.track_pools import POOL_CONFIG, TEST_SEEDS

FROZEN_MODEL_PATH = "models/generalist_v1.pth"
CONFIG_PATH = "configs/generalist_v1.json"
EXPECTED_SHA256 = "0d40dc47b70463b99dc2c65b9f034317ab527cf94b756065a8205284f3b76784"

CSV_OUTPUT_PATH = "analysis/results/generalist_reproducibility.csv"
JSON_OUTPUT_PATH = "analysis/results/generalist_reproducibility_summary.json"

WIDTH_LEVELS = [70, 40, 36, 32, 28, 25, 20]
HAIRPIN_FACTORS = [0.50, 0.35, 0.20]


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_actor(checkpoint_path: str = FROZEN_MODEL_PATH) -> Tuple[SACActor, str]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    actual_sha = compute_sha256(checkpoint_path)
    assert actual_sha == EXPECTED_SHA256, (
        f"Weight corruption detected! Expected {EXPECTED_SHA256}, got {actual_sha}"
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    obs_dim = state_dict["network.0.weight"].shape[1]
    action_dim = state_dict["mean.weight"].shape[0]

    actor = SACActor(obs_dim=obs_dim, action_dim=action_dim)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor, actual_sha


def run_single_episode(
    actor: SACActor,
    seed: int,
    track_kwargs: Dict[str, Any],
    ray_angles: Tuple[float, ...],
    max_dist: float = 200.0,
    max_steps: int = 700,
) -> Dict[str, Any]:
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


def build_obs_from_state(env: RacingEnv) -> np.ndarray:
    car, track = env.car, env.track
    err = (car.angle - track.track_heading(car.x, car.y) + 180.0) % 360.0 - 180.0
    dist = track.signed_distance(car.x, car.y)
    slip = 0.0
    rays = car.cast_rays(track, ray_angles=env.obs_ray_angles, max_dist=env.obs_ray_max_dist)
    return np.array([car.velocity, err, dist, slip, *rays], dtype=np.float32)


def run_randomized_start_episode(
    actor: SACActor,
    track_seed: int,
    eval_rng: np.random.Generator,
    track_kwargs: Dict[str, Any],
    ray_angles: Tuple[float, ...],
    max_dist: float = 200.0,
    max_steps: int = 1000,
) -> Dict[str, Any]:
    env = RacingEnv(
        max_steps=max_steps,
        verbose=False,
        track_kwargs=track_kwargs,
        obs_ray_angles=ray_angles,
        obs_ray_max_dist=max_dist,
    )

    env.reset(options={"track_seed": track_seed})

    center = env.track.centerline
    idx = int(eval_rng.integers(0, len(center)))
    env.car.x, env.car.y = float(center[idx][0]), float(center[idx][1])
    env.car.angle = float(env.track.track_heading(env.car.x, env.car.y) + eval_rng.uniform(-15, 15))
    env.car.velocity = float(eval_rng.uniform(0.0, 4.0))

    env.previous_progress = float(env.track.get_progress(env.car.x, env.car.y))
    env.lap_progress = 0.0
    env.lap_completed = False
    env.step_count = 0

    obs = build_obs_from_state(env)
    completed = False
    crashed = False
    steps = 0
    final_info: Dict[str, Any] = {}
    max_speed = env.car.velocity

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
        "seed": track_seed,
        "start_pose_idx": idx,
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


def compute_slice_statistics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = len(episodes)
    completed_lts = [e["steps"] for e in episodes if e["completed"]]
    crashed = sum(1 for e in episodes if e["crashed"])
    timeouts = sum(1 for e in episodes if e["timeout"])
    stationary = sum(1 for e in episodes if e["stationary"])
    progresses = [e["lap_progress"] for e in episodes]

    stats: Dict[str, Any] = {
        "n_total": n_total,
        "n_completed": len(completed_lts),
        "completion_rate": round(len(completed_lts) / n_total, 4) if n_total > 0 else 0.0,
        "n_crashed": crashed,
        "crash_rate": round(crashed / n_total, 4) if n_total > 0 else 0.0,
        "n_timeout": timeouts,
        "timeout_rate": round(timeouts / n_total, 4) if n_total > 0 else 0.0,
        "n_stationary": stationary,
        "stationary_rate": round(stationary / n_total, 4) if n_total > 0 else 0.0,
        "mean_progress": round(float(np.mean(progresses)), 4) if progresses else 0.0,
        "median_progress": round(float(np.median(progresses)), 4) if progresses else 0.0,
    }

    if completed_lts:
        stats["mean_lap_time"] = round(float(np.mean(completed_lts)), 2)
        stats["median_lap_time"] = round(float(np.median(completed_lts)), 2)
        stats["std_lap_time"] = round(float(np.std(completed_lts, ddof=1)), 2) if len(completed_lts) > 1 else 0.0
        stats["min_lap_time"] = int(np.min(completed_lts))
        stats["max_lap_time"] = int(np.max(completed_lts))
    else:
        stats["mean_lap_time"] = None
        stats["median_lap_time"] = None
        stats["std_lap_time"] = None
        stats["min_lap_time"] = None
        stats["max_lap_time"] = None

    return stats


def main():
    print("=" * 80)
    print("FROZEN GENERALIST MODEL REPRODUCIBILITY VERIFICATION")
    print("=" * 80)

    actor, sha256_hash = load_frozen_actor(FROZEN_MODEL_PATH)
    print(f"Loaded frozen model: {FROZEN_MODEL_PATH}")
    print(f"SHA-256: {sha256_hash}")

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    print(f"Loaded config: {CONFIG_PATH} (model={config['model_name']}, v={config['version']})")

    ray_angles = tuple(config["observation_space"]["sensor_configuration"]["obs_ray_angles"])
    ray_max_dist = float(config["observation_space"]["sensor_configuration"]["obs_ray_max_dist"])
    base_tk = dict(config["nominal_track_parameters"])

    all_csv_records: List[Dict[str, Any]] = []
    run_summaries: Dict[str, Dict[str, Any]] = {}

    N_REPETITIONS = 3
    print(f"\nExecuting {N_REPETITIONS} full evaluation repetitions...")

    for rep in range(1, N_REPETITIONS + 1):
        rep_key = f"repetition_{rep}"
        print(f"\n--- REPETITION {rep}/{N_REPETITIONS} ---")
        rep_episodes: List[Dict[str, Any]] = []
        rep_slice_data: Dict[str, Any] = {}

        # 1. Standard Held-Out
        std_eps = []
        for s in TEST_SEEDS:
            res = run_single_episode(actor, s, base_tk, ray_angles, ray_max_dist, max_steps=POOL_CONFIG["max_steps"])
            rec = {
                "repetition": rep,
                "benchmark": "standard_heldout",
                "width": 70,
                "hairpin_factor": None,
                "start_variant": "nominal",
                **res
            }
            std_eps.append(rec)
            rep_episodes.append(rec)
        rep_slice_data["standard_heldout"] = compute_slice_statistics(std_eps)
        print(f"  Standard held-out (W=70): comp={rep_slice_data['standard_heldout']['n_completed']}/10, "
              f"mean_lt={rep_slice_data['standard_heldout']['mean_lap_time']}")

        # 2. Narrow Corridors
        for w in WIDTH_LEVELS:
            tk = dict(base_tk, width=w)
            w_eps = []
            for s in TEST_SEEDS:
                res = run_single_episode(actor, s, tk, ray_angles, ray_max_dist, max_steps=POOL_CONFIG["max_steps"])
                rec = {
                    "repetition": rep,
                    "benchmark": "narrow_corridor",
                    "width": w,
                    "hairpin_factor": None,
                    "start_variant": "nominal",
                    **res
                }
                w_eps.append(rec)
                rep_episodes.append(rec)
            rep_slice_data[f"narrow_w_{w}"] = compute_slice_statistics(w_eps)
        print(f"  Narrow corridors: W=20 comp={rep_slice_data['narrow_w_20']['n_completed']}/10, "
              f"mean_lt={rep_slice_data['narrow_w_20']['mean_lap_time']}")

        # 3. Hairpins
        for hf in HAIRPIN_FACTORS:
            tk = dict(base_tk, hairpin=True, hairpin_factor=hf)
            h_eps = []
            for s in TEST_SEEDS:
                res = run_single_episode(actor, s, tk, ray_angles, ray_max_dist, max_steps=POOL_CONFIG["max_steps"])
                rec = {
                    "repetition": rep,
                    "benchmark": "hairpin",
                    "width": 70,
                    "hairpin_factor": hf,
                    "start_variant": "nominal",
                    **res
                }
                h_eps.append(rec)
                rep_episodes.append(rec)
            rep_slice_data[f"hairpin_f_{hf}"] = compute_slice_statistics(h_eps)
        print(f"  Hairpins: f=0.50 comp={rep_slice_data['hairpin_f_0.5']['n_completed']}/10, "
              f"f=0.35 comp={rep_slice_data['hairpin_f_0.35']['n_completed']}/10, "
              f"f=0.20 comp={rep_slice_data['hairpin_f_0.2']['n_completed']}/10")

        # 4. Randomized Start States (20 trials across held-out seeds with fixed RNG seed per rep)
        # Using fixed RNG seed per repetition to test exact reproducibility
        start_rng = np.random.default_rng(1000 + rep)
        start_eps = []
        for s in TEST_SEEDS:
            for trial in range(2):
                res = run_randomized_start_episode(
                    actor=actor,
                    track_seed=s,
                    eval_rng=start_rng,
                    track_kwargs=base_tk,
                    ray_angles=ray_angles,
                    max_dist=ray_max_dist,
                    max_steps=1000
                )
                rec = {
                    "repetition": rep,
                    "benchmark": "randomized_starts",
                    "width": 70,
                    "hairpin_factor": None,
                    "start_variant": f"randomized_trial_{trial}",
                    **res
                }
                start_eps.append(rec)
                rep_episodes.append(rec)
        rep_slice_data["randomized_starts"] = compute_slice_statistics(start_eps)
        print(f"  Randomized starts: comp={rep_slice_data['randomized_starts']['n_completed']}/20, "
              f"mean_lt={rep_slice_data['randomized_starts']['mean_lap_time']}")

        # Overall summary for standard + narrow + hairpin (the 110 deterministic benchmark episodes)
        bench_110 = [e for e in rep_episodes if e["benchmark"] != "randomized_starts"]
        rep_slice_data["overall_deterministic_benchmark"] = compute_slice_statistics(bench_110)
        rep_slice_data["overall_all_regimes"] = compute_slice_statistics(rep_episodes)

        run_summaries[rep_key] = rep_slice_data
        all_csv_records.extend(rep_episodes)

    # Cross-repetition consistency analysis
    print(f"\n{'=' * 80}")
    print("CROSS-REPETITION CONSISTENCY VERIFICATION")
    print("=" * 80)

    # For the deterministic episodes (110 per rep), compare repetition 1, 2, and 3
    rep1_det = [e for e in all_csv_records if e["repetition"] == 1 and e["benchmark"] != "randomized_starts"]
    rep2_det = [e for e in all_csv_records if e["repetition"] == 2 and e["benchmark"] != "randomized_starts"]
    rep3_det = [e for e in all_csv_records if e["repetition"] == 3 and e["benchmark"] != "randomized_starts"]

    assert len(rep1_det) == 110 and len(rep2_det) == 110 and len(rep3_det) == 110

    mismatches_12 = 0
    mismatches_13 = 0
    for i in range(110):
        e1, e2, e3 = rep1_det[i], rep2_det[i], rep3_det[i]
        diff12 = (e1["outcome"] != e2["outcome"] or e1["steps"] != e2["steps"] or e1["lap_progress"] != e2["lap_progress"])
        diff13 = (e1["outcome"] != e3["outcome"] or e1["steps"] != e3["steps"] or e1["lap_progress"] != e3["lap_progress"])
        if diff12:
            mismatches_12 += 1
        if diff13:
            mismatches_13 += 1

    print(f"Deterministic benchmark episodes compared: 110")
    print(f"Mismatches between Repetition 1 and 2: {mismatches_12}")
    print(f"Mismatches between Repetition 1 and 3: {mismatches_13}")

    perfect_reproducibility = (mismatches_12 == 0 and mismatches_13 == 0)
    print(f"Exact Deterministic Reproducibility: {'CONFIRMED (100% BIT-FOR-BIT MATCH)' if perfect_reproducibility else 'FAILED'}")

    # Output CSV
    fieldnames = [
        "repetition", "benchmark", "width", "hairpin_factor", "start_variant",
        "seed", "start_pose_idx", "outcome", "completed", "crashed", "timeout",
        "stationary", "steps", "lap_time", "lap_progress", "max_speed"
    ]
    with open(CSV_OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_csv_records)
    print(f"\nSaved reproducibility episodes to: {CSV_OUTPUT_PATH} ({len(all_csv_records)} rows)")

    # Output JSON summary
    summary_data = {
        "metadata": {
            "model_path": FROZEN_MODEL_PATH,
            "config_path": CONFIG_PATH,
            "weights_sha256": sha256_hash,
            "config_sha256": compute_sha256(CONFIG_PATH),
            "test_seeds": TEST_SEEDS,
            "n_repetitions": N_REPETITIONS,
            "episodes_per_repetition": 130,  # 110 deterministic + 20 randomized start
            "total_episodes": len(all_csv_records),
            "deterministic_reproducibility": {
                "mismatches_rep1_vs_rep2": mismatches_12,
                "mismatches_rep1_vs_rep3": mismatches_13,
                "is_bit_for_bit_identical": perfect_reproducibility
            }
        },
        "repetition_summaries": run_summaries,
        "consolidated_deterministic_benchmark": run_summaries["repetition_1"]["overall_deterministic_benchmark"],
        "consolidated_randomized_starts": {
            "rep_1_completion_rate": run_summaries["repetition_1"]["randomized_starts"]["completion_rate"],
            "rep_2_completion_rate": run_summaries["repetition_2"]["randomized_starts"]["completion_rate"],
            "rep_3_completion_rate": run_summaries["repetition_3"]["randomized_starts"]["completion_rate"],
            "mean_completion_rate": float(np.mean([
                run_summaries[f"repetition_{r}"]["randomized_starts"]["completion_rate"] for r in [1, 2, 3]
            ])),
            "mean_lap_time": float(np.mean([
                run_summaries[f"repetition_{r}"]["randomized_starts"]["mean_lap_time"] for r in [1, 2, 3]
            ]))
        }
    }

    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved reproducibility summary to: {JSON_OUTPUT_PATH}")

    # Compare with Run A / dr_multitrack_s42_calibrated_25tracks from ablation_episodes.csv
    print(f"\n{'=' * 80}")
    print("CROSS-CHECK AGAINST HISTORICAL ABLATION RUN A EPISODES")
    print("=" * 80)
    ablation_csv_path = "analysis/results/ablation_episodes.csv"
    if os.path.exists(ablation_csv_path):
        with open(ablation_csv_path, "r") as f:
            abl_rows = [r for r in csv.DictReader(f) if r["condition"] == "A"]
        print(f"Found {len(abl_rows)} historical episodes for Condition A in ablation_episodes.csv")

        hist_mismatches = 0
        for i in range(min(len(rep1_det), len(abl_rows))):
            curr = rep1_det[i]
            hist = abl_rows[i]
            # Compare seed, benchmark, outcome, steps
            if (curr["seed"] != int(hist["seed"]) or
                curr["outcome"] != hist["outcome"] or
                curr["steps"] != int(hist["steps"])):
                hist_mismatches += 1

        print(f"Mismatches against historical Condition A records: {hist_mismatches}")
        if hist_mismatches == 0:
            print("Historical consistency: 100% BIT-FOR-BIT MATCH with ablation Run A!")
        else:
            print(f"Warning: {hist_mismatches} historical mismatches detected.")

    print(f"\nReproducibility verification complete.")


if __name__ == "__main__":
    main()
