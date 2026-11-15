"""Compare 5-ray canonical baseline vs 9-ray candidate on identical unseen hairpins.

Evaluates checkpoints deterministically (tanh(mean)) across multiple hairpin pinch
factors and seeds using identical environment configurations, recording:
  - completed laps, crashes, timeouts
  - completion rate
  - average lap time for completed laps
  - average lap progress
  - per-seed paired outcomes (baseline vs candidate)

Outputs:
  - Formatted console tables
  - analysis/results/hairpin_comparison.csv (per-episode records)
  - analysis/results/hairpin_comparison_summary.json (aggregate metrics and pairings)
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from env.environment import RacingEnv
from sac.actor import SACActor

# -----------------------------------------------------------------------------
# Default configurations matching project conventions
# -----------------------------------------------------------------------------

DEFAULT_BASELINE_CKPT = "models/best_single_track_v2.pth"
DEFAULT_CANDIDATE_CKPT = "runs/exp2_demos_s42_9ray/best_actor.pth"

DEFAULT_FACTORS = [0.5, 0.35, 0.2]
DEFAULT_SEEDS = list(range(1, 21))
DEFAULT_MAX_STEPS = 500

BASE_TRACK_KWARGS: Dict[str, Any] = {
    "width": 70,
    "base_r": 250,
    "n_ctrl": 10,
    "min_radius": 80,
    "cx": 400,
    "cy": 300,
    "hairpin": True,
}

RAY_ANGLES_5 = (-90.0, -45.0, 0.0, 45.0, 90.0)
RAY_ANGLES_9 = (-90.0, -67.5, -45.0, -22.5, 0.0, 22.5, 45.0, 67.5, 90.0)
OBS_RAY_MAX_DIST = 200.0


def inspect_and_load_actor(
    checkpoint_path: str,
    override_ray_angles: Tuple[float, ...] = None,
    override_max_dist: float = None,
) -> Tuple[SACActor, int, Tuple[float, ...], float]:
    """Inspect checkpoint input dimension and return (actor, obs_dim, ray_angles, ray_max_dist)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "network.0.weight" not in state_dict:
        raise KeyError(
            f"Expected key 'network.0.weight' not found in {checkpoint_path}. "
            f"Available keys: {list(state_dict.keys())}"
        )

    obs_dim = state_dict["network.0.weight"].shape[1]
    n_rays = obs_dim - 4

    ray_angles = override_ray_angles
    ray_max_dist = override_max_dist

    # Check for adjacent run_config.json or {name}_config.json
    if ray_angles is None or ray_max_dist is None:
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        candidates_config = [
            os.path.join(ckpt_dir, "run_config.json"),
            os.path.splitext(checkpoint_path)[0] + "_config.json",
        ]
        for cfg_path in candidates_config:
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r") as f:
                        cfg = json.load(f)
                    if ray_angles is None and "obs_ray_angles" in cfg:
                        ray_angles = tuple(float(a) for a in cfg["obs_ray_angles"])
                    if ray_max_dist is None and "obs_ray_max_dist" in cfg:
                        ray_max_dist = float(cfg["obs_ray_max_dist"])
                except Exception:
                    pass
                break

    if ray_max_dist is None:
        ray_max_dist = OBS_RAY_MAX_DIST

    if ray_angles is None:
        if obs_dim == 9:
            ray_angles = RAY_ANGLES_5
        elif obs_dim == 13:
            ray_angles = RAY_ANGLES_9
        else:
            raise ValueError(
                f"Unsupported observation dimension {obs_dim} (requires {n_rays} rays) without "
                f"explicit config or override in {checkpoint_path}."
            )

    if len(ray_angles) != n_rays:
        raise ValueError(
            f"Mismatch for checkpoint {checkpoint_path}: obs_dim={obs_dim} implies {n_rays} rays, "
            f"but detected/provided ray_angles has {len(ray_angles)}: {ray_angles}"
        )

    actor = SACActor(obs_dim=obs_dim, action_dim=2)
    actor.load_state_dict(state_dict)
    actor.eval()

    return actor, obs_dim, ray_angles, ray_max_dist


def run_episode(
    actor: SACActor,
    seed: int,
    hairpin_factor: float,
    ray_angles: Tuple[float, ...],
    max_dist: float = OBS_RAY_MAX_DIST,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Dict[str, Any]:
    """Run a single deterministic rollout on an unseen hairpin track."""
    track_kwargs = dict(BASE_TRACK_KWARGS)
    track_kwargs["hairpin_factor"] = hairpin_factor

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
        "hairpin_factor": hairpin_factor,
        "seed": seed,
        "outcome": outcome,
        "steps": steps,
        "lap_time": steps if completed else None,
        "lap_progress": round(lap_progress, 4),
        "completed": completed,
        "crashed": crashed,
        "timeout": outcome == "timeout",
        "stationary": stationary,
        "max_speed": round(max_speed, 4),
    }


def run_diagnostic_episode(
    actor: SACActor,
    seed: int,
    hairpin_factor: float,
    ray_angles: Tuple[float, ...],
    obs_dim: int,
    checkpoint_path: str,
    max_dist: float = OBS_RAY_MAX_DIST,
    max_steps: int = DEFAULT_MAX_STEPS,
    csv_out: str = "analysis/results/hairpin_trace.csv",
) -> Dict[str, Any]:
    """Run a single episode in diagnostic mode, recording step-by-step telemetry."""
    track_kwargs = dict(BASE_TRACK_KWARGS)
    track_kwargs["hairpin_factor"] = hairpin_factor

    env = RacingEnv(
        max_steps=max_steps,
        verbose=False,
        track_kwargs=track_kwargs,
        obs_ray_angles=ray_angles,
        obs_ray_max_dist=max_dist,
    )

    obs, _ = env.reset(options={"track_seed": seed})

    trace_records: List[Dict[str, Any]] = []
    completed = False
    crashed = False

    print("=" * 80)
    print("DIAGNOSTIC TRACE RUN")
    print("=" * 80)
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Obs dimension:  {obs_dim} ({len(ray_angles)} rays: {ray_angles})")
    print(f"Hairpin factor: {hairpin_factor}")
    print(f"Track seed:     {seed}")
    print(f"Max steps:      {max_steps}")
    print("-" * 80)

    for step in range(max_steps):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean, _ = actor(obs_tensor)
            action = torch.tanh(mean).numpy()[0]

        steering = float(action[0])
        throttle = float(action[1])

        obs, reward, terminated, truncated, info = env.step(action)

        step_crashed = bool(info.get("crashed", False))
        step_completed = bool(info.get("lap_completed", False))
        if step_completed:
            completed = True
        if step_crashed:
            crashed = True

        velocity = float(obs[0])
        heading_err = float(obs[1])
        signed_dist = float(obs[2])
        slip = float(obs[3])
        rays = [float(r) for r in obs[4:]]
        min_ray = min(rays) if rays else 0.0
        lap_progress = float(info.get("lap_progress", 0.0))

        record: Dict[str, Any] = {
            "step": step + 1,
            "velocity": round(velocity, 4),
            "heading_error": round(heading_err, 2),
            "signed_distance": round(signed_dist, 2),
            "slip": round(slip, 2),
            "min_ray": round(min_ray, 2),
            "steering": round(steering, 4),
            "throttle": round(throttle, 4),
            "lap_progress": round(lap_progress, 4),
            "crashed": step_crashed,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "lap_completed": step_completed,
            "reward": round(float(reward), 4),
        }

        # Include individual ray distance columns
        for r_idx, r_val in enumerate(rays):
            record[f"ray_{r_idx}"] = round(r_val, 2)

        # Include full ray list string for convenience
        record["all_rays"] = ";".join(str(round(r, 2)) for r in rays)

        trace_records.append(record)

        if step < 10 or (step + 1) % 25 == 0 or terminated or truncated:
            print(
                f"STEP {step+1:3d} | "
                f"velocity={velocity:5.2f} | "
                f"min_ray={min_ray:5.1f} | "
                f"steer={steering:+5.2f} | "
                f"throttle={throttle:+5.2f} | "
                f"dist={signed_dist:+6.2f} | "
                f"heading={heading_err:+6.1f}° | "
                f"slip={slip:+5.2f}° | "
                f"progress={lap_progress:.4f}"
            )

        if terminated or truncated:
            break

    env.close()

    if completed:
        outcome = "completed"
    elif crashed:
        outcome = "crash"
    else:
        outcome = "timeout"

    print("-" * 80)
    print(f"DIAGNOSTIC RESULT: {outcome.upper()} at step {len(trace_records)} (progress={lap_progress:.4f})")
    print("=" * 80)

    # Save to CSV
    csv_dir = os.path.dirname(csv_out)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    if trace_records:
        fieldnames = list(trace_records[0].keys())
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in trace_records:
                writer.writerow(r)
        print(f"\n[Saved] Diagnostic trace CSV -> {csv_out}")

    return {
        "checkpoint": checkpoint_path,
        "obs_dim": obs_dim,
        "hairpin_factor": hairpin_factor,
        "seed": seed,
        "steps": len(trace_records),
        "outcome": outcome,
        "lap_progress": lap_progress,
        "csv_out": csv_out,
    }


def compute_aggregate(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics for a set of episode runs."""
    total = len(episodes)
    if total == 0:
        return {
            "total": 0,
            "completed": 0,
            "crashes": 0,
            "timeouts": 0,
            "stationary": 0,
            "completion_rate": 0.0,
            "avg_lap_time": None,
            "avg_progress": 0.0,
            "median_progress": 0.0,
        }

    completed = sum(1 for e in episodes if e["completed"])
    crashes = sum(1 for e in episodes if e["crashed"])
    timeouts = sum(1 for e in episodes if e["timeout"])
    stationary = sum(1 for e in episodes if e.get("stationary", False))

    completed_times = [e["lap_time"] for e in episodes if e["completed"] and e["lap_time"] is not None]
    avg_lap_time = round(float(np.mean(completed_times)), 2) if completed_times else None

    progresses = [e["lap_progress"] for e in episodes]
    avg_progress = round(float(np.mean(progresses)), 4)
    median_progress = round(float(np.median(progresses)), 4) if progresses else 0.0

    return {
        "total": total,
        "completed": completed,
        "crashes": crashes,
        "timeouts": timeouts,
        "stationary": stationary,
        "completion_rate": round(completed / total, 4),
        "avg_lap_time": avg_lap_time,
        "avg_progress": avg_progress,
        "median_progress": median_progress,
    }


def format_summary_table(
    summary_by_model_and_factor: Dict[str, Dict[float, Dict[str, Any]]],
    factors: List[float],
) -> str:
    """Format aggregate results into a clear markdown-style table."""
    lines = []
    lines.append(
        "| Model | Factor | Completed | Crashes | Timeouts (Stat) | Completion Rate | Avg Lap Time | Mean Progress | Median Progress |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for model_label, factor_dict in summary_by_model_and_factor.items():
        for factor in factors:
            agg = factor_dict[factor]
            rate_str = f"{agg['completion_rate'] * 100:5.1f}%"
            lap_str = f"{agg['avg_lap_time']:6.1f}" if agg["avg_lap_time"] is not None else "N/A"
            mean_prog_str = f"{agg['avg_progress']:.4f}"
            med_prog_str = f"{agg.get('median_progress', 0.0):.4f}"
            to_str = f"{agg['timeouts']} ({agg.get('stationary', 0)})"
            lines.append(
                f"| {model_label:<20} | {factor:6.2f} | {agg['completed']:9d} | "
                f"{agg['crashes']:7d} | {to_str:>15} | {rate_str:>15} | "
                f"{lap_str:>12} | {mean_prog_str:>13} | {med_prog_str:>15} |"
            )

    return "\n".join(lines)


def format_paired_table(
    paired_data: List[Dict[str, Any]],
    factor: float,
    baseline_label: str = "Baseline",
    candidate_label: str = "Candidate",
) -> str:
    """Format per-seed paired comparison for a single hairpin factor."""
    lines = []
    lines.append(f"\n--- Factor {factor} Per-Seed Comparison: {baseline_label} vs {candidate_label} ---")
    lines.append(
        "| Seed | Baseline Outcome | Base Progress | Base Time | Candidate Outcome | Cand Progress | Cand Time | Change |"
    )
    lines.append(
        "|---:|:---|---:|---:|:---|---:|---:|:---|"
    )

    for row in paired_data:
        if row["hairpin_factor"] != factor:
            continue

        b_out = row["baseline_outcome"]
        c_out = row["candidate_outcome"]
        b_prog = f"{row['baseline_progress']:.3f}"
        c_prog = f"{row['candidate_progress']:.3f}"
        b_time = str(row["baseline_lap_time"]) if row["baseline_lap_time"] is not None else "-"
        c_time = str(row["candidate_lap_time"]) if row["candidate_lap_time"] is not None else "-"

        if b_out != "completed" and c_out == "completed":
            change = "IMPROVED (+)"
        elif b_out == "completed" and c_out != "completed":
            change = "REGRESSED (-)"
        elif b_out == c_out:
            change = f"SAME ({b_out})"
        else:
            change = f"CHANGED ({b_out}->{c_out})"

        lines.append(
            f"| {row['seed']:4d} | {b_out:<16} | {b_prog:>13} | {b_time:>9} | "
            f"{c_out:<17} | {c_prog:>13} | {c_time:>9} | {change:<17} |"
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare SAC checkpoints on unseen hairpins."
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE_CKPT,
        help=f"Path to baseline checkpoint (default: {DEFAULT_BASELINE_CKPT})",
    )
    parser.add_argument(
        "--candidate",
        default=DEFAULT_CANDIDATE_CKPT,
        help=f"Path to candidate checkpoint (default: {DEFAULT_CANDIDATE_CKPT})",
    )
    parser.add_argument(
        "--baseline-label",
        default=None,
        help="Optional custom display label for baseline model",
    )
    parser.add_argument(
        "--candidate-label",
        default=None,
        help="Optional custom display label for candidate model",
    )
    parser.add_argument(
        "--baseline-rays",
        type=float,
        nargs="+",
        default=None,
        help="Override ray angles for baseline",
    )
    parser.add_argument(
        "--candidate-rays",
        type=float,
        nargs="+",
        default=None,
        help="Override ray angles for candidate",
    )
    parser.add_argument(
        "--baseline-max-dist",
        type=float,
        default=None,
        help="Override max ray distance for baseline",
    )
    parser.add_argument(
        "--candidate-max-dist",
        type=float,
        default=None,
        help="Override max ray distance for candidate",
    )
    parser.add_argument(
        "--factors",
        type=float,
        nargs="+",
        default=DEFAULT_FACTORS,
        help=f"Hairpin pinch factors to test (default: {DEFAULT_FACTORS})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Track seeds to test (default: 1..20)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum steps per episode (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--csv-out",
        default="analysis/results/hairpin_comparison.csv",
        help="Filepath for episode-level CSV output",
    )
    parser.add_argument(
        "--json-out",
        default="analysis/results/hairpin_comparison_summary.json",
        help="Filepath for aggregate JSON summary output",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Run in diagnostic mode to record step-by-step telemetry trace for a single run",
    )
    parser.add_argument(
        "--diagnostic-checkpoint",
        default=None,
        help="Checkpoint to evaluate in diagnostic mode (defaults to --candidate if not provided)",
    )
    parser.add_argument(
        "--diagnostic-factor",
        type=float,
        default=0.35,
        help="Hairpin pinch factor for diagnostic mode (default: 0.35)",
    )
    parser.add_argument(
        "--diagnostic-seed",
        type=int,
        default=16,
        help="Track seed for diagnostic mode (default: 16)",
    )
    parser.add_argument(
        "--diagnostic-csv-out",
        default="analysis/results/hairpin_trace.csv",
        help="CSV filepath to save the diagnostic step trace (default: analysis/results/hairpin_trace.csv)",
    )

    args = parser.parse_args()

    # Diagnostic mode: run single trace and exit
    if args.diagnostic:
        diag_ckpt = args.diagnostic_checkpoint or args.candidate
        actor, obs_dim, ray_angles, ray_max_dist = inspect_and_load_actor(
            diag_ckpt,
            override_ray_angles=args.candidate_rays,
            override_max_dist=args.candidate_max_dist,
        )
        run_diagnostic_episode(
            actor=actor,
            seed=args.diagnostic_seed,
            hairpin_factor=args.diagnostic_factor,
            ray_angles=ray_angles,
            obs_dim=obs_dim,
            checkpoint_path=diag_ckpt,
            max_dist=ray_max_dist,
            max_steps=args.max_steps,
            csv_out=args.diagnostic_csv_out,
        )
        return 0

    # 1. Load and inspect checkpoints
    print("=" * 80)
    print("UNSEEN HAIRPIN ZERO-SHOT EVALUATION")
    print("=" * 80)

    print(f"Loading baseline:  {args.baseline}")
    baseline_actor, baseline_dim, baseline_rays, baseline_max_dist = inspect_and_load_actor(
        args.baseline,
        override_ray_angles=args.baseline_rays,
        override_max_dist=args.baseline_max_dist,
    )
    print(f"  -> Detected obs_dim={baseline_dim}, rays={len(baseline_rays)} angles: {baseline_rays}, max_dist={baseline_max_dist}")

    print(f"Loading candidate: {args.candidate}")
    candidate_actor, candidate_dim, candidate_rays, candidate_max_dist = inspect_and_load_actor(
        args.candidate,
        override_ray_angles=args.candidate_rays,
        override_max_dist=args.candidate_max_dist,
    )
    print(f"  -> Detected obs_dim={candidate_dim}, rays={len(candidate_rays)} angles: {candidate_rays}, max_dist={candidate_max_dist}")

    b_label = args.baseline_label or f"Baseline ({len(baseline_rays)}ray-{int(baseline_max_dist)}px)"
    c_label = args.candidate_label or f"Candidate ({len(candidate_rays)}ray-{int(candidate_max_dist)}px)"

    models = [
        (b_label, args.baseline, baseline_actor, baseline_dim, baseline_rays, baseline_max_dist),
        (c_label, args.candidate, candidate_actor, candidate_dim, candidate_rays, candidate_max_dist),
    ]

    # 2. Run evaluations
    all_episodes: List[Dict[str, Any]] = []
    # Index results by (model_label, factor, seed) for easy paired comparison
    indexed_results: Dict[Tuple[str, float, int], Dict[str, Any]] = {}

    total_evals = len(args.factors) * len(args.seeds) * len(models)
    print(f"\nRunning {total_evals} deterministic rollouts ({len(args.factors)} factors x {len(args.seeds)} seeds x {len(models)} models)...")

    for factor in args.factors:
        for seed in args.seeds:
            for label, path, actor, obs_dim, ray_angles, ray_max_dist in models:
                res = run_episode(
                    actor=actor,
                    seed=seed,
                    hairpin_factor=factor,
                    ray_angles=ray_angles,
                    max_dist=ray_max_dist,
                    max_steps=args.max_steps,
                )
                res["model_label"] = label
                res["checkpoint_path"] = path
                res["obs_dim"] = obs_dim
                res["obs_ray_max_dist"] = ray_max_dist

                all_episodes.append(res)
                indexed_results[(label, factor, seed)] = res

    # 3. Compute aggregate summaries
    summary_by_model_and_factor: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for label, _, _, _, _, _ in models:
        summary_by_model_and_factor[label] = {}
        for factor in args.factors:
            subset = [
                e for e in all_episodes
                if e["model_label"] == label and e["hairpin_factor"] == factor
            ]
            summary_by_model_and_factor[label][factor] = compute_aggregate(subset)

    # 4. Compute paired comparisons
    paired_data: List[Dict[str, Any]] = []
    for factor in args.factors:
        for seed in args.seeds:
            b_res = indexed_results[(b_label, factor, seed)]
            c_res = indexed_results[(c_label, factor, seed)]
            paired_data.append({
                "hairpin_factor": factor,
                "seed": seed,
                "baseline_outcome": b_res["outcome"],
                "baseline_progress": b_res["lap_progress"],
                "baseline_lap_time": b_res["lap_time"],
                "candidate_outcome": c_res["outcome"],
                "candidate_progress": c_res["lap_progress"],
                "candidate_lap_time": c_res["lap_time"],
            })

    # 5. Display formatted tables
    print("\n" + "=" * 80)
    print("AGGREGATE EVALUATION SUMMARY")
    print("=" * 80)
    print(format_summary_table(summary_by_model_and_factor, args.factors))

    print("\n" + "=" * 80)
    print("PAIRED PER-SEED COMPARISONS")
    print("=" * 80)
    for factor in args.factors:
        print(format_paired_table(paired_data, factor, baseline_label=b_label, candidate_label=c_label))

    # 6. Save outputs
    # Ensure directory exists
    csv_dir = os.path.dirname(args.csv_out)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    json_dir = os.path.dirname(args.json_out)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)

    # Write episode-level CSV
    csv_fieldnames = [
        "model_label",
        "checkpoint_path",
        "obs_dim",
        "obs_ray_max_dist",
        "hairpin_factor",
        "seed",
        "outcome",
        "completed",
        "crashed",
        "timeout",
        "stationary",
        "max_speed",
        "steps",
        "lap_time",
        "lap_progress",
    ]
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for e in all_episodes:
            writer.writerow({k: e.get(k) for k in csv_fieldnames})
    print(f"\n[Saved] Episode-level details -> {args.csv_out}")

    # Write aggregate JSON
    json_payload = {
        "metadata": {
            "baseline": args.baseline,
            "candidate": args.candidate,
            "baseline_label": b_label,
            "candidate_label": c_label,
            "factors": args.factors,
            "seeds": args.seeds,
            "max_steps": args.max_steps,
            "track_kwargs": BASE_TRACK_KWARGS,
        },
        "aggregate_summary": {
            model_label: {
                str(factor): summary_by_model_and_factor[model_label][factor]
                for factor in args.factors
            }
            for model_label in summary_by_model_and_factor
        },
        "paired_per_seed": paired_data,
    }
    with open(args.json_out, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"[Saved] Aggregate summary   -> {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
