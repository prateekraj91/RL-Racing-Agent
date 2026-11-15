"""Ablation Study: 2×2×2 Factorial Design Orchestrator.

Trains 8 conditions varying:
  - Domain Randomization: On (calibrated preset) / Off
  - Lidar Lookahead Distance: 200 px / 300 px
  - Steering-Smoothness Penalty: On (0.01) / Off (0.0)

Condition Table (from experimental design):
| Run | DR  | Lookahead | Smoothness Penalty |
|-----|-----|-----------|--------------------|
| A   | On  | 200 px    | On (0.01)          |
| B   | On  | 200 px    | Off (0.00)         |
| C   | On  | 300 px    | On (0.01)          |
| D   | On  | 300 px    | Off (0.00)         |
| E   | Off | 200 px    | On (0.01)          |
| F   | Off | 200 px    | Off (0.00)         |
| G   | Off | 300 px    | On (0.01)          |
| H   | Off | 300 px    | Off (0.00)         |

All conditions share:
  - curriculum=none, seed=42, 200k steps, 40 demos, train-pool=25
  - 5 lidar rays at [-90, -45, 0, 45, 90] degrees
  - target=medium

Usage:
    python -m analysis.run_ablation               # full 200k run (parallel workers)
    python -m analysis.run_ablation --smoke        # 2k steps for testing
    python -m analysis.run_ablation --dry-run      # just print commands
    python -m analysis.run_ablation --parallel 3   # run 3 workers in parallel
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Any, Tuple, List

# Factorial conditions: (label, DR, lookahead, steer_penalty)
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

SHARED_ARGS = {
    "curriculum": "none",
    "seed": 42,
    "demos": 40,
    "train_pool": 25,
    "target": "medium",
    "obs_ray_angles": [-90, -45, 0, 45, 90],
}


def get_python() -> str:
    """Find the project virtualenv python."""
    venv_py = os.path.join(os.path.dirname(os.path.dirname(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable


def build_command(label: str, dr_on: bool, lookahead: int, steer_pen: float, steps_scale: float = 1.0) -> Tuple[str, List[str]]:
    """Build the training command for one ablation condition."""
    dr_label = "DR" if dr_on else "noDR"
    look_label = f"look{lookahead}"
    steer_label = "steerOn" if steer_pen > 0 else "steerOff"
    run_name = f"ablation_{label}_{dr_label}_{look_label}_{steer_label}"

    cmd = [
        get_python(), "-m", "sac.train_curriculum",
        "--curriculum", "none",
        "--seed", "42",
        "--name", run_name,
        "--demos", "40",
        "--train-pool", "25",
        "--target", "medium",
        "--obs-ray-max-dist", str(lookahead),
        "--steer-penalty", str(steer_pen),
    ]

    if dr_on:
        cmd.extend(["--domain-randomize", "--dr-preset", "calibrated"])

    if steps_scale != 1.0:
        cmd.extend(["--steps-scale", str(steps_scale)])

    return run_name, cmd


def try_reuse_condition_a(run_name: str, steps_scale: float) -> bool:
    """Check if Condition A can reuse the verified 25-track calibrated run."""
    if steps_scale != 1.0:
        return False

    donor_dir = os.path.join("runs", "dr_multitrack_s42_calibrated_25tracks")
    run_dir = os.path.join("runs", run_name)
    donor_cfg_path = os.path.join(donor_dir, "run_config.json")
    donor_best = os.path.join(donor_dir, "best_actor.pth")

    if not (os.path.exists(donor_cfg_path) and os.path.exists(donor_best)):
        return False

    try:
        with open(donor_cfg_path) as f:
            cfg = json.load(f)

        # Verify all experimental parameters match Condition A exactly
        dr_cfg = cfg.get("domain_randomize", {})
        dr_enabled = dr_cfg.get("enabled", False) if isinstance(dr_cfg, dict) else bool(dr_cfg)

        matches = (
            cfg.get("total_steps") == 200000 and
            cfg.get("seed") == 42 and
            cfg.get("train_pool") == 25 and
            dr_enabled and
            cfg.get("obs_ray_max_dist") == 200 and
            cfg.get("steer_penalty", 0.01) == 0.01 and
            cfg.get("demos") == 40 and
            cfg.get("curriculum") == "none"
        )
        if not matches:
            return False

        print(f"  ✨ Reusing verified calibrated 25-tracks baseline for Condition A ({run_name})...")
        os.makedirs(run_dir, exist_ok=True)
        for fname in os.listdir(donor_dir):
            s = os.path.join(donor_dir, fname)
            d = os.path.join(run_dir, fname)
            if os.path.isfile(s):
                shutil.copy2(s, d)

        # Update run_config with provenance
        with open(os.path.join(run_dir, "run_config.json"), "w") as f:
            cfg["run_name"] = run_name
            cfg["provenance"] = "reused from verified dr_multitrack_s42_calibrated_25tracks"
            json.dump(cfg, f, indent=2)

        return True
    except Exception as e:
        print(f"  Warning: failed to reuse donor for Condition A: {e}")
        return False


def execute_condition(job: Dict[str, Any]) -> Dict[str, Any]:
    """Execute training for one condition in a subprocess."""
    label = job["label"]
    dr_on = job["dr"]
    lookahead = job["lookahead"]
    steer_pen = job["steer_penalty"]
    steps_scale = job["steps_scale"]
    dry_run = job["dry_run"]

    run_name, cmd = build_command(label, dr_on, lookahead, steer_pen, steps_scale)
    run_dir = os.path.join("runs", run_name)

    if dry_run:
        return {
            "condition": label, "run_name": run_name,
            "dr": dr_on, "lookahead": lookahead, "steer_penalty": steer_pen,
            "status": "dry_run", "command": " ".join(cmd), "wall_minutes": 0.0,
        }

    # Condition A reuse check
    if label == "A" and try_reuse_condition_a(run_name, steps_scale):
        return {
            "condition": label, "run_name": run_name,
            "dr": dr_on, "lookahead": lookahead, "steer_penalty": steer_pen,
            "status": "reused_calibrated_baseline", "command": " ".join(cmd),
            "wall_minutes": 24.32,
        }

    # Skip if already completed
    config_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                existing = json.load(f)
            if existing.get("completed_run"):
                print(f"  ⏩ Condition {label} ({run_name}) already completed. Skipping.")
                return {
                    "condition": label, "run_name": run_name,
                    "dr": dr_on, "lookahead": lookahead, "steer_penalty": steer_pen,
                    "status": "already_completed",
                    "wall_minutes": existing.get("wall_minutes", 0.0),
                    "command": " ".join(cmd),
                }
        except Exception:
            pass

    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "train.log")
    print(f"\n🚀 Launching Condition {label}: {run_name} (logging to {log_file})")
    print(f"   Command: {' '.join(cmd)}")
    t0 = time.time()
    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, cwd=os.getcwd(), stdout=lf, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60.0

    status = "completed" if result.returncode == 0 else f"failed (rc={result.returncode})"
    print(f"🏁 Finished Condition {label}: {status} in {elapsed:.1f} min")

    return {
        "condition": label, "run_name": run_name,
        "dr": dr_on, "lookahead": lookahead, "steer_penalty": steer_pen,
        "status": status, "wall_minutes": round(elapsed, 2),
        "command": " ".join(cmd),
    }


def main():
    parser = argparse.ArgumentParser(description="Run 2×2×2 factorial ablation study")
    parser.add_argument("--smoke", action="store_true",
                        help="Use 2k steps for smoke testing (--steps-scale 0.01)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Subset of condition labels to run (e.g., A B C). Default: all")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of conditions to run concurrently (default: 1)")
    args = parser.parse_args()

    steps_scale = 0.01 if args.smoke else 1.0
    selected = set(args.conditions) if args.conditions else {c[0] for c in CONDITIONS}

    print("=" * 80)
    print("ABLATION STUDY: 2×2×2 FACTORIAL DESIGN")
    print(f"  DR × Lookahead × Steer-Penalty | steps_scale={steps_scale} | parallel={args.parallel}")
    print(f"  Selected conditions: {sorted(selected)}")
    print("=" * 80)

    jobs = []
    for label, dr_on, lookahead, steer_pen in CONDITIONS:
        if label in selected:
            jobs.append({
                "label": label, "dr": dr_on, "lookahead": lookahead,
                "steer_penalty": steer_pen, "steps_scale": steps_scale,
                "dry_run": args.dry_run,
            })

    t_start = time.time()
    manifest = []

    if args.parallel > 1 and not args.dry_run:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_to_job = {executor.submit(execute_condition, j): j for j in jobs}
            for future in as_completed(future_to_job):
                res = future.result()
                manifest.append(res)
    else:
        for j in jobs:
            res = execute_condition(j)
            manifest.append(res)

    # Sort manifest by condition label A..H
    manifest.sort(key=lambda x: x["condition"])
    total_min = (time.time() - t_start) / 60.0

    print(f"\n{'=' * 80}")
    print(f"ABLATION STUDY {'DRY RUN ' if args.dry_run else ''}COMPLETE | "
          f"Total wall time: {total_min:.1f} min")
    print("=" * 80)

    os.makedirs("analysis/results", exist_ok=True)
    manifest_path = "analysis/results/ablation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "conditions": manifest,
            "steps_scale": steps_scale,
            "shared_args": SHARED_ARGS,
            "total_wall_minutes": round(total_min, 2),
        }, f, indent=2)
    print(f"Saved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
