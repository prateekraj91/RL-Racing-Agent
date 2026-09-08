"""Focused hyperparameter sweep: lr x entropy_target on the demo-seeded trainer.

Runs a small grid of full training runs (reusing sac.train_curriculum), reads
each run's run_config.json for its result, and prints a comparison table ranked
by lap time among STABLE configs (those that solve most evals).

Usage:
    python -m sac.sweep
"""

import itertools
import json
import os
import subprocess
import sys

# ─── The grid: the two knobs most tied to our failure modes ──────────────────
LR_GRID = [1e-4, 3e-4, 1e-3]              # stability axis
ENTROPY_GRID = [-1.0, -2.0, -4.0]         # explore/exploit axis

# ─── Fixed sweep settings (small, to keep the sweep tractable on CPU) ────────
SEED = 42
DEMOS = 12                # fewer demos than 40 -> faster collection per run
STEPS_SCALE = 0.1        # shorter runs; enough to see solve + a few evals
CURRICULUM = "none"       # demo-seeding alone solves; no curriculum needed
EVAL_EVERY = 2000


def run_one(lr, ent):
    """Launch one training run with these hyperparameters, return its manifest."""
    name = f"sweep_lr{lr:.0e}_ent{ent:+.1f}".replace("+", "p").replace("-", "m")
    cmd = [
        sys.executable, "-m", "sac.train_curriculum",
        "--curriculum", CURRICULUM,
        "--seed", str(SEED),
        "--name", name,
        "--demos", str(DEMOS),
        "--steps-scale", str(STEPS_SCALE),
        "--eval-every", str(EVAL_EVERY),
        "--lr", str(lr),
        "--entropy-target", str(ent),
    ]
    print(f"\n{'='*80}\nRUN  lr={lr:.0e}  entropy_target={ent:+.1f}  -> {name}\n{'='*80}")
    subprocess.run(cmd, check=True)

    run_dir = os.path.join("runs", name)
    with open(os.path.join(run_dir, "run_config.json")) as f:
        manifest = json.load(f)

    # Count solved evals from the REAL per-eval record (history.json), not from
    # a summary field. An eval "solved" iff it completed the lap without crashing.
    with open(os.path.join(run_dir, "history.json")) as f:
        history = json.load(f)
    n_solved = sum(
        1 for h in history
        if h.get("target_eval", {}).get("completed") and not h.get("target_eval", {}).get("crashed")
    )
    manifest["_n_solved_evals"] = n_solved   # inject the derived, trustworthy count
    return manifest


def main():
    results = []
    for lr, ent in itertools.product(LR_GRID, ENTROPY_GRID):
        m = run_one(lr, ent)
        solved_eval = m.get("solved_eval") or {}
        results.append({
            "lr": lr,
            "entropy_target": ent,
            "solved_at": m.get("solved_at_step"),
            "n_solved_evals": m.get("_n_solved_evals", 0),
            "best_progress": m.get("best_target_progress", 0.0),
            "lap_time": solved_eval.get("lap_time"),
        })

    # ─── Ranked table ───
    print(f"\n{'='*80}\nSWEEP RESULTS  ({len(results)} configs)\n{'='*80}")
    print(f"{'lr':>8} {'entropy':>8} {'solved_at':>10} {'n_solved':>9} "
          f"{'lap_time':>9} {'best_prog':>10}")
    print("-" * 80)

    # Sort: stable-and-fast first. A config is "stable" if it solved >=2 evals.
    # Among stable ones, lower lap_time is better. Unstable/unsolved sink to bottom.
    def sort_key(r):
        stable = (r["n_solved_evals"] or 0) >= 2
        lap = r["lap_time"] if r["lap_time"] is not None else 10**9
        return (not stable, lap)   # stable first, then fastest lap

    for r in sorted(results, key=sort_key):
        lap = r["lap_time"] if r["lap_time"] is not None else "—"
        solved = r["solved_at"] if r["solved_at"] is not None else "—"
        print(f"{r['lr']:>8.0e} {r['entropy_target']:>+8.1f} {str(solved):>10} "
              f"{r['n_solved_evals']:>9} {str(lap):>9} {r['best_progress']:>10.4f}")

    # ─── The pick ───
    stable = [r for r in results if (r["n_solved_evals"] or 0) >= 2 and r["lap_time"]]
    if stable:
        best = min(stable, key=lambda r: r["lap_time"])
        print(f"\nBEST STABLE CONFIG: lr={best['lr']:.0e} "
              f"entropy_target={best['entropy_target']:+.1f} "
              f"-> lap_time={best['lap_time']}, solved {best['n_solved_evals']} evals")
    else:
        print("\nNo config solved stably at this budget — widen steps-scale or grid.")

    with open("runs/sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote runs/sweep_results.json")


if __name__ == "__main__":
    main()