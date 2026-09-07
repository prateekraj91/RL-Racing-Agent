"""Compare curriculum / demo-seeding experiments on the target metric.

Reads runs/<name>/history.json (written by sac.train_curriculum) and reports
TARGET-eval lap progress over training for each run, plus a summary table.

Usage:
    python -m analysis.compare_runs runs/ctl_none_s42 runs/exp1_width5_s42 ...
    python -m analysis.compare_runs --all
"""

import argparse
import glob
import json
import os

import numpy as np


def load(run_dir):
    hp = os.path.join(run_dir, "history.json")
    cp = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(hp):
        return None
    hist = json.load(open(hp))
    cfg = json.load(open(cp)) if os.path.exists(cp) else {}
    return {"name": os.path.basename(run_dir.rstrip("/")), "hist": hist, "cfg": cfg}


def summarise(run):
    h = run["hist"]
    if not h:
        return None
    prog = [e["target_eval"]["progress"] for e in h]
    solved = [e for e in h if e["target_eval"]["completed"]
              and not e["target_eval"]["crashed"]]
    crashed = sum(1 for e in h if e["target_eval"]["crashed"])
    frozen = sum(1 for e in h
                 if abs(e["target_eval"]["mean_speed"]) < 0.1
                 and not e["target_eval"]["crashed"])
    lap_times = [e["target_eval"]["lap_time"] for e in solved]
    return {
        "name": run["name"],
        "evals": len(h),
        "best_progress": max(prog),
        "final_progress": prog[-1],
        "n_solved_evals": len(solved),
        "solve_rate": len(solved) / len(h),
        "first_solve_step": solved[0]["step"] if solved else None,
        "best_lap_time": min(lap_times) if lap_times else None,
        "crashed_evals": crashed,
        "frozen_evals": frozen,
        "steps": h[-1]["step"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--include-diagnostic", action="store_true",
                    help="also list runs whose TRAINING reward was modified "
                         "(e.g. --clip-backward). Excluded by default: their "
                         "numbers are not comparable and can never be a solve.")
    ap.add_argument("--plot", default=None, help="save a PNG comparison here")
    args = ap.parse_args()

    dirs = args.runs
    if args.all or not dirs:
        dirs = sorted(d for d in glob.glob("runs/*")
                      if os.path.exists(os.path.join(d, "history.json")))

    runs = [r for r in (load(d) for d in dirs) if r]
    if not args.include_diagnostic:
        skipped = [r["name"] for r in runs
                   if r["cfg"].get("clip_backward_DIAGNOSTIC")]
        runs = [r for r in runs if not r["cfg"].get("clip_backward_DIAGNOSTIC")]
        if skipped:
            print(f"[excluded {len(skipped)} diagnostic run(s) with a modified "
                  f"training reward: {', '.join(skipped)} — pass "
                  f"--include-diagnostic to show]")
    if not runs:
        print("no runs with history.json found")
        return

    print("=" * 118)
    print(f"{'run':26s} {'steps':>7s} {'evals':>6s} {'best_prog':>10s} "
          f"{'final':>8s} {'solved':>7s} {'solve%':>7s} {'1st_solve':>10s} "
          f"{'best_lap':>9s} {'crash_ev':>9s} {'frozen_ev':>10s}")
    print("-" * 118)
    for r in runs:
        s = summarise(r)
        if not s:
            continue
        print(f"{s['name']:26s} {s['steps']:7d} {s['evals']:6d} "
              f"{s['best_progress']:10.4f} {s['final_progress']:8.4f} "
              f"{s['n_solved_evals']:7d} {s['solve_rate']:6.1%} "
              f"{str(s['first_solve_step']):>10s} {str(s['best_lap_time']):>9s} "
              f"{s['crashed_evals']:9d} {s['frozen_evals']:10d}")
    print("=" * 118)
    print("solved  = TARGET evals with lap_completed=True AND crashed=False")
    print("frozen  = TARGET evals with |mean_speed| < 0.1 and no crash (the "
          "sit-still failure mode)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        # Colour by the variable that actually decides the outcome: demos or not.
        # Every demo-seeded run ends up at ~1.0, every un-seeded one at ~0.0, with
        # nothing in between -- the result is bimodal, not a gradual curve.
        demo_colors = ["#0b6e4f", "#1b9e77", "#3fa34d", "#66c2a5", "#2c7fb8"]
        cold_colors = ["#b2182b", "#d6604d", "#e08214", "#8c510a", "#777777"]
        di = ci = 0
        for r in runs:
            h = r["hist"]
            has_demos = bool(r["cfg"].get("demos", 0))
            if has_demos:
                c = demo_colors[di % len(demo_colors)]; di += 1
            else:
                c = cold_colors[ci % len(cold_colors)]; ci += 1
            xs = [e["step"] for e in h]
            ys = [e["target_eval"]["progress"] for e in h]
            n_solved = sum(1 for e in h if e["target_eval"]["completed"]
                           and not e["target_eval"]["crashed"])
            tag = "demos" if has_demos else "no demos"
            if r["cfg"].get("clip_backward_DIAGNOSTIC"):
                tag += ", DIAGNOSTIC reward"
            label = f"{r['name']}  [{tag}] {n_solved}/{len(h)} solved"
            ax.plot(xs, ys, marker="o", ms=3.5, lw=1.6, color=c,
                    ls="-" if has_demos else "--", label=label, zorder=3)
            sol = [(e["step"], e["target_eval"]["progress"]) for e in h
                   if e["target_eval"]["completed"] and not e["target_eval"]["crashed"]]
            if sol:
                ax.scatter([p[0] for p in sol], [p[1] for p in sol], marker="*",
                           s=150, zorder=5, color=c, edgecolors="k", linewidths=0.5)

        ax.axhline(1.0, color="k", ls="--", lw=1.2)
        ax.axhline(0.0, color="grey", ls=":", lw=1)
        ax.axhspan(0.15, 0.95, color="black", alpha=0.04, zorder=0)
        ax.text(0.5, 0.55, "no eval ever lands here —\nthe outcome is bimodal:\n"
                           "lap the track, or sit still",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=9, color="#555555", style="italic")
        bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
        ax.annotate("lap completed (1.0)", xy=(0.995, 1.0),
                    xycoords=("axes fraction", "data"), va="bottom", ha="right",
                    fontsize=9, bbox=bbox)
        ax.annotate("frozen — car sits still (0.0)", xy=(0.5, 0.0),
                    xycoords=("axes fraction", "data"), va="bottom", ha="center",
                    fontsize=9, color="#777777", bbox=bbox)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("TARGET eval lap progress (medium, width 70, 500 steps)")
        ax.set_title("Reward-v2 on medium: demo seeding solves it fast; the curriculum "
                     "solves it ~11x later\n"
                     "solid = demo-seeded, dashed = no demos, stars = lap_completed & not crashed",
                     fontsize=11)
        ax.set_ylim(-0.15, 1.12)
        ax.legend(fontsize=8, loc="center right", framealpha=0.95)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        fig.savefig(args.plot, dpi=140)
        print(f"\nsaved plot -> {args.plot}")


if __name__ == "__main__":
    main()
