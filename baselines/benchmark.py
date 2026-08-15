"""Benchmark both baselines on the 5 fixed tracks with lap timing.

Runs random-policy and pure-pursuit baselines, records the step at which
the first full lap is completed, and outputs a comparison table + JSON file.

Usage:
    python -m baselines.benchmark
"""

import json
import time

from env.environment import RacingEnv
from baselines.pure_pursuit import choose_action as pp_choose_action

# ─── Configuration ───────────────────────────────────────────────────────────

TRACK_SEEDS = [101, 202, 303, 404, 505]
MAX_STEPS = 2000


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_episode(track_seed, policy_fn):
    """Run a single episode, returning metrics with first-lap timing.

    policy_fn(env, step) -> action
    """
    env = RacingEnv(max_steps=MAX_STEPS, verbose=False)
    obs, info = env.reset(options={"track_seed": track_seed})

    total_reward = 0.0
    first_lap_step = None

    for step in range(MAX_STEPS):
        action = policy_fn(env, step)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Detect first full lap completion.
        if first_lap_step is None and env.lap_progress >= 1.0:
            first_lap_step = env.step_count

        if terminated or truncated:
            break

    result = {
        "track_seed": track_seed,
        "steps": env.step_count,
        "lap_progress": round(env.lap_progress, 4),
        "first_lap_step": first_lap_step,
        "total_reward": round(total_reward, 4),
        "crashed": info.get("crashed", False),
    }

    env.close()
    return result


# ─── Policy functions ────────────────────────────────────────────────────────

def random_policy(env, step):
    return env.action_space.sample()


def pure_pursuit_policy(env, step):
    action, _ = pp_choose_action(env, step_count=step)
    return action


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    policies = [
        ("random", random_policy),
        ("pure_pursuit", pure_pursuit_policy),
    ]

    all_results = {}

    for policy_name, policy_fn in policies:
        print(f"\n{'─' * 60}")
        print(f"  Running: {policy_name}")
        print(f"{'─' * 60}")

        results = []
        for seed in TRACK_SEEDS:
            r = run_episode(seed, policy_fn)
            r["policy"] = policy_name

            lap_str = (
                f"step {r['first_lap_step']}"
                if r["first_lap_step"] is not None
                else "no lap"
            )
            status = (
                "CRASH" if r["crashed"]
                else ("LAP" if r["first_lap_step"] else "TIMEOUT")
            )

            print(
                f"  Track {seed}: "
                f"first_lap={lap_str:>10s}  "
                f"progress={r['lap_progress']:.4f}  "
                f"reward={r['total_reward']:+8.4f}  "
                f"[{status}]"
            )
            results.append(r)

        all_results[policy_name] = results

    # ── Comparison table ─────────────────────────────────────────────────

    print(f"\n{'=' * 72}")
    print("  BASELINE COMPARISON — Fixed seeds [101, 202, 303, 404, 505]")
    print(f"{'=' * 72}")
    print(
        f"{'Track':>6s}  │  "
        f"{'Random':^22s}  │  "
        f"{'Pure Pursuit':^22s}"
    )
    print(
        f"{'':>6s}  │  "
        f"{'1st Lap':>8s}  {'Progress':>8s}  │  "
        f"{'1st Lap':>8s}  {'Progress':>8s}"
    )
    print(f"{'─' * 72}")

    for i, seed in enumerate(TRACK_SEEDS):
        rr = all_results["random"][i]
        pp = all_results["pure_pursuit"][i]

        rr_lap = str(rr["first_lap_step"]) if rr["first_lap_step"] else "—"
        pp_lap = str(pp["first_lap_step"]) if pp["first_lap_step"] else "—"

        print(
            f"  {seed:>4d}  │  "
            f"{rr_lap:>8s}  {rr['lap_progress']:>8.4f}  │  "
            f"{pp_lap:>8s}  {pp['lap_progress']:>8.4f}"
        )

    # Averages
    rr_avg_prog = sum(r["lap_progress"] for r in all_results["random"]) / 5
    pp_avg_prog = sum(r["lap_progress"] for r in all_results["pure_pursuit"]) / 5
    rr_avg_rew = sum(r["total_reward"] for r in all_results["random"]) / 5
    pp_avg_rew = sum(r["total_reward"] for r in all_results["pure_pursuit"]) / 5
    pp_laps = sum(1 for r in all_results["pure_pursuit"] if r["first_lap_step"])
    rr_laps = sum(1 for r in all_results["random"] if r["first_lap_step"])
    pp_first_steps = [
        r["first_lap_step"]
        for r in all_results["pure_pursuit"]
        if r["first_lap_step"]
    ]
    pp_avg_first = (
        sum(pp_first_steps) / len(pp_first_steps) if pp_first_steps else None
    )

    print(f"{'─' * 72}")
    print(
        f"  {'Avg':>4s}  │  "
        f"{'':>8s}  {rr_avg_prog:>8.4f}  │  "
        f"{'':>8s}  {pp_avg_prog:>8.4f}"
    )
    print(f"{'─' * 72}")
    print(f"  Random:       {rr_laps}/5 laps, avg progress {rr_avg_prog:.4f}, avg reward {rr_avg_rew:+.2f}")
    print(f"  Pure Pursuit: {pp_laps}/5 laps, avg progress {pp_avg_prog:.4f}, avg reward {pp_avg_rew:+.2f}")
    if pp_avg_first:
        print(f"                avg first-lap step: {pp_avg_first:.0f}")
    print()

    # ── Save to JSON ─────────────────────────────────────────────────────

    output = {
        "track_seeds": TRACK_SEEDS,
        "max_steps": MAX_STEPS,
        "results": all_results,
        "summary": {
            "random": {
                "laps_completed": rr_laps,
                "avg_progress": round(rr_avg_prog, 4),
                "avg_reward": round(rr_avg_rew, 4),
            },
            "pure_pursuit": {
                "laps_completed": pp_laps,
                "avg_progress": round(pp_avg_prog, 4),
                "avg_reward": round(pp_avg_rew, 4),
                "avg_first_lap_step": round(pp_avg_first) if pp_avg_first else None,
            },
        },
    }

    out_path = "baselines/results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
