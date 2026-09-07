"""Three-way eval: v1 (best_actor.pth) vs v2 (solved) vs pure_pursuit.

Reuses baselines/benchmark.py's metrics/reporting; uses a runner that mirrors
run_episode but keeps the observation in scope so a neural-net actor can see it.
Metrics are computed identically, so the comparison is apples-to-apples.

Usage:
    python -m baselines.benchmark_agents
"""

import json
import numpy as np
import torch
from datetime import datetime, timezone

from env.environment import RacingEnv
from baselines.benchmark import summarise, print_block, MAX_STEPS, run_episode, pure_pursuit_policy
from baselines.pure_pursuit import TRACK_CONFIGS, TRACK_SEEDS
from sac.agent import SACAgent


# ─── Load an actor, return a deterministic action-from-obs function ──────────

def load_actor(checkpoint_path):
    agent = SACAgent()
    agent.actor.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    agent.actor.eval()

    def act(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _ = agent.actor(obs_t)
            return torch.tanh(mean).numpy()[0]     # deterministic, same as eval

    return act


# ─── Agent runner: mirrors run_episode but feeds the obs to the actor ────────

def run_agent_episode(track_seed, act_fn, track_kwargs):
    env = RacingEnv(max_steps=MAX_STEPS, verbose=False, track_kwargs=track_kwargs)
    obs, info = env.reset(options={"track_seed": track_seed})

    total_reward = 0.0
    off_track_steps = 0

    for step in range(MAX_STEPS):
        action = act_fn(obs)                       # <-- actor sees the observation
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["crashed"]:
            off_track_steps += 1
        if terminated or truncated:
            break

    steps = env.step_count
    if env.lap_completed:
        outcome = "LAP"
    elif info.get("crashed", False):
        outcome = "CRASH"
    else:
        outcome = "TIMEOUT"

    result = {
        "track_seed": track_seed,
        "steps": steps,
        "lap_completed": env.lap_completed,
        "lap_time": steps if env.lap_completed else None,
        "lap_progress": round(env.lap_progress, 4),
        "progress_pct": round(100.0 * env.lap_progress, 2),
        "off_track_rate": round(off_track_steps / max(steps, 1), 4),
        "total_reward": round(total_reward, 4),
        "crashed": info.get("crashed", False),
        "outcome": outcome,
    }
    env.close()
    return result


# ─── The three controllers ───────────────────────────────────────────────────

V1 = "best_actor.pth"
V2 = "runs/exp2_demos_s42/solved_actor.pth"

act_v1 = load_actor(V1)
act_v2 = load_actor(V2)


if __name__ == "__main__":
    print("=" * 88)
    print("Three-way eval — v1 vs v2 vs pure_pursuit")
    print(f"  seeds={TRACK_SEEDS}  max_steps={MAX_STEPS}")
    print(f"  v1 = {V1}")
    print(f"  v2 = {V2}")
    print("=" * 88)

    all_results = {}
    all_summaries = {}

    for config_name, track_kwargs in TRACK_CONFIGS.items():
        all_results[config_name] = {}
        all_summaries[config_name] = {}

        # v1 and v2 via the agent runner
        for name, act_fn in [("v1", act_v1), ("v2", act_v2)]:
            results = [run_agent_episode(seed, act_fn, track_kwargs) for seed in TRACK_SEEDS]
            all_results[config_name][name] = results
            all_summaries[config_name][name] = summarise(results)
            print_block(config_name, name, results)

        # pure_pursuit via the original runner (reads env directly)
        results = [run_episode(seed, pure_pursuit_policy, track_kwargs) for seed in TRACK_SEEDS]
        all_results[config_name]["pure_pursuit"] = results
        all_summaries[config_name]["pure_pursuit"] = summarise(results)
        print_block(config_name, "pure_pursuit", results)

    # ─── Comparison table ───
    print(f"\n{'=' * 88}")
    print("SUMMARY — v1 vs v2 vs pure_pursuit")
    print(f"{'=' * 88}")
    print(f"{'config':<10} {'policy':<14} {'laps':>7} {'avg_lap_time':>13} "
          f"{'avg_progress':>13} {'off_track':>10} {'avg_reward':>12}")
    print("-" * 88)
    order = ["v1", "v2", "pure_pursuit"]
    for config_name in TRACK_CONFIGS:
        for name in order:
            s = all_summaries[config_name][name]
            alt = "—" if s["avg_lap_time"] is None else f"{s['avg_lap_time']:.1f}"
            print(
                f"{config_name:<10} {name:<14} "
                f"{s['laps_completed']:>3}/{s['tracks']:<3} "
                f"{alt:>13} {s['avg_progress']:>13.4f} "
                f"{s['avg_off_track_rate']:>9.2%} {s['avg_reward']:>12.2f}"
            )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "controllers": {"v1": V1, "v2": V2, "pure_pursuit": "baselines/pure_pursuit.py"},
        "track_seeds": TRACK_SEEDS,
        "max_steps": MAX_STEPS,
        "track_configs": TRACK_CONFIGS,
        "results": all_results,
        "summary": all_summaries,
    }
    out_path = "baselines/eval_v1_v2_heuristic.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")