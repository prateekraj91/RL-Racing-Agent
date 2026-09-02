"""Benchmark both baselines on the 5 fixed tracks with lap timing.

Runs the random-policy and pure-pursuit baselines on both track configurations,
records lap time (or DNF + progress reached) and off-track rate per track, and
writes a comparison table + baselines/results.json.

Usage:
    python -m baselines.benchmark
"""

# ─── ACTION CONTRACT (verified against env/environment.py:123-128, env/car.py:39-54) ──
#
#   action = np.array([steering, throttle], dtype=np.float32)    Box(-1.0, 1.0, (2,))
#
#   action[0]  STEERING  in [-1, 1]  ->  car.steering = action[0] * car.max_steering (30°)
#                +1.0 = full LEFT   (heading angle increases, CCW on screen)
#                -1.0 = full RIGHT  (heading angle decreases, CW on screen)
#
#   action[1]  THROTTLE  in [-1, 1]  ->  car.velocity += action[1] * car.acceleration (0.08)
#                +1.0 = full accelerate       -1.0 = full brake / reverse
#                velocity clamped to [-2.0, 4.0]; friction 0.03/step decays toward 0
#
#   Error signals (env/track.py):
#     heading_err = wrap180(car.angle - track.track_heading(x, y))
#                   > 0  ->  car points LEFT of road    ->  correct with NEGATIVE steering
#     signed_dist = track.signed_distance(x, y)
#                   > 0  ->  car is RIGHT of centerline ->  correct with POSITIVE steering
#
#   Car min turning radius at full steer = wheelbase / tan(30°) = 86.6 px
# ──────────────────────────────────────────────────────────────────────────────────────

import json
from datetime import datetime, timezone

from env.environment import RacingEnv
from baselines.pure_pursuit import (
    choose_action as pp_choose_action,
    TRACK_CONFIGS,
    TRACK_SEEDS,
)

# ─── Configuration ───────────────────────────────────────────────────────────

MAX_STEPS = 2000
RANDOM_ACTION_SEED = 12345   # fixes the random policy's action stream


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_episode(track_seed, policy_fn, track_kwargs, action_seed=None):
    """Run a single episode, returning metrics with lap timing.

    policy_fn(env, step) -> continuous action, np.float32 array [steer, throttle]

    Note on off_track_rate: the env terminates the episode as soon as the car
    leaves the track (environment.py:188), so this is 1/steps on a crash episode
    and 0.0 otherwise. It is reported as a rate for consistency with the SAC
    training loop, which logs the same quantity.
    """
    env = RacingEnv(max_steps=MAX_STEPS, verbose=False, track_kwargs=track_kwargs)
    if action_seed is not None:
        env.action_space.seed(action_seed)
    obs, info = env.reset(options={"track_seed": track_seed})

    total_reward = 0.0
    off_track_steps = 0

    for step in range(MAX_STEPS):
        action = policy_fn(env, step)
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


# ─── Policy functions ────────────────────────────────────────────────────────

def random_policy(env, step):
    return env.action_space.sample()


def pure_pursuit_policy(env, step):
    action, _ = pp_choose_action(env, step_count=step)
    return action


POLICIES = {
    "random": random_policy,
    "pure_pursuit": pure_pursuit_policy,
}


# ─── Reporting ───────────────────────────────────────────────────────────────

def summarise(results):
    laps = [r for r in results if r["lap_completed"]]
    lap_times = [r["lap_time"] for r in laps]
    return {
        "laps_completed": len(laps),
        "tracks": len(results),
        "crashes": sum(1 for r in results if r["crashed"]),
        "avg_lap_time": round(sum(lap_times) / len(lap_times), 1) if lap_times else None,
        "avg_progress": round(sum(r["lap_progress"] for r in results) / len(results), 4),
        "avg_off_track_rate": round(
            sum(r["off_track_rate"] for r in results) / len(results), 4
        ),
        "avg_reward": round(sum(r["total_reward"] for r in results) / len(results), 2),
    }


def print_block(config_name, policy_name, results):
    print(f"\n{'─' * 60}")
    print(f"  {config_name} tracks — {policy_name}")
    print(f"{'─' * 60}")
    for r in results:
        lap_str = str(r["lap_time"]) if r["lap_time"] is not None else "DNF"
        print(
            f"  Track {r['track_seed']}: "
            f"lap_time={lap_str:>6s}  "
            f"progress={r['progress_pct']:7.2f}%  "
            f"off_track={r['off_track_rate']:6.2%}  "
            f"reward={r['total_reward']:+9.2f}  "
            f"[{r['outcome']}]"
        )


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 88)
    print("Baseline Benchmark — random vs pure_pursuit")
    print(f"  seeds={TRACK_SEEDS}  max_steps={MAX_STEPS}")
    print("=" * 88)

    all_results = {}
    all_summaries = {}

    for config_name, track_kwargs in TRACK_CONFIGS.items():
        all_results[config_name] = {}
        all_summaries[config_name] = {}

        for policy_name, policy_fn in POLICIES.items():
            results = [
                run_episode(
                    seed,
                    policy_fn,
                    track_kwargs,
                    action_seed=RANDOM_ACTION_SEED if policy_name == "random" else None,
                )
                for seed in TRACK_SEEDS
            ]
            all_results[config_name][policy_name] = results
            all_summaries[config_name][policy_name] = summarise(results)
            print_block(config_name, policy_name, results)

    # ─── Comparison table ───
    print(f"\n{'=' * 88}")
    print("SUMMARY")
    print(f"{'=' * 88}")
    print(f"{'config':<10} {'policy':<14} {'laps':>7} {'avg_lap_time':>13} "
          f"{'avg_progress':>13} {'off_track':>10} {'avg_reward':>12}")
    print("-" * 88)
    for config_name in TRACK_CONFIGS:
        for policy_name in POLICIES:
            s = all_summaries[config_name][policy_name]
            alt = "—" if s["avg_lap_time"] is None else f"{s['avg_lap_time']:.1f}"
            print(
                f"{config_name:<10} {policy_name:<14} "
                f"{s['laps_completed']:>3}/{s['tracks']:<3} "
                f"{alt:>13} "
                f"{s['avg_progress']:>13.4f} "
                f"{s['avg_off_track_rate']:>9.2%} "
                f"{s['avg_reward']:>12.2f}"
            )

    # ─── Write fresh results.json ───
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action_space": "Box(-1.0, 1.0, (2,), float32) — [steering, throttle]",
        "track_seeds": TRACK_SEEDS,
        "max_steps": MAX_STEPS,
        "random_action_seed": RANDOM_ACTION_SEED,
        "track_configs": TRACK_CONFIGS,
        "results": all_results,
        "summary": all_summaries,
    }

    out_path = "baselines/results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {out_path}")
