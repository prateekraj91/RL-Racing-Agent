"""Independently verify a saved actor against the medium-track success criterion.

Deliberately does NOT import the training loop: a success claim that can only be
reproduced by the code that produced it is not a verified success. This rebuilds
the env from sac.curricula.TARGET_CONFIG, loads the checkpoint cold, and runs the
same deterministic tanh(mean) policy the training eval uses.

Usage:
    python -m analysis.verify_checkpoint runs/exp2_demos_s42/best_actor.pth
    python -m analysis.verify_checkpoint runs/<name>/solved_actor.pth --other-seeds

Exit code 0 iff the target criterion is met:
    lap_completed=True AND crashed=False on medium, track_seed 101, 500 steps.
"""

import argparse
import sys

import numpy as np
import torch

from env.environment import RacingEnv
from sac.actor import SACActor
from sac.curricula import TARGET_CONFIG


def rollout(actor, track_kwargs, track_seed, max_steps):
    env = RacingEnv(max_steps=max_steps, track_kwargs=track_kwargs)
    obs, _ = env.reset(options={"track_seed": track_seed})
    total_r = 0.0
    off = 0
    crashed = False
    speeds, steers, throts = [], [], []

    for n in range(max_steps):
        with torch.no_grad():
            mean, _ = actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            act = torch.tanh(mean).numpy()[0]
        obs, r, term, trunc, info = env.step(act)
        total_r += r
        speeds.append(float(obs[0]))
        steers.append(float(act[0]))
        throts.append(float(act[1]))
        if info["crashed"]:
            crashed = True
            off += 1
        if term or trunc:
            break

    steps = n + 1
    return {
        "track_seed": track_seed,
        "steps": steps,
        "lap_progress": env.lap_progress,
        "lap_completed": env.lap_completed,
        "crashed": crashed,
        "lap_time": steps if env.lap_completed else None,
        "off_track_rate": off / max(steps, 1),
        "total_reward": total_r,
        "mean_speed": float(np.mean(speeds)),
        "min_throttle": float(np.min(throts)),
        "max_throttle": float(np.max(throts)),
        "min_steer": float(np.min(steers)),
        "max_steer": float(np.max(steers)),
    }


def fmt(r):
    verdict = ("LAP" if r["lap_completed"] and not r["crashed"]
               else ("CRASH" if r["crashed"] else "TIMEOUT"))
    return (f"  seed={r['track_seed']:4d} steps={r['steps']:4d} "
            f"progress={r['lap_progress']:7.4f} "
            f"lap_completed={str(r['lap_completed']):5s} "
            f"crashed={str(r['crashed']):5s} "
            f"lap_time={str(r['lap_time']):>5s} "
            f"off_track={r['off_track_rate']:6.2%} "
            f"R={r['total_reward']:9.2f}  [{verdict}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--other-seeds", action="store_true",
                    help="also report unseen track seeds (generalisation, not "
                         "part of the success criterion)")
    args = ap.parse_args()

    actor = SACActor(obs_dim=9, action_dim=2)
    actor.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    actor.eval()

    tk = TARGET_CONFIG["track_kwargs"]
    ts = TARGET_CONFIG["track_seed"]
    ms = TARGET_CONFIG["max_steps"]

    print("=" * 100)
    print(f"VERIFY  {args.checkpoint}")
    print(f"TARGET  medium {tk}")
    print(f"        track_seed={ts}  max_steps={ms}  policy=deterministic tanh(mean)")
    print("=" * 100)

    r = rollout(actor, tk, ts, ms)
    print(fmt(r))
    solved = r["lap_completed"] and not r["crashed"]

    print(f"\n  action ranges over the lap: "
          f"steer [{r['min_steer']:+.3f}, {r['max_steer']:+.3f}]  "
          f"throttle [{r['min_throttle']:+.3f}, {r['max_throttle']:+.3f}]  "
          f"mean_speed={r['mean_speed']:.3f}")

    if args.other_seeds:
        print("\nUNSEEN TRACK SEEDS (generalisation only — not the success bar):")
        laps = 0
        for s in [202, 303, 404, 505]:
            rr = rollout(actor, tk, s, ms)
            print(fmt(rr))
            laps += int(rr["lap_completed"] and not rr["crashed"])
        print(f"  -> {laps}/4 unseen seeds completed")

    print("\n" + "=" * 100)
    print(f"VERDICT: {'SOLVED' if solved else 'NOT SOLVED'} "
          f"(lap_completed={r['lap_completed']}, crashed={r['crashed']})")
    print("=" * 100)
    return 0 if solved else 1


if __name__ == "__main__":
    sys.exit(main())
