"""Evaluate a trained agent across a POOL of tracks (train or test).

This is the generalization testbed's measuring instrument: given a checkpoint and
a list of seeds, it runs a deterministic lap on each track and reports per-track
results + pool averages. Run it on TRAIN_SEEDS and TEST_SEEDS separately, and the
difference is the generalization gap.
"""

import numpy as np
import torch

from env.environment import RacingEnv
from sac.agent import SACAgent
from sac.track_pools import POOL_CONFIG


def load_agent(checkpoint):
    agent = SACAgent()
    agent.actor.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    agent.actor.eval()
    return agent


def eval_on_track(agent, seed):
    """One deterministic lap on the track for `seed`. Returns outcome metrics."""
    env = RacingEnv(max_steps=POOL_CONFIG["max_steps"],
                    track_kwargs=POOL_CONFIG["track_kwargs"])
    obs, _ = env.reset(options={"track_seed": seed})

    completed, crashed, steps = False, False, 0
    for _ in range(POOL_CONFIG["max_steps"]):
        with torch.no_grad():
            mean, _ = agent.actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = torch.tanh(mean).numpy()[0]
        obs, r, term, trunc, info = env.step(action)
        steps += 1
        if info["crashed"]:  crashed = True
        if info["lap_completed"]: completed = True
        if term or trunc: break

    return {
        "seed": seed,
        "completed": completed,
        "crashed": crashed,
        "lap_time": steps if completed else None,
        "progress": round(info["lap_progress"], 3),
    }


def evaluate_pool(agent, seeds, label=""):
    """Evaluate the agent across all seeds in a pool. Returns per-track + summary."""
    results = [eval_on_track(agent, s) for s in seeds]

    n = len(results)
    n_completed = sum(r["completed"] for r in results)
    lap_times = [r["lap_time"] for r in results if r["lap_time"] is not None]
    avg_lap = round(sum(lap_times) / len(lap_times), 1) if lap_times else None
    avg_progress = round(sum(r["progress"] for r in results) / n, 3)

    print(f"\n=== POOL: {label} ({n} tracks) ===")
    for r in results:
        lt = r["lap_time"] if r["lap_time"] is not None else "DNF"
        print(f"  seed {r['seed']:5d}: {'LAP ' if r['completed'] else 'FAIL'} "
              f"lap_time={str(lt):>5}  progress={r['progress']:.2f}")
    print(f"  --> completed {n_completed}/{n}  |  avg lap (of completed): {avg_lap}  "
          f"|  avg progress: {avg_progress}")

    return {"label": label, "n": n, "completed": n_completed,
            "avg_lap_time": avg_lap, "avg_progress": avg_progress, "per_track": results}


if __name__ == "__main__":
    import argparse
    from sac.track_pools import TRAIN_SEEDS, TEST_SEEDS

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/best_single_track_v2.pth")
    args = p.parse_args()

    agent = load_agent(args.checkpoint)
    train_res = evaluate_pool(agent, TRAIN_SEEDS, "TRAIN")
    test_res  = evaluate_pool(agent, TEST_SEEDS,  "TEST (held-out)")

    # The generalization gap: train-pool vs test-pool completion.
    print(f"\n=== GENERALIZATION GAP ===")
    print(f"  train completed: {train_res['completed']}/{train_res['n']}")
    print(f"  test  completed: {test_res['completed']}/{test_res['n']}")