"""Roll out one lap with the pure-pursuit heuristic and save the trace to .npz.

The heuristic reference line for the Day 24 racing-line overlay. Mirrors
analysis/collect_run.py's schema exactly (same keys, same "record state BEFORE
stepping" convention) so plot_overlay.py can read either interchangeably.

Physics note: grip_limit=False (default). The agent was trained and evaluated
under default physics, so the heuristic line must be captured under the same
physics for the comparison to be apples-to-apples.

Usage:
    python -m analysis.collect_pure_pursuit
    python -m analysis.collect_pure_pursuit --seed 303
"""

import argparse
import os

import numpy as np

from env.environment import RacingEnv
from baselines.pure_pursuit import choose_action, MEDIUM_TRACK

MAX_STEPS = 2000   # generous cap so a full lap isn't cut short

parser = argparse.ArgumentParser(description="Collect one lap from the pure-pursuit heuristic")
parser.add_argument("--seed", type=int, default=101,
                    help="track seed (default: 101)")
args = parser.parse_args()

TRACK_SEED = args.seed
out_name = f"medium_seed{TRACK_SEED}_pure_pursuit"


# -------------------------
# Build env — default physics, matching how the agent was trained
# -------------------------

env = RacingEnv(
    max_steps=MAX_STEPS,
    verbose=False,
    track_kwargs=MEDIUM_TRACK,
    grip_limit=False,
)

print("controller: pure-pursuit (baselines/pure_pursuit.py)")
print(f"config: medium  |  track_seed={TRACK_SEED}  |  max_steps={MAX_STEPS}")
print(f"grip_limit: {env.car.grip_limit}")


# -------------------------
# Roll out ONE lap, record everything
# -------------------------

log = {
    "steer": [],
    "throttle": [],
    "velocity": [],
    "x": [],
    "y": [],
    "reward": [],
    "lap_progress": [],
    "obs": [],
}

observation, info = env.reset(options={"track_seed": TRACK_SEED})

terminated = False
truncated = False
steps = 0

while not (terminated or truncated):
    # pure_pursuit reads the live env (car + track), not the observation vector
    action, _dbg = choose_action(env)

    # record the state we are in BEFORE stepping,
    # plus the action we took from it
    log["steer"].append(float(action[0]))
    log["throttle"].append(float(action[1]))
    log["velocity"].append(float(env.car.velocity))
    log["x"].append(float(env.car.x))
    log["y"].append(float(env.car.y))
    log["obs"].append(np.asarray(observation, dtype=np.float32))

    observation, reward, terminated, truncated, info = env.step(action)

    log["reward"].append(float(reward))
    log["lap_progress"].append(float(info.get("lap_progress", 0.0)))

    steps += 1


# -------------------------
# Save + summarise
# -------------------------

os.makedirs("analysis/runs", exist_ok=True)
out_path = f"analysis/runs/{out_name}.npz"

np.savez(
    out_path,
    steer=np.array(log["steer"]),
    throttle=np.array(log["throttle"]),
    velocity=np.array(log["velocity"]),
    x=np.array(log["x"]),
    y=np.array(log["y"]),
    reward=np.array(log["reward"]),
    lap_progress=np.array(log["lap_progress"]),
    obs=np.array(log["obs"]),
)

print("\n--- RUN SUMMARY ---")
print("steps:          ", steps)
print("final progress: ", round(log["lap_progress"][-1], 4))
print("lap_completed:  ", info.get("lap_completed", None))
print("crashed:        ", info.get("crashed", None))
print("mean speed:     ", round(float(np.mean(log["velocity"])), 4))
print("terminated:     ", terminated, " truncated:", truncated)
print("saved to:       ", out_path)
