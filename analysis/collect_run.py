import os
import numpy as np
import torch

from env.environment import RacingEnv
from sac.agent import SACAgent


import argparse

MEDIUM_TRACK = {
    "width": 70,
    "base_r": 250,
    "n_ctrl": 10,
    "min_radius": 80,
    "cx": 400,
    "cy": 300,
}

MAX_STEPS = 2000   # generous cap so a full lap isn't cut short

parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=["medium", "default"], default="medium")
parser.add_argument("--seed", type=int, default=101)
args = parser.parse_args()

TRACK_SEED = args.seed


if args.config == "medium":
    track_kwargs = MEDIUM_TRACK
    out_name = f"medium_seed{TRACK_SEED}"
else:
    track_kwargs = {}          # default config = the one it fails
    out_name = f"default_seed{TRACK_SEED}"


# -------------------------
# Build env + load actor
# -------------------------

env = RacingEnv(
    max_steps=MAX_STEPS,
    verbose=False,
    track_kwargs=track_kwargs,
)

agent = SACAgent()

agent.actor.load_state_dict(
    torch.load("best_actor.pth", map_location="cpu")
)
agent.actor.eval()

print("Loaded trained actor from best_actor.pth")


# -------------------------
# Deterministic action helper
# (same as visualize.py: tanh(mean), no sampling)
# -------------------------

def policy_action(observation):
    obs_tensor = torch.tensor(
        observation, dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        mean, _ = agent.actor(obs_tensor)
        action = torch.tanh(mean).numpy()[0]

    return action


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
    action = policy_action(observation)

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
print("terminated:     ", terminated, " truncated:", truncated)
print("saved to:       ", out_path)