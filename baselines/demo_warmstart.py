import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from env.environment import RacingEnv
from sac.agent import SACAgent


TRACK_SEED = 101

MEDIUM_TRACK = {
    "width": 70,
    "base_r": 250,
    "n_ctrl": 10,
    "min_radius": 80,
    "cx": 400,
    "cy": 300,
}


env = RacingEnv(
    max_steps=500,
    verbose=False,
    track_kwargs=MEDIUM_TRACK,
)

agent = SACAgent()


observation, info = env.reset(
    options={"track_seed": TRACK_SEED}
)


# --------------------------------
# Collect heuristic demonstrations
# --------------------------------
total_reward = 0.0

for step in range(500):

    track_heading = env.track.track_heading(
        env.car.x,
        env.car.y,
    )

    heading_error = (
        env.car.angle - track_heading + 180
    ) % 360 - 180

    steering = np.clip(
        -heading_error / 45.0,
        -1.0,
        1.0,
    )

    throttle = 0.7

    action = np.array(
        [steering, throttle],
        dtype=np.float32,
    )

    next_observation, reward, terminated, truncated, info = env.step(
        action
    )

    total_reward += reward

    agent.replay_buffer.add(
        observation,
        action,
        reward,
        next_observation,
        terminated,
    )

    observation = next_observation

    if terminated or truncated:
        break


print("Demo transitions:", len(agent.replay_buffer))
print("Demo progress:", info["lap_progress"])
print("Demo reward:", total_reward)


# --------------------------------
# Add stationary transitions
# --------------------------------

observation, info = env.reset(
    options={"track_seed": TRACK_SEED}
)

for _ in range(20):

    action = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    next_observation, reward, terminated, truncated, info = env.step(
        action
    )

    agent.replay_buffer.add(
        observation,
        action,
        reward,
        next_observation,
        terminated,
    )

    observation = next_observation

    if terminated or truncated:
        break

print("Transitions after stationary data:", len(agent.replay_buffer))

# --------------------------------
# SAC learns from demo data
# --------------------------------

if len(agent.replay_buffer) >= 64:

    for update in range(100):

        result = agent.update(
            batch_size=64
        )

        if update % 10 == 0:
            print(
                "Update:",
                update,
                "| Actor:",
                round(result["actor_loss"], 3),
                "| Critic 1:",
                round(result["critic1_loss"], 3),
                "| Critic 2:",
                round(result["critic2_loss"], 3),
            )

# Compare heuristic vs learned SAC action at the start state

observation, info = env.reset(
    options={"track_seed": TRACK_SEED}
)

track_heading = env.track.track_heading(
    env.car.x,
    env.car.y,
)

heading_error = (
    env.car.angle - track_heading + 180
) % 360 - 180

heuristic_steering = np.clip(
    -heading_error / 45.0,
    -1.0,
    1.0,
)

heuristic_action = np.array(
    [heuristic_steering, 0.7],
    dtype=np.float32,
)

obs_tensor = torch.tensor(
    observation,
    dtype=torch.float32,
).unsqueeze(0)

with torch.no_grad():
    mean, _ = agent.actor(obs_tensor)
    sac_action = torch.tanh(mean).numpy()[0]

print()
print("START ACTION COMPARISON")
print("Heuristic:", heuristic_action)
print("SAC:", sac_action)
# --------------------------------
# Evaluate learned SAC policy
# --------------------------------

observation, info = env.reset(
    options={"track_seed": TRACK_SEED}
)

total_reward = 0.0
max_progress = 0.0

for step in range(500):


    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        mean, _ = agent.actor(observation_tensor)
        action = torch.tanh(mean).numpy()[0]

    observation, reward, terminated, truncated, info = env.step(
        action
    )

    total_reward += reward
    max_progress = max(
        max_progress,
        info["lap_progress"],
    )

    if terminated or truncated:
        break

print()
print("SAC DEMO-WARMSTART EVALUATION")
print("Reward:", total_reward)
print("Max progress:", max_progress)
print("Final velocity:", env.car.velocity)
print("Crashed:", info["crashed"])
print("Lap completed:", info["lap_completed"])

