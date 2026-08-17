import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from env.environment import RacingEnv


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

observation, info = env.reset(
    options={"track_seed": TRACK_SEED}
)

transitions = []

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

    transitions.append(
        (
            observation.copy(),
            action.copy(),
            reward,
            next_observation.copy(),
            terminated,
        )
    )

    observation = next_observation

    if terminated or truncated:
        break

print("Collected transitions:", len(transitions))
print("Total reward:", sum(t[2] for t in transitions))
print("Final progress:", info["lap_progress"])
print("Completed:", info["lap_completed"])
print("Crashed:", info["crashed"])