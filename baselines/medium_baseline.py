import numpy as np

import sys
from pathlib import Path

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

total_reward = 0.0


for step in range(500):

    # -------------------------
    # Find road direction
    # -------------------------

    track_heading = env.track.track_heading(
        env.car.x,
        env.car.y,
    )

    # Difference between where the car points
    # and where the road points.
    heading_error = (
        env.car.angle - track_heading + 180
    ) % 360 - 180

    # -------------------------
    # Simple steering controller
    # -------------------------

    steering = np.clip(
        -heading_error / 45.0,
        -1.0,
        1.0,
    )

    # Constant forward throttle
    throttle = 0.7

    action = np.array(
        [steering, throttle],
        dtype=np.float32,
    )

    # -------------------------
    # Step environment
    # -------------------------

    observation, reward, terminated, truncated, info = env.step(
        action
    )

    total_reward += reward

    if terminated or truncated:
        break


print("TOTAL REWARD:", total_reward)
print("STEPS:", step + 1)
print("LAP PROGRESS:", info["lap_progress"])
print("VELOCITY:", env.car.velocity)
print("CRASHED:", info["crashed"])
print("LAP COMPLETED:", info["lap_completed"])