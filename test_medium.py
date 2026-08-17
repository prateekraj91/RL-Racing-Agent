import numpy as np

from env.environment import RacingEnv

MEDIUM = {
    "width": 70,
    "base_r": 250,
    "n_ctrl": 10,
    "min_radius": 80,
    "cx": 400,
    "cy": 300,
}

env = RacingEnv(
    max_steps=500,
    track_kwargs=MEDIUM,
)

obs, _ = env.reset(
    options={"track_seed": 101}
)

total = 0.0

for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 1.0], dtype=np.float32)
    )

    total += reward

    if terminated or truncated:
        break

print("TOTAL REWARD:", total)
print("PROGRESS:", info["lap_progress"])
print("VELOCITY:", env.car.velocity)
print("CRASHED:", info["crashed"])