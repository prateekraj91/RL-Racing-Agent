import numpy as np
from env.environment import RacingEnv

env = RacingEnv(max_steps=500)

obs, _ = env.reset(seed=101)

mins = obs.copy()
maxs = obs.copy()

for _ in range(500):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    mins = np.minimum(mins, obs)
    maxs = np.maximum(maxs, obs)

    if terminated or truncated:
        obs, _ = env.reset(seed=101)

print("MIN:", mins)
print("MAX:", maxs)


def hello():
    print("hello")
    print("world")

