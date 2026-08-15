from env.environment import RacingEnv

TRACK_SEEDS = [101, 202, 303, 404, 505]

def run_random_baseline(track_seed):
    env = RacingEnv(max_steps=2000, verbose=False)

    observation, info = env.reset(
        options={"track_seed": track_seed}
    )

    total_reward = 0.0

    for step in range(2000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        
        if terminated or truncated:
            break

    print(
    f"Track {track_seed}: "
    f"steps={step + 1}, "
    f"lap_progress={info['lap_progress']:.4f}, "
    f"lap_completed={info['lap_completed']}, "
    f"reward={total_reward:.4f}, "
    f"crashed={info['crashed']}"
    )

    env.close()

if __name__ == "__main__":
    for seed in TRACK_SEEDS:
        run_random_baseline(seed)


    

