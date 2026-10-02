from env.environment import RacingEnv


env = RacingEnv(
    max_steps=500,
    track_kwargs={
        "width": 70,
        "base_r": 250,
        "n_ctrl": 10,
        "min_radius": 80,
        "cx": 400,
        "cy": 300,
        "hairpin": True,
        "hairpin_factor": 0.35,
    },
)

obs, _ = env.reset(options={"track_seed": 16})

for step in range(300):

    # Start with a gentle forward speed.
    if env.car.velocity < 0.8:
        throttle = 1.0
    else:
        throttle = 0.0

    # Try maximum steering in the direction of the hairpin.
    steering = 1.0

    action = [steering, throttle]

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"STEP {step:3d} | "
        f"speed={info['speed']:5.2f} | "
        f"steer={steering:+.2f} | "
        f"throttle={throttle:+.2f} | "
        f"dist={info['signed_dist']:+6.2f} | "
        f"heading={info['heading_err']:+6.1f}°"
    )

    if terminated or truncated:
        break

env.close()