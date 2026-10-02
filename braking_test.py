import torch
import torch.nn.functional as F

from env.environment import RacingEnv


# --------------------------------------------------
# 1. Load the trained V2 policy
# --------------------------------------------------

state = torch.load(
    "models/best_single_track_v2.pth",
    map_location="cpu"
)


def act(obs, s):
    """
    Run the trained neural network.

    The network still chooses both:
        action[0] = steering
        action[1] = throttle

    We will later override ONLY the throttle.
    """

    x = torch.tensor(obs, dtype=torch.float32)

    x = F.relu(
        F.linear(
            x,
            s["network.0.weight"],
            s["network.0.bias"]
        )
    )

    x = F.relu(
        F.linear(
            x,
            s["network.2.weight"],
            s["network.2.bias"]
        )
    )

    mean = F.linear(
        x,
        s["mean.weight"],
        s["mean.bias"]
    )

    return torch.tanh(mean).detach().numpy()


# --------------------------------------------------
# 2. Create the same hard track
# --------------------------------------------------

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
    }
)


# --------------------------------------------------
# 3. Reset on the same problematic seed
# --------------------------------------------------

obs, _ = env.reset(
    options={"track_seed": 16}
)


# --------------------------------------------------
# 4. Run V2 with manual braking
# --------------------------------------------------

for step in range(500):

    # Let the neural network choose the action.
    action = act(obs, state)

    # Keep the neural network's steering.
    steering = float(action[0])

    # Start with the neural network's throttle.
    throttle = float(action[1])

    # Look at the current observation.
    #
    # Observation layout:
    # [speed, heading_error, distance, slip,
    #  ray1, ray2, ray3, ray4, ray5]
    #
    # The five ray values are at positions 4:9.
    min_ray = min(obs[4:9])

    # --------------------------------------------------
    # MANUAL INTERVENTION
    # --------------------------------------------------
    #
    # If the corner is extremely close,
    # force the car to brake.
    #
    # This is NOT training.
    # We are only testing whether braking helps.
    #

    if min_ray < 28:
        throttle = -0.5

    action = [steering, throttle]

    # Apply the action.
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"STEP {step:3d} | "
        f"speed={info['speed']:5.2f} | "
        f"min_ray={min_ray:5.1f} | "
        f"steer={steering:+5.2f} | "
        f"throttle={throttle:+5.2f} | "
        f"dist={info['signed_dist']:+6.2f} | "
        f"heading={info['heading_err']:+6.1f}°"
    )

    if terminated or truncated:
        print("\nEpisode ended.")

        if info["lap_completed"]:
            print("RESULT: LAP COMPLETED")
        elif info["crashed"]:
            print("RESULT: CRASH")
        else:
            print("RESULT: TIMEOUT")

        break