import torch
import torch.nn.functional as F

from env.environment import RacingEnv


# Load the trained V2 policy
state = torch.load(
    "models/best_single_track_v2.pth",
    map_location="cpu"
)


def act(obs, s):
    """
    Run the trained neural network manually.
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


# Same hard-track environment used for our crash analysis
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


# Analyze seed 16
obs, _ = env.reset(
    options={"track_seed": 16}
)


for step in range(500):

    # Ask V2 what action it wants to take
    action = act(obs, state)

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Extract the quantities we care about
    speed = info["speed"]
    steering = info["steering"]
    throttle = info["throttle"]
    distance = info["signed_dist"]
    heading_error = info["heading_err"]
    slip = info["slip"]

    # The closest ray tells us how close the car is
    # to an obstacle/track boundary in its sensor field.
    min_ray = min(info["rays"])

    print(
        f"STEP {step:3d} | "
        f"speed={speed:5.2f} | "
        f"min_ray={min_ray:5.1f} | "
        f"steer={steering:+5.2f} | "
        f"throttle={throttle:+5.2f} | "
        f"dist={distance:+6.2f} | "
        f"heading={heading_error:+6.1f}° | "
        f"slip={slip:+5.2f}°"
    )

    if terminated or truncated:
        print("\nEpisode ended.")
        break