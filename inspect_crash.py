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
    Manually run the neural network.

    obs:
        Current observation from the racing environment.

    s:
        Saved neural-network weights.
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


# Create the same hard hairpin environment
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


# IMPORTANT:
# We are analyzing seed 16 now.
obs, _ = env.reset(
    options={"track_seed": 16}
)


# Run the trained policy
for step in range(500):

    action = act(obs, state)

    obs, reward, terminated, truncated, info = env.step(action)

    print("\nSTEP:", step)
    print("ACTION:", action)
    print("REWARD:", reward)
    print("INFO:", info)

    if terminated or truncated:
        print("\nEpisode ended.")
        break