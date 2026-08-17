import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, obs_dim=9, action_dim=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
        )

    def forward(self, observation, action):

        x = torch.cat(
            [observation, action],
            dim=-1,
        )

        return self.network(x)