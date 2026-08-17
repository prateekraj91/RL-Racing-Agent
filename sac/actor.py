import torch
import torch.nn as nn


class SACActor(nn.Module):

    def __init__(self, obs_dim=9, action_dim=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, observation):

        x = self.network(observation)

        mean = self.mean(x)

        log_std = self.log_std(x)

        log_std = torch.clamp(
            log_std,
            -20,
            2,
        )

        return mean, log_std

    def sample(self, observation):

        mean, log_std = self.forward(observation)

        std = log_std.exp()

        distribution = torch.distributions.Normal(mean, std)

        raw_action = distribution.rsample()

        action = torch.tanh(raw_action)

        log_prob = distribution.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)

        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob