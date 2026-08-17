import torch
import copy
import torch.optim as optim

from sac.actor import SACActor
from sac.critic import Critic
from sac.replay_buffer import ReplayBuffer


class SACAgent:

    def __init__(self, obs_dim=9, action_dim=2):

        self.actor = SACActor(
            obs_dim,
            action_dim,
        )

        self.critic1 = Critic(
            obs_dim,
            action_dim,
        )

        self.critic2 = Critic(
            obs_dim,
            action_dim,
        )

        self.log_alpha = torch.tensor(
            [-3.0],
            requires_grad=True,
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=3e-4,
        )

        self.target_entropy = -action_dim

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.replay_buffer = ReplayBuffer()

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=3e-4,
        )

        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(),
            lr=3e-4,
        )

        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=3e-4,
        )

    def soft_update(self, source, target, tau=0.005):
        
        for target_param, source_param in zip(
            target.parameters(),
            source.parameters(),
        ):
            target_param.data.copy_(
                tau * source_param.data
                + (1.0 - tau) * target_param.data
            )

    def update_critics(self, batch_size=64, gamma=0.99):

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(batch_size)
        )

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
        ).unsqueeze(1)

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
        ).unsqueeze(1)

        with torch.no_grad():

            next_actions, next_log_probs = (
                self.actor.sample(next_states)
            )

            target_q1 = self.target_critic1(
                next_states,
                next_actions,
            )

            target_q2 = self.target_critic2(
                next_states,
                next_actions,
            )

            target_q = torch.min(
                target_q1,
                target_q2,
            )

            alpha = self.log_alpha.exp()

            target = rewards + gamma * (
                1 - dones
            ) * (
                target_q - alpha * next_log_probs
            )

        current_q1 = self.critic1(
            states,
            actions,
        )

        current_q2 = self.critic2(
            states,
            actions,
        )

        loss1 = torch.mean(
            (current_q1 - target) ** 2
        )

        loss2 = torch.mean(
            (current_q2 - target) ** 2
        )

        self.critic1_optimizer.zero_grad()
        loss1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        loss2.backward()
        self.critic2_optimizer.step()

        return loss1.item(), loss2.item()

    def update_actor(self, states):

        actions, log_probs = self.actor.sample(states)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        q = torch.min(q1, q2)

        alpha = self.log_alpha.exp()

        actor_loss = (
            alpha * log_probs - q
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def update_alpha(self, log_probs):

        alpha_loss = -(
            self.log_alpha
            * (log_probs.detach() + self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def update(self, batch_size=64, gamma=0.99):

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(batch_size)
        )

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
        ).unsqueeze(1)

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
        ).unsqueeze(1)

        # 1. Update critics
        critic1_loss, critic2_loss = self.update_critics(
            batch_size,
            gamma,
        )

        # 2. Update actor
        actor_loss = self.update_actor(states)

        # 3. Update entropy temperature
        _, log_probs = self.actor.sample(states)
        alpha_loss = self.update_alpha(log_probs)

        # 4. Slowly update target critics
        self.soft_update(
            self.critic1,
            self.target_critic1,
        )

        self.soft_update(
            self.critic2,
            self.target_critic2,
        )

        return {
            "critic1_loss": critic1_loss,
            "critic2_loss": critic2_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.log_alpha.exp().item(),
        }
        