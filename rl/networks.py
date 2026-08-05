import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(9, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 8)
        )

    def forward(self, state):
        output = self.actor(state)

        mu, log_std = output.chunk(2, dim=1)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z).sum(dim=1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        action = torch.cat(
            [action[:, :3] * 0.05, ((action[:, 3] + 1) / 2).unsqueeze(1)], dim=1
        )
        return action, log_prob

    def action(self, state):
        mu, _ = self.forward(state)
        action = torch.tanh(mu)
        action = torch.cat(
            [action[:, :3] * 0.05, ((action[:, 3] + 1) / 2).unsqueeze(1)], dim=1
        )
        return action


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(13, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.critic(x)
