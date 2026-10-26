import torch
from torch import nn


class RunningNorm(nn.Module):
    """Online (Welford) mean/std normalization applied to states. Buffers live
    in the state_dict, so checkpoints carry the stats with zero checkpoint-code
    changes; drifting stats from eval calls are negligible."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("M2", torch.ones(dim))
        self.register_buffer("count", torch.zeros(1))

    def forward(self, x):
        x = x.float()
        with torch.no_grad():
            if x.dim() > 1:
                n = x.shape[0]
                batch_mean = x.mean(dim=0)
                batch_M2 = ((x - batch_mean) ** 2).sum(dim=0)
                total = self.count + n
                delta = batch_mean - self.mean
                self.mean += delta * (n / total)
                self.M2 += batch_M2 + delta**2 * (self.count * n / total)
                self.count += n
            var = self.M2 / torch.clamp(self.count - 1, min=1)
        return (x - self.mean) / torch.sqrt(var + 1e-8)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_norm = RunningNorm(18)
        self.actor = nn.Sequential(
            nn.Linear(18, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 8)
        )

    def forward(self, state):
        output = self.actor(self.state_norm(state))

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
        self.state_norm = RunningNorm(18)
        self.critic = nn.Sequential(
            nn.Linear(22, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, state, action):
        x = torch.cat([self.state_norm(state), action], dim=1)
        return self.critic(x)
