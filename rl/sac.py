import copy
from math import log
import torch
from torch import nn
from .networks import Actor, Critic


class SAC:
    def __init__(self):
        self.actor = Actor()
        self.critic1 = Critic()
        self.critic2 = Critic()

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4
        )

        self.alpha = 0.2
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action

    def update(self, replay_buffer, batch_size):
        # 1 Sample a batch
        # 2 Update both critics
        # 3 Update the actor
        # 4 Soft update the target critics
        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size=batch_size
        )
        self._update_critics(states, actions, rewards, next_states, dones)
        self._update_actor(states)
        self._soft_update_targets()

    def _update_critics(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, log_prob = self.actor.sample(next_states)

            target_q1 = self.target_critic1.forward(next_states, next_actions)
            target_q2 = self.target_critic2.forward(next_states, next_actions)

            target_q = torch.min(target_q1, target_q2)
            target = rewards + self.gamma * (1 - dones) * (
                target_q - self.alpha * log_prob
            )
        current_q1 = self.critic1.forward(states, actions)
        current_q2 = self.critic2.forward(states, actions)

        loss1 = nn.functional.mse_loss(current_q1, target)
        loss2 = nn.functional.mse_loss(current_q2, target)

        critic_loss = loss1 + loss2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _update_actor(self, states):
        actions, log_prob = self.actor.sample(states)
        q1 = self.critic1.forward(states, actions)
        q2 = self.critic2.forward(states, actions)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def _soft_update_targets(self):

        for target_param, param in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
