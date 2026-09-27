import copy
from math import log
import torch
from torch import nn
from .networks import Actor, Critic


class SAC:
    def __init__(self, alpha=0.2):
        self.actor = Actor()
        self.critic1 = Critic()
        self.critic2 = Critic()

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4
        )

        # Learnable temperature (SAC automatic entropy tuning): `alpha` is only
        # the starting value, from here alpha moves so policy entropy tracks
        # target_entropy. Higher alpha = keep exploring, lower = exploit.
        self.log_alpha = nn.Parameter(torch.tensor(log(alpha), dtype=torch.float32))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)
        self.target_entropy = -4  # -action_dim (Actor emits 4 dims)
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action

    def select_best_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.actor.action(state)

    def update(self, replay_buffer, batch_size):
        # 1 Sample a batch
        # 2 Update both critics
        # 3 Update the actor
        # 4 Update the temperature
        # 5 Soft update the target critics
        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size=batch_size
        )
        self._update_critics(states, actions, rewards, next_states, dones)
        log_prob = self._update_actor(states)
        self._update_alpha(log_prob)
        self._soft_update_targets()

    def _update_critics(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, _ = self.actor.sample(next_states)

            target_q1 = self.target_critic1.forward(next_states, next_actions)
            target_q2 = self.target_critic2.forward(next_states, next_actions)

            target_q = torch.min(target_q1, target_q2)
            target = rewards + self.gamma * (1 - dones) * target_q
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

        # detach: the temperature has its own loss, J(pi) must not train it
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return log_prob

    def _update_alpha(self, log_prob):
        # Automatic entropy tuning. Entropy is H = -E[log_prob] and the fixed
        # point is H == target_entropy, so the residual is (log_prob +
        # target_entropy): positive = policy more peaked than target -> alpha up
        # (more bonus, more entropy), negative = too random -> alpha down.
        # target_entropy is negative, and its magnitude must stay under the
        # 4*ln2 cap on a tanh policy's entropy over [-1,1]^4 -- above that the
        # residual never changes sign and alpha runs away. That invariant is
        # asserted in scripts/test_alpha.py.
        alpha_loss = (
            -self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

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
