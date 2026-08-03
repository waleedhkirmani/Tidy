from collections import deque
from random import sample
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_len=10_000):
        self.buffer = deque(maxlen=max_len)
        self.max_len = max_len

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action.detach(), reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        return len(self.buffer) >= batch_size

    def sample(self, batch_size):
        batch = sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(np.array(states), dtype=torch.float32)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.cat(actions, dim=0)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        dones = torch.as_tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones
