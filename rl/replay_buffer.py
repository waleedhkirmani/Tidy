from collections import deque
from random import sample
import torch


class ReplayBuffer:
    def __init__(self, max_len=10_000):
        self.buffer = deque(maxlen=max_len)
        self.max_len = max_len

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action.detach(), reward, next_state, done))

    def is_ready(self, batch_size):
        if len(self.buffer) >= batch_size:
            return True
        return False

    def sample(self, batch_size):
        batch = sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(states, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones
