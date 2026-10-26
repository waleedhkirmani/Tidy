import numpy as np
import torch


class ReplayBuffer:
    # Ring buffer in preallocated arrays. The deque version copied the whole
    # buffer into a tuple on every sample() call (~12ms at 400k, +25% step time);
    # indexing arrays directly is ~0.05ms and costs 67MB instead of ~370MB.
    # 400k = ~1000 episodes of 400 steps, which is what a single 12h CPU session
    # actually produces. The buffer is checkpointed into b_model_latest.pt
    # (written prefix only), so it survives Colab resumes and a fresh run
    # flushes it by deleting that file.
    def __init__(self, max_len=400_000, obs_dim=18, act_dim=4):
        self.max_len = max_len
        self.states = np.empty((max_len, obs_dim), dtype=np.float32)
        self.actions = np.empty((max_len, act_dim), dtype=np.float32)
        self.rewards = np.empty((max_len, 1), dtype=np.float32)
        self.nstates = np.empty((max_len, obs_dim), dtype=np.float32)
        self.dones = np.empty((max_len, 1), dtype=np.float32)
        self.i = 0  # write cursor
        self.n = 0  # fill count, capped at max_len

    def add(self, state, action, reward, next_state, done):
        j = self.i
        self.states[j] = state
        self.actions[j] = action.detach().cpu().numpy().ravel()
        self.rewards[j] = reward
        self.nstates[j] = next_state
        self.dones[j] = done
        self.i = (self.i + 1) % self.max_len
        self.n = min(self.n + 1, self.max_len)

    def __len__(self):
        return self.n

    def is_ready(self, batch_size):
        return self.n >= batch_size

    def sample(self, batch_size):
        # ponytail: index only the written prefix, so the unwritten tail of a
        # partially filled ring is never sampled.
        idx = np.random.randint(0, self.n, size=batch_size)
        get = lambda a: torch.as_tensor(a[idx], dtype=torch.float32)
        return (
            get(self.states),
            get(self.actions),
            get(self.rewards),
            get(self.nstates),
            get(self.dones),
        )

    def save_state(self):
        # Save only the written prefix: sampling is order-agnostic (SAC draws
        # random batches and each transition carries its own done flag), so the
        # ring rotation never needs to be linearized.
        return {
            "states": self.states[: self.n].copy(),
            "actions": self.actions[: self.n].copy(),
            "rewards": self.rewards[: self.n].copy(),
            "nstates": self.nstates[: self.n].copy(),
            "dones": self.dones[: self.n].copy(),
            "i": self.i,
            "n": self.n,
        }

    def restore_state(self, d):
        # ponytail: rebuild into the same preallocated arrays, truncating a
        # large saved ring to max_len if the capacity was ever reduced.
        n = min(d["n"], self.max_len)
        self.states[:n] = d["states"][:n]
        self.actions[:n] = d["actions"][:n]
        self.rewards[:n] = d["rewards"][:n]
        self.nstates[:n] = d["nstates"][:n]
        self.dones[:n] = d["dones"][:n]
        self.i = d["i"] % self.max_len
        self.n = n