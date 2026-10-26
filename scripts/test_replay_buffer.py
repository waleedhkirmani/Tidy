import numpy as np
import torch

from rl.replay_buffer import ReplayBuffer
from rl.sac import SAC

OBS, ACT = 18, 4


def filled(max_len):
    """Ring holding transitions whose reward encodes their insertion index."""
    buf = ReplayBuffer(max_len=max_len)
    for k in range(max_len + 4):
        buf.add(
            np.full(OBS, k, dtype=np.float32),
            torch.full((1, ACT), k, dtype=torch.float32),
            float(k),
            np.full(OBS, k, dtype=np.float32),
            k % 2 == 0,
        )
    return buf


# Wrap-around keeps the newest and evicts the oldest, same as deque(maxlen=...).
buf = filled(8)
assert len(buf) == 8, len(buf)
assert buf.i == 4, buf.i
rewards = np.sort(buf.rewards.ravel())
assert rewards.tolist() == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], rewards

# A partially filled ring must never sample its unwritten tail.
part = ReplayBuffer(max_len=8)
for k in range(3):
    part.add(
        np.zeros(OBS, dtype=np.float32),
        torch.zeros((1, ACT), dtype=torch.float32),
        float(k),
        np.zeros(OBS, dtype=np.float32),
        False,
    )
assert len(part) == 3 and part.n == 3, (len(part), part.n)
assert not part.is_ready(4) and part.is_ready(3)
assert np.random.randint(0, part.n, size=1000).max() < 3

# Shapes/dtypes are the contract rl/sac.py's cat([state, action], dim=1) needs.
states, actions, rewards_b, nstates, dones = filled(16).sample(8)
for name, t, shape in (
    ("states", states, (8, OBS)),
    ("actions", actions, (8, ACT)),
    ("rewards", rewards_b, (8, 1)),
    ("nstates", nstates, (8, OBS)),
    ("dones", dones, (8, 1)),
):
    assert tuple(t.shape) == shape, (name, t.shape)
    assert t.dtype == torch.float32, (name, t.dtype)
assert torch.cat([states, actions], dim=1).shape == (8, OBS + ACT)

# A real batch out of the buffer trains the agent without shape surprises.
sac = SAC()
sac.update(replay_buffer=filled(64), batch_size=32)

# save_state/restore_state roundtrip: written prefix survives, ring unchanged.
src = filled(8)
restored = ReplayBuffer(max_len=8)
restored.restore_state(src.save_state())
assert len(restored) == len(src) and restored.i == src.i and restored.n == src.n
for a, b in (
    (restored.states, src.states),
    (restored.actions, src.actions),
    (restored.rewards, src.rewards),
    (restored.nstates, src.nstates),
    (restored.dones, src.dones),
):
    assert (a == b).all()
assert src.save_state()["states"].shape == (8, OBS)

print("replay buffer: OK")