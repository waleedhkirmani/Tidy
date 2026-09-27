import math
import os
import tempfile

import torch

from rl.checkpoint import load_checkpoint, save_checkpoint
from rl.sac import SAC

# alpha starts at the constructor value and target entropy is -action_dim.
sac = SAC(alpha=0.05)
assert abs(sac.log_alpha.exp().item() - 0.05) < 1e-6, sac.log_alpha.exp().item()
assert sac.target_entropy == -4, sac.target_entropy

# The fixed point is H == target_entropy, and a tanh policy over [-1,1]^4 cannot
# exceed 4*ln2 nats. A target above that cap leaves the residual sign constant
# and alpha runs away; this assert is what catches that.
assert sac.target_entropy <= 4 * math.log(2), sac.target_entropy

# Sign check on the tuner: the fixed point is H == target_entropy == -4, so a
# policy more peaked than that (H = -10, log_prob = +10) must push alpha up and
# a more random one (H = 1, log_prob = -1) must push it down. This is the check
# that catches a flipped sign. Both values are reachable: this policy spans
# H in (-inf, 4*ln2].
def drive(log_prob, steps=60):
    batch_log_prob = torch.full((8, 1), log_prob, dtype=torch.float32)
    for _ in range(steps):
        sac._update_alpha(batch_log_prob)


a_start = sac.log_alpha.exp().item()
drive(10.0)
a_hot = sac.log_alpha.exp().item()
assert a_hot > a_start, (a_start, a_hot)
drive(-1.0)
a_cold = sac.log_alpha.exp().item()
assert a_cold < a_hot, (a_hot, a_cold)

# The actor loss must not train the temperature: J(pi) and J(alpha) are separate
# objectives, so log_alpha.grad stays untouched by an actor step.
states = torch.zeros(8, 18)
sac.log_alpha.grad = None
sac._update_actor(states)
assert sac.log_alpha.grad is None, sac.log_alpha.grad

# A resumed run keeps the learned temperature...
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "c.pt")
    sac.log_alpha.data.fill_(torch.log(torch.tensor(0.123)).item())
    save_checkpoint(sac, 5, path=path)

    resumed = SAC(alpha=0.2)
    assert load_checkpoint(resumed, path=path) == 6
    assert abs(resumed.log_alpha.exp().item() - 0.123) < 1e-5, resumed.log_alpha.exp()

    # ...and a checkpoint saved before alpha was learnable still resumes the
    # policy instead of falling into the "architecture changed, start fresh"
    # path that a bare checkpoint["log_alpha"] would trigger.
    old = torch.load(path, map_location="cpu")
    del old["log_alpha"]
    old_path = os.path.join(tmp, "old.pt")
    torch.save(old, old_path)

    from_old = SAC(alpha=0.2)
    assert load_checkpoint(from_old, path=old_path) == 6
    assert abs(from_old.log_alpha.exp().item() - 0.2) < 1e-6, from_old.log_alpha.exp()

print("alpha: OK")
