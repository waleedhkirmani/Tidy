import os
import tempfile

import torch

from rl.checkpoint import load_checkpoint, save_checkpoint
from rl.sac import SAC

# Alpha is a plain fixed hyperparameter now.
sac = SAC(alpha=0.05)
assert isinstance(sac.alpha, float), type(sac.alpha)
assert sac.alpha == 0.05, sac.alpha
assert not hasattr(sac, "log_alpha"), "auto entropy tuning should be gone"
assert not hasattr(sac, "alpha_optimizer"), "auto entropy tuning should be gone"
assert not hasattr(sac, "_update_alpha"), "auto entropy tuning should be gone"


# The critic target must carry the entropy bonus. Two agents identical except for
# alpha, one critic step on the same batch: log_prob is negative for a continuous
# density, so -alpha*log_prob RAISES the target and the alpha=0.5 critic must end
# up strictly higher. Equal outputs catch "term removed", lower catches a flipped
# sign. This is the invariant whose absence made the old temperature a one-way
# countdown to zero.
def trained_q(alpha):
    torch.manual_seed(0)
    a = SAC(alpha=alpha)
    torch.manual_seed(1)
    states = torch.randn(64, 18)
    actions = torch.tanh(torch.randn(64, 4))
    rewards = torch.randn(64, 1)
    nstates = torch.randn(64, 18)
    dones = torch.zeros(64, 1)
    a._update_critics(states, actions, rewards, nstates, dones)
    with torch.no_grad():
        return a.critic1(states, actions)


q_zero, q_hot = trained_q(0.0), trained_q(0.5)
# Compared on the mean, not elementwise: a single Adam step at lr 3e-4 does not
# move the net monotonically in the target, sample by sample.
gap = (q_hot - q_zero).mean().item()
assert gap > 0, (gap, q_hot.mean().item(), q_zero.mean().item())

# Bound: the target gap is gamma*alpha*|log_prob|, and one small step can only
# close a fraction of it. Zero alpha must move the critic exactly as much as an
# entropy-free target does.
with torch.no_grad():
    _, lp = SAC().actor.sample(torch.randn(512, 18))
target_gap = 0.99 * 0.5 * -lp.mean().item()
assert 0 < gap < target_gap, (gap, target_gap)
q_zero2 = trained_q(0.0)
assert torch.equal(q_zero, q_zero2), "alpha=0 must be deterministic in the target"

# A resumed run keeps the policy and does NOT inherit a stale temperature.
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "c.pt")
    sac.actor.actor[0].weight.data.fill_(0.123)
    save_checkpoint(sac, 5, path=path)

    resumed = SAC(alpha=0.2)
    assert load_checkpoint(resumed, path=path) == 6
    assert abs(resumed.alpha - 0.2) == 0.0, resumed.alpha
    assert abs(resumed.actor.actor[0].weight.data[0, 0].item() - 0.123) < 1e-6

    # ...and a checkpoint from the auto-tuning era resumes the policy at the
    # constructor alpha instead of the collapsed log_alpha it carries.
    old = torch.load(path, map_location="cpu")
    del old["alpha"]
    old["log_alpha"] = torch.tensor(-6.2)  # alpha ~0.002
    old_path = os.path.join(tmp, "old.pt")
    torch.save(old, old_path)

    from_old = SAC(alpha=0.2)
    assert load_checkpoint(from_old, path=old_path) == 6
    assert from_old.alpha == 0.2, from_old.alpha
    assert abs(from_old.actor.actor[0].weight.data[0, 0].item() - 0.123) < 1e-6

print("sac: OK")