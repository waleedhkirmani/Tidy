# %%

from statistics import mean
from rl.checkpoint import load_checkpoint
from rl.sac import SAC
from envs.tidy_env import TidyEnv

env = TidyEnv(gui=True)
sac = SAC()

start = load_checkpoint(sac)
if start == 0:
    print("WARNING: no checkpoint found, evaluating an untrained policy")

# %%

KEYS = ["reach", "grasp", "lift", "approach", "lower", "success"]
n_episodes = 10
rows = []

for _ in range(n_episodes):
    state, info = env.reset()
    done = False
    ep_reward = 0
    comps = dict.fromkeys(KEYS, 0.0)
    while not done:
        action = sac.select_best_action(state)
        next_state, reward, terminated, truncated, info = env.step(
            action.detach().cpu().numpy()[0]
        )
        done = terminated or truncated
        ep_reward += reward
        for k in KEYS:
            comps[k] += info["rewards"][k]
        state = next_state
    rows.append((ep_reward, comps, terminated))
    print(
        f"Episode reward: {ep_reward:8.2f}"
        + "".join(f" {k}={comps[k]:.2f}" for k in KEYS)
    )

rewards, comps, successes = zip(*rows)
print(
    f"\nSuccess rate: {sum(successes)}/{len(rows)} ({100 * sum(successes) / len(rows):.0f}%)"
)
print(f"avg_reward={mean(rewards):.2f}")
for k in KEYS:
    print(f"  avg_{k}={mean(c[k] for c in comps):.2f}")
