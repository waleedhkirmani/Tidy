from statistics import mean
from envs.tidy_env import TidyEnv

env = TidyEnv(gui=True)
print("Initialized")
KEYS = ["reach", "grasp", "lift", "approach", "lower", "success"]
rows = []

for episode in range(10):
    env.reset()
    comps = dict.fromkeys(KEYS, 0.0)
    reward = length = 0
    while True:
        _, r, terminated, truncated, info = env.step(env.action_space.sample())
        reward += r
        length += 1
        for k in KEYS:
            comps[k] += info["rewards"][k]
        if terminated or truncated:
            break
    print(
        f"Episode {episode + 1}: reward={reward:.2f} len={length}"
        + "".join(f" {k}={comps[k]:.2f}" for k in KEYS)
    )
    rows.append((reward, length, comps, terminated))

rewards, lengths, comps, successes = zip(*rows)
print(
    f"\nSuccess rate: {sum(successes)}/{len(rows)} ({100 * sum(successes) / len(rows):.0f}%)"
)
print(f"avg_reward={mean(rewards):.2f} avg_len={mean(lengths):.1f}")
for k in KEYS:
    print(f"  avg_{k}={mean(c[k] for c in comps):.2f}")
