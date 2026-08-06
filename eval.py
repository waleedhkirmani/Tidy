# %%

from statistics import mean
import numpy as np
import pybullet as p
from rl.checkpoint import load_checkpoint
from rl.sac import SAC
from envs.tidy_env import TidyEnv

env = TidyEnv(gui=True)
sac = SAC()

cam = {"dist": 1.7, "yaw": 90.0, "pitch": -35.0, "target": np.array([0.5, 0.0, 0.2])}
p.resetDebugVisualizerCamera(cam["dist"], cam["yaw"], cam["pitch"], cam["target"])


def move_camera():
    keys = p.getKeyboardEvents()
    if p.B3G_LEFT_ARROW in keys:
        cam["yaw"] += 4
    if p.B3G_RIGHT_ARROW in keys:
        cam["yaw"] -= 4
    if p.B3G_UP_ARROW in keys:
        cam["pitch"] += 4
    if p.B3G_DOWN_ARROW in keys:
        cam["pitch"] -= 4
    if ord("q") in keys:
        cam["dist"] = max(0.2, cam["dist"] - 0.1)
    if ord("e") in keys:
        cam["dist"] += 0.1
    for key, axis, sgn in (
        (ord("w"), 1, 0.05),
        (ord("s"), 1, -0.05),
        (ord("a"), 0, -0.05),
        (ord("d"), 0, 0.05),
    ):
        if key in keys:
            cam["target"][axis] += sgn
    p.resetDebugVisualizerCamera(cam["dist"], cam["yaw"], cam["pitch"], cam["target"])


# start = load_checkpoint(sac, path="models/best.pt") or load_checkpoint(sac)
start = load_checkpoint(sac)
if start == 0:
    print("WARNING: no checkpoint found, evaluating an untrained policy")

# %%

KEYS = ["reach", "grasp", "lift", "hold", "approach", "lower", "success"]
n_episodes = 10
rows = []

for _ in range(n_episodes):
    state, info = env.reset()
    done = False
    ep_reward = 0
    comps = dict.fromkeys(KEYS, 0.0)
    while not done:
        move_camera()
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
