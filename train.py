# %%

import argparse
import os
import time
from collections import deque

from rl.checkpoint import MODELS_DIR, load_checkpoint, save_checkpoint
from rl.sac import SAC
from rl.replay_buffer import ReplayBuffer
from envs.tidy_env import TidyEnv

# %%

parser = argparse.ArgumentParser()
parser.add_argument(
    "--minutes",
    type=float,
    default=None,
    help="Wall-clock budget in minutes; stop cleanly after this (cloud sessions)",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=200,
    help="Episodes to run this session (resumes from checkpoint)",
)
args = parser.parse_args()

# %%

episode_len = args.episodes
batch_size = 64
RAMP_EPISODES = 2000  # spread reaches full randomization after this many episodes
KEYS = ["reach", "grasp", "lift", "hold", "approach", "lower", "success"]

# %%

env = TidyEnv(gui=False, randomize_cube_spawn=True)
sac = SAC()
replay_buffer = ReplayBuffer()

# %%

start_episode = load_checkpoint(sac)

# %%
j = 0
best_reward = -1e9
recent = deque(maxlen=10)
t_start = time.time()
for i in range(start_episode, start_episode + episode_len):
    if args.minutes and time.time() - t_start > args.minutes * 60:
        print(
            f"Time budget ({args.minutes:.0f} min) hit after episode {j}; saving and stopping"
        )
        break
    # env.set_cube_spawn_spread(min(1.0, (i - start_episode) / RAMP_EPISODES))
    state, info = env.reset()
    ep_start = time.time()
    done = False
    ep_reward = 0
    comps = dict.fromkeys(KEYS, 0.0)
    while not done:
        # Select Action
        action = sac.select_action(state)
        # Perform on Environment
        next_state, reward, terminated, truncated, info = env.step(
            action.detach().cpu().numpy()[0]
        )

        done = terminated or truncated
        ep_reward += reward
        for k in KEYS:
            comps[k] += info["rewards"][k]

        # Save episode
        replay_buffer.add(state, action, reward, next_state, terminated)
        state = next_state

        # Train
        if replay_buffer.is_ready(batch_size=batch_size):
            sac.update(replay_buffer=replay_buffer, batch_size=batch_size)

    print(
        f"Episode {i:4d} | Reward: {ep_reward:8.2f} | Buffer: {len(replay_buffer.buffer)} | Time: {time.time() - ep_start:5.1f}s"
        + "".join(f" | {k}={comps[k]:.2f}" for k in KEYS),
        flush=True,
    )
    recent.append(ep_reward)
    avg = sum(recent) / len(recent)
    if avg > best_reward:
        best_reward = avg
        save_checkpoint(sac, i, path=os.path.join(MODELS_DIR, "b_best.pt"))
    if i % 100 == 0:
        save_checkpoint(sac, i)
    j = i
save_checkpoint(sac, j)
