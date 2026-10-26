# %%

import argparse
import os
import time
from collections import deque

from rl.checkpoint import MODELS_DIR, load_checkpoint, save_checkpoint
from rl.sac import SAC
from rl.replay_buffer import ReplayBuffer
from envs.tidy_env import (
    RAMP_EPISODES,
    TidyEnv,
    ramp_curriculum_prob,
    ramp_spread,
)

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
parser.add_argument(
    "--alpha",
    type=float,
    default=0.1,
    help="Fixed SAC entropy temperature (not auto-tuned)",
)
parser.add_argument(
    "--probe",
    choices=["grasp", "place"],
    default="grasp",
    help="Curriculum probe type: grasp (start above the cube) or place (start "
    "above the tray, cube in hand; fires at HALF the grasp probability)",
)
args = parser.parse_args()

# %%

episode_len = args.episodes
batch_size = 256
KEYS = ["reach", "grasp", "lift", "hold", "approach", "lower", "success"]

# %%

env = TidyEnv(
    gui=False,
    randomize_cube_spawn=True,
    randomize_tray_spawn=True,
    probe=args.probe,
)
sac = SAC(alpha=args.alpha)
replay_buffer = ReplayBuffer()

# %%

start_episode = load_checkpoint(sac, restore_buffer=replay_buffer)

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
    # Ramp over ABSOLUTE episode count (i is cumulative across Colab sessions
    # via load_checkpoint), so resuming a checkpoint keeps the ramp advancing.
    spread = ramp_spread(i)
    env.set_spawn_spread(spread)
    env.curriculum_prob = ramp_curriculum_prob(spread)
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
            action.cpu().numpy()[0]
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
        f"Episode {i:4d} | Reward: {ep_reward:8.2f} | Buffer: {len(replay_buffer)} | Time: {time.time() - ep_start:5.1f}s"
        + f" | alpha={sac.alpha:g}"
        + "".join(f" | {k}={comps[k]:.2f}" for k in KEYS),
        flush=True,
    )
    recent.append(ep_reward)
    avg = sum(recent) / len(recent)
    if avg > best_reward:
        best_reward = avg
        save_checkpoint(sac, i, path=os.path.join(MODELS_DIR, "b_best.pt"))
    if i % 100 == 0:
        save_checkpoint(sac, i, buffer=replay_buffer)
    j = i
save_checkpoint(sac, j, buffer=replay_buffer)
