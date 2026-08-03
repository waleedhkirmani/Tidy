# %%

from rl.checkpoint import load_checkpoint, save_checkpoint
from rl.sac import SAC
from rl.replay_buffer import ReplayBuffer
from envs.tidy_env import TidyEnv

# %%

episode_len = 200
batch_size = 64

# %%

env = TidyEnv(gui=False)
sac = SAC()
replay_buffer = ReplayBuffer()

start_episode = load_checkpoint(sac)

# %%

for i in range(start_episode, start_episode + episode_len):
    state, info = env.reset()
    done = False
    ep_reward = 0
    while not done:
        # Select Action
        action = sac.select_action(state)
        # Perform on Environment
        next_state, reward, terminated, truncated, info = env.step(
            action.detach().cpu().numpy()[0]
        )

        done = terminated or truncated
        ep_reward += reward

        # Save episode
        replay_buffer.add(state, action, reward, next_state, terminated)
        state = next_state

        # Train
        if replay_buffer.is_ready(batch_size=batch_size):
            sac.update(replay_buffer=replay_buffer, batch_size=batch_size)

    print(
        f"Episode {i:4d} | Reward: {ep_reward:8.2f} | Buffer: {len(replay_buffer.buffer)}"
    )
    if i % 100 == 0:
        save_checkpoint(sac, i)
save_checkpoint(sac, i)
