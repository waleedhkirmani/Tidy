from envs.tidy_env import TidyEnv

env = TidyEnv()
obs, info = env.reset()

for _ in range(1):
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
