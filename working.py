from envs.tidy_env import TidyEnv

env = TidyEnv()
obs, info = env.reset()

env.motion()
