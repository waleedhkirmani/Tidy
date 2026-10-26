import numpy as np
from envs.tidy_env import TidyEnv


def obs(
    ee,
    cube,
    tray,
    fingers=(0.04, 0.04),
    gripping=0,
    ee_vel=(0.0, 0.0, 0.0),
):
    return np.asarray(
        [*ee, *cube, *tray, *fingers, gripping, *np.subtract(cube, ee), *ee_vel],
        dtype=np.float32,
    )


env = TidyEnv(gui=False)

# Markov observation: shape and finiteness
o, _ = env.reset()
assert env.observation_space.shape == (18,), env.observation_space.shape
assert o.shape == (18,) and o.dtype == np.float32 and np.isfinite(o).all(), o

# Run a random episode; every returned reward must be exactly reproducible
# from (prev_obs, obs, action) given the pre-step tracker state (_grasp_claimed
# is the one intentional exception -- it's snapshotted and restored here).
for _ in range(40):
    prev = env._prev_obs.copy()
    action = env.action_space.sample()
    claimed = env._grasp_claimed
    o, reward, terminated, truncated, info = env.step(action)
    env._grasp_claimed = claimed
    assert o.shape == (18,) and np.isfinite(o).all(), o
    recomputed = env._calculate_reward(prev, o, action)
    assert reward == recomputed, (reward, recomputed)
    for k in ("reach", "grasp", "lift", "hold", "approach", "lower", "success"):
        assert k in info["rewards"]
    assert abs(sum(info["rewards"].values()) - 0.01 - max(0.0, 1.0 * (0.02 - o[2])) - reward) < 1e-4
    if terminated:
        o, _ = env.reset()

# No episode-scoped trackers left on the env (_grasp_claimed is the one
# intentional one, handled above; the recompute test reproduces its state).
env._grasp_claimed = False
banned = (
    "min", "band", "start_landing", "close_bonus", "prev_gripping",
)
assert not [k for k in vars(env) if any(b in k for b in banned)], [
    k for k in vars(env) if any(b in k for b in banned)
]

# Reach is a stateless potential: closer pays positive, backing off pays
# negative, and the sum over a go out-and-back telescopes to the same net.
tray = (0.625, -0.30, 0.0)
a = np.zeros(4, dtype=np.float32)
x_cur = obs(ee=(0.5, 0.0, 0.3), cube=(0.4, 0.0, 0.02), tray=tray)
x_far = obs(ee=(0.5, 0.0, 0.3), cube=(0.2, 0.0, 0.02), tray=tray)
def cmp(prev, cur):
    env._calculate_reward(prev, cur, a)
    return env.step_rewards["reach"]


r_in = cmp(x_far, x_cur)
r_out = cmp(x_cur, x_far)
assert r_in > 0 and r_out < 0, (r_in, r_out)
assert abs(r_in + r_out) < 1e-4, (r_in, r_out)  # telescopes: out-and-back nets ~0

# Grasp pays +10 exactly once per episode, on the first clamp. Hold, release,
# and any re-grasp pay 0 -- releasing at the tray is free, and re-grip cycling
# can't farm anything.
env._grasp_claimed = False
x_grip = obs(ee=(0.4, 0.0, 0.3), cube=(0.4, 0.0, 0.02), tray=tray, gripping=1)
env._calculate_reward(x_far, x_grip, a)
assert env.step_rewards["grasp"] == 10.0
env._calculate_reward(x_grip, x_grip, a)
assert env.step_rewards["grasp"] == 0.0
env._calculate_reward(x_grip, x_far, a)
assert env.step_rewards["grasp"] == 0.0  # release: no penalty
env._calculate_reward(x_far, x_grip, a)
assert env.step_rewards["grasp"] == 0.0  # one-shot: re-grasp doesn't re-pay
env._calculate_reward(x_grip, x_far, a)
assert env.step_rewards["grasp"] == 0.0  # ...and releasing an unclaimed grip never paid anyway

print("reward/obs smoke: OK")