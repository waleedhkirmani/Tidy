import numpy as np
from envs.tidy_env import TidyEnv

env = TidyEnv(gui=False, randomize_cube_spawn=True, randomize_tray_spawn=True)


def run(f, n=30):
    env.set_spawn_spread(f)
    trays, cubes = [], []
    for _ in range(n):
        env.reset()
        trays.append(env.tray.get_current_pos())
        cubes.append(env.small_cube.get_current_pos())
        tray, cube = trays[-1], cubes[-1]
        # cube never overlaps the tray footprint (+0.05 margin)
        assert not (
            abs(cube[0] - tray[0]) < env.tray.half_length + 0.05
            and abs(cube[1] - tray[1]) < env.tray.half_width + 0.05
        ), (f, cube, tray)
    return np.asarray(trays), np.asarray(cubes)


# Tray spread expands with the ramp: variance strictly grows with f.
vars_low = run(0.1)[0][:, :2].var(axis=0)
vars_mid = run(0.5)[0][:, :2].var(axis=0)
vars_full = run(1.0)[0][:, :2].var(axis=0)
assert (vars_low < vars_mid).all() and (vars_mid < vars_full).all(), (
    vars_low,
    vars_mid,
    vars_full,
)

# Neutral anchors (small f) sit symmetric about the workspace center x=0.575:
# cube 0.475 on the left, tray 0.675 on the right.
trays, cubes = run(0.1, n=60)
cube_mean_x = cubes[:, 0].mean()
tray_mean_x = trays[:, 0].mean()
center = 0.575
assert abs(cube_mean_x - 0.475) < 0.05, cube_mean_x
assert abs(tray_mean_x - 0.675) < 0.05, tray_mean_x
assert abs(cube_mean_x + tray_mean_x - 2 * center) < 0.1, (cube_mean_x, tray_mean_x)

print("spawn ramp: OK")