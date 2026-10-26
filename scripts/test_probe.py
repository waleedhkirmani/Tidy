import random

import numpy as np

from envs.tidy_env import TidyEnv


def classify(env):
    """Return "grasp", "place", or "normal" for the current episode state."""
    cube = np.array(env.small_cube.get_current_pos())
    gripping = bool(env._get_obs()[11])
    ee_z = env.robot.get_end_effector_pos()[2]
    on_table = abs(cube[2] - 0.01) < 0.04  # descend nudges the cube a bit
    descended = ee_z < 0.3  # grasp probe parks low over the cube; normal starts at home 0.7
    hovering = cube[2] > 0.3
    if gripping and hovering:
        return "place"
    if not gripping and on_table and descended:
        return "grasp"
    return "normal"


def dists_to_tray(env, n=20):
    tray = np.array(env.tray.get_current_pos()[:2])
    return [float(np.hypot(*(np.array(env.small_cube.get_current_pos()[:2]) - tray))) for _ in range(n)]


def make(probe):
    random.seed(0)
    env = TidyEnv(
        gui=False,
        randomize_cube_spawn=True,
        randomize_tray_spawn=True,
        probe=probe,
    )
    env.set_spawn_spread(0.1)
    env.curriculum_prob = 1.0
    return env


def kinds_for(probe, n=20):
    env = make(probe)
    kinds = []
    for _ in range(n):
        env.reset()
        kinds.append(classify(env))
    return kinds


ks = kinds_for("grasp")
assert ks.count("grasp") == len(ks), f"grasp mode: all episodes should be grasp probes, got {ks}"

ps = kinds_for("place")
assert ps.count("grasp") == 0, f"place mode must not emit grasp probes, got {ps}"
assert ps.count("place") > 0, f"place probe never fired, got {ps}"
assert 4 <= ps.count("place") <= 16, f"place rate not ~half, got {ps}"

env = make("place")
random.seed(0)
env.curriculum_prob = 1.0
jitter = [float("nan")] * 30
seen = 0
while seen < 12:  # collect up to 12 real place starts
    env.reset()
    if classify(env) != "place":
        continue
    tray = env.tray.get_current_pos()[:2]
    cube = env.small_cube.get_current_pos()[:2]
    jitter[seen] = float(np.hypot(*[a - b for a, b in zip(cube, tray)]))
    seen += 1
assert seen == 12, f"could not collect 12 place starts (got {seen})"
assert min(jitter[:seen]) > 0.15, f"place starts too close to dead-center: {jitter[:seen]}"
assert max(jitter[:seen]) > 0.22, f"place starts never far from tray: {jitter[:seen]}"

print("ok: grasp mode -> all grasp probes")
print(f"ok: place mode -> 0 grasp, {ps.count('place')}/20 place (~half)")
print(f"ok: place hover jitter {min(jitter[:seen]):.2f}..{max(jitter[:seen]):.2f} m off-center (x or y)")