import time

import pybullet as p

from envs.tidy_env import TidyEnv

env = TidyEnv(gui=True, randomize_cube_spawn=True, randomize_tray_spawn=True)
env.set_spawn_spread(0.1)
p.setRealTimeSimulation(1)
p.resetDebugVisualizerCamera(
    cameraDistance=1.2, cameraYaw=60, cameraPitch=-40, cameraTargetPosition=[0.575, 0, 0.2]
)


def show(label, probe, curriculum_prob):
    env.probe = probe
    env.curriculum_prob = curriculum_prob
    env.reset()
    print(f"--- {label} ---")
    time.sleep(3)


show("normal start", "grasp", 0.0)
show("grasp probe (above cube)", "grasp", 1.0)
show("place probe (above tray, cube in hand)", "place", 1.0)

p.disconnect()
print("done")