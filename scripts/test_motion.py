import math

import pybullet as p

from envs.tidy_env import TidyEnv

PLANE_Z = 0.0


def plane_id(env):
    known = {env.robot.id, env.small_cube.id, env.tray.id}
    for i in range(p.getNumBodies()):
        body = p.getBodyUniqueId(i)
        if body not in known:
            return body
    raise RuntimeError("plane not found")


def descent_trace(env, target, n_calls):
    min_z = float("inf")
    floor_contacts = 0
    plane = plane_id(env)
    for _ in range(n_calls):
        env._advance_toward(target)
        z = env.robot.get_end_effector_pos()[2]
        min_z = min(min_z, z)
        if p.getContactPoints(bodyA=env.robot.id, bodyB=plane):
            floor_contacts += 1
    return min_z, floor_contacts


env = TidyEnv(gui=False)
env.reset()

# 1. home sits in the feasible basin: forearm roll away from its 0 upper limit
j4 = p.getJointState(env.robot.id, 3)[0]
assert j4 <= -0.3, f"home forearm roll j4={j4:.3f} must be in feasible basin (<= -0.3)"
ee = env.robot.get_end_effector_pos()
assert math.dist(ee, [0.5, 0, 0.7]) < 0.05, f"home ee {ee} far from nominal"

# 2. descent toward the cube approach stays above the floor, no contacts
cube = env.small_cube.get_current_pos()
target = [cube[0], cube[1], 0.15]
min_z, floor_contacts = descent_trace(env, target, 12)
assert min_z > 0.10, f"ee dived into floor during descent: min z {min_z:.3f}"
assert floor_contacts == 0, f"robot touched floor {floor_contacts} steps during descent"
j4 = p.getJointState(env.robot.id, 3)[0]
assert j4 <= -0.3, f"forearm roll left feasible basin during descent (j4={j4:.3f})"

# 3. descend onto the cube and grip it
env._advance_toward([cube[0], cube[1], 0.055])
env.robot.close_claw()
for _ in range(20):
    p.stepSimulation()
assert env.robot.is_gripping(), "failed to grip cube after clean descent"

print("OK: home feasible, descent clean, grasp held")
