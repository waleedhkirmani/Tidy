# %%

import pybullet as p
import pybullet_data
import time
import math
# %%


# Connect to the PyBullet GUI
p.connect(p.GUI)

# Use PyBullet's built-in assets
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Physics settings
p.setGravity(0, 0, -9.81)

# Load the ground plane
p.loadURDF("plane.urdf")

# Load the Franka Panda robot
robot = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION,
)

target_pos = [0.5, 0.0, 0.4]
target_orn = p.getQuaternionFromEuler([0, -math.pi, 0])

joint_angles = p.calculateInverseKinematics(
    robot,
    11,
    target_pos,
    targetOrientation=target_orn,
)

for i in range(7):
    p.resetJointState(robot, i, joint_angles[i])

# Camera
p.resetDebugVisualizerCamera(
    cameraDistance=1.7,
    cameraYaw=90,
    cameraPitch=-35,
    cameraTargetPosition=[0.5, 0.0, 0.2],
)

# Print joint information
print(f"Number of joints: {p.getNumJoints(robot)}\n")

for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(f"{i:2d}: {info[1].decode()}")

lower_limits = []
upper_limits = []
joint_ranges = []
rest_poses = []

for joint in range(7):
    info = p.getJointInfo(robot, joint)

    lower = info[8]
    upper = info[9]

    lower_limits.append(lower)
    upper_limits.append(upper)
    joint_ranges.append(upper - lower)

    # current joint position
    rest_poses.append(p.getJointState(robot, joint)[0])
print("Lower:", lower_limits)
print("Upper:", upper_limits)
print("Ranges:", joint_ranges)
print("Rest :", rest_poses)

# %%
# Helper function


def move_to(target_position, max_steps=2000, stall_steps=50, stall_eps=1e-4):
    last_distance = float("inf")
    stall_count = 0
    for _ in range(max_steps):
        target_orientation = p.getQuaternionFromEuler([0, -math.pi, 0])
        joint_angles = p.calculateInverseKinematics(
            robot,
            11,
            target_position,
            target_orientation,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
        )
        # print(joint_angles[:7])
        for i in range(7):
            p.setJointMotorControl2(
                bodyUniqueId=robot,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[i],
                force=500,
            )
            # for i in range(7):
            # print(
            #     f"Joint {i}:",
            #     "Current =",
            #     round(p.getJointState(robot, i)[0], 3),
            #     "Target =",
            #     round(joint_angles[i], 3),
            # )
        p.stepSimulation()
        time.sleep(1 / 240)

        current_position = p.getLinkState(robot, 11)[4]
        distance = math.dist(current_position, target_position)

        print("Target :", target_position)
        print("Current:", current_position)
        print("Distance:", distance)
        if distance < 0.009:
            return True
        if abs(last_distance - distance) < stall_eps:
            stall_count += 1
            if stall_count > stall_steps:
                print(f"move_to: stalled at distance={distance:.3f}")
                return False
        else:
            stall_count = 0
        last_distance = distance
    print("move_to: did not converge")
    return False


# %%

tray = p.loadURDF(
    "tray/traybox.urdf",
    basePosition=[0.625, -0.30, 0.0],
    useFixedBase=True,
    globalScaling=0.25,
)
# %%
p.removeBody(tray)
# %%


# opening the claw to full extent
def open_claw():
    for joint in [9, 10]:
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=joint,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.04,
            force=200,
        )
    for _ in range(1):
        p.stepSimulation()
        time.sleep(1 / 240)


def close_claw():
    for joint in [9, 10]:
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=joint,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0,
            force=200,
        )
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1 / 240)


# %%
cube = p.loadURDF("cube_small.urdf", basePosition=[0.45, 0.4, 0.01])

# %%


def place_cube():
    p.resetBasePositionAndOrientation(cube, [0.45, 0.4, 0.01], [0, 0, 0, 1])

    move_to([0.65, -0.4, 0.5])
    open_claw()
    move_to([0.65, -0.4, 0.5])

    xyz = [0.43, 0.42, 0.01]
    for i in range(10):
        move_to([0.43, 0.42, 0.5])
    for i in range(10):
        move_to([0.43, 0.42, 0.01])
    close_claw()
    zz = 0.1
    yy = 0.42
    xx = 0.43

    zz += 0.5
    move_to([xx, yy, zz])
    yy -= 0.425
    move_to([xx, yy, zz])
    xx = 0.525
    yy = -0.30
    move_to([xx, yy, zz])
    zz -= 0.25
    move_to([xx, yy, zz])
    zz -= 0.25
    move_to([xx, yy, zz])
    zz -= 0.25
    move_to([xx, yy, zz])
    time.sleep(1 / 140)
    open_claw()
    zz += 0.5
    move_to([xx, yy, zz])


# %%

place_cube()
# %%
for i in [8, 9, 10, 11]:
    state = p.getLinkState(robot, i)
    print(i, state[4])
# %%

close_claw()
# %%

p.disconnect()

# %%
p.resetDebugVisualizerCamera(
    cameraDistance=1.7,
    cameraYaw=90,
    cameraPitch=-35,
    cameraTargetPosition=[0.5, 0.0, 0.2],
)

# %%
p.changeVisualShape(cube, linkIndex=-1, rgbaColor=[0, 1, 0, 1])

# %%

move_to([0.20, 0, 0.7])
