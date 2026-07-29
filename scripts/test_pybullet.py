import math
import time

import pybullet as p
import pybullet_data

# Connect to the PyBullet GUI
p.connect(p.GUI)

# Use PyBullet's built-in assets
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Physics settings
p.setGravity(0, 0, -9.81)

# Load the ground plane
p.loadURDF("plane.urdf")

# Load the Franka Panda robot
robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

# Camera
p.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=[0.5, 0, 0.2],
)

# Print joint information
print(f"Number of joints: {p.getNumJoints(robot)}\n")

for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(f"{i:2d}: {info[1].decode()}")

# Disable all default motors
for joint in range(p.getNumJoints(robot)):
    p.setJointMotorControl2(
        bodyUniqueId=robot, jointIndex=joint, controlMode=p.VELOCITY_CONTROL, force=0
    )

t = 0.0

# Main simulation loop
while p.isConnected():
    # Oscillate Joint 1 between -1.5 and +1.5 radians
    target = 1.5 * math.sin(t)

    p.setJointMotorControl2(
        bodyUniqueId=robot,
        jointIndex=0,  # panda_joint1
        controlMode=p.POSITION_CONTROL,
        targetPosition=target,
        force=5000,
    )

    p.stepSimulation()

    time.sleep(1 / 240)

    t += 0.01

p.disconnect()
