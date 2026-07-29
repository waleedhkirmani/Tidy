import pybullet as p
import pybullet_data
import time
import math
from .small_cube import SmallCube
from .tray import Tray


class TidyEnv:
    def __init__(self):
        p.connect(p.GUI)
        # Use pybullet's built in assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        initial_pos = [0.5, 0.0, 0.7]
        initial_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        joint_angles = p.calculateInverseKinematics(
            self.robot,
            11,
            initial_pos,
            targetOrientation=initial_orn,
        )
        for i in range(7):
            p.resetJointState(self.robot, i, joint_angles[i])

        p.resetDebugVisualizerCamera(
            cameraDistance=1.7,
            cameraYaw=90,
            cameraPitch=-35,
            cameraTargetPosition=[0.5, 0.0, 0.2],
        )

        self.lower_limits = []
        self.upper_limits = []
        self.joint_ranges = []
        self.rest_poses = []

        for joint in range(7):
            info = p.getJointInfo(self.robot, joint)
            lower = info[8]
            upper = info[9]

            self.lower_limits.append(lower)
            self.upper_limits.append(upper)
            self.joint_ranges.append(upper - lower)
            self.rest_poses.append(p.getJointState(self.robot, joint)[0])

    def reset(self):
        if hasattr(self, "small_cube") and self.small_cube:
            p.removeBody(self.small_cube.id)
        if hasattr(self, "tray") and self.tray:
            p.removeBody(self.tray.id)

        self.small_cube = self.spawn_small_cube()
        self.tray = self.spawn_tray()

        initial_pos = [0.5, 0.0, 0.7]
        initial_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        joint_angles = p.calculateInverseKinematics(
            self.robot,
            11,
            initial_pos,
            targetOrientation=initial_orn,
        )
        for i in range(7):
            p.resetJointState(self.robot, i, joint_angles[i])

        self.open_claw()

    def spawn_small_cube(self, basePos=[0.45, 0.4, 0.01], rgba=[0, 1, 0, 1]):
        self.small_cube = SmallCube(basePos, rgba)
        return self.small_cube

    def spawn_tray(self, basePos=[0.625, -0.30, 0.0], globScale=0.25):
        self.tray = Tray(basePos, globScale)

        return self.tray

    def open_claw(self, f=200, target_position=0.04):
        for joint in [9, 10]:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=f,
            )
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1 / 240)

    def close_claw(self, f=200, target_position=0.0):
        for joint in [9, 10]:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=f,
            )
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1 / 240)

    def check_ik_reachability(self, target_position, target_orientation):
        saved = [p.getJointState(self.robot, i)[0] for i in range(7)]
        joint_angles = p.calculateInverseKinematics(
            self.robot,
            11,
            target_position,
            target_orientation,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.rest_poses,
        )
        for i in range(7):
            p.resetJointState(self.robot, i, joint_angles[i])
        ee_pos = p.getLinkState(self.robot, 11)[4]
        residual = math.dist(ee_pos, target_position)
        for i, pos in enumerate(saved):
            p.resetJointState(self.robot, i, pos)  # restore real state
        return residual

    def move_to(
        self, target_position, f=200, max_steps=2000, stall_steps=50, stall_eps=1e-4
    ):
        last_distance = float("inf")
        stall_count = 0
        target_orientation = p.getQuaternionFromEuler([0, -math.pi, 0])
        forces = [f] * 7
        forces[1] = 1000
        # debugging shit
        residual = self.check_ik_reachability(target_position, target_orientation)
        print(f"IK residual: {residual:.4f}")

        for _ in range(max_steps):
            joint_angles = p.calculateInverseKinematics(
                self.robot,
                11,
                target_position,
                target_orientation,
                lowerLimits=self.lower_limits,
                upperLimits=self.upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=self.rest_poses,
            )
            # print(joint_angles[:7])
            for i in range(7):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_angles[i],
                    force=forces[i],
                    positionGain=0.3,
                    velocityGain=1.0,
                    maxVelocity=4.0,
                )
            p.stepSimulation()
            time.sleep(1 / 240)

            current_position = p.getLinkState(self.robot, 11)[4]
            distance = math.dist(current_position, target_position)

            # print("Target :", target_position)
            # print("Current:", current_position)
            # print("Distance:", distance)
            if distance < 0.009:
                return True
            if abs(last_distance - distance) < stall_eps:
                stall_count += 1
                if stall_count > stall_steps:
                    current_angles = [
                        p.getJointState(self.robot, i)[0] for i in range(7)
                    ]
                    print("commanded:", [round(a, 3) for a in joint_angles[:7]])
                    print("actual:   ", [round(a, 3) for a in current_angles])
                    print(f"move_to: stalled at distance={distance:.3f}")
                    return False
            else:
                stall_count = 0
            last_distance = distance
        print("move_to: did not converge")
        return False

    def motion(self):
        self.move_to([0.43, 0.42, 0.7])
        self.move_to([0.43, 0.42, 0.3])

        self.move_to([0.43, 0.42, 0.01])
        self.close_claw()
        zz = 0.1
        yy = 0.42
        xx = 0.43

        zz += 0.5
        self.move_to([xx, yy, zz])
        yy -= 0.425
        self.move_to([xx, yy, zz])
        xx = 0.525
        yy = -0.30
        self.move_to([xx, yy, zz])
        zz -= 0.25
        self.move_to([xx, yy, zz])
        zz -= 0.25
        self.move_to([xx, yy, zz])
        zz -= 0.25
        self.move_to([xx, yy, zz])
        time.sleep(1 / 140)
        self.open_claw()
        zz += 0.5
        self.move_to([xx, yy, zz])
