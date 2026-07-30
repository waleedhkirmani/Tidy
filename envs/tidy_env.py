import pybullet as p
import pybullet_data
from gymnasium import spaces
import numpy as np
import time
import math
from .robot import Robot
from .small_cube import SmallCube
from .tray import Tray


class TidyEnv:
    def __init__(self):

        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, 0], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        p.connect(p.GUI)
        # Use pybullet's built in assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.robot = Robot()

        p.resetDebugVisualizerCamera(
            cameraDistance=1.7,
            cameraYaw=90,
            cameraPitch=-35,
            cameraTargetPosition=[0.5, 0.0, 0.2],
        )

    def reset(self):
        if hasattr(self, "small_cube") and self.small_cube:
            p.removeBody(self.small_cube.id)
        if hasattr(self, "tray") and self.tray:
            p.removeBody(self.tray.id)

        self.small_cube = self._spawn_small_cube()
        self.tray = self._spawn_tray()

        initial_pos = [0.5, 0.0, 0.7]
        initial_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        joint_angles = p.calculateInverseKinematics(
            self.robot.id,
            11,
            initial_pos,
            targetOrientation=initial_orn,
        )
        for i in range(7):
            p.resetJointState(self.robot.id, i, joint_angles[i])

        self.robot.open_claw()

        info = {}

        return self._get_obs(), info

    def step(self, action):
        # The action would be delta x, delta y, delta z, and gripper 0/1
        target_position = self._calculate_target_position(
            action[:3], self.robot.get_end_effector_pos()
        )
        converged = self._move_to(target_position)

        if not converged:
            # Handle failure
            print("Not Converged")

        if action[3] > 0.5:
            self.robot.open_claw()
        else:
            self.robot.close_claw()

        terminated = self._is_cube_in_tray()
        return (self._get_obs(), 0, terminated, False, {})

    def _get_obs(self):
        return np.array(
            [
                *self.robot.get_end_effector_pos(),
                *self.small_cube.get_current_pos(),
                *self.tray.get_current_pos(),
            ],
            dtype=np.float32,
        )

    def _spawn_small_cube(self, basePos=[0.45, 0.4, 0.01], rgba=[0, 1, 0, 1]):
        self.small_cube = SmallCube(basePos, rgba)
        return self.small_cube

    def _spawn_tray(self, basePos=[0.625, -0.30, 0.0], globScale=0.25):
        self.tray = Tray(basePos, globScale)

        return self.tray

    def _is_cube_in_tray(self):
        cube_pos = self.small_cube.get_current_pos()
        tray_pos = self.tray.get_current_pos()

        x_ok = abs(cube_pos[0] - tray_pos[0]) < self.tray.half_length
        y_ok = abs(cube_pos[1] - tray_pos[1]) < self.tray.half_width
        z_ok = cube_pos[2] < self.tray.top

        return x_ok and y_ok and z_ok

    def check_ik_reachability(self, target_position, target_orientation):
        saved = [p.getJointState(self.robot.id, i)[0] for i in range(7)]
        joint_angles = p.calculateInverseKinematics(
            self.robot.id,
            11,
            target_position,
            target_orientation,
            lowerLimits=self.robot.lower_limits,
            upperLimits=self.robot.upper_limits,
            jointRanges=self.robot.joint_ranges,
            restPoses=self.robot.rest_poses,
        )
        for i in range(7):
            p.resetJointState(self.robot.id, i, joint_angles[i])
        ee_pos = p.getLinkState(self.robot.id, 11)[4]
        residual = math.dist(ee_pos, target_position)
        for i, pos in enumerate(saved):
            p.resetJointState(self.robot.id, i, pos)  # restore real state
        return residual

    def _calculate_target_position(self, action, current_position):
        current_position = np.asarray(current_position, dtype=np.float32)
        return current_position + action[:3]

    def _move_to(
        self, target_position, f=200, max_steps=2000, stall_steps=50, stall_eps=1e-4
    ):
        target_position = np.asarray(target_position, dtype=np.float32)
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
                self.robot.id,
                11,
                target_position,
                target_orientation,
                lowerLimits=self.robot.lower_limits,
                upperLimits=self.robot.upper_limits,
                jointRanges=self.robot.joint_ranges,
                restPoses=self.robot.rest_poses,
            )
            # print(joint_angles[:7])
            for i in range(7):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot.id,
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

            current_position = np.asarray(
                p.getLinkState(self.robot.id, 11)[4], dtype=np.float32
            )
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
                        p.getJointState(self.robot.id, i)[0] for i in range(7)
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
        self.robot.close_claw()
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
        self.robot.open_claw()
        zz += 0.5
        self.move_to([xx, yy, zz])
