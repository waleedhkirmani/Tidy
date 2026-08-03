import pybullet as p
import pybullet_data
from gymnasium import spaces
import numpy as np
import time
import math
from .robot import Robot
from .small_cube import SmallCube
from .tray import Tray

WORKSPACE_LOW = np.array([0.15, -0.4, 0.1], dtype=np.float32)
WORKSPACE_HIGH = np.array([1.0, 0.4, 0.8], dtype=np.float32)


class TidyEnv:
    def __init__(self, gui=True):

        # RL
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, 0], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        # Environment
        self.workspace_low = WORKSPACE_LOW
        self.workspace_high = WORKSPACE_HIGH
        self.max_episode_steps = 200

        # Physics
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self._setup_physics()

        if gui:
            self._setup_graphics()

        # Robot
        self.robot = Robot()

    def reset(self):
        if hasattr(self, "small_cube") and self.small_cube:
            p.removeBody(self.small_cube.id)
        if hasattr(self, "tray") and self.tray:
            p.removeBody(self.tray.id)

        self.small_cube = self._spawn_small_cube()
        self.robot.cube_id = self.small_cube.id
        self.tray = self._spawn_tray()

        self.robot.reset()

        self.robot.open_claw()
        for _ in range(40):
            p.stepSimulation()
        self._previous_cube_height = None
        self._old_distance_from_cube = None
        self._old_cube_distance_from_tray = None
        self.success_status = False
        self.step_count = 0

        info = {}

        return self._get_obs(), info

    def step(self, action):

        self.step_count += 1

        # The action would be delta x, delta y, delta z, and gripper 0/1
        target_position = self._calculate_target_position(
            action[:3], self.robot.get_end_effector_pos()
        )

        target_position = self._clip_target_position(target_position)
        self._advance_toward(target_position)

        if action[3] > 0.5:
            self.robot.open_claw()
        else:
            self.robot.close_claw()

        reward = self._calculate_reward()
        terminated = self.success_status
        truncated = self.step_count >= self.max_episode_steps
        info = {"rewards": self.step_rewards}
        return (self._get_obs(), reward, terminated, truncated, info)

    def _calculate_reward(self):
        r = {"reach": 0, "grasp": 0, "lift": 0, "approach": 0, "lower": 0, "success": 0}
        if not self.robot.is_gripping():
            r["reach"] = self._reach_cube_reward()
            r["grasp"] = self._grasp_reward()
        else:
            r["lift"] = self._lift_reward()
            if not self._is_cube_centralized_wrt_tray():
                r["approach"] = self._approach_tray_reward()
            else:
                r["lower"] = self._lower_into_tray_reward()
                r["success"], self.success_status = self._success_reward()
        self.step_rewards = r
        return sum(r.values()) - 0.001

    def _reach_cube_reward(self):
        current_distance = self._get_distance_from_cube()
        if (
            self._old_distance_from_cube is None
        ):  # this is for the first step after reset only
            self._old_distance_from_cube = current_distance
            return 0
        reward = self._old_distance_from_cube - current_distance
        self._old_distance_from_cube = current_distance
        return reward

    def _grasp_reward(self):
        return 1 if self.robot.is_gripping() else 0

    def _lift_reward(self):
        if self.robot.is_gripping():
            current_height = self.small_cube.get_current_pos()[2]

            if self._previous_cube_height is None:
                self._previous_cube_height = self.small_cube.get_current_pos()[2]
            height_diff = current_height - self._previous_cube_height
            self._previous_cube_height = current_height
            if current_height > 0.6:
                self._previous_cube_height = current_height
                return 0
            return height_diff
        else:
            return 0

    def _approach_tray_reward(
        self,
    ):  # TODO: Maybe add a feature where it does not count z at all
        current_distance = self._get_cube_distance_from_tray()
        if self._old_cube_distance_from_tray is None:
            self._old_cube_distance_from_tray = current_distance
            return 0
        reward = self._old_cube_distance_from_tray - current_distance
        self._old_cube_distance_from_tray = current_distance
        return reward

    def _lower_into_tray_reward(self):
        # Reward staying centralized wrt to the tray
        if self._is_cube_centralized_wrt_tray():
            reward = 0.5
        else:
            reward = 0

        current_height = self.small_cube.get_current_pos()[2]

        if self._previous_cube_height is None:
            self._previous_cube_height = current_height
        else:
            height_diff = self._previous_cube_height - current_height
            reward += height_diff
            self._previous_cube_height = current_height
        if not self.robot.gripper_closed:
            reward -= 1
        return reward

    def _success_reward(self):
        reward = 0
        if self._is_cube_in_tray():
            reward += 50
            return reward, True
        return 0, False

    def _setup_physics(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

    def _setup_graphics(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=1.7,
            cameraYaw=90,
            cameraPitch=-35,
            cameraTargetPosition=[0.5, 0.0, 0.2],
        )

    def _get_distance_from_cube(self):
        return np.linalg.norm(
            np.array(self.robot.get_end_effector_pos())
            - np.array(self.small_cube.get_current_pos())
        )

    def _clip_target_position(self, target_position):
        return np.clip(target_position, self.workspace_low, self.workspace_high)

    def _get_obs(self):
        return np.array(
            [
                *self.robot.get_end_effector_pos(),
                *self.small_cube.get_current_pos(),
                *self.tray.get_current_pos(),
            ],
            dtype=np.float32,
        )

    def _get_cube_distance_from_tray(self):
        return np.linalg.norm(
            np.array(self.small_cube.get_current_pos())
            - np.array(self.tray.get_current_pos())
        )

    def _is_cube_centralized_wrt_tray(self):
        cube_pos = self.small_cube.get_current_pos()
        tray_pos = self.tray.get_current_pos()
        return (
            abs(cube_pos[0] - tray_pos[0]) < self.tray.half_length
            and abs(cube_pos[1] - tray_pos[1]) < self.tray.half_width
        )

    def _spawn_small_cube(self, basePos=[0.45, 0.4, 0.01], rgba=[0, 1, 0, 1]):
        self.small_cube = SmallCube(basePos, rgba)
        return self.small_cube

    def _spawn_tray(self, basePos=[0.625, -0.30, 0.0], globScale=0.25):
        self.tray = Tray(basePos, globScale)

        return self.tray

    def _is_cube_in_tray(self):
        return (
            self._is_cube_centralized_wrt_tray()
            and self.small_cube.get_current_pos()[2] < self.tray.top
        )

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

    def _advance_toward(self, target_position, n_steps=60, f=200):
        # fixed-tick chase: one env step = one short burst, no convergence wait
        target_position = np.asarray(target_position, dtype=np.float32)
        target_orientation = p.getQuaternionFromEuler([0, -math.pi, 0])
        forces = [f] * 7
        forces[1] = 1000

        for _ in range(n_steps):
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

    def _move_to(
        self, target_position, f=200, max_steps=2000, stall_steps=50, stall_eps=1e-4
    ):
        target_position = np.asarray(target_position, dtype=np.float32)
        last_distance = float("inf")
        stall_count = 0
        target_orientation = p.getQuaternionFromEuler([0, -math.pi, 0])
        forces = [f] * 7
        forces[1] = 1000

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

            if distance < 0.009:
                return True
            if abs(last_distance - distance) < stall_eps:
                stall_count += 1
                if stall_count > stall_steps:
                    return False
            else:
                stall_count = 0
            last_distance = distance
        print("move_to: did not converge")
        return False

    def motion(self):
        self._move_to([0.43, 0.42, 0.7])
        self._move_to([0.43, 0.42, 0.3])

        self._move_to([0.43, 0.42, 0.01])
        self.robot.close_claw()
        zz = 0.1
        yy = 0.42
        xx = 0.43

        zz += 0.5
        self._move_to([xx, yy, zz])
        yy -= 0.425
        self._move_to([xx, yy, zz])
        xx = 0.525
        yy = -0.30
        self._move_to([xx, yy, zz])
        zz -= 0.25
        self._move_to([xx, yy, zz])
        zz -= 0.25
        self._move_to([xx, yy, zz])
        zz -= 0.25
        self._move_to([xx, yy, zz])
        time.sleep(1 / 140)
        self.robot.open_claw()
        zz += 0.5
        self._move_to([xx, yy, zz])
