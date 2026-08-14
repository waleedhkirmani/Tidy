import pybullet as p
import pybullet_data
from gymnasium import spaces
import numpy as np
import time
import math
import random
from .robot import Robot
from .small_cube import SmallCube
from .tray import Tray

WORKSPACE_LOW = np.array([0.15, -0.4, 0.0], dtype=np.float32)
WORKSPACE_HIGH = np.array([1.0, 0.4, 0.8], dtype=np.float32)

# Curriculum: start this fraction of episodes already hovering in-band above
# the cube so the grasp->lift->place chain is actually sampled by the critic.
CURRICULUM_PROB = 0.0
CURRICULUM_DESCEND_Z = 0.02  # lands EE ~0.045 (IK residual), proven firm grip
HOLD_INCOME = 0.04  # per-step reward while holding a real (clamped) grasp
CUBE_SPAWN_X = (0.2, 0.7)
CUBE_SPAWN_Y = (-0.4, 0.4)


class TidyEnv:
    def __init__(
        self, gui=True, randomize_cube_spawn=False, randomize_tray_spawn=False
    ):
        self.randomize_cube_spawn = randomize_cube_spawn
        self.randomize_tray_spawn = randomize_tray_spawn
        self.cube_spawn_spread = 0.1  # 0 = trained anchor, 1 = full range

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
        self.max_episode_steps = 400

        # Physics
        # pybullet corrupts state when multiple connections coexist in one
        # process; this codebase only ever uses one env at a time, so drop any
        # previous connection first.
        try:
            p.disconnect()
        except Exception:
            pass
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

        self.tray = self._spawn_tray()
        self.small_cube = self._spawn_small_cube()
        self.robot.cube_id = self.small_cube.id

        self.robot.reset()
        self.robot.open_claw()
        if random.random() < CURRICULUM_PROB:
            # Descend onto the cube with the claw open (same motion as the
            # scripted demo) so episodes start one close away from a firm grip.
            cx, cy = self.small_cube.get_current_pos()[:2]
            z = 0.30
            while z > CURRICULUM_DESCEND_Z:
                z = max(z - 0.05, CURRICULUM_DESCEND_Z)
                self._advance_toward([cx, cy, z])
        for _ in range(40):
            p.stepSimulation()
        self._previous_cube_height = None
        self._old_distance_from_cube = None
        self._band_bonus_claimed = False
        self._close_executed = False
        self._old_cube_distance_from_tray = None
        self._prev_gripping = False
        self._grasp_claimed = False
        self._tray_min_cube_height = None  # monotone-min descent tracker
        self._old_center_distance = None  # center-gradient tracker
        self._old_band_dist = None  # band-center gradient tracker
        self.success_status = False
        self._start_landing = False
        self.step_count = 0

        info = {}

        return self._get_obs(), info

    def step(self, action):

        self.step_count += 1

        # The action would be delta x, delta y, delta z, and gripper 0/1
        if action[3] <= 0.5 and not self.robot.gripper_closed:
            # Gated like a gripper controller refusing a misaligned close:
            # in the window -> pure close (moving + closing in one step never
            # clamps); outside the window -> wait in place with the claw open,
            # so a premature close can't waste a step and force a slow reopen.
            if self._in_grasp_window():
                self.robot.close_claw()
                self._close_executed = True
        else:
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
        prev_band_dist = self._old_band_dist
        self._old_band_dist = abs(self.robot.get_end_effector_pos()[2] - 0.03)
        r = {
            "reach": 0,
            "grasp": 0,
            "lift": 0,
            "hold": 0,
            "approach": 0,
            "lower": 0,
            "success": 0,
        }
        gripping_now = self.robot.is_gripping()
        if gripping_now and not self._prev_gripping and not self._grasp_claimed:
            r["grasp"] = self._grasp_reward()
            self._grasp_claimed = True
        if self._close_executed:
            r["grasp"] += 1.0  # one-shot: executed the close inside the grasp band
        self._close_executed = False
        self._prev_gripping = gripping_now
        if not gripping_now:
            r["reach"] = self._reach_cube_reward()
            if self._in_grasp_window():
                r["reach"] += 0.1  # window income: last-cm gradient into the grasp band
            # descent-into-window: a claw-open descent over the cube pays, so a
            # far spawn hovering just above the band gets a gradient to drop in
            if not self.robot.gripper_closed:
                ee = self.robot.get_end_effector_pos()
                cube = self.small_cube.get_current_pos()
                aligned = abs(ee[0] - cube[0]) < 0.04 and abs(ee[1] - cube[1]) < 0.04
                band_dist = abs(ee[2] - 0.03)
                if aligned and prev_band_dist is not None:
                    # ponytail: gradient to the band center — descending above
                    # the band pays, overshooting below it pays negative, so
                    # the EE parks in the window where the close can fire
                    r["reach"] += (prev_band_dist - band_dist) * 10
        else:
            if not self._start_landing:
                r["lift"] = self._lift_reward()
            r["hold"] = HOLD_INCOME  # firm grip income; makes gripping net-positive
            if not self._is_cube_centralized_wrt_tray():
                r["approach"] = self._approach_tray_reward()
            else:
                if not self._start_landing:
                    self._start_landing = True
                r["lower"] = self._lower_into_tray_reward()
        if self._get_cube_distance_from_tray() < 0.15:
            r["success"], self.success_status = self._success_reward()
        self.step_rewards = r
        floor_penalty = max(0.0, 5.0 * (0.02 - self.robot.get_end_effector_pos()[2]))
        return sum(r.values()) - 0.01 - floor_penalty

    def _reach_cube_reward(self):
        current_distance = self._get_distance_from_cube()
        if (
            self._old_distance_from_cube is None
        ):  # this is for the first step after reset only
            self._old_distance_from_cube = current_distance
            return 0
        reward = (self._old_distance_from_cube - current_distance) * 10
        if not self._band_bonus_claimed and self._in_grasp_window():
            self._band_bonus_claimed = True
            reward += 0.5  # one-shot: reached the grasp band
        self._old_distance_from_cube = current_distance
        return reward

    def _in_grasp_window(self):
        ee = self.robot.get_end_effector_pos()
        cube = self.small_cube.get_current_pos()
        aligned = abs(ee[0] - cube[0]) < 0.03 and abs(ee[1] - cube[1]) < 0.03
        # ponytail: z-bottom widened (0.02 -> 0.005) so a far spawn that dives
        # just below the old band (EE ~0.013, aligned, close intent) can close
        return aligned and 0.005 <= ee[2] <= 0.04

    def _grasp_reward(self):
        return 3 if self.robot.is_gripping() else 0

    def _lift_reward(self):
        if self.robot.is_gripping():
            current_height = self.small_cube.get_current_pos()[2]

            if self._previous_cube_height is None:
                self._previous_cube_height = self.small_cube.get_current_pos()[2]
            height_diff = current_height - self._previous_cube_height
            self._previous_cube_height = current_height
            if current_height > 0.75:
                self._previous_cube_height = current_height
                return 0
            return height_diff * 5  # lift pays like carry
        else:
            return 0

    def _approach_tray_reward(
        self,
    ):  # TODO: Maybe add a feature where it does not count z at all
        current_distance = self._get_cube_distance_from_tray()
        if self._old_cube_distance_from_tray is None:
            self._old_cube_distance_from_tray = current_distance
            return 0
        reward = (self._old_cube_distance_from_tray - current_distance) * 10
        self._old_cube_distance_from_tray = current_distance
        # ponytail: approach pays only once the cube is lifted above the tray
        # rim (0.25 = ~20cm clearance); carrying low clips the rim and is the
        # "into the side of the tray" failure.
        if self.small_cube.get_current_pos()[2] < 0.25:
            return reward / 2
        return reward

    def _lower_into_tray_reward(self):
        # Reward staying centralized wrt to the tray
        if self._is_cube_centralized_wrt_tray():
            reward = 0.005
        else:
            reward = 0

        if self.robot.gripper_closed and not self._is_cube_in_tray():
            current_height = self.small_cube.get_current_pos()[2]
            if self._tray_min_cube_height is None:
                self._tray_min_cube_height = current_height
            elif current_height < self._tray_min_cube_height:
                # ponytail: monotone-min — the full drop credits once; raising
                # never resets, so up/down oscillation can't re-farm the descent
                reward += (self._tray_min_cube_height - current_height) * 13
                self._tray_min_cube_height = current_height
            # ponytail: continuous center-gradient — approach switches off once
            # the cube enters the tray footprint, so pull toward the mouth
            # center while descending instead of dropping off-center
            center_dist = self._get_cube_distance_from_tray()
            if self._old_center_distance is None:
                self._old_center_distance = center_dist
            else:
                reward += (self._old_center_distance - center_dist) * 10
                self._old_center_distance = center_dist
        # if not self.robot.gripper_closed:
        # reward -= 1
        return reward

    def _success_reward(self):
        reward = 0
        fingers_low = self.robot.get_end_effector_pos()[2] < self.tray.top + 0.02
        if self._is_cube_in_tray() and fingers_low:
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
        cube = np.array(self.small_cube.get_current_pos())
        grasp_target = cube  # EE is the finger frame; grip happens at cube center
        return np.linalg.norm(
            np.array(self.robot.get_end_effector_pos()) - grasp_target
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
            np.array(self.small_cube.get_current_pos())[:2]
            - np.array(self.tray.get_current_pos())[:2]
        )

    def _is_cube_centralized_wrt_tray(self):
        cube_pos = self.small_cube.get_current_pos()
        tray_pos = self.tray.get_current_pos()
        return (
            abs(cube_pos[0] - tray_pos[0]) < self.tray.half_length
            and abs(cube_pos[1] - tray_pos[1]) < self.tray.half_width
        )

    def set_cube_spawn_spread(self, fraction):
        self.cube_spawn_spread = fraction

    def _sample_cube_spawn(self):
        f = self.cube_spawn_spread
        cx = 0.45
        cy = 0.40 * (1 - f)  # lerp center 0.40 -> 0.0
        tray_pos = self.tray.get_current_pos()
        while True:
            x = random.uniform(cx - 0.25 * f, cx + 0.25 * f)
            y = random.uniform(cy - 0.40 * f, cy + 0.40 * f)
            if not (
                abs(x - tray_pos[0]) < self.tray.half_length + 0.05
                and abs(y - tray_pos[1]) < self.tray.half_width + 0.05
            ):
                return [x, y, 0.01]

    def _spawn_small_cube(self, basePos=[0.45, 0.4, 0.01], rgba=[0, 1, 0, 1]):
        if self.randomize_cube_spawn:
            basePos = self._sample_cube_spawn()
        self.small_cube = SmallCube(basePos, rgba)
        return self.small_cube

    def _sample_tray_spawn(self):
        return [random.uniform(*CUBE_SPAWN_X), random.uniform(*CUBE_SPAWN_Y), 0.0]

    def _spawn_tray(self, basePos=[0.625, -0.30, 0.0], globScale=0.25):
        if self.randomize_tray_spawn:
            basePos = self._sample_tray_spawn()
        self.tray = Tray(basePos, globScale)

        return self.tray

    def _is_cube_in_tray(self):
        # ponytail: +0.01 margin — the funnel-shaped tray wedges the hand with
        # the cube ~0.5mm above the rim; the cube is seated in the tray mouth.
        return (
            self._is_cube_centralized_wrt_tray()
            and self.small_cube.get_current_pos()[2] < self.tray.top + 0.01
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
