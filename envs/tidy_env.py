import pybullet as p
import pybullet_data
import gymnasium
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
# Hover-point jitter for the place probe: outside the funnel mouth
# (full-spread half-extents 0.175 x 0.25) so a straight drop doesn't land.
PLACE_HOVER_JITTER = (0.18, 0.28)
# EE_x stalls (cube droops) past ~0.77 at the low hover height; keep targets
# inside the arm's reach box or the offset silently collapses.
PLACE_REACH_X = 0.77
HOLD_INCOME = 0.1  # per-step reward while holding a real (clamped) grasp
# Ramp over ABSOLUTE episode count (cumulative across Colab sessions via
# load_checkpoint), so resuming a checkpoint keeps the ramp advancing.
RAMP_EPISODES = 4000  # spread reaches full randomization after this many episodes
CURRICULUM_PROBE = (
    0.5  # fraction of early episodes starting above the cube; decays with spread
)


def ramp_spread(episode):
    return max(0.1, min(1.0, episode / RAMP_EPISODES))


def ramp_curriculum_prob(spread):
    return CURRICULUM_PROBE * (1 - spread)


# Spawn ramp anchors, symmetric about the workspace center (0.575, 0): the
# cube and tray expand outward from these neutral points as ramp_spread(f)
# grows over RAMP_EPISODES. HALF are the full-range (f=1) half-extents.
CUBE_ANCHOR = (0.475, 0.0)
CUBE_HALF = (0.25, 0.40)
TRAY_ANCHOR = (0.675, 0.0)
TRAY_HALF = (0.175, 0.25)


class TidyEnv(gymnasium.Env):
    def __init__(
        self,
        gui=True,
        randomize_cube_spawn=False,
        randomize_tray_spawn=False,
        curriculum_prob=None,
        probe="grasp",
    ):
        self.randomize_cube_spawn = randomize_cube_spawn
        self.randomize_tray_spawn = randomize_tray_spawn
        self.cube_spawn_spread = 1.0  # 0 = trained anchor, 1 = full range
        self.tray_spawn_spread = 1.0
        self.curriculum_prob = (
            CURRICULUM_PROB if curriculum_prob is None else curriculum_prob
        )
        self.probe = probe  # "grasp" (above cube) or "place" (above tray, held)

        # RL
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, 0], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05, 1], dtype=np.float32),
            dtype=np.float32,
        )
        # [ee_xyz(3), cube_xyz(3), tray_xyz(3), fingers(2), gripping(1),
        #  cube-ee(3), ee_vel(3)] — everything the reward branches on is in the
        # observation, so the critic sees the real state, not a projection.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
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

    def reset(self, *, seed=None, options=None):
        # ponytail: seed accepted for the gymnasium API but not applied — SB3
        # re-passes the same seed on every episode, which would freeze the cube
        # spawn. train.py/eval.py call reset() with no args, unchanged.
        if hasattr(self, "small_cube") and self.small_cube:
            p.removeBody(self.small_cube.id)
        if hasattr(self, "tray") and self.tray:
            p.removeBody(self.tray.id)

        self.tray = self._spawn_tray()
        self.small_cube = self._spawn_small_cube()
        self.robot.cube_id = self.small_cube.id

        self.robot.reset()
        self.robot.open_claw()
        self._window_income_claimed = False
        # Single probe mode at a time. Place fires at HALF the grasp
        # probability so the harder chain (grasp->carry->place) over the
        # table-cube probe isn't drowned out by grab-world early reward.
        gate = self.curriculum_prob * (0.5 if self.probe == "place" else 1.0)
        if random.random() < gate:
            if self.probe == "place":
                # Place probe: start above the tray, cube already in hand, so
                # the lower->open->success chain is actually sampled by the
                # critic early in the ramp. Grasp failure falls back to a
                # normal start.
                if self.grasp_at_hand():
                    tray_x, tray_y = self.tray.get_current_pos()[:2]
                    cube = self.small_cube.get_current_pos()
                    # Hover off dead-center (one axis at a time, random sign)
                    # so the probe can't be solved by dropping straight down:
                    # the agent must translate to align over the tray mouth.
                    # Draws outside the arm's reach box are retried; if none
                    # land (all-x-blocked), fall back to an inward +0.18 x.
                    hover = None
                    for _ in range(8):
                        if random.random() < 0.5:
                            d = random.uniform(*PLACE_HOVER_JITTER) * random.choice((-1.0, 1.0))
                            hx, hy = tray_x + d, tray_y
                        else:
                            d = random.uniform(*PLACE_HOVER_JITTER) * random.choice((-1.0, 1.0))
                            hx, hy = tray_x, tray_y + d
                        if (
                            0.15 <= hx <= PLACE_REACH_X
                            and WORKSPACE_LOW[1] <= hy <= WORKSPACE_HIGH[1]
                        ):
                            hover = [hx, hy, 0.42]
                            break
                    if hover is None:
                        hover = [max(0.15, tray_x - PLACE_HOVER_JITTER[0]), tray_y, 0.42]
                    self._advance_toward([cube[0], cube[1], 0.6])  # lift
                    self._advance_toward([tray_x, tray_y, 0.6])  # fly to tray (clear rim)
                    for _ in range(8):  # converge; bursts move the loaded arm ~0.03 m each
                        self._advance_toward(hover)  # hover off-center
                        if math.dist(self.robot.get_end_effector_pos()[:2], hover[:2]) < 0.03:
                            break
            else:
                # Grasp probe: descend onto the cube with the claw open (same
                # motion as the scripted demo) so episodes start one close away
                # from a firm grip.
                cx, cy = self.small_cube.get_current_pos()[:2]
                z = 0.30
                while z > CURRICULUM_DESCEND_Z:
                    z = max(z - 0.05, CURRICULUM_DESCEND_Z)
                    self._advance_toward([cx, cy, z])

        for _ in range(40):
            p.stepSimulation()
        self.success_status = False
        self.step_count = 0
        self._grasp_claimed = False
        self._window_income_claimed = False

        info = {}

        obs = self._get_obs()
        self._prev_obs = obs
        return obs, info

    def step(self, action):

        self.step_count += 1

        # Movement is always driven by action[:3]; the gripper channel only
        # decides open/close, so a close intent can never discard the actor's
        # xyz output and snap-vertical (the old whiplash source near 0.5).
        target_position = self._calculate_target_position(
            action[:3], self.robot.get_end_effector_pos()
        )
        target_position = self._clip_target_position(target_position)
        self._advance_toward(target_position)

        if action[3] > 0.5:
            # A clamped cube can only be released over the tray; a failed
            # close can still be reopened to retry.
            if not self.robot.is_gripping() or self._in_release_window():
                self.robot.open_claw()
        elif self._in_grasp_window() and not self.robot.gripper_closed:
            # Gated like a gripper controller refusing a misaligned close:
            # in the window -> pure close (moving + closing in one step never
            # clamps); outside the window -> just keep moving, don't close.
            self.robot.close_claw()

        obs = self._get_obs()
        reward = self._calculate_reward(self._prev_obs, obs, action)
        self._prev_obs = obs
        terminated = self.success_status
        truncated = self.step_count >= self.max_episode_steps
        info = {"rewards": self.step_rewards}
        return (obs, reward, terminated, truncated, info)

    def _calculate_reward(self, prev_obs, obs, action):
        # Stateless: every term is a function of (s, a, s') -- the previous
        # observation, the current action, and the current observation. One
        # exception: _grasp_claimed pays the +10 grasp once per episode, so
        # re-grip cycling can't farm it and releasing is always free.
        # Everything else is potential differences that telescope, so
        # bouncing can't farm them.
        ee, cube, tray = obs[0:3], obs[3:6], obs[6:9]
        ee_prev, cube_prev = prev_obs[0:3], prev_obs[3:6]
        gripping = bool(obs[11])
        r = {
            "reach": 0,
            "grasp": 0,
            "lift": 0,
            "hold": 0,
            "approach": 0,
            "lower": 0,
            "success": 0,
        }
        # +10 on the first real clamp of the episode, never again. Releases
        # and re-grasps cost/pay nothing, so the policy never fears opening the
        # claw at the tray.
        if gripping and not bool(prev_obs[11]) and not self._grasp_claimed:
            r["grasp"] = 10.0
            self._grasp_claimed = True

        if not gripping:
            # potential: reach = -13 * ||cube - ee||; telescopes, backing off
            # pays negative, so in/out hunting collects nothing extra.
            r["reach"] = 13 * (
                np.linalg.norm(cube_prev - ee_prev) - np.linalg.norm(cube - ee)
            )
            if self._in_grasp_window_obs(obs):
                r["reach"] += 0.1  # window income: parks the EE where close can fire
        else:
            r["hold"] = HOLD_INCOME  # firm grip income; makes gripping net-positive
            if not self._is_cube_centralized_obs(obs):
                if cube[2] <= 0.75:
                    r["lift"] = 5 * (cube[2] - cube_prev[2])  # lift pays like carry
                r["approach"] = 10 * (
                    self._xy_dist(cube_prev, tray) - self._xy_dist(cube, tray)
                )
                # ponytail: approach pays only once the cube is lifted above
                # the tray rim (0.25 = ~20cm clearance); carrying low clips the
                # rim and is the "into the side of the tray" failure.
                if cube[2] < 0.25:
                    r["approach"] *= 0.5
            else:
                lower = 0.005  # staying centralized wrt the tray
                if not self._is_cube_in_tray_obs(obs):
                    # potential: descent = -13 * cube_z, center = -10 * xy_dist;
                    # both telescope, so drop/rise oscillation can't re-farm.
                    lower += 13 * (cube_prev[2] - cube[2])
                    lower += 10 * (
                        self._xy_dist(cube_prev, tray) - self._xy_dist(cube, tray)
                    )
                r["lower"] = lower
        if self._xy_dist(cube, tray) < 0.15:
            fingers_low = ee[2] < self.tray.top + 0.02
            if self._is_cube_in_tray_obs(obs) and fingers_low:
                r["success"] = 60.0
                self.success_status = True
        self.step_rewards = r
        floor_penalty = max(0.0, 1.0 * (0.02 - ee[2]))
        return sum(r.values()) - 0.01 - floor_penalty

    @staticmethod
    def _xy_dist(cube, tray):
        return np.linalg.norm((cube - tray)[:2])

    def _in_grasp_window_obs(self, obs):
        ee, cube = obs[0:3], obs[3:6]
        aligned = abs(ee[0] - cube[0]) < 0.03 and abs(ee[1] - cube[1]) < 0.03
        # ponytail: z-bottom widened (0.02 -> 0.005) so a far spawn that dives
        # just below the old band (EE ~0.013, aligned, close intent) can close
        return aligned and 0.005 <= ee[2] <= 0.04

    def _is_cube_centralized_obs(self, obs):
        cube, tray = obs[3:6], obs[6:9]
        return (
            abs(cube[0] - tray[0]) < self.tray.half_length
            and abs(cube[1] - tray[1]) < self.tray.half_width
        )

    def _is_cube_in_tray_obs(self, obs):
        # ponytail: +0.01 margin — the funnel-shaped tray wedges the hand with
        # the cube ~0.5mm above the rim; the cube is seated in the tray mouth.
        return self._is_cube_centralized_obs(obs) and obs[5] < self.tray.top + 0.01

    def _in_grasp_window(self):
        ee = self.robot.get_end_effector_pos()
        cube = self.small_cube.get_current_pos()
        aligned = abs(ee[0] - cube[0]) < 0.03 and abs(ee[1] - cube[1]) < 0.03
        # ponytail: z-bottom widened (0.02 -> 0.005) so a far spawn that dives
        # just below the old band (EE ~0.013, aligned, close intent) can close
        return aligned and 0.005 <= ee[2] <= 0.04

    def _in_release_window(self):
        # Cube is over the tray mouth and low enough that opening drops it in.
        cube = self.small_cube.get_current_pos()
        tray = self.tray.get_current_pos()
        over_tray = (
            abs(cube[0] - tray[0]) < self.tray.half_length
            and abs(cube[1] - tray[1]) < self.tray.half_width
        )
        return over_tray and cube[2] < self.tray.top + 0.02

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

    def _clip_target_position(self, target_position):
        return np.clip(target_position, self.workspace_low, self.workspace_high)

    def _get_obs(self):
        ee = np.array(self.robot.get_end_effector_pos(), dtype=np.float32)
        cube = np.array(self.small_cube.get_current_pos(), dtype=np.float32)
        tray = np.array(self.tray.get_current_pos(), dtype=np.float32)
        fingers = np.array(
            [
                p.getJointState(self.robot.id, 9)[0],
                p.getJointState(self.robot.id, 10)[0],
            ],
            dtype=np.float32,
        )
        ee_vel = np.array(
            p.getLinkState(self.robot.id, 11, computeLinkVelocity=1)[6],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                ee,
                cube,
                tray,
                fingers,
                [float(self.robot.is_gripping())],
                cube - ee,
                ee_vel,
            ]
        ).astype(np.float32)

    def set_spawn_spread(self, fraction):
        # One ramp variable controls cube AND tray randomization together.
        self.cube_spawn_spread = fraction
        self.tray_spawn_spread = fraction

    def _sample_cube_spawn(self):
        f = self.cube_spawn_spread
        cx, cy = CUBE_ANCHOR
        hx, hy = CUBE_HALF
        tray_pos = self.tray.get_current_pos()
        while True:
            x = random.uniform(cx - hx * f, cx + hx * f)
            y = random.uniform(cy - hy * f, cy + hy * f)
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
        f = self.tray_spawn_spread
        tx, ty = TRAY_ANCHOR
        hx, hy = TRAY_HALF
        x = random.uniform(tx - hx * f, tx + hx * f)
        y = random.uniform(ty - hy * f, ty + hy * f)
        return np.clip(
            [x, y, 0.0], WORKSPACE_LOW, [WORKSPACE_HIGH[0], WORKSPACE_HIGH[1], 0.0]
        )

    def _spawn_tray(self, basePos=[0.625, -0.30, 0.0], globScale=0.25):
        if self.randomize_tray_spawn:
            basePos = self._sample_tray_spawn()
        self.tray = Tray(basePos, globScale)

        return self.tray

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

    def grasp_at_hand(self):
        # Scripted: descend onto the cube with the claw open (closing during
        # the descend would turn it into a push and wedge the fingers over the
        # cube top), then close once low, and nudge down to seat a real clamp.
        # Returns True on a firm grip, else the cube stays ungrasped.
        cx, cy = self.small_cube.get_current_pos()[:2]
        for z in (0.12, 0.045, 0.02):
            self._advance_toward([cx, cy, z])
        self.robot.close_claw()
        if not self.robot.is_gripping():
            self._advance_toward([cx, cy, 0.005])
        return self.robot.is_gripping()

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
