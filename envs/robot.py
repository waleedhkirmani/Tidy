import pybullet as p
import numpy as np
import math


class Robot:
    def __init__(self, initial_pos=[0.5, 0.0, 0.7]):
        self.id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )

        self.initial_pos = initial_pos
        self.initial_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        # pybullet IK seeds from the current joint state and can get stuck in
        # local minima; iterate it to a converged, reproducible home pose.
        # Seed the forearm roll (joint 3) mid-range: parked at its upper limit
        # (0.0) the solver returns infeasible solutions and jams the arm.
        joint_angles = [0.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0]
        for i in range(7):
            p.resetJointState(self.id, i, joint_angles[i])
        for _ in range(12):
            joint_angles = p.calculateInverseKinematics(
                self.id,
                11,
                self.initial_pos,
                targetOrientation=self.initial_orn,
            )
            for i in range(7):
                p.resetJointState(self.id, i, joint_angles[i])
        self.home_joint_angles = list(joint_angles)

        self.lower_limits = []
        self.upper_limits = []
        self.joint_ranges = []
        self.rest_poses = []

        for joint in range(7):
            info = p.getJointInfo(self.id, joint)
            lower = info[8]
            upper = info[9]

            self.lower_limits.append(lower)
            self.upper_limits.append(upper)
            self.joint_ranges.append(upper - lower)
            self.rest_poses.append(p.getJointState(self.id, joint)[0])

        self.gripper_closed = False
        self.cube_id = None

    def get_end_effector_pos(self):
        return p.getLinkState(self.id, 11)[0]

    def is_gripping(self):
        if self.cube_id is None:
            return False
        contact_points = p.getContactPoints(bodyA=self.id, bodyB=self.cube_id)
        contacts = {c[3]: c for c in contact_points if c[3] in (9, 10)}
        if 9 not in contacts or 10 not in contacts or not self.gripper_closed:
            return False
        # A real clamp: the fingers press opposite faces of the cube, so the
        # cube-surface contact normals are opposed. Closed fingers merely
        # touching (pushing the top or a side) yield same/adjacent normals.
        n1, n2 = np.array(contacts[9][7]), np.array(contacts[10][7])
        return float(np.dot(n1, n2)) < -0.5

    def open_claw(self, f=200, target_position=0.04):
        for joint in [9, 10]:
            p.setJointMotorControl2(
                bodyUniqueId=self.id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=f,
            )
        self.gripper_closed = False

    def close_claw(self, f=200, target_position=0.0):
        if self.gripper_closed:
            return
        for joint in [9, 10]:
            p.setJointMotorControl2(
                bodyUniqueId=self.id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=f,
            )
        for _ in range(10):
            p.stepSimulation()
        self.gripper_closed = True

    def reset(self):
        for i in range(7):
            p.resetJointState(self.id, i, self.home_joint_angles[i])
            p.setJointMotorControl2(
                bodyUniqueId=self.id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.home_joint_angles[i],
                force=200,
            )
