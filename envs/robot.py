import pybullet as p
import math
import time


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
        joint_angles = p.calculateInverseKinematics(
            self.id,
            11,
            self.initial_pos,
            targetOrientation=self.initial_orn,
        )

        for i in range(7):
            p.resetJointState(self.id, i, joint_angles[i])

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

    def get_end_effector_pos(self):
        return p.getLinkState(self.id, 11)[0]

    def is_gripping(self):
        contact_points = p.getContactPoints(bodyA=self.id, bodyB=-1)
        left_finger_touching = any(contact[3] == 10 for contact in contact_points)
        right_finger_touching = any(contact[3] == 11 for contact in contact_points)
        return left_finger_touching and right_finger_touching and self.gripper_closed

    def open_claw(self, f=200, target_position=0.04):
        for joint in [9, 10]:
            p.setJointMotorControl2(
                bodyUniqueId=self.id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=f,
            )
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1 / 240)
        self.gripper_closed = False

    def close_claw(self, f=200, target_position=0.0):
        for joint in [9, 10]:
            p.setJointMotorControl2(
                bodyUniqueId=self.id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=f,
            )
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1 / 240)
        self.gripper_closed = True

    def reset(self):
        joint_angles = p.calculateInverseKinematics(
            self.id,
            11,
            self.initial_pos,
            targetOrientation=self.initial_orn,
        )

        for i in range(7):
            p.resetJointState(self.id, i, joint_angles[i])
