"""
A simple biped body in PyBullet for the embodied space substrate.

The body: a box torso with two legs, each leg has a hip joint and a knee
joint.  Built from scratch via PyBullet's createMultiBody so there are no
external URDF dependencies.

Joints:
    0: left hip  (revolute, sagittal plane)
    1: left knee  (revolute, sagittal plane)
    2: right hip  (revolute, sagittal plane)
    3: right knee  (revolute, sagittal plane)

The body stands on a flat ground plane.  Gravity pulls it down.
Foot segments have friction with the ground.
"""

from __future__ import annotations

from pathlib import Path

import pybullet as p
import pybullet_data
import numpy as np


# Body dimensions (meters)
TORSO_HALF = [0.15, 0.1, 0.25]     # x, y, z half-extents
UPPER_LEG_LEN = 0.35
UPPER_LEG_RAD = 0.04
LOWER_LEG_LEN = 0.35
LOWER_LEG_RAD = 0.035
FOOT_HALF = [0.08, 0.04, 0.02]

# Mass (kg)
TORSO_MASS = 10.0
UPPER_LEG_MASS = 2.0
LOWER_LEG_MASS = 1.5
FOOT_MASS = 0.5

# Joint limits (radians)
HIP_RANGE = (-1.2, 1.2)
KNEE_RANGE = (-0.05, 2.2)    # knees bend backward


URDF_PATH = Path(__file__).parent / "biped.urdf"

# Standing height: torso half-height + upper leg + lower leg + foot half-height
STAND_HEIGHT = TORSO_HALF[2] + UPPER_LEG_LEN + LOWER_LEG_LEN + FOOT_HALF[2] + 0.01


def create_biped(physics_client: int) -> int:
    """Load the biped URDF and return the body ID."""

    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physics_client)
    p.loadURDF("plane.urdf", physicsClientId=physics_client)

    body_id = p.loadURDF(
        str(URDF_PATH),
        basePosition=[0, 0, STAND_HEIGHT],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
        physicsClientId=physics_client,
    )

    # Disable default velocity motors so we have torque control
    for joint_idx in ACTIVE_JOINTS:
        p.setJointMotorControl2(
            body_id, joint_idx,
            controlMode=p.VELOCITY_CONTROL,
            force=0,
            physicsClientId=physics_client,
        )

    # Set friction on feet
    for foot_idx in [2, 5]:  # left foot, right foot (fixed joints)
        p.changeDynamics(body_id, foot_idx, lateralFriction=1.0,
                         physicsClientId=physics_client)

    return body_id


# Joint name mapping for convenience
JOINT_NAMES = {
    0: "left_hip_ab",
    1: "left_hip",
    2: "left_knee",
    4: "right_hip_ab",
    5: "right_hip",
    6: "right_knee",
}

# Active (controllable) joint indices — skip fixed ankle joints (3, 7)
ACTIVE_JOINTS = [0, 1, 2, 4, 5, 6]


def get_body_state(physics_client: int, body_id: int) -> dict:
    """Read the full body state from PyBullet.

    Returns torso orientation (quaternion), joint angles, ground contacts,
    and center of mass height — everything the adapter needs.
    """
    pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
    vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=physics_client)

    joint_angles = {}
    joint_velocities = {}
    for idx in ACTIVE_JOINTS:
        state = p.getJointState(body_id, idx, physicsClientId=physics_client)
        joint_angles[idx] = state[0]
        joint_velocities[idx] = state[1]

    # Ground contact: check if feet (links 2, 5) touch the ground (body 0)
    contacts_left = p.getContactPoints(body_id, 0, 3, -1, physicsClientId=physics_client)
    contacts_right = p.getContactPoints(body_id, 0, 7, -1, physicsClientId=physics_client)

    return {
        "position": np.array(pos),
        "orientation": np.array(orn),   # [x, y, z, w] PyBullet convention
        "linear_velocity": np.array(vel),
        "angular_velocity": np.array(ang_vel),
        "joint_angles": joint_angles,
        "joint_velocities": joint_velocities,
        "left_foot_contact": len(contacts_left) > 0,
        "right_foot_contact": len(contacts_right) > 0,
        "height": pos[2],
    }


def apply_joint_torques(physics_client: int, body_id: int,
                         torques: dict[int, float]) -> None:
    """Apply torques to the active joints."""
    for idx, torque in torques.items():
        p.setJointMotorControl2(
            body_id, idx,
            controlMode=p.TORQUE_CONTROL,
            force=torque,
            physicsClientId=physics_client,
        )


def orientation_to_s3(pybullet_orn: np.ndarray) -> np.ndarray:
    """Convert PyBullet quaternion [x,y,z,w] to our S³ convention [w,x,y,z]."""
    return np.array([pybullet_orn[3], pybullet_orn[0],
                     pybullet_orn[1], pybullet_orn[2]])