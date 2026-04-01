"""
Walking adapter — wires Enkidu cells into a bipedal locomotion lattice.

This is the substrate-specific layer for embodied space.  Same role as
the music adapter in the Bach track: it provides directional truth and
maps between the substrate (physics engine) and the kernel (S³ algebra).

The lattice:
    body_cell    — drives, lean, overall coordination
    gait_cell    — stride phase, alternates legs on foot contact
    left_hip     — swing/stance target from gait cell
    right_hip    — anti-phase
    left_knee    — flex during swing
    right_knee   — flex during swing
    left_ankle   — lateral balance
    right_ankle  — lateral balance

Each cell is an Enkidu with its own kernel.  Targets flow downward
from body → gait → joints.  Closures flow upward from joints → gait → body.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from closure_ea.kernel import Kernel, compose, inverse, sigma, identity
from closure_ea.cell import Adapter, Cell


def angle_to_q(angle: float, axis: int = 0) -> np.ndarray:
    """Angle (radians) → unit quaternion.  axis: 0=x, 1=y, 2=z."""
    q = np.array([math.cos(angle / 2), 0.0, 0.0, 0.0])
    q[1 + axis] = math.sin(angle / 2)
    return q


class JointAdapter(Adapter):
    """Adapter for one joint cell.  Embeds the gap between current
    angle and target as a quaternion on S³."""

    def __init__(self, name: str, axis: int = 0):
        super().__init__(damping=0.05)
        self.name = name
        self.axis = axis
        self.target = 0.0

    def embed(self, event_key):
        """Event = current joint angle.  Embedded as gap from target."""
        angle = float(event_key)
        gap = angle - self.target
        return angle_to_q(gap, self.axis)


class GaitAdapter(Adapter):
    """Adapter for the gait coordinator.  Embeds foot contact events."""

    def __init__(self):
        super().__init__(damping=0.05)
        self.phase = 0  # 0 = left stance, 1 = right stance

    def embed(self, event_key):
        """Event = 'left_contact' or 'right_contact' or 'no_contact'."""
        if event_key == "left_contact":
            self.phase = 0  # left foot planted → left stance
            return angle_to_q(0.0)  # near identity = stable
        elif event_key == "right_contact":
            self.phase = 1
            return angle_to_q(0.0)
        else:
            return angle_to_q(0.5)  # flight phase = displaced from identity


class WalkingLattice:
    """The full lattice of cells that coordinate walking.

    Body level sets the lean from the drive.
    Gait level alternates legs from foot contacts.
    Joint cells follow their targets.
    """

    def __init__(self):
        # Joint cells — each is an Enkidu
        self.left_hip = Cell(JointAdapter("left_hip", axis=0), epsilon=0.2)
        self.right_hip = Cell(JointAdapter("right_hip", axis=0), epsilon=0.2)
        self.left_knee = Cell(JointAdapter("left_knee", axis=0), epsilon=0.2)
        self.right_knee = Cell(JointAdapter("right_knee", axis=0), epsilon=0.2)
        self.left_ankle = Cell(JointAdapter("left_ankle", axis=1), epsilon=0.2)
        self.right_ankle = Cell(JointAdapter("right_ankle", axis=1), epsilon=0.2)

        # Gait coordinator
        self.gait = Cell(GaitAdapter(), epsilon=0.3)
        self.gait_phase_time = 0.0  # continuous phase within current stride
        self.stride_period = 0.8    # seconds per stride cycle
        self.in_left_stance = True  # which leg is supporting

        # Control parameters (the genome will learn to adjust these)
        self.swing_amplitude = 0.25   # hip swing angle during swing phase
        self.knee_flex = 0.35         # knee flex during swing
        self.lean_sensitivity = 0.10  # how strongly drive maps to lean

        # Stats
        self.total_strides = 0
        self.total_closures = {
            "left_hip": 0, "right_hip": 0,
            "left_knee": 0, "right_knee": 0,
        }

    def step(self, body_correction: np.ndarray, drive_strength: float,
             joint_angles: dict, left_contact: bool, right_contact: bool,
             dt: float) -> dict:
        """One tick of the lattice.  Returns joint targets as a dict.

        body_correction: C⁻¹ of torso orientation composed with desired lean.
            The vector part tells each joint which direction to move.
        drive_strength: 0 (rest) to 1 (full drive).  Scales all movement.
        joint_angles: {pybullet_idx: current_angle} for revolute joints.
        left_contact / right_contact: foot contact from physics (adapter truth).
        dt: timestep.

        Returns: {pybullet_joint_idx: target_angle_or_quaternion}
        """

        # ── GAIT COORDINATOR ──
        # Foot contacts are boundary operators (like BAR in music)
        if left_contact and not self.in_left_stance:
            self.gait.ingest("left_contact")
            self.in_left_stance = True
            self.gait_phase_time = 0.0
            self.total_strides += 1
        elif right_contact and self.in_left_stance:
            self.gait.ingest("right_contact")
            self.in_left_stance = False
            self.gait_phase_time = 0.0
            self.total_strides += 1
        else:
            self.gait.ingest("no_contact")

        self.gait_phase_time += dt

        # Gait phase: 0→1 within each half-stride
        phase = min(self.gait_phase_time / (self.stride_period / 2.0), 1.0)
        swing_shape = math.sin(phase * math.pi)  # smooth swing profile

        # ── BODY LEVEL → JOINT TARGETS ──
        # The body correction's pitch component (i-axis) = balance + lean
        pitch_correction = body_correction[1]
        roll_correction = body_correction[2]

        # Scale by drive strength: no drive = just balance, full drive = lean + stride
        lean = pitch_correction  # already includes balance + drive lean
        stride = self.swing_amplitude * swing_shape * drive_strength

        # ── HIP TARGETS ──
        if self.in_left_stance:
            # Left leg is stance (supporting), right leg swings forward
            left_hip_target = lean - stride * 0.2   # stance: push back slightly
            right_hip_target = lean + stride          # swing: forward
            left_knee_target = 0.0                    # stance: extend
            right_knee_target = self.knee_flex * swing_shape * drive_strength  # swing: flex
        else:
            # Right leg is stance, left leg swings
            left_hip_target = lean + stride
            right_hip_target = lean - stride * 0.2
            left_knee_target = self.knee_flex * swing_shape * drive_strength
            right_knee_target = 0.0

        # Ankle targets: lateral balance from roll correction
        ankle_target = roll_correction * 0.3

        # ── FEED EACH JOINT CELL ──
        # Each cell receives its current angle, composes against target,
        # checks closure, teaches directionally

        self.left_hip.adapter.target = left_hip_target
        self.right_hip.adapter.target = right_hip_target
        self.left_knee.adapter.target = left_knee_target
        self.right_knee.adapter.target = right_knee_target
        self.left_ankle.adapter.target = ankle_target
        self.right_ankle.adapter.target = -ankle_target  # opposite side

        # Ingest current angles into each cell
        for cell, name in [
            (self.left_hip, "left_hip"),
            (self.right_hip, "right_hip"),
            (self.left_knee, "left_knee"),
            (self.right_knee, "right_knee"),
        ]:
            if name.startswith("left"):
                idx = 13 if "knee" in name else 12  # humanoid joint indices
            else:
                idx = 10 if "knee" in name else 9
            if idx in joint_angles:
                result, gap, incident = cell.ingest(str(joint_angles[idx]))
                if result == "closure":
                    self.total_closures[name] += 1

        # Return targets for the physics engine
        # Hips are spherical → quaternion targets [x,y,z,w]
        # Knees are revolute → scalar targets
        return {
            "left_hip_q": [math.sin(left_hip_target/2), 0, 0,
                           math.cos(left_hip_target/2)],
            "right_hip_q": [math.sin(right_hip_target/2), 0, 0,
                            math.cos(right_hip_target/2)],
            "left_knee": left_knee_target,
            "right_knee": right_knee_target,
            "left_ankle_q": [0, math.sin(ankle_target/2), 0,
                             math.cos(ankle_target/2)],
            "right_ankle_q": [0, math.sin(-ankle_target/2), 0,
                              math.cos(-ankle_target/2)],
        }

    def summary(self) -> str:
        return (f"strides={self.total_strides}  "
                f"closures=L_hip:{self.total_closures['left_hip']} "
                f"R_hip:{self.total_closures['right_hip']} "
                f"L_knee:{self.total_closures['left_knee']} "
                f"R_knee:{self.total_closures['right_knee']}  "
                f"genome_size="
                f"{self.left_hip.genome_size + self.right_hip.genome_size + self.left_knee.genome_size + self.right_knee.genome_size}")
