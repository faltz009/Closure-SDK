"""
A body is a lattice of joint-Enkidus coordinated by Gilgamesh.

Each joint is one Enkidu cell.  It receives its own angle as an event,
composes toward a target, and produces a correction (C⁻¹) that IS the
torque signal.

This is Enkidu Alive with articulated limbs.  Each joint is a creature
with a drive.  The drives come from two sources:
    1. Balance: keep the body upright (σ_tilt → 0)
    2. Forward: reach the shelter (σ_goal → 0)

The joint cells don't know about "walking."  They know about closing
toward their current target.  Walking emerges when the body discovers
a gait pattern that maintains balance while making forward progress.

Learning happens through evolution of the gait genome:
    - The gait genome is a set of parameters (amplitude, frequency,
      phase offsets, knee ratios) that define the joint target pattern
    - Each episode runs with the current genome plus small mutations
    - Episodes that survive longer and move further reinforce their mutations
    - The gait pattern IS the learned motor memory
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from closure_ea.kernel import compose, inverse, sigma, identity
from closure_ea.cell import Adapter, Cell, geodesic_step


def angle_to_quaternion(angle: float, axis: int = 0) -> np.ndarray:
    """Joint angle (radians) → unit quaternion.  Rotation around the given axis."""
    half = angle / 2.0
    q = np.array([math.cos(half), 0.0, 0.0, 0.0])
    q[1 + axis] = math.sin(half)
    return q


class JointAdapter(Adapter):
    """Adapter for one joint.  Embeds joint angles as quaternions on S³."""

    def __init__(self, joint_name: str, axis: int = 0):
        super().__init__(damping=0.08)
        self.joint_name = joint_name
        self.axis = axis
        self.target = 0.0

    def embed_angle(self, angle: float) -> np.ndarray:
        gap_angle = angle - self.target
        return angle_to_quaternion(gap_angle, self.axis)

    def embed(self, event_key):
        angle = float(event_key)
        return self.embed_angle(angle)


class GaitGenome:
    """The learned gait pattern — the motor memory.

    Parameters that define how the joints move:
        hip_amplitude:  how far the hips swing (radians)
        hip_frequency:  how fast the oscillation (Hz)
        knee_ratio:     how much the knees flex relative to hips
        lean:           forward lean angle
        balance_gain:   how strongly to correct for tilt

    These start at sensible defaults and evolve through experience.
    Each episode mutates them.  Better episodes overwrite worse ones.
    """

    def __init__(self):
        self.hip_amplitude = 0.4
        self.hip_frequency = 2.0
        self.knee_ratio = 0.7
        self.lean = 0.08
        self.balance_gain = 0.5
        self.best_fitness = -1.0
        self.generation = 0

    def params(self) -> np.ndarray:
        return np.array([
            self.hip_amplitude,
            self.hip_frequency,
            self.knee_ratio,
            self.lean,
            self.balance_gain,
        ])

    def set_params(self, p: np.ndarray):
        self.hip_amplitude = float(np.clip(p[0], 0.05, 0.8))
        self.hip_frequency = float(np.clip(p[1], 0.5, 4.0))
        self.knee_ratio = float(np.clip(p[2], 0.1, 1.2))
        self.lean = float(np.clip(p[3], -0.1, 0.3))
        self.balance_gain = float(np.clip(p[4], 0.1, 1.5))

    def mutate(self, scale: float = 0.12) -> np.ndarray:
        """Return mutated params.  Keep the originals until we know if it's better."""
        base = self.params()
        noise = np.random.randn(len(base)) * scale
        return base + noise

    def update_if_better(self, candidate_params: np.ndarray, fitness: float):
        """If this candidate performed better, adopt its parameters."""
        if fitness > self.best_fitness:
            self.set_params(candidate_params)
            self.best_fitness = fitness
            self.generation += 1
            return True
        return False

    def __repr__(self):
        return (f"GaitGenome(amp={self.hip_amplitude:.3f}, freq={self.hip_frequency:.2f}, "
                f"knee={self.knee_ratio:.2f}, lean={self.lean:.3f}, "
                f"bal={self.balance_gain:.2f}, fit={self.best_fitness:.2f}, "
                f"gen={self.generation})")


class Body:
    """A body = a lattice of joint-Enkidus + a gait genome.

    Four joints: left hip, left knee, right hip, right knee.
    Each is an Enkidu cell with its own adapter.

    The gait genome determines the joint target pattern.
    The kernel in each cell detects closure (joint reached target)
    and provides directional teaching.
    """

    def __init__(self, epsilon: float = 0.15):
        self.joints = {
            "left_hip": JointAdapter("left_hip", axis=0),
            "left_knee": JointAdapter("left_knee", axis=0),
            "right_hip": JointAdapter("right_hip", axis=0),
            "right_knee": JointAdapter("right_knee", axis=0),
        }

        self.cells = {}
        for name, adapter in self.joints.items():
            self.cells[name] = Cell(adapter, epsilon=epsilon)

        self.joint_indices = {
            "left_hip": 0, "left_knee": 1,
            "right_hip": 3, "right_knee": 4,
        }

        self.gait = GaitGenome()
        self.phase = 0.0
        self.current_params = self.gait.params()  # params for this episode
        self.total_closures = {name: 0 for name in self.joints}

    def begin_episode(self, forward_drive: float, mutation_scale: float = 0.10,
                       use_best: bool = False):
        """Start a new episode with mutated gait params."""
        self.phase = 0.0
        for cell in self.cells.values():
            cell.reset()

        if use_best or forward_drive < 0.01:
            # Replay the best known gait (for testing / visualization)
            self.current_params = self.gait.params()
        else:
            self.current_params = self.gait.mutate(scale=mutation_scale)

    def end_episode(self, survived_time: float, forward_distance: float,
                     height_gained: float = 0.0):
        """Score this episode and update the genome if it was better.

        Fitness rewards forward distance and height gained.  The body MUST
        move to score well — survival alone scores low.
        """
        fitness = forward_distance * 20.0 + height_gained * 50.0 + survived_time * 0.5
        improved = self.gait.update_if_better(self.current_params, fitness)
        return improved, fitness

    def set_targets(self, forward_drive: float, body_pitch: float, dt: float):
        """Set joint targets from the current (possibly mutated) gait params + body state."""
        amp = self.current_params[0] * forward_drive
        freq = self.current_params[1]
        knee_ratio = self.current_params[2]
        lean = self.current_params[3] * forward_drive
        balance_gain = self.current_params[4]

        # Advance gait phase
        self.phase += freq * 2.0 * math.pi * dt

        left_phase = math.sin(self.phase)
        right_phase = math.sin(self.phase + math.pi)

        # Balance correction from body pitch
        balance = -body_pitch * balance_gain

        # Hip targets
        self.joints["left_hip"].target = amp * left_phase + lean + balance
        self.joints["right_hip"].target = amp * right_phase + lean + balance

        # Knee targets: flex during swing
        left_swing = max(0, left_phase)
        right_swing = max(0, right_phase)
        self.joints["left_knee"].target = left_swing * amp * knee_ratio
        self.joints["right_knee"].target = right_swing * amp * knee_ratio

    def step(self, joint_angles: dict, joint_velocities: dict,
             body_sigma: float, dt: float) -> dict[int, float]:
        """Each joint-Enkidu processes its angle, returns target positions."""
        targets = {}

        for name, adapter in self.joints.items():
            idx = self.joint_indices[name]
            angle = joint_angles.get(idx, 0.0)

            q = adapter.embed_angle(angle)
            cell = self.cells[name]
            sigma_before = cell.kernel.gap

            result, vote = cell.kernel.ingest(q)
            sigma_after = cell.kernel.gap

            if sigma_after < sigma_before or result == "closure":
                adapter.teach(str(angle), vote)
                if result == "closure":
                    self.total_closures[name] += 1

            targets[idx] = adapter.target

        return targets

    @property
    def genome_summary(self) -> str:
        return repr(self.gait)
