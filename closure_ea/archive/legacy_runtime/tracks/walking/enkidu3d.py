"""
Enkidu Alive 3D — limb groups, persistent genome across runs.

Each run: the body spawns, loads the genome from the previous run,
explores from where it left off, saves the updated genome.
The body gets better at standing with each run because the
learned joint targets accumulate.

Run repeatedly:
    python3 enkidu3d.py              # headless, saves genome
    python3 enkidu3d.py --watch      # visual, saves genome
    python3 enkidu3d.py --watch      # starts from saved genome
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

TIMESTEP = 1.0 / 240.0
MAX_TICKS = 7200        # 30 seconds per run — fast runs, accumulate genome
PG = 0.6
VG = 0.8
EXPLORE_TICKS = 30
TEACH_RATE = 0.25
SPHERICAL_RANGE = math.pi    # spherical joints: explore full ±π

GENOME_PATH = Path(__file__).parent / "output" / "walking_genome.json"


def _qmul_pb(a, b):
    x1,y1,z1,w1 = a
    x2,y2,z2,w2 = b
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    n = math.sqrt(w*w+x*x+y*y+z*z)
    if n < 1e-8: return [0,0,0,1]
    return [x/n, y/n, z/n, w/n]


def _angles_to_quat(angles):
    a, b, c = angles
    qx = [math.sin(a/2), 0, 0, math.cos(a/2)]
    qy = [0, math.sin(b/2), 0, math.cos(b/2)]
    qz = [0, 0, math.sin(c/2), math.cos(c/2)]
    return _qmul_pb(_qmul_pb(qx, qy), qz)


@dataclass
class JointInfo:
    idx: int
    name: str
    jtype: int
    n_axes: int
    targets: list = field(default_factory=list)
    lower_limit: float = -math.pi
    upper_limit: float = math.pi
    max_force: float = 500

    def __post_init__(self):
        if not self.targets:
            self.targets = [0.0] * self.n_axes

    @property
    def explore_delta(self):
        """Explore half the full range of this joint."""
        return (self.upper_limit - self.lower_limit) / 2.0

    @property
    def range_center(self):
        return (self.upper_limit + self.lower_limit) / 2.0


@dataclass
class LimbGroup:
    name: str
    joints: list
    anchor_link: int
    closures: int = 0
    explore_axis: int = 0
    n_explore_axes: int = 3

    def teach(self, direction: float, helped: bool):
        """direction: +1 or -1.  Each joint uses its own delta scale.
        Clamp to joint limits — the body can't move past its physical range."""
        if not helped:
            return
        self.closures += 1
        head = self.joints[0]
        axis = self.explore_axis % head.n_axes
        head.targets[axis] += direction * head.explore_delta * TEACH_RATE
        head.targets[axis] = max(head.lower_limit, min(head.upper_limit, head.targets[axis]))
        for j in self.joints[1:]:
            fa = self.explore_axis % j.n_axes
            j.targets[fa] += direction * j.explore_delta * TEACH_RATE * 0.5
            j.targets[fa] = max(j.lower_limit, min(j.upper_limit, j.targets[fa]))

    def advance_axis(self):
        self.explore_axis = (self.explore_axis + 1) % self.n_explore_axes


class HierarchicalLattice:
    def __init__(self, pc: int, body_id: int):
        self.pc = pc
        self.body_id = body_id
        self.max_head_height = 0.0
        self.tick_count = 0
        self.total_runs = 0

        n = p.getNumJoints(body_id, physicsClientId=pc)
        all_joints = {}
        for i in range(n):
            info = p.getJointInfo(body_id, i, physicsClientId=pc)
            name = info[1].decode()
            jtype = info[2]
            lower = info[8]
            upper = info[9]
            max_force = info[10]

            if jtype in (0, 2):
                n_axes = 3 if jtype == 2 else 1

                # Read limits from the URDF — the adapter's truth about this joint
                if jtype == 0 and upper > lower:  # revolute with valid limits
                    jlower, jupper, jforce = lower, upper, max(max_force, 500)
                else:  # spherical or no limits
                    jlower, jupper, jforce = -SPHERICAL_RANGE, SPHERICAL_RANGE, 500

                all_joints[name] = JointInfo(
                    i, name, jtype, n_axes,
                    lower_limit=jlower, upper_limit=jupper, max_force=jforce,
                )

        self.groups: list[LimbGroup] = []

        group_defs = [
            ("left_leg", ["left_hip", "left_knee", "left_ankle"], -1),
            ("right_leg", ["right_hip", "right_knee", "right_ankle"], -1),
            ("left_arm", ["left_shoulder", "left_elbow"], 1),
            ("right_arm", ["right_shoulder", "right_elbow"], 1),
            ("spine", ["chest", "neck"], -1),
        ]

        for gname, jnames, anchor in group_defs:
            if all(n in all_joints for n in jnames):
                self.groups.append(LimbGroup(
                    gname, [all_joints[n] for n in jnames], anchor))

        self.all_joints = all_joints
        self._group_idx = 0
        self._phase = 0
        self._tick = 0
        self._pos_h = []
        self._neg_h = []

    def save_genome(self):
        """Save learned targets to JSON."""
        GENOME_PATH.parent.mkdir(exist_ok=True)
        data = {
            "total_runs": self.total_runs,
            "max_head_height": self.max_head_height,
            "joints": {},
        }
        for name, j in self.all_joints.items():
            data["joints"][name] = {
                "targets": j.targets,
                "idx": j.idx,
                "jtype": j.jtype,
            }
        data["groups"] = {}
        for g in self.groups:
            data["groups"][g.name] = {"closures": g.closures}

        GENOME_PATH.write_text(json.dumps(data, indent=2))

    def load_genome(self):
        """Load targets from previous run."""
        if not GENOME_PATH.exists():
            return False
        data = json.loads(GENOME_PATH.read_text())
        self.total_runs = data.get("total_runs", 0)
        prev_max = data.get("max_head_height", 0)

        loaded = 0
        for name, jdata in data.get("joints", {}).items():
            if name in self.all_joints:
                j = self.all_joints[name]
                j.targets = jdata["targets"][:j.n_axes]
                loaded += 1

        for gname, gdata in data.get("groups", {}).items():
            for g in self.groups:
                if g.name == gname:
                    g.closures = gdata.get("closures", 0)

        return loaded > 0

    def _link_height(self, link_idx):
        if link_idx == -1:
            pos, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.pc)
            return pos[2]
        return p.getLinkState(self.body_id, link_idx, physicsClientId=self.pc)[0][2]

    def step(self):
        self.tick_count += 1
        head_h = self._link_height(2)
        if head_h > self.max_head_height:
            self.max_head_height = head_h

        if not self.groups:
            return head_h

        group = self.groups[self._group_idx % len(self.groups)]
        anchor_h = self._link_height(group.anchor_link)

        if self._phase == 0:
            self._pos_h.append(anchor_h)
        else:
            self._neg_h.append(anchor_h)

        self._tick += 1

        if self._tick >= EXPLORE_TICKS:
            if self._phase == 0:
                self._phase = 1
                self._tick = 0
            else:
                pm = sum(self._pos_h) / max(len(self._pos_h), 1)
                nm = sum(self._neg_h) / max(len(self._neg_h), 1)

                if pm > nm + 0.001:
                    group.teach(+1.0, True)
                elif nm > pm + 0.001:
                    group.teach(-1.0, True)

                group.advance_axis()
                if group.explore_axis == 0:
                    self._group_idx = (self._group_idx + 1) % len(self.groups)

                self._phase = 0
                self._tick = 0
                self._pos_h = []
                self._neg_h = []

        # Apply ALL joints — each uses its own limits and force
        active_group = group
        for g in self.groups:
            for ji, joint in enumerate(g.joints):
                angles = list(joint.targets)
                if g is active_group:
                    axis = g.explore_axis % joint.n_axes
                    scale = 1.0 if ji == 0 else 0.5
                    delta = joint.explore_delta * scale
                    if self._phase == 0:
                        angles[axis] += delta
                    else:
                        angles[axis] -= delta

                # Clamp to joint limits
                for k in range(len(angles)):
                    angles[k] = max(joint.lower_limit,
                                    min(joint.upper_limit, angles[k]))

                if joint.jtype == 2:
                    q = _angles_to_quat(angles)
                    p.setJointMotorControlMultiDof(
                        self.body_id, joint.idx, p.POSITION_CONTROL,
                        targetPosition=q, positionGain=PG, velocityGain=VG,
                        force=[joint.max_force]*3, physicsClientId=self.pc)
                elif joint.jtype == 0:
                    p.setJointMotorControl2(
                        self.body_id, joint.idx, p.POSITION_CONTROL,
                        targetPosition=angles[0], force=joint.max_force,
                        positionGain=PG, velocityGain=VG,
                        physicsClientId=self.pc)

        return head_h

    def summary(self):
        total = sum(g.closures for g in self.groups)
        parts = [f"{g.name}:{g.closures}" for g in self.groups]
        return (f"run={self.total_runs} closures={total} "
                f"[{', '.join(parts)}] max_head={self.max_head_height:.3f}")

    def print_learned(self):
        print(f"\n  Learned targets (run {self.total_runs}):")
        for g in self.groups:
            print(f"    {g.name} ({g.closures} closures):")
            for j in g.joints:
                a = ", ".join(f"{t:>6.2f}" for t in j.targets)
                print(f"      {j.name:<20} [{a}]")


def main():
    watch = "--watch" in sys.argv
    pc = p.connect(p.GUI if watch else p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=pc)
    p.setTimeStep(TIMESTEP, physicsClientId=pc)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pc)
    p.loadURDF("plane.urdf", physicsClientId=pc)

    body_id = p.loadURDF("humanoid/humanoid.urdf", [0, 0, 0.5],
                          useFixedBase=False, physicsClientId=pc)

    if watch:
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0, cameraYaw=30, cameraPitch=-20,
            cameraTargetPosition=[0, 0, 1.0], physicsClientId=pc)

    lattice = HierarchicalLattice(pc, body_id)

    # Load genome from previous runs
    loaded = lattice.load_genome()
    lattice.total_runs += 1

    print(f"ENKIDU ALIVE 3D — run {lattice.total_runs}")
    print(f"Genome: {'loaded from previous run' if loaded else 'fresh start'}")
    if loaded:
        lattice.print_learned()
    print(f"Mode: {'visual' if watch else 'headless'}")
    print("=" * 60, flush=True)

    t0 = time.time()
    for tick in range(MAX_TICKS):
        head_h = lattice.step()
        p.stepSimulation(physicsClientId=pc)

        if watch:
            elapsed = time.time() - t0
            sim_time = (tick + 1) * TIMESTEP
            if elapsed < sim_time:
                time.sleep(sim_time - elapsed)

        if (tick + 1) % 720 == 0:  # every 3 seconds
            print(f"  t={tick/240+1:>6.1f}s  head={head_h:.3f}  "
                  f"{lattice.summary()}", flush=True)

    p.disconnect(pc)

    # Save genome for next run
    lattice.save_genome()

    print(f"\n{'='*60}")
    print(f"  Run {lattice.total_runs} complete.  Genome saved.")
    print(f"  {lattice.summary()}")
    lattice.print_learned()
    print(f"  Genome: {GENOME_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
