"""
Phase 1: Stand.

A biped body in PyBullet.  Gravity pulls it down.  The closure engine
drives joint positions to keep it upright.

Identity = standing upright.  σ = tilt from vertical.
Falling = death (σ → π).

The adapter reads the torso orientation quaternion from PyBullet.
The kernel produces C⁻¹ — the correction that would return the body
to upright.  The vector part of C⁻¹ is the correction direction on S³.
Joint position targets are derived from that correction.

This is Enkidu Alive in 3D: the drive is gravity (always pulling away
from identity), the closure is balance (returning to upright).  The kernel
provides the WHAT (correction direction).  The physics engine handles the
HOW (motor control).

Pass condition: the body stands without falling for 10 seconds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from closure_ea.kernel import inverse, sigma, identity
from biped import (
    create_biped,
    get_body_state,
    orientation_to_s3,
    ACTIVE_JOINTS,
)

DURATION = 10.0           # seconds to survive
TIMESTEP = 1.0 / 240.0   # PyBullet default
DEATH_TILT = 0.8          # σ > this = fallen
REPORT_EVERY = 240        # print every N steps (1 second)

# How strongly the kernel's correction maps to joint target offsets
CORRECTION_GAIN = 0.5     # radians of hip offset per unit of pitch error
HIP_FORCE = 80.0          # max torque the position controller can apply
KNEE_FORCE = 50.0


def main() -> None:
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.setTimeStep(TIMESTEP, physicsClientId=physics_client)

    body_id = create_biped(physics_client)

    print("PHASE 1: STAND")
    print(f"Duration target: {DURATION}s")
    print(f"Death threshold: σ > {DEATH_TILT}")
    print(f"Timestep: {TIMESTEP}s ({int(1/TIMESTEP)} Hz)")
    print("=" * 60)

    total_steps = int(DURATION / TIMESTEP)
    alive = True
    max_sigma = 0.0
    sigma_sum = 0.0
    closures = 0
    epsilon_balance = 0.05

    t0 = time.time()

    for step in range(total_steps):
        # 1. READ STATE — S1: measurement
        state = get_body_state(physics_client, body_id)
        torso_q = orientation_to_s3(state["orientation"])

        # 2. MEASURE GAP — σ from identity (upright)
        current_sigma = sigma(torso_q)

        # 3. COMPUTE CORRECTION — C⁻¹ points back toward identity
        correction = inverse(torso_q)
        pitch_error = correction[1]  # i-axis: forward/back tilt

        # 4. TRACK
        sigma_sum += current_sigma
        if current_sigma > max_sigma:
            max_sigma = current_sigma
        if current_sigma < epsilon_balance:
            closures += 1

        # 5. CHECK DEATH
        if current_sigma > DEATH_TILT:
            alive = False
            elapsed = step * TIMESTEP
            print(f"\n   DEATH at t={elapsed:.2f}s  σ={current_sigma:.4f}")
            break

        # 6. ACT — kernel correction → joint position targets
        # The pitch error from C⁻¹ tells the hips how much to adjust.
        # Hips lean to counteract the tilt.  Knees stay straight.
        hip_target = -pitch_error * CORRECTION_GAIN

        for idx in ACTIVE_JOINTS:
            if idx in (0, 3):  # hips
                p.setJointMotorControl2(
                    body_id, idx, p.POSITION_CONTROL,
                    targetPosition=hip_target,
                    force=HIP_FORCE,
                    positionGain=0.3, velocityGain=0.5,
                    physicsClientId=physics_client,
                )
            else:  # knees
                p.setJointMotorControl2(
                    body_id, idx, p.POSITION_CONTROL,
                    targetPosition=0.0,
                    force=KNEE_FORCE,
                    positionGain=0.2, velocityGain=0.5,
                    physicsClientId=physics_client,
                )

        # 7. STEP PHYSICS
        p.stepSimulation(physicsClientId=physics_client)

        # Report
        if (step + 1) % REPORT_EVERY == 0:
            elapsed = (step + 1) * TIMESTEP
            mean_sigma = sigma_sum / (step + 1)
            contacts = ("L" if state["left_foot_contact"] else ".") + \
                       ("R" if state["right_foot_contact"] else ".")
            print(f"   t={elapsed:>5.1f}s  σ={current_sigma:.4f}  "
                  f"mean_σ={mean_sigma:.4f}  max_σ={max_sigma:.4f}  "
                  f"contacts={contacts}  closures={closures}  "
                  f"h={state['height']:.3f}")

    elapsed = (step + 1) * TIMESTEP
    wall_time = time.time() - t0
    mean_sigma = sigma_sum / (step + 1) if step > 0 else 0

    p.disconnect(physics_client)

    print(f"\n{'='*60}")
    if alive:
        print(f"PHASE 1: PASS — stood for {elapsed:.1f}s")
    else:
        print(f"PHASE 1: FAIL — fell at {elapsed:.2f}s")
    print(f"  Mean σ: {mean_sigma:.4f}")
    print(f"  Max σ: {max_sigma:.4f}")
    print(f"  Closures (σ < {epsilon_balance}): {closures}/{step+1}")
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"{'='*60}")

    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    audit = out / "embodied_phase1_stand_audit.txt"
    with audit.open("w") as f:
        f.write(f"Phase 1: Stand\n")
        f.write(f"Result: {'PASS' if alive else 'FAIL'}\n")
        f.write(f"Duration: {elapsed:.2f}s / {DURATION}s\n")
        f.write(f"Mean σ: {mean_sigma:.6f}\n")
        f.write(f"Max σ: {max_sigma:.6f}\n")
        f.write(f"Closures: {closures}/{step+1}\n")
        f.write(f"Death threshold: {DEATH_TILT}\n")
    print(f"Audit: {audit.name}")


if __name__ == "__main__":
    main()
