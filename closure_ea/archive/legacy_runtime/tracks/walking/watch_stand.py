"""
Watch the biped stand — visual demo with PyBullet GUI.

Opens a 3D window showing the biped body maintained upright by the
closure engine.  Runs in real-time so you can see it.

The torso orientation quaternion is read each frame.  C⁻¹ points back
toward identity (upright).  The hip targets adjust from that correction.
The σ trace is printed as the body stands.

Close the window or press Ctrl+C to stop.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).parent))

from closure_ea.kernel import inverse, sigma, identity
from biped import (
    create_biped,
    get_body_state,
    orientation_to_s3,
    ACTIVE_JOINTS,
)

DURATION = 30.0           # watch for 30 seconds
TIMESTEP = 1.0 / 240.0
CORRECTION_GAIN = 0.5
HIP_FORCE = 80.0
KNEE_FORCE = 50.0


def main() -> None:
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.setTimeStep(TIMESTEP, physicsClientId=physics_client)
    p.setRealTimeSimulation(0, physicsClientId=physics_client)

    # Camera setup — look at the biped from a nice angle
    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=30,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 0.8],
        physicsClientId=physics_client,
    )

    body_id = create_biped(physics_client)

    print("WATCHING BIPED STAND")
    print(f"Duration: {DURATION}s (close window or Ctrl+C to stop)")
    print("The closure engine reads torso orientation → C⁻¹ → hip correction")
    print("=" * 60)

    total_steps = int(DURATION / TIMESTEP)
    t0 = time.time()

    try:
        for step in range(total_steps):
            state = get_body_state(physics_client, body_id)
            torso_q = orientation_to_s3(state["orientation"])
            current_sigma = sigma(torso_q)
            correction = inverse(torso_q)
            pitch_error = correction[1]

            hip_target = -pitch_error * CORRECTION_GAIN

            for idx in ACTIVE_JOINTS:
                if idx in (0, 3):
                    p.setJointMotorControl2(
                        body_id, idx, p.POSITION_CONTROL,
                        targetPosition=hip_target,
                        force=HIP_FORCE,
                        positionGain=0.3, velocityGain=0.5,
                        physicsClientId=physics_client,
                    )
                else:
                    p.setJointMotorControl2(
                        body_id, idx, p.POSITION_CONTROL,
                        targetPosition=0.0,
                        force=KNEE_FORCE,
                        positionGain=0.2, velocityGain=0.5,
                        physicsClientId=physics_client,
                    )

            p.stepSimulation(physicsClientId=physics_client)

            # Real-time pacing
            elapsed_sim = (step + 1) * TIMESTEP
            elapsed_wall = time.time() - t0
            if elapsed_wall < elapsed_sim:
                time.sleep(elapsed_sim - elapsed_wall)

            if (step + 1) % 240 == 0:
                contacts = ("L" if state["left_foot_contact"] else ".") + \
                           ("R" if state["right_foot_contact"] else ".")
                print(f"   t={elapsed_sim:>5.1f}s  σ={current_sigma:.4f}  "
                      f"h={state['height']:.3f}  contacts={contacts}")

    except KeyboardInterrupt:
        print("\nStopped by user")

    p.disconnect(physics_client)
    print("Done.")


if __name__ == "__main__":
    main()
