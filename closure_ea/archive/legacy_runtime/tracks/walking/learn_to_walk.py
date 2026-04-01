"""
Watch Gilgamesh learn to stand and walk.

Multiple episodes.  Each episode the body spawns upright with a shelter
target ahead.  The lattice of joint-Enkidus tries to keep the body
upright (balance drive) and move forward (cold drive).  Falling = death.

The genome persists between episodes.  Early episodes: the body falls.
Later episodes: the body stands longer.  Eventually: the body walks.

Run with:
    python3 learn_to_walk.py          # headless, fast
    python3 learn_to_walk.py --watch  # PyBullet GUI, real-time
"""

from __future__ import annotations

import sys
import time
import math
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).parent))

from closure_ea.kernel import inverse, sigma, identity
from biped import (
    create_biped,
    get_body_state,
    apply_joint_torques,
    orientation_to_s3,
    ACTIVE_JOINTS,
    STAND_HEIGHT,
)
from body import Body


EPISODES = 300
MAX_EPISODE_SECONDS = 10.0
TIMESTEP = 1.0 / 240.0
DEATH_TILT = 0.7          # σ > this = fallen
SHELTER_DISTANCE = 5.0     # meters ahead

FIXED_DRIVE = 0.35        # constant drive — let the genome learn to walk at this level


def run_episode(physics_client: int, body_obj: Body, episode: int,
                forward_drive: float, watch: bool) -> dict:
    """Run one episode.  Returns stats."""

    # Reset the body position
    body_id = None
    # We need to reload because PyBullet doesn't cleanly reset multibodies
    p.resetSimulation(physicsClientId=physics_client)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.setTimeStep(TIMESTEP, physicsClientId=physics_client)
    if watch:
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0, cameraYaw=60, cameraPitch=-15,
            cameraTargetPosition=[1, 0, 0.5],
            physicsClientId=physics_client,
        )
    body_id = create_biped(physics_client)

    # Show episode info in the GUI
    debug_text_id = None
    if watch:
        debug_text_id = p.addUserDebugText(
            f"Episode {episode}  drive={forward_drive:.2f}  genome={body_obj.genome_size}",
            [0, 0, 2.0], textColorRGB=[1, 1, 1], textSize=1.5,
            physicsClientId=physics_client,
        )

    body_obj.begin_episode(forward_drive)
    total_steps = int(MAX_EPISODE_SECONDS / TIMESTEP)
    t0 = time.time()

    survived_time = 0.0
    max_forward = 0.0
    sigma_sum = 0.0
    closures = 0

    for step in range(total_steps):
        state = get_body_state(physics_client, body_id)
        torso_q = orientation_to_s3(state["orientation"])
        current_sigma = sigma(torso_q)
        body_pitch = torso_q[1]  # i-axis component = forward/back tilt

        # Forward progress
        forward_pos = state["position"][0]  # x-axis is forward
        if forward_pos > max_forward:
            max_forward = forward_pos

        # Set targets from the (mutated) gait genome + balance correction
        body_obj.set_targets(forward_drive, body_pitch, TIMESTEP)

        # Each joint-Enkidu produces a target position
        targets = body_obj.step(
            state["joint_angles"], state["joint_velocities"],
            current_sigma, TIMESTEP,
        )

        # Apply via position control
        for idx, target_pos in targets.items():
            force = 80.0 if idx in (0, 3) else 50.0
            p.setJointMotorControl2(
                body_id, idx, p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=force,
                positionGain=0.3, velocityGain=0.5,
                physicsClientId=physics_client,
            )

        p.stepSimulation(physicsClientId=physics_client)

        sigma_sum += current_sigma
        if current_sigma < 0.1:
            closures += 1
        survived_time = (step + 1) * TIMESTEP

        # Death check
        if current_sigma > DEATH_TILT or state["height"] < 0.3:
            break

        # Real-time pacing for visual mode
        if watch:
            elapsed = time.time() - t0
            target = survived_time
            if elapsed < target:
                time.sleep(target - elapsed)

    mean_sigma = sigma_sum / max(step + 1, 1)

    # Score this episode and update gait genome if better
    improved, fitness = body_obj.end_episode(survived_time, max_forward)

    return {
        "episode": episode,
        "survived": survived_time,
        "max_forward": max_forward,
        "mean_sigma": mean_sigma,
        "closures": closures,
        "total_steps": step + 1,
        "alive": survived_time >= MAX_EPISODE_SECONDS - TIMESTEP,
        "improved": improved,
        "fitness": fitness,
        "gait_gen": body_obj.gait.generation,
    }


def main() -> None:
    watch = "--watch" in sys.argv

    physics_client = p.connect(p.GUI if watch else p.DIRECT)

    body_obj = Body(epsilon=0.3)

    print(f"GILGAMESH LEARNS TO WALK")
    print(f"Episodes: {EPISODES}")
    print(f"Max episode: {MAX_EPISODE_SECONDS}s")
    print(f"Death: σ > {DEATH_TILT}")
    print(f"Shelter: {SHELTER_DISTANCE}m ahead")
    print(f"Mode: {'visual' if watch else 'headless'}")
    print("=" * 70)

    results = []
    forward_drive = FIXED_DRIVE

    for ep in range(EPISODES):
        result = run_episode(physics_client, body_obj, ep + 1, forward_drive, watch)
        results.append(result)

        status = "ALIVE" if result["alive"] else f"FELL at {result['survived']:.1f}s"
        star = " ★" if result.get("improved") else ""
        print(f"  ep {ep+1:>3}: {status:<16} "
              f"fwd={result['max_forward']:>5.2f}m  "
              f"σ={result['mean_sigma']:.3f}  "
              f"fit={result['fitness']:>5.1f}  "
              f"gen={result['gait_gen']}  "
              f"drive={forward_drive:.2f}{star}")

    p.disconnect(physics_client)

    # Summary
    survived_eps = sum(1 for r in results if r["alive"])
    max_fwd = max(r["max_forward"] for r in results)
    best_fitness = max(r["fitness"] for r in results)
    improvements = sum(1 for r in results if r.get("improved"))

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Survived full episode: {survived_eps}/{EPISODES}")
    print(f"  Max forward distance: {max_fwd:.2f}m")
    print(f"  Best fitness: {best_fitness:.1f}")
    print(f"  Gait improvements: {improvements}")
    print(f"  Gait genome: {body_obj.genome_summary}")
    print(f"{'='*70}")

    # Save audit
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    audit = out / "learn_to_walk_audit.txt"
    with audit.open("w") as f:
        f.write("Gilgamesh learns to walk\n\n")
        f.write(f"Episodes: {EPISODES}\n")
        f.write(f"Survived: {survived_eps}/{EPISODES}\n")
        f.write(f"Max forward: {max_fwd:.2f}m\n")
        f.write(f"Best fitness: {best_fitness:.1f}\n")
        f.write(f"Gait genome: {body_obj.genome_summary}\n\n")
        for r in results:
            status = "ALIVE" if r["alive"] else f"FELL {r['survived']:.1f}s"
            f.write(f"ep {r['episode']:>3}: {status:<12} "
                    f"fwd={r['max_forward']:.2f}m  "
                    f"sigma={r['mean_sigma']:.3f}  "
                    f"closures={r['closures']}\n")
    print(f"Audit: {audit.name}")


if __name__ == "__main__":
    main()
