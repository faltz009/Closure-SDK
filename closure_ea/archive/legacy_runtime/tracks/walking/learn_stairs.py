"""
Gilgamesh learns to climb stairs to get food.

Phase 1: Learn to walk on flat ground (300 episodes)
Phase 2: Add a staircase.  The body must walk to the stairs and climb them.
         Food (drive target) is at the top.

The staircase is built from box obstacles at increasing heights.
The gait genome must discover a modified pattern — higher knee flex,
shorter stride — to clear the steps.

Run with:
    python3 learn_stairs.py           # headless
    python3 learn_stairs.py --watch   # visual
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

FLAT_EPISODES = 200       # learn to walk on flat ground first
STAIR_EPISODES = 300      # then learn stairs
MAX_EPISODE_SECONDS = 12.0
TIMESTEP = 1.0 / 240.0
DEATH_TILT = 0.7
FLAT_DRIVE = 0.35
STAIR_DRIVE = 0.40

# Staircase parameters
STAIR_START_X = 0.5       # meters from spawn — close enough to reach
STEP_WIDTH = 0.5          # depth of each step — generous
STEP_HEIGHT = 0.03        # height of each step — very shallow to start
NUM_STEPS = 8
STEP_DEPTH = 1.0          # lateral width


def create_staircase(physics_client: int) -> list:
    """Build a staircase from box obstacles."""
    step_ids = []
    for i in range(NUM_STEPS):
        x = STAIR_START_X + i * STEP_WIDTH
        z = (i + 1) * STEP_HEIGHT / 2.0
        half_h = (i + 1) * STEP_HEIGHT / 2.0

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[STEP_WIDTH / 2, STEP_DEPTH / 2, half_h],
            physicsClientId=physics_client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[STEP_WIDTH / 2, STEP_DEPTH / 2, half_h],
            rgbaColor=[0.6, 0.4, 0.2, 1.0],
            physicsClientId=physics_client,
        )
        step_id = p.createMultiBody(
            baseMass=0,  # static
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, 0, half_h],
            physicsClientId=physics_client,
        )
        step_ids.append(step_id)

    return step_ids


def run_episode(physics_client: int, body_obj: Body, episode: int,
                forward_drive: float, use_stairs: bool, watch: bool,
                use_best: bool = False) -> dict:
    """Run one episode."""
    p.resetSimulation(physicsClientId=physics_client)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.setTimeStep(TIMESTEP, physicsClientId=physics_client)

    if watch:
        p.resetDebugVisualizerCamera(
            cameraDistance=3.5, cameraYaw=45, cameraPitch=-20,
            cameraTargetPosition=[1.5, 0, 0.5],
            physicsClientId=physics_client,
        )

    body_id = create_biped(physics_client)

    if use_stairs:
        create_staircase(physics_client)

    if watch:
        phase_name = "STAIRS" if use_stairs else "FLAT"
        p.addUserDebugText(
            f"Ep {episode}  {phase_name}  drive={forward_drive:.2f}  gen={body_obj.gait.generation}",
            [0, 0, 2.0], textColorRGB=[1, 1, 1], textSize=1.5,
            physicsClientId=physics_client,
        )

    body_obj.begin_episode(forward_drive, use_best=use_best)
    total_steps = int(MAX_EPISODE_SECONDS / TIMESTEP)
    t0 = time.time()

    survived_time = 0.0
    max_forward = 0.0
    max_height = 0.0
    sigma_sum = 0.0

    for step in range(total_steps):
        state = get_body_state(physics_client, body_id)
        torso_q = orientation_to_s3(state["orientation"])
        current_sigma = sigma(torso_q)
        body_pitch = torso_q[1]

        forward_pos = state["position"][0]
        height = state["position"][2]
        if forward_pos > max_forward:
            max_forward = forward_pos
        if height > max_height:
            max_height = height

        body_obj.set_targets(forward_drive, body_pitch, TIMESTEP)
        targets = body_obj.step(
            state["joint_angles"], state["joint_velocities"],
            current_sigma, TIMESTEP,
        )

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
        survived_time = (step + 1) * TIMESTEP

        if current_sigma > DEATH_TILT or height < 0.2:
            break

        if watch:
            elapsed = time.time() - t0
            if elapsed < survived_time:
                time.sleep(survived_time - elapsed)

    height_gained = max(0, max_height - STAND_HEIGHT)
    improved, fitness = body_obj.end_episode(survived_time, max_forward, height_gained)
    mean_sigma = sigma_sum / max(step + 1, 1)

    return {
        "episode": episode,
        "survived": survived_time,
        "max_forward": max_forward,
        "max_height": max_height,
        "mean_sigma": mean_sigma,
        "alive": survived_time >= MAX_EPISODE_SECONDS - TIMESTEP,
        "improved": improved,
        "fitness": fitness,
        "gait_gen": body_obj.gait.generation,
    }


def main() -> None:
    watch = "--watch" in sys.argv
    physics_client = p.connect(p.GUI if watch else p.DIRECT)
    body_obj = Body(epsilon=0.3)

    print(f"GILGAMESH LEARNS STAIRS")
    print(f"Phase 1: {FLAT_EPISODES} episodes flat ground")
    print(f"Phase 2: {STAIR_EPISODES} episodes with stairs")
    print(f"Mode: {'visual' if watch else 'headless'}")
    print("=" * 70)

    results = []

    # ── PHASE 1: Learn to walk on flat ground ─────────────────────────
    print(f"\nPHASE 1: WALK ({FLAT_EPISODES} episodes, drive={FLAT_DRIVE})")
    for ep in range(FLAT_EPISODES):
        result = run_episode(physics_client, body_obj, ep + 1, FLAT_DRIVE,
                             use_stairs=False, watch=watch)
        results.append(result)

        if (ep + 1) % 20 == 0 or result.get("improved"):
            status = "ALIVE" if result["alive"] else f"FELL {result['survived']:.1f}s"
            star = " ★" if result.get("improved") else ""
            print(f"  ep {ep+1:>3}: {status:<14} fwd={result['max_forward']:>5.2f}m  "
                  f"fit={result['fitness']:>5.1f}  gen={result['gait_gen']}{star}")

    flat_best = max(r["max_forward"] for r in results)
    flat_survived = sum(1 for r in results if r["alive"])
    print(f"\n  Flat ground: best={flat_best:.2f}m  survived={flat_survived}/{FLAT_EPISODES}")
    print(f"  Gait genome: {body_obj.genome_summary}")

    # ── PHASE 2: Stairs ───────────────────────────────────────────────
    print(f"\nPHASE 2: STAIRS ({STAIR_EPISODES} episodes, drive={STAIR_DRIVE})")
    stair_results = []

    for ep in range(STAIR_EPISODES):
        result = run_episode(physics_client, body_obj, FLAT_EPISODES + ep + 1,
                             STAIR_DRIVE, use_stairs=True, watch=watch)
        results.append(result)
        stair_results.append(result)

        if (ep + 1) % 20 == 0 or result.get("improved"):
            status = "ALIVE" if result["alive"] else f"FELL {result['survived']:.1f}s"
            star = " ★" if result.get("improved") else ""
            print(f"  ep {FLAT_EPISODES+ep+1:>3}: {status:<14} "
                  f"fwd={result['max_forward']:>5.2f}m  h={result['max_height']:.3f}m  "
                  f"fit={result['fitness']:>5.1f}  gen={result['gait_gen']}{star}")

    p.disconnect(physics_client)

    stair_best_fwd = max(r["max_forward"] for r in stair_results)
    stair_best_h = max(r["max_height"] for r in stair_results)
    stair_survived = sum(1 for r in stair_results if r["alive"])

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Flat: best_fwd={flat_best:.2f}m  survived={flat_survived}/{FLAT_EPISODES}")
    print(f"  Stairs: best_fwd={stair_best_fwd:.2f}m  best_height={stair_best_h:.3f}m  "
          f"survived={stair_survived}/{STAIR_EPISODES}")
    print(f"  Gait genome: {body_obj.genome_summary}")
    print(f"{'='*70}")

    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    audit = out / "learn_stairs_audit.txt"
    with audit.open("w") as f:
        f.write("Gilgamesh learns stairs\n\n")
        f.write(f"Flat: best_fwd={flat_best:.2f}m  survived={flat_survived}/{FLAT_EPISODES}\n")
        f.write(f"Stairs: best_fwd={stair_best_fwd:.2f}m  best_height={stair_best_h:.3f}m  "
                f"survived={stair_survived}/{STAIR_EPISODES}\n")
        f.write(f"Gait genome: {body_obj.genome_summary}\n")
    print(f"Audit: {audit.name}")


if __name__ == "__main__":
    main()
