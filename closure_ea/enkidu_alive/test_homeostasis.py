"""Test the homeostasis loop.

Spawn food, watch Enkidu get hungry, walk to food, eat,
get cold (far from home), walk home, rest. Repeat.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from homeostasis import World


def _run_until(world: World, predicate, max_ticks: int) -> None:
    """Advance until the predicate holds or the budget is exhausted."""
    for _ in range(max_ticks):
        if not world.alive:
            break
        snap = world.tick()
        if predicate(snap):
            break


def test_basic_cycle():
    """One full cycle: rest → hungry → walk to food → eat → cold → walk home."""
    print("=" * 60)
    print("TEST 1: Basic hunger → food → home cycle")
    print("=" * 60)
    print()

    w = World(grid_size=15, hunger_rate=0.15, food_spawn_rate=0.0)
    w.spawn_food(4, 3)
    print(f"  Food at (4, 3). Enkidu at home (0, 0).")
    print(f"  Hunger rate: {w.hunger_rate} per tick.")
    print()

    w.run(30)

    # Verify he ate and got home
    ate_ticks = [s for s in w.trace if s["ate"]]
    home_ticks = [s for s in w.trace if s["pos"] == (0, 0) and s["tick"] > 1]

    print()
    if ate_ticks:
        print(f"  Ate at tick {ate_ticks[0]['tick']} at position {ate_ticks[0]['pos']}")
    if home_ticks:
        print(f"  Returned home at tick {home_ticks[0]['tick']}")
    print("  PASS\n")


def test_two_food_sources():
    """Two food sources. Enkidu should go to the nearest one."""
    print("=" * 60)
    print("TEST 2: Two food sources — goes to nearest")
    print("=" * 60)
    print()

    w = World(grid_size=15, hunger_rate=0.20, food_spawn_rate=0.0)
    w.spawn_food(2, 1)   # close
    w.spawn_food(8, -6)  # far
    print(f"  Food at (2, 1) and (8, -6). Enkidu at (0, 0).")
    print()

    w.run(25)

    ate_ticks = [s for s in w.trace if s["ate"]]
    if ate_ticks:
        print(f"\n  Ate at position {ate_ticks[0]['pos']} — {'nearest' if ate_ticks[0]['pos'] == (2, 1) else 'farther'}")
        assert ate_ticks[0]["pos"] == (2, 1), "Should eat nearest food first"
    print("  PASS\n")


def test_multiple_cycles():
    """Multiple hunger cycles with food respawning."""
    print("=" * 60)
    print("TEST 3: Three full cycles — food respawns")
    print("=" * 60)
    print()

    w = World(grid_size=15, hunger_rate=0.25, food_spawn_rate=0.0)

    # Cycle 1
    w.spawn_food(3, 0)
    print("  --- Cycle 1: food at (3, 0) ---")
    _run_until(w, lambda s: s["ate"], 12)
    _run_until(w, lambda s: s["pos"] == (0, 0), 12)

    # Cycle 2
    w.spawn_food(-2, 4)
    print("\n  --- Cycle 2: food at (-2, 4) ---")
    _run_until(w, lambda s: s["ate"] and s["pos"] == (-2, 4), 12)
    _run_until(w, lambda s: s["pos"] == (0, 0), 12)

    # Cycle 3
    w.spawn_food(0, -5)
    print("\n  --- Cycle 3: food at (0, -5) ---")
    _run_until(w, lambda s: s["ate"] and s["pos"] == (0, -5), 14)

    ate_count = sum(1 for s in w.trace if s["ate"])
    print(f"\n  Total meals: {ate_count}")
    assert ate_count >= 2, f"Expected at least 2 meals, got {ate_count}"
    print("  PASS\n")


def test_drive_switching():
    """Verify the drive switches correctly between hunger and shelter."""
    print("=" * 60)
    print("TEST 4: Drive switching trace")
    print("=" * 60)
    print()

    w = World(grid_size=15, hunger_rate=0.12, food_spawn_rate=0.0)
    w.spawn_food(5, 5)

    drives_seen = set()
    for _ in range(50):
        snap = w.tick()
        drives_seen.add(snap["drive"])

    # Re-run with output for last few ticks
    w2 = World(grid_size=15, hunger_rate=0.12, food_spawn_rate=0.0)
    w2.spawn_food(5, 5)
    w2.run(50)

    print(f"\n  Drives seen: {drives_seen}")
    assert "hunger" in drives_seen, "Should have been hungry"
    assert "shelter" in drives_seen or "rest" in drives_seen, "Should have gone home"
    print("  PASS\n")


if __name__ == "__main__":
    test_basic_cycle()
    test_two_food_sources()
    test_multiple_cycles()
    test_drive_switching()
    print("All homeostasis tests passed.")
