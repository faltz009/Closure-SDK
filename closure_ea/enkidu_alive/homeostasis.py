"""Homeostasis — competing drives select which target Enkidu walks toward.

Two drives, same function. Hunger rises with time. Coldness rises
with distance from home (σ). Each drive is tracked as a live stream
against identity using the SDK's Enkidu classifier. Whichever open
drive is larger picks the target.
"""

import random
import sys
from pathlib import Path
from geometry import EnkiduState, hopf_decompose

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closure_sdk.canon import Enkidu as StreamEnkidu


HUNGER_LETHAL = 3.14  # π — max hunger before death


class DriveChannel:
    """One internal drive tracked by the SDK Enkidu stream classifier.

    `source` means the drive is active (away from identity).
    `target` means the drive resolved back to identity.
    """

    def __init__(self, label: str):
        self.label = label.encode()
        self.detector = StreamEnkidu()
        self._active = False
        self.last_incidents = []

    def observe(self, active_now: bool, tick: int) -> bool:
        incidents = []
        if active_now != self._active:
            side = "source" if active_now else "target"
            report = self.detector.ingest(self.label, tick, side)
            if report is not None:
                incidents.append(report)
            self._active = active_now
        incidents.extend(self.detector.advance_cycle())
        self.last_incidents = incidents
        return self.away_from_identity

    @property
    def away_from_identity(self) -> bool:
        return self.detector.unresolved_source > 0

    def snapshot(self) -> dict:
        return {
            "away": self.away_from_identity,
            "cycle": self.detector.cycle,
            "unresolved_source": self.detector.unresolved_source,
            "unresolved_target": self.detector.unresolved_target,
            "reclassified": self.detector.reclassified_count,
            "incidents": [
                {
                    "type": inc.incident_type,
                    "source_index": inc.source_index,
                    "target_index": inc.target_index,
                }
                for inc in self.last_incidents
            ],
        }


class World:
    """The grid world. Enkidu, home, food sources, drives."""

    def __init__(self, grid_size=15, hunger_rate=0.04, food_spawn_rate=0.0):
        self.grid_size = grid_size
        self.hunger_rate = hunger_rate
        self.food_spawn_rate = food_spawn_rate  # probability per tick

        # Enkidu starts at home
        self.enkidu = EnkiduState(0, 0)

        # Drives — both start at 0 (homeostasis)
        self.hunger = 0.0
        self._hunger_channel = DriveChannel("hunger")
        self._cold_channel = DriveChannel("cold")

        # State
        self.alive = True

        # Food sources: list of (x, y)
        self.food = []

        # Trace for visualization
        self.trace = []
        self.tick_count = 0

    @property
    def coldness(self):
        """Coldness = σ(C) = distance from home. Computed by geometry."""
        return self.enkidu.sigma_home()

    @property
    def active_drive(self):
        """Which open drive is dominant right now?"""
        hunger_open = self._hunger_channel.away_from_identity and bool(self.food)
        cold_open = self._cold_channel.away_from_identity
        if hunger_open and (not cold_open or self.hunger > self.coldness):
            return "hunger"
        elif cold_open:
            return "shelter"
        else:
            return "rest"

    @property
    def target(self):
        """Current navigation target based on active drive."""
        drive = self.active_drive
        if drive == "hunger":
            # Find nearest food
            return self._nearest_food()
        elif drive == "shelter":
            return (0, 0)
        else:
            return None  # at rest

    def spawn_food(self, x=None, y=None):
        """Place food on the grid. Random if no position given."""
        if x is None or y is None:
            x = random.randint(-self.grid_size, self.grid_size)
            y = random.randint(-self.grid_size, self.grid_size)
            # Don't spawn on home
            while x == 0 and y == 0:
                x = random.randint(-self.grid_size, self.grid_size)
                y = random.randint(-self.grid_size, self.grid_size)
        self.food.append((x, y))
        return (x, y)

    def tick(self):
        """One time step. Returns a snapshot of the state."""
        if not self.alive:
            return self.trace[-1] if self.trace else None

        self.tick_count += 1

        # Random food spawn
        if random.random() < self.food_spawn_rate:
            pos = self.spawn_food()

        # Hunger rises
        self.hunger = min(self.hunger + self.hunger_rate, HUNGER_LETHAL)

        # Classify the live drive streams against identity.
        hunger_active = self.hunger > 0.0 and bool(self.food)
        cold_active = self.coldness > 0.01
        self._hunger_channel.observe(hunger_active, self.tick_count)
        self._cold_channel.observe(cold_active, self.tick_count)

        # Death check
        died = self.hunger >= HUNGER_LETHAL
        if died:
            self.alive = False

        # Pick target and move (if alive)
        drive = self.active_drive if self.alive else "dead"
        tgt = self.target if self.alive else None
        move_name = None
        if self.alive and tgt is not None:
            move_name, _ = self.enkidu.best_step_toward(*tgt)
            self.enkidu.move(move_name)

        # Check food arrival
        ate = False
        if self.alive:
            for food_pos in self.food[:]:
                if self.enkidu.position == food_pos:
                    self.hunger = 0.0
                    self.food.remove(food_pos)
                    ate = True
                    break

        # Snapshot
        snap = {
            "tick": self.tick_count,
            "pos": self.enkidu.position,
            "move": move_name,
            "drive": drive,
            "target": tgt,
            "hunger": self.hunger,
            "coldness": self.coldness,
            "sigma_home": self.enkidu.sigma_home(),
            "ate": ate,
            "died": died,
            "alive": self.alive,
            "food": list(self.food),
            "hopf": self.enkidu.hopf(),
            "drive_channels": {
                "hunger": self._hunger_channel.snapshot(),
                "cold": self._cold_channel.snapshot(),
            },
        }
        self.trace.append(snap)
        return snap

    def _nearest_food(self):
        """Grid-distance nearest food."""
        if not self.food:
            return None
        ex, ey = self.enkidu.position
        return min(self.food, key=lambda f: abs(f[0] - ex) + abs(f[1] - ey))

    def run(self, ticks, verbose=True):
        """Run the world for N ticks. Stops on death."""
        for _ in range(ticks):
            if not self.alive:
                break
            snap = self.tick()
            if verbose:
                self._print_snap(snap)
            if snap.get("died"):
                break

    def _print_snap(self, s):
        pos = s["pos"]
        move = s["move"] or "rest"
        drive = s["drive"]
        tgt = s["target"] or "—"
        h_bar = "█" * int(s["hunger"] * 6)
        c_bar = "█" * int(s["coldness"] * 6)
        event = ""
        if s.get("died"):
            event = " DIED!"
        elif s.get("ate"):
            event = " ATE!"
        print(
            f"  t={s['tick']:>3}  ({pos[0]:>3},{pos[1]:>3})  "
            f"{move:>5}  drive={drive:<8}  tgt={str(tgt):<10}  "
            f"H={s['hunger']:.2f} {h_bar:<10}  "
            f"C={s['coldness']:.2f} {c_bar:<10}"
            f"{event}"
        )
