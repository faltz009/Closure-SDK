"""Closure SDK — identity composition on S³.

Any ordered data can be composed on the ball (S³, unit quaternions).
Two copies of the same data land on the same point. When they diverge,
the distance from identity tells you how much, and the color channels
tell you what kind.

The ball (composition and measurement):
    embed         — bytes become a point on the ball
    compose       — two points become one (the closure operation)
    invert        — the opposite of any point (the undo)
    sigma         — distance from identity (the thermometer)
    diff          — the gap between two points
    compare       — the gap as a yes/no verdict

Lenses (observe the composition at different focal lengths):
    Seer          — the sensor. Streaming, O(1), detects drift.
    Oracle        — the recorder. Full history, O(log n), finds where.
    Witness       — the template. Reference vs test, finds corruption.

The chain (translates ball geometry into color channels):
    expose            — any point → Valence(σ, RGB, W)
    incident_drift    — incident + both paths → local gap quaternion
    expose_incident   — incident + drift → IncidentValence with labels
    bind              — two points → Binding (equal, inverse, or disordered)
    Valence           — σ + base(R,G,B) + phase(W)
    IncidentValence   — channels + positions + payload + axis
    Binding           — relation + gap valence + σ

The canon (finds what broke):
    gilgamesh          — static: compose, narrow, classify
    gilgamesh_detailed — static + paths: same incidents, returns paths for local coloring
    Enkidu             — stream: match, wait, promote, reclassify
    IncidentReport     — one incident: type, positions, payload
    DetailedFaults     — incidents + both paths for per-incident color

Answer formats:
    ClosureState       — a point on the ball
    CompareResult      — drift number + coherent flag
    LocalizationResult — position + search steps
"""

__version__ = "1.0.0"

from .lenses import Seer, Oracle, Witness
from .state import ClosureState, CompareResult, LocalizationResult
from .ops import embed, compose, invert, sigma, diff, compare
from .valence import Valence, IncidentValence, Binding, expose, expose_incident, incident_drift, bind
from .canon import RetentionWindow, IncidentReport, DetailedFaults, gilgamesh, gilgamesh_detailed, Enkidu

__all__ = [
    # Primitives
    "embed",
    "compose",
    "invert",
    "sigma",
    # Derived
    "diff",
    "compare",
    # Lenses
    "Seer",
    "Oracle",
    "Witness",
    # Answer formats
    "ClosureState",
    "CompareResult",
    "LocalizationResult",
    # Canon
    "gilgamesh",
    "gilgamesh_detailed",
    "DetailedFaults",
    "Enkidu",
    "IncidentReport",
    "RetentionWindow",
    # Valence
    "Valence",
    "IncidentValence",
    "Binding",
    "expose",
    "expose_incident",
    "incident_drift",
    "bind",
]
