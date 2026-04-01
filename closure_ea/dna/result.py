"""Result types returned by the stable Closure DNA surface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResonanceHit:
    """One search result from a resonance query.

    position  — where in the table (0-indexed)
    drift     — geodesic distance from query (0 = exact match)
    base      — S² direction (R, G, B) — what kind of relationship
    phase     — S¹ fiber (W) — positional context
    """

    position: int
    drift: float
    base: tuple[float, float, float]
    phase: float

    def __repr__(self) -> str:
        return f"ResonanceHit(position={self.position}, drift={self.drift:.6f})"

