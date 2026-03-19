"""Answer formats — the shapes that come back from the tools.

When the tools in ops.py and the machines in monitor/path/tree do
their work, they return their answers in one of these three formats.
These are containers, not operations — they hold results, they don't
compute anything.

Formats:
    ClosureState        — a point on the ball. Returned by embed, compose,
                          invert, diff, and the .state() method on machines.
    CompareResult       — a drift number and a coherent flag. Returned by compare.
    LocalizationResult  — a position and a step count. Returned by localize_against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ClosureState:
    """A point on the ball. Contains the group name ("Sphere") and the
    quaternion (4 floats). This is a composition frozen at a moment in
    time — the running product as a snapshot.
    """

    group: str
    element: NDArray[np.float64]

    @property
    def dim(self) -> int:
        """Number of components in the quaternion (always 4 for S³)."""
        return len(self.element)

    def __repr__(self) -> str:
        return f"ClosureState(group={self.group!r}, dim={self.dim})"


@dataclass(frozen=True, slots=True)
class CompareResult:
    """The verdict from compare(). drift is how far apart two points
    are on the ball. coherent is True if drift is below threshold,
    False if something broke.
    """

    drift: float
    coherent: bool

    def __repr__(self) -> str:
        label = "COHERENT" if self.coherent else f"DIVERGED (drift={self.drift:.6f})"
        return f"CompareResult({label})"


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """The answer from binary search. index is where in the stream the
    first divergence was found (None if streams are coherent). checks
    is how many steps the search took (always O(log n)).
    """

    index: int | None
    checks: int

    def __repr__(self) -> str:
        return f"LocalizationResult(index={self.index}, checks={self.checks})"
