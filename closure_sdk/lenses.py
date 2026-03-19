"""The lenses — three ways to observe composition on S³.

All three lenses compose records into running products on the ball.
They differ in how much they remember and what questions they can
answer — different focal lengths on the same geometry.

    Seer     — the sensor. Holds one point. Constant memory. Detects
               drift but cannot say where. The cheap always-on watcher.

    Oracle   — the recorder. Holds every intermediate point. O(n) memory.
               Detects drift AND locates where it happened via binary
               search in O(log n) steps. You consult the Oracle when
               the Seer raises an alarm.

    Witness  — the reference template. Built once from known-good data.
               Checks any test data against the reference. Answers
               "does this match the original?" and "where does it diverge?"
"""

from __future__ import annotations
from typing import Any

import closure_rs
import numpy as np

# The geometry — always S³.
GROUP_SPEC = "Sphere"

from .state import ClosureState, CompareResult, LocalizationResult


# ── Seer ──────────────────────────────────────────────────────────────

class Seer:
    """The sensor. Feed it records, ask it if the streams still match.

    Example::

        source = closure.Seer()
        target = closure.Seer()

        for record in stream:
            source.ingest(record)
            target.ingest(transform(record))

        result = source.compare(target)
        if not result.coherent:
            print(f"Drift detected: drift = {result.drift}")
    """

    __slots__ = ("_monitor", "_ingest", "_ingest_many")

    def __init__(self) -> None:
        mon = closure_rs.StreamMonitor(GROUP_SPEC)
        self._monitor = mon
        self._ingest = mon.ingest
        self._ingest_many = mon.ingest_many

    def ingest(self, record: bytes) -> None:
        """Feed one record into the running product."""
        self._ingest(record)

    def ingest_many(self, records: list[bytes]) -> None:
        """Feed many records at once."""
        self._ingest_many(records)

    def drift(self) -> float:
        """How far the current point is from identity (north pole).
        Zero means the composition is clean.
        """
        return self._monitor.sigma()

    def state(self) -> ClosureState:
        """Snapshot the current point on the ball."""
        return ClosureState(
            group=GROUP_SPEC,
            element=np.array(self._monitor.closure_element()),
        )

    def compare(self, other: "Seer", *, threshold: float = 1e-10) -> CompareResult:
        """Compare this sensor's point against another's."""
        drift = self._monitor.compare_against(other._monitor)
        return CompareResult(drift=drift, coherent=drift < threshold)

    def reset(self) -> None:
        """Back to identity — start fresh."""
        mon = self._monitor
        mon.reset()
        self._ingest = mon.ingest
        self._ingest_many = mon.ingest_many

    @property
    def group(self) -> str:
        """The geometry spec (always "Sphere")."""
        return GROUP_SPEC

    def __len__(self) -> int:
        return len(self._monitor)

    def __repr__(self) -> str:
        return f"Seer(n={len(self)}, drift={self.drift():.6f})"


# ── Oracle ────────────────────────────────────────────────────────────

class Oracle:
    """The recorder. Builds a complete composition history on S³.

    Build all at once::

        trace = closure.Oracle.from_records(records)

    Or feed incrementally::

        trace = closure.Oracle()
        for record in stream:
            trace.append(record)
    """

    __slots__ = ("_path", "_records")

    def __init__(self) -> None:
        self._path: Any | None = None
        self._records: list[bytes] = []

    @classmethod
    def from_records(cls, records: list[bytes]) -> "Oracle":
        """Build a full trace from a list of raw records."""
        obj = cls.__new__(cls)
        obj._path = closure_rs.path_from_raw_bytes(GROUP_SPEC, records)
        obj._records = list(records)
        return obj

    def append(self, record: bytes) -> None:
        """Add one record to the trace."""
        self._records.append(record)
        if self._path is None:
            self._path = closure_rs.path_from_raw_bytes(GROUP_SPEC, self._records)
        else:
            elem = closure_rs.closure_element_from_raw_bytes(GROUP_SPEC, [record])
            self._path.append(np.array(elem, dtype=np.float64))

    def check_global(self) -> float:
        """Sigma of the final running product. O(1).
        Zero means the whole sequence is coherent.
        """
        self._ensure_built()
        return self._path.check_global()

    def check_range(self, i: int, j: int) -> float:
        """Sigma of a sub-segment [i, j]. O(1).
        Lets you narrow down which part of the stream drifted.
        """
        self._ensure_built()
        return self._path.check_range(i, j)

    def recover(self, t: int) -> np.ndarray:
        """The embedded element at position t (1-indexed). O(1).
        Retrieves the individual point that was composed in at step t.
        """
        self._ensure_built()
        return np.array(self._path.recover(t))

    def localize_against(self, other: "Oracle") -> LocalizationResult:
        """Binary search for the first divergence against another trace.
        O(log n) comparisons. Returns the position and step count.
        """
        self._ensure_built()
        other._ensure_built()
        index, checks = self._path.localize_against(other._path)
        return LocalizationResult(index=index, checks=checks)

    def state(self) -> ClosureState:
        """Snapshot the current point on the ball (final running product)."""
        self._ensure_built()
        return ClosureState(
            group=GROUP_SPEC,
            element=np.array(self._path.closure_element()),
        )

    def compare(self, other: "Oracle", *, threshold: float = 1e-10) -> CompareResult:
        """Compare final running products of two traces."""
        s1 = self.state()
        s2 = other.state()
        from .ops import compare
        return compare(s1, s2, threshold=threshold)

    @property
    def group(self) -> str:
        return GROUP_SPEC

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"Oracle(n={len(self._records)})"

    def _ensure_built(self) -> None:
        if self._path is None:
            if not self._records:
                raise ValueError("Oracle is empty — append records first")
            self._path = closure_rs.path_from_raw_bytes(GROUP_SPEC, self._records)


# ── Witness ───────────────────────────────────────────────────────────

class Witness:
    """The reference template. Built from known-good data, checks test
    data against it.

    Example::

        ref = closure.Witness.from_records(good_records)

        drift = ref.check(test_records)
        if drift > 1e-6:
            result = ref.localize(test_records)
            print(f"Corruption at index {result.index}")
    """

    __slots__ = ("_group", "_elements", "_tree")

    def __init__(self, elements: list[np.ndarray]) -> None:
        self._group = closure_rs.sphere()
        self._elements = np.array(elements)
        self._tree = closure_rs.HierarchicalClosure(self._group, self._elements)

    @classmethod
    def from_records(cls, records: list[bytes]) -> "Witness":
        """Build a reference from raw records."""
        elements = []
        for record in records:
            elem = closure_rs.closure_element_from_raw_bytes(GROUP_SPEC, [record])
            elements.append(np.array(elem))
        return cls(elements)

    def check(self, test_data: list[bytes] | np.ndarray) -> float:
        """How much drift between the reference and the test data.
        Returns sigma — zero means they match.
        """
        test_elements = self._to_elements(test_data)
        return self._tree.check(test_elements)

    def localize(
        self,
        test_data: list[bytes] | np.ndarray,
        *,
        threshold: float = 1e-6,
    ) -> LocalizationResult:
        """Where the test data first diverges from the reference.
        O(log n) binary search. Returns position and step count.
        """
        test_elements = self._to_elements(test_data)
        result = self._tree.localize(test_elements, threshold)
        index, checks, _depth = result
        return LocalizationResult(
            index=index,
            checks=checks,
        )

    def _to_elements(self, data: list[bytes] | np.ndarray) -> np.ndarray:
        """Convert raw bytes or element arrays to numpy elements."""
        if isinstance(data, np.ndarray):
            return data
        elements = []
        for record in data:
            elem = closure_rs.closure_element_from_raw_bytes(GROUP_SPEC, [record])
            elements.append(np.array(elem))
        return np.array(elements)

    @property
    def group(self) -> str:
        return GROUP_SPEC

    def state(self) -> ClosureState:
        """The reference composition as a point on the ball."""
        path = closure_rs.GeometricPath.from_elements(self._group, self._elements)
        return ClosureState(group=GROUP_SPEC, element=np.array(path.closure_element()))

    def __repr__(self) -> str:
        n = len(self._elements)
        return f"Witness(n={n})"
