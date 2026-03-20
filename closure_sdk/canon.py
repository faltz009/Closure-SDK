"""The canon — finds what's true, where it broke, and how.

Two composition modes, same classification:

    Static mode (gilgamesh):
        Both streams are complete. Compose both on S³, binary-search for
        the first divergence, classify it, remove the pair, recompose,
        repeat. Because we have the full picture, we know instantly
        whether a record is missing or reordered.
        O(k · n · log n) where k is the number of incidents.

    Stream mode (Enkidu):
        Records arrive one at a time. We cannot see the future, so every
        unmatched record is initially a missing record — that is literally
        what it is at the present moment. The question is whether it STAYS
        missing or gets reclassified as a reorder when its match arrives
        late on the other side.

        The rolling counter makes this uncertainty explicit. Each cycle,
        for every unresolved record, it asks one binary question:

            "Is this an invertibility of payload (has or hasn't)
             or an invertibility of position (arrived late)?"

        Three things can happen when a new record arrives:

            1. Both sides receive the same package → coherent, no incident.
            2. One side receives a package that will arrive late → reorder.
            3. One side receives a package that will never arrive → missing.

        We can't distinguish 2 from 3 at arrival time. The grace period
        is a bounded wait that turns this into a decision: hold the record
        as "unresolved" for one cycle. If its match arrives next cycle,
        reclassify missing → reorder. If not, promote to missing.

        After a misalignment, the NEXT step has only two possible outcomes:
        either the late record arrives (good — reclassify to reorder), or
        another missing record arrives (bad — but it's the same case again,
        same binary question, same grace period). The solution is recursive:
        each new unresolved record gets the same treatment, the same bounded
        wait, the same binary decision. Errors chain but the tool chains
        with them.

Also here:

    IncidentReport   — the report. One incident: what type (missing or
                       reorder), which positions in each stream, the actual
                       record bytes, and how many search steps it took.

    RetentionWindow  — the evidence archive. A rolling buffer that stores
                       raw records so they're available when the Seer
                       detects drift and you need to investigate.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import closure_rs
import numpy as np

from .lenses import GROUP_SPEC


@dataclass
class RetainedBlock:
    """One block of raw records, tagged with its block number."""

    block_number: int
    records: list[bytes]


class RetentionWindow:
    """A bounded rolling buffer of raw records. When a Monitor detects
    drift, this is where the raw bytes are stored so gilgamesh can
    work on them. Plumbing — no algebra happens here.
    """

    def __init__(self, maxlen: int = 64) -> None:
        self._blocks: deque[RetainedBlock] = deque(maxlen=maxlen)

    def append(self, block_number: int, records: list[bytes]) -> None:
        """Store a block's records."""
        self._blocks.append(RetainedBlock(block_number, records))

    def flatten(self) -> list[bytes]:
        """All stored records as a single flat list."""
        result: list[bytes] = []
        for block in self._blocks:
            result.extend(block.records)
        return result

    def block_map(self) -> list[tuple[int, int]]:
        """Return [(block_number, record_count), ...] for index mapping."""
        return [(b.block_number, len(b.records)) for b in self._blocks]

    @property
    def total_records(self) -> int:
        """Total records across all stored blocks."""
        return sum(len(b.records) for b in self._blocks)

    @property
    def blocks(self) -> deque[RetainedBlock]:
        """The underlying block deque."""
        return self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        return f"RetentionWindow(blocks={len(self)}, records={self.total_records})"


@dataclass(frozen=True)
class IncidentReport:
    """One incident found by the detective.

    incident_type   — "missing" (W broke, record absent from one side)
                      or "reorder" (RGB broke, record moved).
    source_index    — position in stream A (None if absent from A).
    target_index    — position in stream B (None if absent from B).
    record          — the actual bytes.
    checks          — how many binary search steps it took to find this.
    """

    incident_type: str
    source_index: int | None
    target_index: int | None
    record: bytes
    checks: int


class Enkidu:
    """The online matcher. Classifies stream records in real time.

    Every record that arrives on one side gets checked against the other
    side's unmatched pool. Three outcomes:

        Match found, no incident yet → silent resolve (was just unresolved).
        Match found, had a missing incident → reclassify to reorder.
        No match → hold as unresolved for one grace period.

    Call advance_cycle() at each comparison tick. Any record that survived
    a full cycle without a match gets promoted to missing. If its match
    arrives later, the missing incident is reclassified to reorder —
    the record wasn't absent, it was late.

    The classifier returns IncidentReport objects (the SDK's clean type).
    The application layer wraps these into whatever richer format it needs
    for logging, persistence, or UI.
    """

    def __init__(self) -> None:
        # Unmatched pools: payload → (position, cycle_added, incident_or_None)
        # incident is None while the record is still in its grace period.
        # Once promoted, it holds the IncidentReport so we can reclassify.
        self._unmatched_source: dict[bytes, tuple[int, int, IncidentReport | None]] = {}
        self._unmatched_target: dict[bytes, tuple[int, int, IncidentReport | None]] = {}
        self._cycle: int = 0
        self._reclassified: int = 0

    def ingest(self, payload: bytes, position: int, side: str) -> IncidentReport | None:
        """Classify one incoming record.

        Call this immediately after ingesting a record into the Seer.
        Returns an IncidentReport only when a reclassification happens
        (missing → reorder). New missing incidents are created by
        advance_cycle(), not here — because at arrival time we don't
        yet know if the record is truly missing or just late.
        """
        if side == "source":
            own = self._unmatched_source
            other = self._unmatched_target
        elif side == "target":
            own = self._unmatched_target
            other = self._unmatched_source
        else:
            raise ValueError(f"side must be 'source' or 'target', got {side!r}")

        if payload in other:
            # Match found on the opposite side.
            other_pos, _, incident = other.pop(payload)
            src_pos = position if side == "source" else other_pos
            tgt_pos = position if side == "target" else other_pos

            if incident is not None:
                # Was already promoted to missing — reclassify to reorder.
                # The record wasn't absent, it was late.
                reclassified = IncidentReport(
                    incident_type="reorder",
                    source_index=src_pos,
                    target_index=tgt_pos,
                    record=payload,
                    checks=0,
                )
                self._reclassified += 1
                return reclassified
            # Still in grace period — silent resolve, no incident ever created.
            return None
        else:
            # No match yet. Hold as unresolved for one grace period.
            own[payload] = (position, self._cycle, None)
            return None

    def advance_cycle(self) -> list[IncidentReport]:
        """Roll the counter. Promote unresolved records from previous cycles.

        Any record added before the current cycle that still has no match
        has exhausted its grace period. It is now classified as missing.
        Returns the list of new missing IncidentReports created this cycle.

        This is the binary question made explicit: for each unresolved
        record, the grace period has passed and no match arrived. The
        answer is "invertibility of payload" — the record is missing.
        """
        self._cycle += 1
        new_incidents: list[IncidentReport] = []

        for payload, (pos, cycle_added, incident) in list(self._unmatched_source.items()):
            if incident is None and cycle_added < self._cycle:
                inc = IncidentReport(
                    incident_type="missing",
                    source_index=pos,
                    target_index=None,
                    record=payload,
                    checks=0,
                )
                self._unmatched_source[payload] = (pos, cycle_added, inc)
                new_incidents.append(inc)

        for payload, (pos, cycle_added, incident) in list(self._unmatched_target.items()):
            if incident is None and cycle_added < self._cycle:
                inc = IncidentReport(
                    incident_type="missing",
                    source_index=None,
                    target_index=pos,
                    record=payload,
                    checks=0,
                )
                self._unmatched_target[payload] = (pos, cycle_added, inc)
                new_incidents.append(inc)

        return new_incidents

    @property
    def reclassified_count(self) -> int:
        """How many times a missing record was reclassified to reorder."""
        return self._reclassified

    @property
    def unresolved_source(self) -> int:
        """Records on the source side still waiting for a match."""
        return len(self._unmatched_source)

    @property
    def unresolved_target(self) -> int:
        """Records on the target side still waiting for a match."""
        return len(self._unmatched_target)

    @property
    def cycle(self) -> int:
        return self._cycle

    def reset(self) -> None:
        """Clear all state."""
        self._unmatched_source.clear()
        self._unmatched_target.clear()
        self._cycle = 0
        self._reclassified = 0

    def __repr__(self) -> str:
        return (
            f"Enkidu(cycle={self._cycle}, "
            f"unresolved={self.unresolved_source + self.unresolved_target}, "
            f"reclassified={self._reclassified})"
        )


def gilgamesh(
    source_records: list[bytes],
    target_records: list[bytes],
    *,
    max_faults: int = 1000,
) -> list[IncidentReport]:
    """Compose both sequences on S³. Walk source. Classify from the fiber.

    Coheres  — record at expected position. Do nothing.
    Reorder  — record exists in target, different position. Flag. No offset.
    Missing  — record absent from target. Flag. Offset += 1.
    """
    n_src = len(source_records)
    n_tgt = len(target_records)

    if n_src == 0 and n_tgt == 0:
        return []

    # --- compose both on S³, localize ---
    group = closure_rs.sphere()

    def _embed_all(records: list[bytes]) -> np.ndarray:
        if not records:
            return np.empty((0, 4), dtype=np.float64)
        elems = [
            np.asarray(
                closure_rs.closure_element_from_raw_bytes(GROUP_SPEC, [r]),
                dtype=np.float64,
            )
            for r in records
        ]
        return np.ascontiguousarray(np.vstack(elems), dtype=np.float64)

    src_path = closure_rs.GeometricPath.from_elements(group, _embed_all(source_records))
    tgt_path = closure_rs.GeometricPath.from_elements(group, _embed_all(target_records))
    first_fault, checks = src_path.localize_against(tgt_path)

    if first_fault is None:
        return []

    # Lookup: does this payload exist on the other side? (W axis)
    # And where? (RGB axis). Built from the composed sequences.
    src_set = Counter(source_records)
    tgt_set = Counter(target_records)

    # Position lookup: payload → list of positions on each side.
    tgt_positions: dict[bytes, list[int]] = {}
    for idx, rec in enumerate(target_records):
        tgt_positions.setdefault(rec, []).append(idx)
    src_positions: dict[bytes, list[int]] = {}
    for idx, rec in enumerate(source_records):
        src_positions.setdefault(rec, []).append(idx)

    # Track consumed positions so reordered records get skipped.
    consumed_src: set[int] = set()
    consumed_tgt: set[int] = set()

    faults: list[IncidentReport] = []
    i = 0  # source pointer
    j = 0  # target pointer

    while i < n_src and j < n_tgt:
        if len(faults) >= max_faults:
            break

        # Skip positions already consumed by earlier reorder pairings.
        while i < n_src and i in consumed_src:
            i += 1
        while j < n_tgt and j in consumed_tgt:
            j += 1
        if i >= n_src or j >= n_tgt:
            break

        if source_records[i] == target_records[j]:
            # Coheres. Advance both.
            src_set[source_records[i]] -= 1
            tgt_set[target_records[j]] -= 1
            i += 1
            j += 1
            continue

        # Mismatch. Check both payloads against the other side.
        src_in_tgt = tgt_set.get(source_records[i], 0) > 0
        tgt_in_src = src_set.get(target_records[j], 0) > 0

        if not src_in_tgt:
            # src[i] not in target at all. W axis broke. Missing.
            faults.append(IncidentReport(
                "missing", i, None, source_records[i],
                checks if not faults else 0,
            ))
            src_set[source_records[i]] -= 1
            i += 1
        elif not tgt_in_src:
            # tgt[j] not in source at all. W axis broke. Missing.
            faults.append(IncidentReport(
                "missing", None, j, target_records[j],
                checks if not faults else 0,
            ))
            tgt_set[target_records[j]] -= 1
            j += 1
        else:
            # Both exist on the other side. RGB axis broke. Reorder.
            # The geometry tells us which one is displaced:
            # look up actual positions, compare distance.
            src_rec = source_records[i]
            tgt_rec = target_records[j]

            # Where does src[i] actually live in target?
            actual_tgt = None
            for pos in tgt_positions[src_rec]:
                if pos not in consumed_tgt:
                    actual_tgt = pos
                    break

            # Where does tgt[j] actually live in source?
            actual_src = None
            for pos in src_positions[tgt_rec]:
                if pos not in consumed_src:
                    actual_src = pos
                    break

            src_dist = abs(actual_tgt - j) if actual_tgt is not None else 0
            tgt_dist = abs(actual_src - i) if actual_src is not None else 0

            if src_dist > tgt_dist:
                # src[i] is further from where it should be — it's displaced.
                faults.append(IncidentReport(
                    "reorder", i, actual_tgt, src_rec,
                    checks if not faults else 0,
                ))
                src_set[src_rec] -= 1
                tgt_set[src_rec] -= 1
                consumed_tgt.add(actual_tgt)
                i += 1
            elif tgt_dist > src_dist:
                # tgt[j] is further — it's displaced.
                faults.append(IncidentReport(
                    "reorder", actual_src, j, tgt_rec,
                    checks if not faults else 0,
                ))
                src_set[tgt_rec] -= 1
                tgt_set[tgt_rec] -= 1
                consumed_src.add(actual_src)
                j += 1
            else:
                # Equal distance — swap. Both displaced. Flag both.
                faults.append(IncidentReport(
                    "reorder", i, actual_tgt, src_rec,
                    checks if not faults else 0,
                ))
                faults.append(IncidentReport(
                    "reorder", actual_src, j, tgt_rec, 0,
                ))
                src_set[src_rec] -= 1
                tgt_set[src_rec] -= 1
                src_set[tgt_rec] -= 1
                tgt_set[tgt_rec] -= 1
                consumed_tgt.add(actual_tgt)
                consumed_src.add(actual_src)
                i += 1
                j += 1

    # Remaining source records are missing from target.
    while i < n_src and len(faults) < max_faults:
        if i not in consumed_src:
            faults.append(IncidentReport(
                "missing", i, None, source_records[i], 0,
            ))
        i += 1

    # Remaining target records are missing from source.
    while j < n_tgt and len(faults) < max_faults:
        if j not in consumed_tgt:
            faults.append(IncidentReport(
                "missing", None, j, target_records[j], 0,
            ))
        j += 1

    return faults[:max_faults]
