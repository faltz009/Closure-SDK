"""The canon — finds what's true, where it broke, and how.

Two composition modes, same classification:

    Static mode (localize_all):
        Both streams are complete. Compose both on S³, binary-search for
        the first divergence, classify it, remove the pair, recompose,
        repeat. Because we have the full picture, we know instantly
        whether a record is missing or reordered.
        O(k · n · log n) where k is the number of incidents.

    Stream mode (StreamClassifier):
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

from collections import deque
from dataclasses import dataclass

from .lenses import Oracle
from .lenses import GROUP_SPEC


@dataclass
class RetainedBlock:
    """One block of raw records, tagged with its block number."""

    block_number: int
    records: list[bytes]


class RetentionWindow:
    """A bounded rolling buffer of raw records. When a Monitor detects
    drift, this is where the raw bytes are stored so localize_all can
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
                      or "content_mismatch" (RGB broke, record moved).
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


class StreamClassifier:
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
                    incident_type="content_mismatch",
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
            f"StreamClassifier(cycle={self._cycle}, "
            f"unresolved={self.unresolved_source + self.unresolved_target}, "
            f"reclassified={self._reclassified})"
        )


def localize_all(
    source_records: list[bytes],
    target_records: list[bytes],
    *,
    max_faults: int = 1000,
) -> list[IncidentReport]:
    """The recursive technique. Takes two complete lists of raw records
    and finds every point where they disagree. Compose both on S³,
    binary-search for first divergence, classify (missing or reorder),
    remove the pair, recompose, repeat until the streams match.
    Returns one IncidentReport per incident found.
    """
    src = [(i, rec) for i, rec in enumerate(source_records)]
    tgt = [(i, rec) for i, rec in enumerate(target_records)]

    faults: list[IncidentReport] = []

    for _ in range(max_faults):
        src_bytes = [b for _, b in src]
        tgt_bytes = [b for _, b in tgt]

        if not src_bytes and not tgt_bytes:
            break
        if not src_bytes:
            for orig_idx, rec in tgt:
                faults.append(IncidentReport("missing", None, orig_idx, rec, 0))
            break
        if not tgt_bytes:
            for orig_idx, rec in src:
                faults.append(IncidentReport("missing", orig_idx, None, rec, 0))
            break

        src_trace = Oracle.from_records(src_bytes)
        tgt_trace = Oracle.from_records(tgt_bytes)

        result = src_trace.localize_against(tgt_trace)

        if result.index is None:
            break

        idx = result.index
        src_rec = src_bytes[idx] if idx < len(src_bytes) else None
        tgt_rec = tgt_bytes[idx] if idx < len(tgt_bytes) else None

        tgt_set = set(tgt_bytes)
        src_set = set(src_bytes)

        if src_rec is not None and src_rec not in tgt_set:
            faults.append(IncidentReport("missing", src[idx][0], None, src_rec, result.checks))
            src.pop(idx)
        elif tgt_rec is not None and tgt_rec not in src_set:
            faults.append(IncidentReport("missing", None, tgt[idx][0], tgt_rec, result.checks))
            tgt.pop(idx)
        else:
            tgt_pos = next(i for i, (_, b) in enumerate(tgt) if b == src_rec)
            faults.append(IncidentReport(
                "content_mismatch", src[idx][0], tgt[tgt_pos][0], src_rec, result.checks,
            ))
            src.pop(idx)
            tgt.pop(tgt_pos)

    return faults
