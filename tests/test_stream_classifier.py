"""Tests for StreamClassifier — the online matcher in canon.py.

The classifier answers one binary question per cycle for each unresolved
record: is this an invertibility of payload (missing) or position (reorder)?

Three scenarios at each tick:
    1. Both sides receive the same record → coherent, no incident.
    2. One side receives a record that arrives late → reorder.
    3. One side receives a record that never arrives → missing.

The grace period turns the oracle problem (we can't see the future)
into a bounded binary decision.
"""

from __future__ import annotations

import closure_sdk as closure
from closure_sdk import StreamClassifier, IncidentReport


# ── Basic matching ───────────────────────────────────────────────────


def test_coherent_records_produce_no_incidents() -> None:
    """Same record on both sides within the same cycle → silent match."""
    c = StreamClassifier()

    # Source gets record, then target gets same record
    assert c.ingest(b"a", 0, "source") is None
    assert c.ingest(b"a", 0, "target") is None

    # Advance cycle — nothing to promote
    incidents = c.advance_cycle()
    assert incidents == []
    assert c.unresolved_source == 0
    assert c.unresolved_target == 0


def test_multiple_coherent_records() -> None:
    """A full coherent stream produces zero incidents."""
    c = StreamClassifier()

    for i in range(20):
        rec = f"record-{i}".encode()
        c.ingest(rec, i, "source")
        c.ingest(rec, i, "target")

    incidents = c.advance_cycle()
    assert incidents == []


# ── Missing records ──────────────────────────────────────────────────


def test_missing_from_target_promoted_after_grace() -> None:
    """Record on source with no match → promoted to missing after one cycle."""
    c = StreamClassifier()

    c.ingest(b"only-source", 0, "source")

    # Still in grace period — advance once
    incidents = c.advance_cycle()
    assert len(incidents) == 1

    inc = incidents[0]
    assert inc.incident_type == "missing"
    assert inc.source_index == 0
    assert inc.target_index is None
    assert inc.record == b"only-source"


def test_missing_from_source_promoted_after_grace() -> None:
    """Record on target with no match → promoted to missing (absent from source)."""
    c = StreamClassifier()

    c.ingest(b"only-target", 5, "target")
    incidents = c.advance_cycle()

    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.incident_type == "missing"
    assert inc.source_index is None
    assert inc.target_index == 5


def test_grace_period_holds_during_same_cycle() -> None:
    """Records added in the current cycle are NOT promoted yet."""
    c = StreamClassifier()

    # Advance to cycle 1 first (empty)
    c.advance_cycle()

    # Add a record in cycle 1
    c.ingest(b"new", 0, "source")

    # Advance to cycle 2 — now it should promote (added in cycle 1 < cycle 2)
    incidents = c.advance_cycle()
    assert len(incidents) == 1
    assert incidents[0].record == b"new"


# ── Late arrival → reclassification ─────────────────────────────────


def test_late_arrival_reclassifies_to_reorder() -> None:
    """Record arrives on source, promoted to missing, then arrives on target
    → reclassified from missing to reorder (content_mismatch)."""
    c = StreamClassifier()

    # Source gets record
    c.ingest(b"late", 0, "source")

    # Grace period expires → promoted to missing
    missing = c.advance_cycle()
    assert len(missing) == 1
    assert missing[0].incident_type == "missing"

    # Now the same record arrives on target (late)
    result = c.ingest(b"late", 3, "target")

    # Should be reclassified to reorder
    assert result is not None
    assert result.incident_type == "content_mismatch"
    assert result.source_index == 0
    assert result.target_index == 3
    assert c.reclassified_count == 1


def test_silent_resolve_within_grace_period() -> None:
    """Record arrives on both sides within the grace period → no incident at all."""
    c = StreamClassifier()

    # Source gets record
    c.ingest(b"quick", 0, "source")
    assert c.unresolved_source == 1

    # Target gets same record before cycle advances
    result = c.ingest(b"quick", 0, "target")
    assert result is None  # silent resolve, no incident
    assert c.unresolved_source == 0

    # Nothing to promote
    incidents = c.advance_cycle()
    assert incidents == []
    assert c.reclassified_count == 0


# ── Chained errors ───────────────────────────────────────────────────


def test_multiple_missing_records_chain() -> None:
    """Two missing records in a row — each gets the same treatment."""
    c = StreamClassifier()

    c.ingest(b"first", 0, "source")
    c.ingest(b"second", 1, "source")

    incidents = c.advance_cycle()
    assert len(incidents) == 2
    assert all(i.incident_type == "missing" for i in incidents)
    records = {i.record for i in incidents}
    assert records == {b"first", b"second"}


def test_mixed_missing_and_late() -> None:
    """One record stays missing, another arrives late → one missing, one reorder."""
    c = StreamClassifier()

    c.ingest(b"will-stay-missing", 0, "source")
    c.ingest(b"will-arrive-late", 1, "source")

    # Both promoted to missing
    missing = c.advance_cycle()
    assert len(missing) == 2

    # The late one arrives on target
    result = c.ingest(b"will-arrive-late", 5, "target")
    assert result is not None
    assert result.incident_type == "content_mismatch"

    # The other one stays missing — no further reclassification
    assert c.reclassified_count == 1
    assert c.unresolved_source == 1  # "will-stay-missing" still there


# ── The recursive property ───────────────────────────────────────────


def test_error_chain_same_solution_each_step() -> None:
    """After a misalignment, the next step is the same case.
    Either a late record arrives (reorder) or another missing arrives.
    The grace period handles both identically."""
    c = StreamClassifier()

    # Step 1: source gets A, target doesn't
    c.ingest(b"A", 0, "source")
    c.advance_cycle()  # A promoted to missing

    # Step 2: source gets B, target doesn't (another missing — same case)
    c.ingest(b"B", 1, "source")
    more_missing = c.advance_cycle()  # B promoted to missing
    assert len(more_missing) == 1
    assert more_missing[0].record == b"B"

    # Step 3: target finally gets A (late arrival → reorder)
    result_a = c.ingest(b"A", 0, "target")
    assert result_a is not None
    assert result_a.incident_type == "content_mismatch"

    # Step 4: target finally gets B (late arrival → reorder)
    result_b = c.ingest(b"B", 1, "target")
    assert result_b is not None
    assert result_b.incident_type == "content_mismatch"

    assert c.reclassified_count == 2
    assert c.unresolved_source == 0
    assert c.unresolved_target == 0


# ── Properties ───────────────────────────────────────────────────────


def test_reset_clears_all_state() -> None:
    c = StreamClassifier()
    c.ingest(b"a", 0, "source")
    c.advance_cycle()
    assert c.unresolved_source == 1

    c.reset()
    assert c.unresolved_source == 0
    assert c.unresolved_target == 0
    assert c.reclassified_count == 0
    assert c.cycle == 0


def test_repr() -> None:
    c = StreamClassifier()
    c.ingest(b"a", 0, "source")
    r = repr(c)
    assert "StreamClassifier" in r
    assert "unresolved=1" in r


# ── Exchange symmetry (documented invariant from section 5) ──────────


def test_exchange_symmetry_documented() -> None:
    """Identical bytes are interchangeable in the algebra.

    Two copies of the same record are the same group element.
    The algebra cannot distinguish them — if copies must be
    distinguished, the adapter includes a signifier in the payload.
    This is a property of the geometry, not a limitation.
    """
    # Static mode: localize_all treats identical records as interchangeable
    src = [b"a", b"dup", b"dup", b"b"]
    tgt = [b"a", b"dup", b"dup", b"b"]
    faults = closure.localize_all(src, tgt)
    assert faults == []  # identical sequences → no faults

    # Swapping identical records is invisible to the algebra
    tgt_swapped = [b"a", b"dup", b"dup", b"b"]  # same bytes, same result
    faults_swapped = closure.localize_all(src, tgt_swapped)
    assert faults_swapped == []

    # Stream mode: identical payloads match by content, not position
    c = StreamClassifier()
    c.ingest(b"dup", 0, "source")
    result = c.ingest(b"dup", 5, "target")  # different position, same bytes
    assert result is None  # silent resolve — they're the same element
    assert c.unresolved_source == 0

    # Embedding: same bytes always produce the same point
    e1 = closure.embed(b"identical")
    e2 = closure.embed(b"identical")
    assert closure.compare(e1, e2).coherent


def test_invalid_side_raises_value_error() -> None:
    c = StreamClassifier()
    try:
        c.ingest(b"x", 0, "soruce")
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "soruce" in str(e)
