"""Aggressive tests for gilgamesh — the static multi-fault localizer.

Covers: single faults, multi-fault, mixed types, duplicates, empty
streams, length mismatches, large sequences, and edge positions.
"""

from __future__ import annotations

import closure_sdk as closure


# ── Coherent (no faults) ─────────────────────────────────────────────


def test_identical_sequences_no_faults() -> None:
    records = [b"a", b"b", b"c", b"d", b"e"]
    faults = closure.gilgamesh(records, records)
    assert faults == []


def test_empty_sequences_no_faults() -> None:
    faults = closure.gilgamesh([], [])
    assert faults == []


def test_single_record_coherent() -> None:
    faults = closure.gilgamesh([b"x"], [b"x"])
    assert faults == []


# ── Single missing record ────────────────────────────────────────────


def test_missing_from_target_at_start() -> None:
    src = [b"a", b"b", b"c"]
    tgt = [b"b", b"c"]
    faults = closure.gilgamesh(src, tgt)
    missing = [f for f in faults if f.incident_type == "missing"]
    assert len(missing) == 1
    assert missing[0].record == b"a"
    assert missing[0].source_index == 0
    assert missing[0].target_index is None


def test_missing_from_target_at_end() -> None:
    src = [b"a", b"b", b"c"]
    tgt = [b"a", b"b"]
    faults = closure.gilgamesh(src, tgt)
    missing = [f for f in faults if f.incident_type == "missing"]
    assert len(missing) == 1
    assert missing[0].record == b"c"
    assert missing[0].source_index == 2


def test_missing_from_target_in_middle() -> None:
    src = [b"a", b"b", b"c", b"d"]
    tgt = [b"a", b"c", b"d"]
    faults = closure.gilgamesh(src, tgt)
    missing = [f for f in faults if f.incident_type == "missing"]
    assert len(missing) >= 1
    assert any(f.record == b"b" for f in missing)


def test_missing_from_source() -> None:
    src = [b"a", b"c"]
    tgt = [b"a", b"b", b"c"]
    faults = closure.gilgamesh(src, tgt)
    missing = [f for f in faults if f.incident_type == "missing"]
    assert len(missing) >= 1
    assert any(f.record == b"b" and f.source_index is None for f in missing)


# ── Single reorder ───────────────────────────────────────────────────


def test_simple_swap() -> None:
    src = [b"a", b"b", b"c"]
    tgt = [b"a", b"c", b"b"]
    faults = closure.gilgamesh(src, tgt)
    reorders = [f for f in faults if f.incident_type == "reorder"]
    assert len(reorders) >= 1
    # At least one reorder involves b or c
    reorder_records = {f.record for f in reorders}
    assert reorder_records & {b"b", b"c"}


def test_full_reversal() -> None:
    src = [b"a", b"b", b"c", b"d"]
    tgt = [b"d", b"c", b"b", b"a"]
    faults = closure.gilgamesh(src, tgt)
    reorders = [f for f in faults if f.incident_type == "reorder"]
    # All records exist on both sides, just reordered
    assert len(reorders) >= 1
    missing = [f for f in faults if f.incident_type == "missing"]
    assert len(missing) == 0


def test_single_element_moved_to_end() -> None:
    src = [b"a", b"b", b"c", b"d"]
    tgt = [b"b", b"c", b"d", b"a"]
    faults = closure.gilgamesh(src, tgt)
    reorders = [f for f in faults if f.incident_type == "reorder"]
    assert len(reorders) >= 1
    missing = [f for f in faults if f.incident_type == "missing"]
    assert len(missing) == 0


# ── Mixed: missing + reorder ─────────────────────────────────────────


def test_missing_and_reorder_together() -> None:
    src = [b"a", b"b", b"c", b"d"]
    tgt = [b"a", b"d", b"c"]  # b missing, c and d swapped
    faults = closure.gilgamesh(src, tgt)
    missing = [f for f in faults if f.incident_type == "missing"]
    reorders = [f for f in faults if f.incident_type == "reorder"]
    # b should be missing
    assert any(f.record == b"b" for f in missing)
    # c or d should be reordered
    assert len(reorders) >= 1


def test_multiple_missing() -> None:
    src = [b"a", b"b", b"c", b"d", b"e"]
    tgt = [b"a", b"d"]  # b, c, e missing
    faults = closure.gilgamesh(src, tgt)
    missing = [f for f in faults if f.incident_type == "missing"]
    missing_records = {f.record for f in missing}
    assert b"b" in missing_records
    assert b"c" in missing_records
    assert b"e" in missing_records


def test_target_has_extra_records() -> None:
    src = [b"a", b"b"]
    tgt = [b"a", b"x", b"b", b"y"]
    faults = closure.gilgamesh(src, tgt)
    missing = [f for f in faults if f.incident_type == "missing"]
    missing_from_src = [f for f in missing if f.source_index is None]
    missing_records = {f.record for f in missing_from_src}
    assert b"x" in missing_records
    assert b"y" in missing_records


# ── Duplicates ────────────────────────────────────────────────────────


def test_identical_duplicates_no_faults() -> None:
    """Same duplicates on both sides → no faults."""
    src = [b"a", b"dup", b"dup", b"b"]
    tgt = [b"a", b"dup", b"dup", b"b"]
    faults = closure.gilgamesh(src, tgt)
    assert faults == []


def test_extra_duplicate_on_source() -> None:
    """Source has one more copy of a record."""
    src = [b"a", b"dup", b"dup", b"b"]
    tgt = [b"a", b"dup", b"b"]
    faults = closure.gilgamesh(src, tgt)
    missing = [f for f in faults if f.incident_type == "missing"]
    assert len(missing) >= 1
    assert any(f.record == b"dup" for f in missing)


def test_extra_duplicate_on_target() -> None:
    """Target has one more copy of a record."""
    src = [b"a", b"dup", b"b"]
    tgt = [b"a", b"dup", b"dup", b"b"]
    faults = closure.gilgamesh(src, tgt)
    missing = [f for f in faults if f.incident_type == "missing"]
    assert len(missing) >= 1
    assert any(
        f.record == b"dup" and f.source_index is None for f in missing
    )


# ── Length mismatches ─────────────────────────────────────────────────


def test_source_empty_target_has_records() -> None:
    faults = closure.gilgamesh([], [b"a", b"b"])
    assert len(faults) == 2
    assert all(f.incident_type == "missing" for f in faults)
    assert all(f.source_index is None for f in faults)


def test_target_empty_source_has_records() -> None:
    faults = closure.gilgamesh([b"a", b"b"], [])
    assert len(faults) == 2
    assert all(f.incident_type == "missing" for f in faults)
    assert all(f.target_index is None for f in faults)


def test_completely_different_sequences() -> None:
    src = [b"a", b"b", b"c"]
    tgt = [b"x", b"y", b"z"]
    faults = closure.gilgamesh(src, tgt)
    # All 6 records are missing (3 from each side)
    assert len(faults) == 6
    assert all(f.incident_type == "missing" for f in faults)


# ── Larger sequences ──────────────────────────────────────────────────


def test_large_coherent() -> None:
    """1000 records, identical → no faults."""
    records = [f"record-{i}".encode() for i in range(1000)]
    faults = closure.gilgamesh(records, records)
    assert faults == []


def test_large_single_missing() -> None:
    """1000 records, one dropped from target."""
    records = [f"record-{i}".encode() for i in range(1000)]
    target = records[:500] + records[501:]  # drop record-500
    faults = closure.gilgamesh(records, target)
    missing = [f for f in faults if f.incident_type == "missing"]
    assert any(f.record == b"record-500" for f in missing)


def test_large_single_reorder() -> None:
    """1000 records, one moved to a different position."""
    records = [f"record-{i}".encode() for i in range(1000)]
    target = list(records)
    # Move record-100 to position 900
    moved = target.pop(100)
    target.insert(900, moved)
    faults = closure.gilgamesh(records, target)
    reorders = [f for f in faults if f.incident_type == "reorder"]
    assert len(reorders) >= 1


def test_large_multi_fault() -> None:
    """1000 records, 5 dropped from target."""
    records = [f"record-{i}".encode() for i in range(1000)]
    drop_indices = {100, 300, 500, 700, 999}
    target = [r for i, r in enumerate(records) if i not in drop_indices]
    faults = closure.gilgamesh(records, target)
    missing = [f for f in faults if f.incident_type == "missing"]
    missing_records = {f.record for f in missing}
    for idx in drop_indices:
        assert f"record-{idx}".encode() in missing_records


# ── Edge: positions and indices ───────────────────────────────────────


def test_missing_reports_correct_source_index() -> None:
    src = [b"a", b"b", b"c"]
    tgt = [b"a", b"c"]
    faults = closure.gilgamesh(src, tgt)
    missing = [f for f in faults if f.record == b"b"]
    assert len(missing) == 1
    assert missing[0].source_index == 1
    assert missing[0].target_index is None


def test_reorder_reports_both_indices() -> None:
    src = [b"a", b"b", b"c"]
    tgt = [b"c", b"b", b"a"]
    faults = closure.gilgamesh(src, tgt)
    reorders = [f for f in faults if f.incident_type == "reorder"]
    # b is at position 1 in both sides — it didn't move, only a and c swapped
    assert len(reorders) == 2
    for r in reorders:
        assert r.source_index is not None
        assert r.target_index is not None


# ── max_faults cap ────────────────────────────────────────────────────


def test_max_faults_caps_output() -> None:
    src = [f"s-{i}".encode() for i in range(20)]
    tgt = [f"t-{i}".encode() for i in range(20)]
    faults = closure.gilgamesh(src, tgt, max_faults=5)
    assert len(faults) <= 5
