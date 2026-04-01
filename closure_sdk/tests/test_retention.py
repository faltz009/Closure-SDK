"""Tests for closure_sdk.retention — RetentionWindow and gilgamesh."""

from __future__ import annotations

import closure_sdk as closure


def test_retention_window_append_and_flatten() -> None:
    window = closure.RetentionWindow(maxlen=10)
    window.append(100, [b"a", b"b"])
    window.append(101, [b"c"])

    assert len(window) == 2
    assert window.total_records == 3
    assert window.flatten() == [b"a", b"b", b"c"]


def test_retention_window_block_map() -> None:
    window = closure.RetentionWindow(maxlen=10)
    window.append(100, [b"a", b"b"])
    window.append(101, [b"c", b"d", b"e"])

    bmap = window.block_map()
    assert bmap == [(100, 2), (101, 3)]


def test_retention_window_bounded() -> None:
    window = closure.RetentionWindow(maxlen=3)
    for i in range(5):
        window.append(i, [bytes([i])])

    assert len(window) == 3
    assert window.block_map() == [(2, 1), (3, 1), (4, 1)]


def test_retention_window_repr() -> None:
    window = closure.RetentionWindow(maxlen=10)
    window.append(1, [b"a", b"b"])
    assert "blocks=1" in repr(window)
    assert "records=2" in repr(window)


def test_gilgamesh_finds_missing() -> None:
    src = [b"a", b"b", b"c", b"d"]
    tgt = [b"a", b"c", b"d"]  # b"b" missing

    faults = closure.gilgamesh(src, tgt)
    assert len(faults) >= 1
    missing = [f for f in faults if f.incident_type == "missing"]
    assert len(missing) >= 1
    assert missing[0].record == b"b"


def test_gilgamesh_finds_reorder() -> None:
    src = [b"a", b"b", b"c"]
    tgt = [b"a", b"c", b"b"]  # b and c swapped

    faults = closure.gilgamesh(src, tgt)
    assert len(faults) >= 1
    reorders = [f for f in faults if f.incident_type == "reorder"]
    assert len(reorders) >= 1


def test_gilgamesh_coherent_returns_empty() -> None:
    records = [b"a", b"b", b"c"]
    faults = closure.gilgamesh(records, records)
    assert faults == []


def test_incident_result_fields() -> None:
    result = closure.IncidentReport(
        incident_type="missing",
        source_index=3,
        target_index=None,
        record=b"test",
        checks=5,
    )
    assert result.incident_type == "missing"
    assert result.source_index == 3
    assert result.target_index is None
    assert result.record == b"test"
    assert result.checks == 5
