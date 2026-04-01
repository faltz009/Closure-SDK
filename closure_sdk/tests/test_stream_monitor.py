"""Tests for streaming raw-byte monitor API."""

from __future__ import annotations

import numpy as np
import pytest

import closure_rs


def _records(n: int, seed: int = 2026) -> list[bytes]:
    rng = np.random.default_rng(seed)
    return [bytes(rng.bytes(96)) for _ in range(n)]


@pytest.mark.parametrize(
    ("group_name", "group_factory", "tol"),
    [
        ("Circle", closure_rs.circle, 1e-12),
        ("Sphere", closure_rs.sphere, 1e-9),
        ("Torus(5)", lambda: closure_rs.torus(5), 1e-12),
        (
            "Hybrid(Torus(5), Sphere)",
            lambda: closure_rs.hybrid(closure_rs.torus(5), closure_rs.sphere()),
            1e-9,
        ),
    ],
    ids=["Circle", "Sphere", "Torus(5)", "Hybrid(Torus(5),Sphere)"],
)
def test_stream_monitor_matches_batch_closure(group_name, group_factory, tol):
    records = _records(2_000)
    mon = closure_rs.StreamMonitor(group_name)
    for r in records:
        mon.ingest(r)

    ce_stream = np.asarray(mon.closure_element(), dtype=np.float64)
    ce_batch = np.asarray(closure_rs.closure_element_from_raw_bytes(group_name, records), dtype=np.float64)
    group = group_factory()
    rel = group.compose(group.inverse(ce_stream), ce_batch)
    d = group.distance_from_identity(rel)
    assert d < tol


def test_stream_monitor_ingest_many_matches_loop():
    records = _records(3_000)
    a = closure_rs.StreamMonitor("Circle")
    b = closure_rs.StreamMonitor("Circle")

    a.ingest_many(records)
    for r in records:
        b.ingest(r)

    assert a.compare_against(b) < 1e-12
    assert len(a) == len(records)


def test_stream_monitor_sigma_matches_group_distance():
    records = _records(1_500)
    mon = closure_rs.StreamMonitor("Sphere")
    mon.ingest_many(records)
    ce = np.asarray(mon.closure_element())
    expected = closure_rs.sphere().distance_from_identity(ce)
    assert abs(mon.sigma() - expected) < 1e-12


def test_stream_monitor_reset():
    mon = closure_rs.StreamMonitor("Circle")
    mon.ingest_many(_records(100))
    assert len(mon) == 100
    assert mon.sigma() >= 0.0
    mon.reset()
    assert len(mon) == 0
    assert mon.sigma() < 1e-15


def test_stream_monitor_compare_group_mismatch_raises():
    a = closure_rs.StreamMonitor("Circle")
    b = closure_rs.StreamMonitor("Sphere")
    with pytest.raises(ValueError, match="group mismatch"):
        a.compare_against(b)


def test_stream_monitor_invalid_group_raises():
    with pytest.raises(ValueError, match="Unknown group"):
        closure_rs.StreamMonitor("SE3")
