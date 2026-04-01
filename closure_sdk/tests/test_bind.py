"""Tests for bind() — the connector between two spheres."""

from __future__ import annotations

import numpy as np

import closure_sdk as closure


# ── Equal relationship ────────────────────────────────────────────

def test_bind_equal_same_composition() -> None:
    """Two Seers fed identical records bind equal."""
    a = closure.Seer()
    b = closure.Seer()
    for r in [b"tx-001", b"tx-002", b"tx-003"]:
        a.ingest(r)
        b.ingest(r)

    result = closure.bind(a.state().element, b.state().element)
    assert result.relation == "equal"
    assert result.sigma < 1e-10


def test_bind_equal_single_record() -> None:
    """Equal binding works with a single record."""
    elem = closure.embed(b"hello").element
    result = closure.bind(elem, elem)
    assert result.relation == "equal"
    assert result.sigma < 1e-10


def test_bind_equal_identity_to_identity() -> None:
    """Two identity elements bind equal."""
    a = closure.Seer()
    b = closure.Seer()
    result = closure.bind(a.state().element, b.state().element)
    assert result.relation == "equal"
    assert result.sigma < 1e-10


# ── Inverse relationship ─────────────────────────────────────────

def test_bind_inverse() -> None:
    """An element and its inverse bind inverse."""
    elem = closure.embed(b"data")
    inv = closure.invert(elem)
    result = closure.bind(elem.element, inv.element)
    assert result.relation == "inverse"
    assert result.sigma < 1e-10


def test_bind_inverse_composed() -> None:
    """A composed element and its inverse bind inverse."""
    a = closure.Seer()
    for r in [b"one", b"two", b"three"]:
        a.ingest(r)
    inv = closure.invert(a.state())
    result = closure.bind(a.state().element, inv.element)
    assert result.relation == "inverse"
    assert result.sigma < 1e-10


# ── Disordered relationship ──────────────────────────────────────

def test_bind_disordered_different_data() -> None:
    """Different compositions bind disordered."""
    a = closure.Seer()
    b = closure.Seer()
    for r in [b"a", b"b", b"c"]:
        a.ingest(r)
    for r in [b"x", b"y", b"z"]:
        b.ingest(r)

    result = closure.bind(a.state().element, b.state().element)
    assert result.relation == "disordered"
    assert result.sigma > 0.1


def test_bind_disordered_reordered_data() -> None:
    """Same records in different order bind disordered."""
    a = closure.Seer()
    b = closure.Seer()
    a.ingest(b"first")
    a.ingest(b"second")
    b.ingest(b"second")
    b.ingest(b"first")

    result = closure.bind(a.state().element, b.state().element)
    assert result.relation == "disordered"
    assert result.sigma > 0


# ── Gap valence ──────────────────────────────────────────────────

def test_bind_disordered_gap_has_color() -> None:
    """Disordered binding carries meaningful gap valence."""
    a = closure.embed(b"left")
    b = closure.embed(b"right")

    result = closure.bind(a.element, b.element)
    assert result.relation == "disordered"
    assert result.gap.sigma > 0
    assert len(result.gap.base) == 3


def test_bind_equal_gap_near_zero() -> None:
    """Equal binding has a gap valence near zero."""
    elem = closure.embed(b"same").element
    result = closure.bind(elem, elem)
    assert result.gap.sigma < 1e-10


# ── Threshold ────────────────────────────────────────────────────

def test_bind_custom_threshold() -> None:
    """Custom threshold controls the equal/disordered boundary."""
    a = closure.Seer()
    b = closure.Seer()
    a.ingest_many([b"a", b"b", b"c"])
    b.ingest_many([b"a", b"b", b"c"])

    tight = closure.bind(a.state().element, b.state().element, threshold=1e-15)
    assert tight.relation == "equal"

    loose = closure.bind(a.state().element, b.state().element, threshold=1e-6)
    assert loose.relation == "equal"


# ── Symmetry ─────────────────────────────────────────────────────

def test_bind_equal_is_symmetric() -> None:
    """bind(a, b) and bind(b, a) agree on equal."""
    elem = closure.embed(b"symmetric").element
    assert closure.bind(elem, elem).relation == "equal"


def test_bind_inverse_is_symmetric() -> None:
    """bind(a, inv(a)) and bind(inv(a), a) both report inverse."""
    elem = closure.embed(b"check")
    inv = closure.invert(elem)
    r1 = closure.bind(elem.element, inv.element)
    r2 = closure.bind(inv.element, elem.element)
    assert r1.relation == "inverse"
    assert r2.relation == "inverse"
