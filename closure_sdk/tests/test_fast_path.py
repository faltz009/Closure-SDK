"""Tests for closure-element fast paths (no intermediate path storage)."""

from __future__ import annotations

import numpy as np
import pytest

import closure_rs


def _distance(group, a: np.ndarray, b: np.ndarray) -> float:
    rel = group.compose(group.inverse(a), b)
    return float(group.distance_from_identity(rel))


def _records(n: int, seed: int = 123) -> list[bytes]:
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
def test_closure_element_from_raw_bytes_matches_full_path(group_name, group_factory, tol):
    records = _records(5_000)
    full = closure_rs.path_from_raw_bytes(group_name, records)
    fast = np.asarray(closure_rs.closure_element_from_raw_bytes(group_name, records))
    ref = np.asarray(full.closure_element())
    d = _distance(group_factory(), ref, fast)
    assert d < tol, f"fast path drifted from full path: d={d}"


def test_closure_element_from_raw_bytes_rejects_unsupported_group():
    with pytest.raises(ValueError, match="Unknown group"):
        closure_rs.closure_element_from_raw_bytes("SE3", [b"abc"])


@pytest.mark.parametrize(
    ("group", "dim", "tol"),
    [
        (closure_rs.circle(), 1, 1e-12),
        (closure_rs.sphere(), 4, 1e-9),
        (closure_rs.torus(5), 5, 1e-12),
    ],
    ids=["Circle", "Sphere", "Torus(5)"],
)
def test_closure_element_from_elements_matches_full_path(group, dim, tol):
    rng = np.random.default_rng(42)
    elems = rng.standard_normal((4_000, dim)).astype(np.float64)
    if dim == 4:
        norms = np.linalg.norm(elems, axis=1, keepdims=True)
        elems = elems / np.where(norms == 0, 1.0, norms)
    full = closure_rs.GeometricPath.from_elements(group, elems)
    fast = np.asarray(closure_rs.closure_element_from_elements(group, elems))
    ref = np.asarray(full.closure_element())
    d = _distance(group, ref, fast)
    assert d < tol, f"fast path drifted from full path: d={d}"


def test_closure_element_from_elements_requires_contiguous_input():
    group = closure_rs.torus(3)
    data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    non_contig = data[:, ::-1]
    assert not non_contig.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="contiguous"):
        closure_rs.closure_element_from_elements(group, non_contig)


def test_closure_element_from_elements_validates_dimension():
    group = closure_rs.circle()
    bad = np.array([[0.1, 0.2]], dtype=np.float64)
    with pytest.raises(ValueError, match="elements has dim"):
        closure_rs.closure_element_from_elements(group, bad)
