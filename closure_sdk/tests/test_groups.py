"""Group axiom verification for all Lie group implementations (Rust backend)."""

from __future__ import annotations

import numpy as np
import pytest

import closure_rs


GROUPS = [
    closure_rs.circle(),
    closure_rs.sphere(),
    closure_rs.torus(3),
    closure_rs.torus(10),
    closure_rs.hybrid(closure_rs.torus(5), closure_rs.sphere()),
]

GROUP_IDS = ["Circle", "Sphere", "Torus(3)", "Torus(10)", "Hybrid"]


@pytest.fixture(params=zip(GROUPS, GROUP_IDS), ids=lambda x: x[1])
def group(request):
    return request.param[0]


def _allclose(a: np.ndarray, b: np.ndarray, atol: float = 1e-10) -> bool:
    """Check approximate equality, handling Circle wrapping."""
    diff = np.asarray(a) - np.asarray(b)
    wrapped = np.abs(diff) % (2 * np.pi)
    wrapped = np.minimum(wrapped, 2 * np.pi - wrapped)
    return bool(np.all(wrapped < atol))


class TestGroupAxioms:
    """Verify the four group axioms for every implementation."""

    def test_left_identity(self, group):
        for seed in range(100):
            a = np.asarray(group.random(seed=seed))
            result = np.asarray(group.compose(group.identity(), a))
            assert _allclose(result, a), f"e*a != a: {result} vs {a}"

    def test_right_identity(self, group):
        for seed in range(100, 200):
            a = np.asarray(group.random(seed=seed))
            result = np.asarray(group.compose(a, group.identity()))
            assert _allclose(result, a), f"a*e != a: {result} vs {a}"

    def test_inverse(self, group):
        for seed in range(200, 300):
            a = np.asarray(group.random(seed=seed))
            prod = np.asarray(group.compose(a, group.inverse(a)))
            d = group.distance_from_identity(prod)
            assert d < 1e-10, f"a*a^-1 not identity: d={d}"

    def test_associativity(self, group):
        for seed in range(0, 150, 3):
            a = np.asarray(group.random(seed=seed))
            b = np.asarray(group.random(seed=seed + 1))
            c = np.asarray(group.random(seed=seed + 2))
            lhs = np.asarray(group.compose(group.compose(a, b), c))
            rhs = np.asarray(group.compose(a, group.compose(b, c)))
            assert _allclose(lhs, rhs, atol=1e-9), "(a*b)*c != a*(b*c)"

    def test_distance_nonnegative(self, group):
        for seed in range(400, 500):
            a = np.asarray(group.random(seed=seed))
            assert group.distance_from_identity(a) >= 0

    def test_distance_zero_at_identity(self, group):
        assert group.distance_from_identity(group.identity()) < 1e-15


class TestSphereSpecific:
    """Sphere-specific properties."""

    def test_norm_preservation(self):
        group = closure_rs.sphere()
        q = np.asarray(group.random(seed=10))
        for seed in range(1000):
            q = np.asarray(group.compose(q, group.random(seed=seed + 100)))
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12

    def test_noncommutative(self):
        group = closure_rs.sphere()
        a = np.asarray(group.random(seed=11))
        b = np.asarray(group.random(seed=12))
        ab = np.asarray(group.compose(a, b))
        ba = np.asarray(group.compose(b, a))
        assert not _allclose(ab, ba, atol=1e-6)


class TestTorusSpecific:
    """Torus-specific properties."""

    def test_commutative(self):
        group = closure_rs.torus(5)
        for seed in range(0, 100, 2):
            a = np.asarray(group.random(seed=seed))
            b = np.asarray(group.random(seed=seed + 1))
            assert _allclose(
                np.asarray(group.compose(a, b)),
                np.asarray(group.compose(b, a)),
            )

    def test_channel_independence(self):
        group = closure_rs.torus(5)
        base = np.asarray(group.identity())
        perturbed = base.copy()
        perturbed[2] = 0.5
        residuals = np.asarray(closure_rs.channel_residuals(5, perturbed))
        assert abs(residuals[2] - 0.5) < 1e-15
        for i in [0, 1, 3, 4]:
            assert abs(residuals[i]) < 1e-15
