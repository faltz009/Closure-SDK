"""Tests for GeometricPath data structure (Rust backend)."""

from __future__ import annotations

import numpy as np
import pytest

import closure_rs


def _make_closed(group, n):
    """Generate n elements composing to identity."""
    elements = [np.asarray(group.random(seed=i)) for i in range(n - 1)]
    path = closure_rs.GeometricPath(group)
    for g in elements:
        path.append(g)
    elements.append(np.asarray(group.inverse(path.closure_element())))
    return elements


class TestRecovery:
    """Recover original elements from running products."""

    @pytest.mark.parametrize(
        "group",
        [closure_rs.circle(), closure_rs.sphere(), closure_rs.torus(4)],
        ids=["Circle", "Sphere", "Torus(4)"],
    )
    def test_recover_all(self, group):
        elements = [np.asarray(group.random(seed=i)) for i in range(100)]
        path = closure_rs.GeometricPath(group)
        for g in elements:
            path.append(g)

        for t in range(1, len(path) + 1):
            recovered = np.asarray(path.recover(t))
            d = group.distance_from_identity(
                group.compose(group.inverse(elements[t - 1]), recovered)
            )
            assert d < 1e-9, f"Recovery failed at t={t}, d={d}"


class TestClosure:
    """Closure checks on coherent and incoherent sequences."""

    @pytest.mark.parametrize(
        "group",
        [closure_rs.circle(), closure_rs.sphere(), closure_rs.torus(4)],
        ids=["Circle", "Sphere", "Torus(4)"],
    )
    def test_closed_sequence(self, group):
        elements = _make_closed(group, 100)
        path = closure_rs.GeometricPath(group)
        for g in elements:
            path.append(g)
        assert path.check_global() < 1e-9

    @pytest.mark.parametrize(
        "group",
        [closure_rs.circle(), closure_rs.sphere(), closure_rs.torus(4)],
        ids=["Circle", "Sphere", "Torus(4)"],
    )
    def test_range_equals_global(self, group):
        elements = [np.asarray(group.random(seed=i)) for i in range(50)]
        path = closure_rs.GeometricPath(group)
        for g in elements:
            path.append(g)
        assert abs(path.check_range(0, len(path)) - path.check_global()) < 1e-12

    def test_closed_subrange(self):
        group = closure_rs.circle()
        # Build: [random...] + [closed block] + [random...]
        pre = [np.asarray(group.random(seed=i)) for i in range(10)]

        # Closed block of 20 elements
        block_path = closure_rs.GeometricPath(group)
        block = []
        for i in range(19):
            g = np.asarray(group.random(seed=100 + i))
            block.append(g)
            block_path.append(g)
        block.append(np.asarray(group.inverse(block_path.closure_element())))

        post = [np.asarray(group.random(seed=i + 200)) for i in range(10)]

        path = closure_rs.GeometricPath(group)
        for g in pre + block + post:
            path.append(g)
        # The block occupies indices 11..30 (1-indexed), range (10, 30)
        assert path.check_range(10, 30) < 1e-9

    def test_length(self):
        group = closure_rs.circle()
        path = closure_rs.GeometricPath(group)
        assert len(path) == 0
        for i in range(5):
            path.append(group.random(seed=i))
            assert len(path) == i + 1


class TestPathComparison:
    def test_localize_against_length_mismatch(self):
        group = closure_rs.circle()
        base = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
        ref = closure_rs.GeometricPath.from_elements(group, base)

        extra = np.vstack([base, np.array([[0.0]], dtype=np.float64)])
        extra_path = closure_rs.GeometricPath.from_elements(group, extra)
        idx_extra, _ = ref.localize_against(extra_path)
        assert idx_extra == len(base)

        short = base[:-1]
        short_path = closure_rs.GeometricPath.from_elements(group, short)
        idx_short, _ = ref.localize_against(short_path)
        assert idx_short == len(short)

    def test_hybrid_group_reordering_detected_via_sphere_component(self):
        rng = np.random.default_rng(2026)
        k = 2
        n = 128
        group = closure_rs.hybrid(closure_rs.torus(k), closure_rs.sphere())

        torus_part = rng.uniform(-1.0, 1.0, size=(n, k))
        quats = rng.standard_normal((n, 4))
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        elems = np.hstack([torus_part, quats]).astype(np.float64)

        ref = closure_rs.GeometricPath.from_elements(group, elems)
        swapped = elems.copy()
        swapped[[10, 11]] = swapped[[11, 10]]
        test = closure_rs.GeometricPath.from_elements(group, swapped)

        sigma = ref.compare_at(test, len(ref))
        assert sigma > 1e-6, "Hybrid should detect reorder via the sphere component"

        found, _ = ref.localize_against(test)
        assert found is not None


class TestInputValidation:
    def test_from_elements_requires_contiguous(self):
        group = closure_rs.torus(2)
        data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        non_contig = data[:, ::-1]
        assert not non_contig.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match="contiguous"):
            closure_rs.GeometricPath.from_elements(group, non_contig)

    def test_invalid_indices_raise_value_error(self):
        group = closure_rs.circle()
        path = closure_rs.GeometricPath.from_elements(
            group,
            np.array([[0.1], [0.2]], dtype=np.float64),
        )
        with pytest.raises(ValueError, match="out of range"):
            path.running_product(3)
        with pytest.raises(ValueError, match="out of range"):
            path.recover(0)
        with pytest.raises(ValueError, match="invalid range"):
            path.check_range(2, 1)
