"""Tests for HierarchicalClosure (Rust backend)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import closure_rs


def _make_closed(group, n):
    elements = [np.asarray(group.random(seed=i)) for i in range(n - 1)]
    path = closure_rs.GeometricPath(group)
    for g in elements:
        path.append(g)
    elements.append(np.asarray(group.inverse(path.closure_element())))
    return elements


def _to_2d(elements):
    return np.array(elements, dtype=np.float64)


class TestCleanSequence:
    @pytest.mark.parametrize(
        "group",
        [closure_rs.circle(), closure_rs.sphere()],
        ids=["Circle", "Sphere"],
    )
    def test_no_corruption_detected(self, group):
        elements = _make_closed(group, 200)
        tree = closure_rs.HierarchicalClosure(group, _to_2d(elements))
        sigma = tree.check(_to_2d(elements))
        assert sigma < 1e-9

        idx, checks, depth = tree.localize(_to_2d(elements))
        assert idx is None


class TestCorruptedSequence:
    def test_circle_finds_corruption(self):
        group = closure_rs.circle()
        elements = _make_closed(group, 500)
        corrupt_idx = 237

        corrupted = [g.copy() for g in elements]
        corrupted[corrupt_idx] = np.asarray(
            group.compose(corrupted[corrupt_idx], np.array([0.5]))
        )

        tree = closure_rs.HierarchicalClosure(group, _to_2d(elements))
        assert tree.check(_to_2d(corrupted)) > 0.1

        idx, checks, depth = tree.localize(_to_2d(corrupted))
        assert idx == corrupt_idx

    @pytest.mark.parametrize("corrupt_idx", [0, 50, 199, 150, 99])
    def test_various_positions(self, corrupt_idx):
        group = closure_rs.circle()
        elements = _make_closed(group, 200)

        corrupted = [g.copy() for g in elements]
        corrupted[corrupt_idx] = np.asarray(
            group.compose(corrupted[corrupt_idx], np.array([0.4]))
        )

        tree = closure_rs.HierarchicalClosure(group, _to_2d(elements))
        idx, checks, depth = tree.localize(_to_2d(corrupted))
        assert idx == corrupt_idx, f"Expected {corrupt_idx}, got {idx}"

    def test_check_count_logarithmic(self):
        group = closure_rs.circle()
        n = 1000
        elements = _make_closed(group, n)

        corrupted = [g.copy() for g in elements]
        corrupted[500] = np.asarray(
            group.compose(corrupted[500], np.array([0.3]))
        )

        tree = closure_rs.HierarchicalClosure(group, _to_2d(elements))
        idx, checks, depth = tree.localize(_to_2d(corrupted))
        assert idx == 500
        assert checks < 5 * math.log2(n), (
            f"Too many checks: {checks} (expected < {5 * math.log2(n):.0f})"
        )


class TestLengthMismatch:
    def test_extra_tail_is_detected(self):
        group = closure_rs.circle()
        elements = _make_closed(group, 50)
        tree = closure_rs.HierarchicalClosure(group, _to_2d(elements))

        extra = [g.copy() for g in elements]
        extra.append(np.asarray(group.identity()))

        sigma = tree.check(_to_2d(extra))
        assert np.isinf(sigma), "length mismatch should not be treated as clean"

        idx, checks, depth = tree.localize(_to_2d(extra))
        assert idx == len(elements)

    def test_short_input_is_detected(self):
        group = closure_rs.circle()
        elements = _make_closed(group, 50)
        tree = closure_rs.HierarchicalClosure(group, _to_2d(elements))

        short = elements[:-1]

        sigma = tree.check(_to_2d(short))
        assert np.isinf(sigma), "length mismatch should not be treated as clean"

        idx, checks, depth = tree.localize(_to_2d(short))
        assert idx == len(short)

    def test_length_mismatch_with_noisy_prefix(self):
        """Matching prefix with tiny floating-point noise + different lengths
        should still report INFINITY, not a small positive number.

        We inject a perturbation small enough to land between the old
        threshold (1e-15) and the current one (1e-9), so the test would
        fail if the threshold regressed back to 1e-15.
        """
        group = closure_rs.sphere()
        elements = _make_closed(group, 100)
        tree = closure_rs.HierarchicalClosure(group, _to_2d(elements))

        # Perturb the last element of a shorter copy by ~1e-12,
        # well above 1e-15 but well below 1e-9.
        short = [g.copy() for g in elements[:-1]]
        eps = 1e-12
        perturbation = np.array([np.cos(eps), np.sin(eps), 0.0, 0.0])
        short[-1] = np.asarray(group.compose(short[-1], perturbation))

        sigma = tree.check(_to_2d(short))
        assert np.isinf(sigma), (
            f"length mismatch with near-clean prefix should return inf, got {sigma}"
        )
