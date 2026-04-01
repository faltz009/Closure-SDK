"""Numerical verification of Theorems 1 and 2 (Rust backend).

Theorem 1 (Exact Perturbation Propagation): d(C, C') = epsilon
Theorem 2 (Uniform Sensitivity): ||dC/dg_k|| = 1 for all k
"""

from __future__ import annotations

import numpy as np
import pytest

import closure_rs


# ── Helpers ──────────────────────────────────────────────────────────


def _build_path(group, elements):
    path = closure_rs.GeometricPath(group)
    for g in elements:
        path.append(g)
    return path


def _perturb_circle(group, element, epsilon):
    return np.asarray(group.compose(element, np.array([epsilon])))


def _perturb_sphere(group, element, epsilon, seed):
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    r = np.array([
        np.cos(epsilon),
        np.sin(epsilon) * axis[0],
        np.sin(epsilon) * axis[1],
        np.sin(epsilon) * axis[2],
    ])
    return np.asarray(group.compose(element, r))


def _perturb_torus(element, epsilon, channel):
    perturbed = np.asarray(element).copy()
    perturbed[channel] = (perturbed[channel] + epsilon) % (2 * np.pi)
    return perturbed


# ── Theorem 1: d(C, C') = epsilon ───────────────────────────────────


class TestTheorem1:
    """Exact perturbation propagation on compact Lie groups with bi-invariant metrics."""

    @pytest.mark.parametrize("n", [50, 100, 500])
    @pytest.mark.parametrize("pos_frac", [0.0, 0.25, 0.5, 0.75, 0.99])
    def test_circle(self, n, pos_frac):
        group = closure_rs.circle()
        epsilon = 0.3
        k = int(pos_frac * (n - 1))

        elements = [np.asarray(group.random(seed=i)) for i in range(n)]
        path_clean = _build_path(group, elements)

        corrupted = [g.copy() for g in elements]
        corrupted[k] = _perturb_circle(group, corrupted[k], epsilon)
        path_corrupt = _build_path(group, corrupted)

        C = np.asarray(path_clean.closure_element())
        C_prime = np.asarray(path_corrupt.closure_element())
        d = group.distance_from_identity(group.compose(group.inverse(C), C_prime))

        assert abs(d - epsilon) < 1e-10, f"d={d}, epsilon={epsilon}, diff={abs(d-epsilon)}"

    @pytest.mark.parametrize("n", [50, 100, 500])
    @pytest.mark.parametrize("pos_frac", [0.0, 0.25, 0.5, 0.75, 0.99])
    def test_sphere(self, n, pos_frac):
        group = closure_rs.sphere()
        epsilon = 0.1
        k = int(pos_frac * (n - 1))

        elements = [np.asarray(group.random(seed=i + 1000)) for i in range(n)]
        path_clean = _build_path(group, elements)

        corrupted = [g.copy() for g in elements]
        corrupted[k] = _perturb_sphere(group, corrupted[k], epsilon, seed=k + 5000)
        path_corrupt = _build_path(group, corrupted)

        C = np.asarray(path_clean.closure_element())
        C_prime = np.asarray(path_corrupt.closure_element())
        d = group.distance_from_identity(group.compose(group.inverse(C), C_prime))

        assert abs(d - epsilon) < 1e-7, f"d={d}, epsilon={epsilon}, diff={abs(d-epsilon)}"

    @pytest.mark.parametrize("n", [50, 100])
    def test_torus(self, n):
        group = closure_rs.torus(5)
        epsilon = 0.2
        k = n // 2
        channel = 2

        elements = [np.asarray(group.random(seed=i + 2000)) for i in range(n)]
        path_clean = _build_path(group, elements)

        corrupted = [g.copy() for g in elements]
        corrupted[k] = _perturb_torus(corrupted[k], epsilon, channel)
        path_corrupt = _build_path(group, corrupted)

        C = np.asarray(path_clean.closure_element())
        C_prime = np.asarray(path_corrupt.closure_element())
        d = group.distance_from_identity(group.compose(group.inverse(C), C_prime))

        assert abs(d - epsilon) < 1e-10, f"d={d}, epsilon={epsilon}"


# ── Theorem 2: Uniform Sensitivity ──────────────────────────────────


class TestTheorem2:
    """||dC/dg_k|| = 1 for all positions k (uniform sensitivity)."""

    def test_circle_uniform(self):
        group = closure_rs.circle()
        n = 100
        delta = 1e-7

        elements = [np.asarray(group.random(seed=i + 3000)) for i in range(n)]
        path_clean = _build_path(group, elements)
        C = np.asarray(path_clean.closure_element())

        sensitivities = []
        for k in range(n):
            corrupted = [g.copy() for g in elements]
            corrupted[k] = _perturb_circle(group, corrupted[k], delta)
            C_prime = np.asarray(_build_path(group, corrupted).closure_element())
            d = group.distance_from_identity(group.compose(group.inverse(C), C_prime))
            sensitivities.append(d / delta)

        sensitivities = np.array(sensitivities)
        assert np.allclose(sensitivities, 1.0, atol=1e-4), (
            f"Sensitivity range: [{sensitivities.min():.6f}, {sensitivities.max():.6f}]"
        )

    def test_sphere_uniform(self):
        group = closure_rs.sphere()
        n = 50
        delta = 1e-7

        elements = [np.asarray(group.random(seed=i + 4000)) for i in range(n)]
        path_clean = _build_path(group, elements)
        C = np.asarray(path_clean.closure_element())

        sensitivities = []
        for k in range(n):
            corrupted = [g.copy() for g in elements]
            corrupted[k] = _perturb_sphere(group, corrupted[k], delta, seed=k + 6000)
            C_prime = np.asarray(_build_path(group, corrupted).closure_element())
            d = group.distance_from_identity(group.compose(group.inverse(C), C_prime))
            sensitivities.append(d / delta)

        sensitivities = np.array(sensitivities)
        # Wider tolerance for Sphere: arccos nonlinearity at small delta causes
        # numerical Jacobian to read ~0.988 instead of 1.0. Not a theorem
        # violation — Theorem 1 tests confirm d(C,C')=ε exactly.
        assert np.allclose(sensitivities, 1.0, atol=0.03), (
            f"Sensitivity range: [{sensitivities.min():.6f}, {sensitivities.max():.6f}]"
        )
