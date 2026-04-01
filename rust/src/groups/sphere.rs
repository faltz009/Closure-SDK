//! Sphere — the order-sensitive mode.
//!
//! Elements are unit quaternions q = (w, x, y, z) on the 3-sphere S³.
//! Compose = Hamilton product (non-commutative: a·b ≠ b·a).
//! Distance = arccos(|w|), the geodesic on S³.
//!
//! Because multiplication is non-commutative, swapping two events
//! changes the summary. Use when ordering matters.

use super::LieGroup;
use rand::RngCore;
use std::f64::consts::TAU;

#[derive(Clone)]
pub struct SphereGroup;

/// Clamp to unit length. Falls back to [1,0,0,0] if near-zero.
#[inline]
fn normalize(q: &mut [f64; 4]) {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-15 {
        *q = [1.0, 0.0, 0.0, 0.0];
    } else {
        let inv = 1.0 / n;
        q[0] *= inv;
        q[1] *= inv;
        q[2] *= inv;
        q[3] *= inv;
    }
}

/// Hamilton product. Scalar-first convention: [w, x, y, z].
#[inline]
fn hamilton(a: &[f64], b: &[f64]) -> [f64; 4] {
    let (w1, x1, y1, z1) = (a[0], a[1], a[2], a[3]);
    let (w2, x2, y2, z2) = (b[0], b[1], b[2], b[3]);
    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
}

// ── Box-Muller: uniform → Gaussian for sphere sampling ──────────────

#[inline]
fn uniform_open01(rng: &mut dyn RngCore) -> f64 {
    ((rng.next_u64() as f64) + 1.0) / ((u64::MAX as f64) + 2.0)
}

#[inline]
fn gaussian_pair(rng: &mut dyn RngCore) -> (f64, f64) {
    let u1 = uniform_open01(rng);
    let u2 = uniform_open01(rng);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = TAU * u2;
    (r * theta.cos(), r * theta.sin())
}

impl LieGroup for SphereGroup {
    /// Hamilton product + renormalize (prevents drift over long chains).
    fn compose(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut q = hamilton(a, b);
        normalize(&mut q);
        q.to_vec()
    }

    fn compose_into(&self, a: &[f64], b: &[f64], out: &mut [f64]) {
        let mut q = hamilton(a, b);
        normalize(&mut q);
        out[..4].copy_from_slice(&q);
    }

    /// Conjugate: q⁻¹ = (w, −x, −y, −z) for unit quaternions.
    fn inverse(&self, a: &[f64]) -> Vec<f64> {
        vec![a[0], -a[1], -a[2], -a[3]]
    }

    fn inverse_into(&self, a: &[f64], out: &mut [f64]) {
        out[0] = a[0];
        out[1] = -a[1];
        out[2] = -a[2];
        out[3] = -a[3];
    }

    fn identity(&self) -> Vec<f64> {
        vec![1.0, 0.0, 0.0, 0.0]
    }

    /// Geodesic on S³. |w| handles the q/−q antipodal equivalence.
    fn distance_from_identity(&self, a: &[f64]) -> f64 {
        a[0].abs().clamp(0.0, 1.0).acos()
    }

    /// Uniform on S³ via normalized Gaussians (Muller 1959).
    fn random(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        loop {
            let (a, b) = gaussian_pair(rng);
            let (c, d) = gaussian_pair(rng);
            let mut q = [a, b, c, d];
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if n > 1e-12 {
                let inv = 1.0 / n;
                for v in &mut q {
                    *v *= inv;
                }
                return q.to_vec();
            }
        }
    }

    fn dim(&self) -> usize {
        4
    }
}

// ── Standalone [f64; 4] functions for direct use by closure_vm ──────

pub const IDENTITY: [f64; 4] = [1.0, 0.0, 0.0, 0.0];

/// Hamilton product + normalize. Standalone, no trait dispatch.
#[inline]
pub fn sphere_compose(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    let mut q = hamilton(a, b);
    normalize(&mut q);
    q
}

/// Conjugate = inverse for unit quaternions.
#[inline]
pub fn sphere_inverse(a: &[f64; 4]) -> [f64; 4] {
    [a[0], -a[1], -a[2], -a[3]]
}

/// Geodesic distance from identity on S³: arccos(|w|).
#[inline]
pub fn sphere_sigma(a: &[f64; 4]) -> f64 {
    a[0].abs().clamp(0.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn identity_laws() {
        let g = SphereGroup;
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = g.random(&mut rng);
            let id = g.identity();
            let la = g.compose(&id, &a);
            let ra = g.compose(&a, &id);
            for i in 0..4 {
                assert!((la[i] - a[i]).abs() < 1e-10, "left identity failed");
                assert!((ra[i] - a[i]).abs() < 1e-10, "right identity failed");
            }
        }
    }

    #[test]
    fn inverse_law() {
        let g = SphereGroup;
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let a = g.random(&mut rng);
            let a_inv = g.inverse(&a);
            let prod = g.compose(&a, &a_inv);
            assert!(g.distance_from_identity(&prod) < 1e-10);
        }
    }

    #[test]
    fn non_commutative() {
        let g = SphereGroup;
        let mut rng = StdRng::seed_from_u64(42);
        let a = g.random(&mut rng);
        let b = g.random(&mut rng);
        let ab = g.compose(&a, &b);
        let ba = g.compose(&b, &a);
        let diff: f64 = ab.iter().zip(&ba).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 0.01, "Sphere should be non-commutative");
    }

    #[test]
    fn norm_stability() {
        let g = SphereGroup;
        let mut rng = StdRng::seed_from_u64(42);
        let mut acc = g.identity();
        for _ in 0..1000 {
            let q = g.random(&mut rng);
            acc = g.compose(&acc, &q);
        }
        let norm: f64 = acc.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "norm drift after 1000 compositions"
        );
    }
}
