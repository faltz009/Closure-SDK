//! Hopf geometry for S^3.
//!
//! This is shared geometry, not Trinity-specific logic.

pub const IDENTITY_BASE: [f64; 3] = [0.0, 0.0, 1.0];
pub const IDENTITY_PHASE: f64 = 0.0;

/// Hopf decomposition: S^3 -> S^2 base x S^1 phase.
#[inline(always)]
pub fn decompose(q: &[f64; 4]) -> ([f64; 3], f64) {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let base = [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        w * w + z * z - x * x - y * y,
    ];
    let phase = (2.0 * (w * x + y * z)).atan2(w * w + z * z - x * x - y * y);
    (base, phase)
}

#[inline(always)]
pub fn base_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
    dot.acos()
}

#[inline(always)]
pub fn circular_distance(a: f64, b: f64) -> f64 {
    let tau = std::f64::consts::PI * 2.0;
    let mut d = (a - b).abs();
    while d > tau {
        d -= tau;
    }
    if d > std::f64::consts::PI {
        tau - d
    } else {
        d
    }
}

#[inline(always)]
pub fn identity_distance(q: &[f64; 4]) -> f64 {
    let (base, phase) = decompose(q);
    base_distance(&base, &IDENTITY_BASE) + circular_distance(phase, IDENTITY_PHASE)
}

pub fn phase_mean(phases: &[f64]) -> f64 {
    let (s, c) = phases
        .iter()
        .fold((0.0, 0.0), |(s, c), p| (s + p.sin(), c + p.cos()));
    s.atan2(c)
}

