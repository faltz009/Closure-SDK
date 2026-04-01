//! Embedding: raw bytes → group elements on S³.
//!
//! Two modes, same function:
//!
//!   embed(data, hashed=false)  — geometric. Each byte composes as a
//!                                rotation on S³. Similar bytes → nearby
//!                                quaternions. The database uses this.
//!
//!   embed(data, hashed=true)   — cryptographic. SHA-256 first, then
//!                                Box-Muller → S³. Destroys similarity.
//!                                Provides confidentiality and forgery
//!                                resistance. The CLI and blockchain use this.
//!
//! The algebra doesn't care which mode produced the quaternion. Compose,
//! verify, search, bind — all the same. The embedding is the adapter's
//! choice, not the engine's concern.

use crate::groups::sphere::SphereGroup;
use crate::groups::LieGroup;
use sha2::{Digest, Sha256};
use std::f64::consts::TAU;
use std::sync::OnceLock;

// ── Byte rotation table for geometric embedding ─────────────────────
// 256 pre-computed quaternions, one per byte value. Generated once from
// SHA-256 for determinism and uniform distribution. After init, SHA-256
// is never called again in the geometric path.

static BYTE_TABLE: OnceLock<[[f64; 4]; 256]> = OnceLock::new();

fn byte_quaternions() -> &'static [[f64; 4]; 256] {
    BYTE_TABLE.get_or_init(|| {
        let mut table = [[0.0f64; 4]; 256];
        for i in 0..256u16 {
            let q = bytes_to_sphere_hashed(&[i as u8]);
            table[i as usize] = [q[0], q[1], q[2], q[3]];
        }
        table
    })
}

// ── Sphere embedding ────────────────────────────────────────────────

/// bytes → unit quaternion on S³.
///
/// hashed=false (geometric): each byte is a rotation, composed
/// sequentially. Similar byte sequences → nearby quaternions.
///
/// hashed=true (cryptographic): SHA-256 → Box-Muller → S³.
/// Same bytes always map to same point. Similarity destroyed.
pub fn bytes_to_sphere(data: &[u8], hashed: bool) -> Vec<f64> {
    bytes_to_sphere4(data, hashed).to_vec()
}

/// Fixed-size quaternion form of `bytes_to_sphere`.
/// This is the natural representation for S³ in the hot path.
pub fn bytes_to_sphere4(data: &[u8], hashed: bool) -> [f64; 4] {
    if hashed {
        bytes_to_sphere_hashed(data)
    } else {
        bytes_to_sphere_geometric(data)
    }
}

/// Native numeric embedding for typed f64 columns.
/// Uses IEEE-754 structure directly instead of treating numbers as 8 raw bytes.
/// This is deterministic, cheap, and preserves local structure better than a
/// byte-wise walk for numeric data.
pub fn f64_to_sphere4(value: f64) -> [f64; 4] {
    if !value.is_finite() {
        return bytes_to_sphere4(&value.to_le_bytes(), false);
    }

    let bits = value.to_bits();
    let sign = if (bits >> 63) == 0 { 1.0 } else { -1.0 };
    let exp = ((bits >> 52) & 0x7ff) as f64 / 2047.0;
    let mant = bits & ((1u64 << 52) - 1);
    let mant_hi = ((mant >> 26) & ((1u64 << 26) - 1)) as f64 / ((1u64 << 26) - 1) as f64;
    let mant_lo = (mant & ((1u64 << 26) - 1)) as f64 / ((1u64 << 26) - 1) as f64;

    let mut q = [
        sign,
        exp * 2.0 - 1.0,
        mant_hi * 2.0 - 1.0,
        mant_lo * 2.0 - 1.0,
    ];

    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-15 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        let inv = 1.0 / norm;
        q[0] *= inv;
        q[1] *= inv;
        q[2] *= inv;
        q[3] *= inv;
        q
    }
}

/// Monotonic numeric chart on a 1D subgroup of S^3.
///
/// Encodes an unbounded real value into a unit quaternion on the x-axis:
///     q(v) = [cos(atan(v)), sin(atan(v)), 0, 0]
///          = [1/sqrt(1+v^2), v/sqrt(1+v^2), 0, 0]
///
/// Order is preserved because atan is monotonic. The original value is
/// recovered by x / w.
pub fn f64_to_order_sphere4(value: f64) -> [f64; 4] {
    if !value.is_finite() {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let denom = (1.0 + value * value).sqrt();
    if denom < 1e-15 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [1.0 / denom, value / denom, 0.0, 0.0]
    }
}

/// Decode a value from the monotonic numeric chart.
#[inline(always)]
pub fn f64_from_order_sphere4(q: &[f64; 4]) -> f64 {
    if q[0].abs() < 1e-15 {
        if q[1].is_sign_negative() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }
    } else {
        q[1] / q[0]
    }
}

/// Geometric embedding for i64 values.
///
/// Encodes the sign and magnitude into a unit quaternion. Nearby
/// integers (close in absolute value, same sign) produce nearby
/// quaternions. For similarity search and identity contribution.
///
/// NOTE: This is for geometric proximity — not for order-preserving
/// comparisons. I64 filter/sort/sum use native integer operations on
/// the raw stored bytes, not this embedding.
pub fn i64_to_sphere4(value: i64) -> [f64; 4] {
    let sign = if value >= 0 { 1.0_f64 } else { -1.0_f64 };
    let abs_bits = value.unsigned_abs();
    // Split into two 32-bit halves for structured spatial encoding
    let hi = ((abs_bits >> 32) as u32) as f64 / u32::MAX as f64;
    let lo = (abs_bits as u32) as f64 / u32::MAX as f64;
    let mag = (abs_bits as f64) / (i64::MAX as f64);
    let q = [sign, mag * 2.0 - 1.0, hi * 2.0 - 1.0, lo * 2.0 - 1.0];
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-15 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let inv = 1.0 / norm;
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

/// Fast opaque bytes embedding for non-indexed payload columns.
/// Order-sensitive, deterministic, cheap to compile/lower.
/// This is for identity contribution, not geometric querying.
pub fn bytes_to_sphere_opaque4(data: &[u8]) -> [f64; 4] {
    if data.is_empty() {
        return [1.0, 0.0, 0.0, 0.0];
    }

    let mut s0: u64 = 0x243f_6a88_85a3_08d3;
    let mut s1: u64 = 0x1319_8a2e_0370_7344;
    let mut s2: u64 = 0xa409_3822_299f_31d0;
    let mut s3: u64 = 0x082e_fa98_ec4e_6c89;

    for (i, &b) in data.iter().enumerate() {
        let x = b as u64 | (((i as u64) + 1) << 8);
        s0 = (s0 ^ x).wrapping_mul(0x9e37_79b1_85eb_ca87).rotate_left(13);
        s1 = (s1 ^ (x << 1)).wrapping_mul(0xc2b2_ae3d_27d4_eb4f).rotate_left(17);
        s2 = (s2 ^ (x << 2)).wrapping_mul(0x1656_67b1_9e37_79f9).rotate_left(29);
        s3 = (s3 ^ (x << 3)).wrapping_mul(0x85eb_ca77_c2b2_ae63).rotate_left(41);
    }

    let to_unit = |u: u64| ((u as f64 / u64::MAX as f64) * 2.0) - 1.0;
    let mut q = [to_unit(s0), to_unit(s1), to_unit(s2), to_unit(s3)];
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-15 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        let inv = 1.0 / norm;
        q[0] *= inv;
        q[1] *= inv;
        q[2] *= inv;
        q[3] *= inv;
        q
    }
}

/// Geometric embedding: compose each byte as a rotation on S³.
/// Similar byte sequences share prefix rotations → nearby results.
/// This IS the closure primitive applied at the byte level.
fn bytes_to_sphere_geometric(data: &[u8]) -> [f64; 4] {
    if data.is_empty() {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let table = byte_quaternions();
    let g = SphereGroup;
    let mut running = [1.0f64, 0.0, 0.0, 0.0];
    let mut buf = [0.0f64; 4];
    for &byte in data {
        g.compose_into(&running, &table[byte as usize], &mut buf);
        running = buf;
    }
    running
}

/// Cryptographic embedding: SHA-256 → Box-Muller → S³.
/// Deterministic but destroys all similarity between inputs.
fn bytes_to_sphere_hashed(data: &[u8]) -> [f64; 4] {
    let hash = Sha256::digest(data);
    let mut u = [0.0f64; 4];
    for i in 0..4 {
        let v = u64::from_le_bytes(hash[i * 8..(i + 1) * 8].try_into().unwrap());
        u[i] = (v as f64 + 1.0) / (u64::MAX as f64 + 2.0);
    }
    let r1 = (-2.0 * u[0].ln()).sqrt();
    let theta1 = TAU * u[1];
    let r2 = (-2.0 * u[2].ln()).sqrt();
    let theta2 = TAU * u[3];
    let vals = [
        r1 * theta1.cos(),
        r1 * theta1.sin(),
        r2 * theta2.cos(),
        r2 * theta2.sin(),
    ];
    let norm =
        (vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2] + vals[3] * vals[3]).sqrt();
    if norm < 1e-10 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let inv = 1.0 / norm;
    [vals[0] * inv, vals[1] * inv, vals[2] * inv, vals[3] * inv]
}

// ── Circle embedding ────────────────────────────────────────────────

/// bytes → Circle element (one phase in [0, 2π)).
pub fn bytes_to_phase(data: &[u8]) -> Vec<f64> {
    let hash = Sha256::digest(data);
    let h = u64::from_le_bytes(hash[..8].try_into().unwrap());
    let angle = (h as f64 / u64::MAX as f64) * TAU;
    vec![angle]
}

// ── Torus embedding ─────────────────────────────────────────────────

/// bytes → Torus element (k phases in [0, 2π), domain-separated).
pub fn bytes_to_torus(data: &[u8], k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let h = hash_u64_with_domain(data, 0x544F5255, i as u32);
        let angle = (h as f64 / u64::MAX as f64) * TAU;
        out.push(angle);
    }
    out
}

/// Domain-separated SHA-256: hash(data || domain || idx) → u64.
fn hash_u64_with_domain(data: &[u8], domain: u32, idx: u32) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.update(domain.to_le_bytes());
    hasher.update(idx.to_le_bytes());
    let hash = hasher.finalize();
    u64::from_le_bytes(hash[..8].try_into().unwrap())
}

// ── Element composition ─────────────────────────────────────────────

/// Compute closure element from pre-embedded elements without storing
/// intermediate products. O(n) time, O(1) memory.
pub fn closure_element_from_elements(group: &dyn LieGroup, data: &[f64], dim: usize) -> Vec<f64> {
    let n = data.len() / dim;
    let mut running = group.identity();
    let mut buf = vec![0.0; dim];
    for i in 0..n {
        let g = &data[i * dim..(i + 1) * dim];
        group.compose_into(&running, g, &mut buf);
        running.copy_from_slice(&buf);
    }
    running
}
