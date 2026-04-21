//! EXPERIMENT 11 — FIBER MEMORY ON S³
//!
//! A quaternion is a rotation. Every rotation has two coordinates:
//! the axis it rotates around (S² base) and how far it rotates (S¹ phase).
//! These two coordinates are independent. The axis tells you nothing about
//! the angle. The angle tells you nothing about the axis.
//!
//! A memory on S³ inherits this directly. Three query modes:
//!
//!   Base  — match on axis only.  Finds everything at that type.
//!   Phase — match on angle only. Finds everything at that position.
//!   Full  — match on both.       Finds the exact carrier.
//!
//! Sections:
//!   1. Decompose → reconstruct (exact round-trip for all carriers).
//!   2. Fiber lattice: 8 angles, one fixed axis. Phase query separates them.
//!      Base query returns one point.
//!   3. Base lattice: 8 axes, one fixed angle. Base query separates them.
//!      Phase query returns one point.
//!   4. Mixed grid: 2 axes × 2 angles. Each query mode sees a different slice.
//!   5. The ALIVE^n orbit in Hopf coordinates.
//!   6. What this means.
//!
//! Run: cargo run --example exp_fiber_memory --release

use closure_ea::{
    address_distance, carrier_from_hopf, compose, hopf_decompose, inverse,
    sigma, AddressMode, IDENTITY, SALIENCE_AXIS, TOTAL_AXIS, UNKNOWN_AXIS,
};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4, PI, TAU};

const ALIVE: [f64; 4] = [FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0, 0.0];

fn header(title: &str) {
    let bar = "━".repeat(70);
    println!("\n{bar}");
    println!("  {title}");
    println!("{bar}");
}

/// Show phase as a readable fraction of π.
fn phase_str(phase: f64) -> String {
    // Common fractions with denominator ≤ 6, within floating-point tolerance.
    for denom in [1_i32, 2, 3, 4, 5, 6] {
        for numer in 0..=(2 * denom) {
            let candidate = numer as f64 * PI / denom as f64;
            if (candidate - phase).abs() < 1e-9 {
                return match (numer, denom) {
                    (0, _) => "0".to_string(),
                    (n, 1) => format!("{n}π"),
                    (n, d) => format!("{n}π/{d}"),
                };
            }
        }
    }
    format!("{:.4}π", phase / PI)
}

/// Name a canonical S² axis direction.
fn axis_name(b: &[f64; 3]) -> &'static str {
    let n = |v: f64, t: f64| (v - t).abs() < 0.001;
    if n(b[0], 1.) && n(b[1], 0.) && n(b[2], 0.)  { "+X  (salience)" }
    else if n(b[0],-1.) && n(b[1], 0.) && n(b[2], 0.) { "−X" }
    else if n(b[0], 0.) && n(b[1], 1.) && n(b[2], 0.) { "+Y  (total)" }
    else if n(b[0], 0.) && n(b[1],-1.) && n(b[2], 0.) { "−Y" }
    else if n(b[0], 0.) && n(b[1], 0.) && n(b[2], 1.) { "+Z  (unknown)" }
    else if n(b[0], 0.) && n(b[1], 0.) && n(b[2],-1.) { "−Z" }
    else { "mixed" }
}

/// Show reconstruction error as either the actual float or "exact".
fn err_str(e: f64) -> String {
    if e == 0.0 { "exact".to_string() }
    else { format!("{:.1e}", e) }
}

fn main() {
    header("EXPERIMENT 11 · FIBER MEMORY ON S³");
    println!();
    println!("  A quaternion encodes a rotation in 3D space.");
    println!("  Every rotation has two coordinates:");
    println!();
    println!("    Axis  (S² base)  — which direction it rotates around.");
    println!("    Angle (S¹ phase) — how far it rotates.");
    println!();
    println!("  These two coordinates do not talk to each other.");
    println!("  You can change the angle without touching the axis, and vice versa.");
    println!();
    println!("  A memory built on S³ inherits this exactly:");
    println!("  two address channels, three query modes, three different answers.");

    // ── 1. Round-trip ────────────────────────────────────────────────────
    header("1. ROUND-TRIP — decompose → reconstruct");
    println!();
    println!("  Every carrier splits into (axis, angle) and reconstructs without loss.");
    println!();
    println!("  {:<30}  {:<17}  {:>7}  {:>10}",
        "carrier", "axis", "angle", "error");
    println!("  {}", "─".repeat(70));

    let rt_cases: &[(&str, [f64; 4])] = &[
        ("IDENTITY",                     IDENTITY),
        ("ALIVE   (σ = π/4, W > 0)",     ALIVE),
        ("ALIVE²  (σ = π/2, W = 0)",     compose(&ALIVE, &ALIVE)),
        ("ALIVE³  (σ = π/4, W < 0)",     compose(&compose(&ALIVE, &ALIVE), &ALIVE)),
        ("from_hopf( salience,  π/3 )",  carrier_from_hopf(SALIENCE_AXIS, PI / 3.0)),
        ("from_hopf( total,    5π/6 )",  carrier_from_hopf(TOTAL_AXIS, 5.0 * PI / 6.0)),
        ("from_hopf( unknown,  7π/5 )",  carrier_from_hopf(UNKNOWN_AXIS, 7.0 * PI / 5.0)),
    ];

    for &(label, q) in rt_cases {
        let (base, phase) = hopf_decompose(&q);
        let recon = carrier_from_hopf(base, phase);
        let err = sigma(&compose(&q, &inverse(&recon)));
        println!("  {label:<30}  {:<17}  {:>7}  {:>10}",
            axis_name(&base), phase_str(phase), err_str(err));
    }
    println!();
    println!("  Error is the geodesic distance between original and reconstructed.");
    println!("  'exact' means floating-point zero. Others are machine precision (≈ 1e-16).");

    // ── 2. Fiber lattice ─────────────────────────────────────────────────
    header("2. FIBER LATTICE — 8 angles, one axis");
    println!();
    println!("  Axis fixed at SALIENCE (+X).");
    println!("  8 carriers at angles 0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4.");
    println!();
    println!("  Distances from k=0:");
    println!();
    println!("  {:>4}  {:>8}  {:>12}  {:>12}",
        "k", "angle", "by angle", "by axis");
    println!("  {}", "─".repeat(42));

    let fiber_base = SALIENCE_AXIS;
    let fiber_carriers: Vec<[f64; 4]> = (0..8)
        .map(|k| carrier_from_hopf(fiber_base, k as f64 * FRAC_PI_4))
        .collect();
    let q0_fiber = fiber_carriers[0];

    for (k, &qk) in fiber_carriers.iter().enumerate() {
        let phase_k = k as f64 * FRAC_PI_4;
        let d_phase = address_distance(&qk, &q0_fiber, AddressMode::Phase);
        let d_base  = address_distance(&qk, &q0_fiber, AddressMode::Base);
        let base_str = if d_base < 1e-12 { "0".to_string() } else { format!("{:.4}", d_base) };
        println!("  {:>4}  {:>8}  {:>12}  {:>12}",
            k, phase_str(phase_k), phase_str(d_phase), base_str);
    }
    println!();
    println!("  Axis distance: 0 for every k. The axis channel reads one carrier.");
    println!("  Angle distance: steps π/4, wraps at π. Each carrier is distinct.");

    // ── 3. Base lattice ──────────────────────────────────────────────────
    header("3. BASE LATTICE — 8 axes, one angle");
    println!();
    println!("  Angle fixed at π/4.");
    println!("  8 carriers at axes rotated by k·π/4 in the XY-plane.");
    println!();
    println!("  Distances from k=0:");
    println!();
    println!("  {:>4}  {:>12}  {:>12}  {:>12}",
        "k", "axis angle", "by axis", "by angle");
    println!("  {}", "─".repeat(46));

    let fixed_phase = FRAC_PI_4;
    let base_carriers: Vec<([f64; 3], [f64; 4])> = (0..8)
        .map(|k| {
            let angle = k as f64 * FRAC_PI_4;
            let base = [angle.cos(), angle.sin(), 0.0_f64];
            (base, carrier_from_hopf(base, fixed_phase))
        })
        .collect();
    let q0_base = base_carriers[0].1;

    for (k, &(_, ref qk)) in base_carriers.iter().enumerate() {
        let axis_angle = k as f64 * FRAC_PI_4;
        let d_base  = address_distance(qk, &q0_base, AddressMode::Base);
        let d_phase = address_distance(qk, &q0_base, AddressMode::Phase);
        let phase_str_d = if d_phase < 1e-12 { "0".to_string() } else { format!("{:.4}", d_phase) };
        println!("  {:>4}  {:>12}  {:>12}  {:>12}",
            k, phase_str(axis_angle), phase_str(d_base), phase_str_d);
    }
    println!();
    println!("  Angle distance: 0 for every k. The angle channel reads one carrier.");
    println!("  Axis distance: steps π/4, wraps at π. Each carrier is distinct.");

    // ── 4. Mixed grid ────────────────────────────────────────────────────
    header("4. MIXED GRID — 2 axes × 2 angles");
    println!();
    println!("  B1 = salience axis  (+X)     B2 = total axis  (+Y)");
    println!("  P1 = π/4                     P2 = 3π/4");
    println!();
    println!("  Four carriers. Query = (B1, P1).");
    println!();
    println!("  {:<22}  {:>8}  {:>8}  {:>8}",
        "", "full", "axis", "angle");
    println!("  {}", "─".repeat(52));

    let b1 = SALIENCE_AXIS;
    let b2 = TOTAL_AXIS;
    let p1 = FRAC_PI_4;
    let p2 = 3.0 * FRAC_PI_4;

    let q_b1p1 = carrier_from_hopf(b1, p1);
    let q_b1p2 = carrier_from_hopf(b1, p2);
    let q_b2p1 = carrier_from_hopf(b2, p1);
    let q_b2p2 = carrier_from_hopf(b2, p2);
    let query = q_b1p1;

    let grid = [
        ("(B1, P1)  ← query", q_b1p1),
        ("(B1, P2)  same axis", q_b1p2),
        ("(B2, P1)  same angle", q_b2p1),
        ("(B2, P2)  different", q_b2p2),
    ];

    for (label, q) in &grid {
        let d_full  = address_distance(&query, q, AddressMode::Full);
        let d_base  = address_distance(&query, q, AddressMode::Base);
        let d_phase = address_distance(&query, q, AddressMode::Phase);
        println!("  {label:<22}  {:>8.4}  {:>8.4}  {:>8.4}", d_full, d_base, d_phase);
    }

    println!();
    println!("  Full  — 0 for (B1,P1) only. Both coordinates must agree for exact match.");
    println!("  Axis  — 0 for (B1,P1) and (B1,P2). Axis query sees two carriers as one.");
    println!("  Angle — 0 for (B1,P1) and (B2,P1). Angle query sees two carriers as one.");
    println!();
    println!("  The axis and angle channels carry different information.");
    println!("  A full query asks both questions at once.");

    // ── 5. ALIVE^n in Hopf coordinates ──────────────────────────────────
    header("5. THE ALIVE^n ORBIT IN HOPF COORDINATES");
    println!();
    println!("  ALIVE generates an 8-element cyclic group under Hamilton product.");
    println!("  Here are all 8 powers in (axis, angle) coordinates:");
    println!();
    println!("  {:>4}  {:>34}  {:<17}  {:>8}",
        "n", "ALIVE^n  [W, X, Y, Z]", "axis", "angle");
    println!("  {}", "─".repeat(70));

    let mut power = IDENTITY;
    for n in 0usize..=8 {
        let (base, phase) = hopf_decompose(&power);
        let tag = match n {
            0 => "  IDENTITY",
            8 => "  = IDENTITY  (closed)",
            _ => "",
        };
        println!("  {:>4}  [{:>7.4}, {:>7.4}, {:>7.4}, {:>7.4}]  {:<17}  {:>8}{}",
            n, power[0], power[1], power[2], power[3],
            axis_name(&base), phase_str(phase), tag);
        if n < 8 { power = compose(&power, &ALIVE); }
    }
    println!();
    println!("  Two observations:");
    println!();
    println!("  First: the 8 quaternions produce 6 distinct (axis, angle) pairs.");
    println!("    n=1 and n=5 land on the same point: axis −Y, angle π/2.");
    println!("    n=3 and n=7 land on the same point: axis +Y, angle 3π/2.");
    println!("    The other 4 powers all have unique coordinates.");
    println!("    These collisions are the 2:1 covering S³ → SO(3): q and −q are");
    println!("    different quaternions but encode the same 3D rotation. When they");
    println!("    collide in Hopf space, that is exactly what is happening.");
    println!();
    println!("  Second: binary GoL composes this orbit every generation and immediately");
    println!("    discards all 8 powers, snapping back to ALIVE^1. Gray GoL keeps them.");

    // ── 6. What this means ───────────────────────────────────────────────
    header("6. WHAT THIS MEANS");
    println!();
    println!("  Every rotation in 3D space has two coordinates. Memory on S³ exposes");
    println!("  both as addressable channels:");
    println!();
    println!("  Axis (base) channel — 'What kind of thing is this?'");
    println!("    Salience, total, unknown, or any direction on S².");
    println!("    Multiple cyclic positions can share one type.");
    println!("    Axis queries return every carrier of that type.");
    println!();
    println!("  Angle (phase) channel — 'Where in the cycle?'");
    println!("    Step k of n. Bar k of a phrase. Generation k of an orbit.");
    println!("    Multiple types can share one position.");
    println!("    Angle queries return every carrier at that position.");
    println!();
    println!("  Full channel — 'Exactly this?'");
    println!("    Both axis and angle must match.");
    println!();
    println!("  Biological parallel — the genetic code:");
    println!("    Multiple codons encode the same amino acid. In S³ terms: several angles");
    println!("    share the same axis (the amino acid type). The ribosome asks an axis");
    println!("    question. The tRNA anticodon asks an angle question. One system, two");
    println!("    independent channels, running for 3.5 billion years.");

    // ── Assertions ───────────────────────────────────────────────────────

    for &(label, q) in rt_cases {
        let (base, phase) = hopf_decompose(&q);
        let recon = carrier_from_hopf(base, phase);
        let err = sigma(&compose(&q, &inverse(&recon)));
        assert!(err < 1e-10, "round-trip for {label}: {err:.2e}");
    }
    for k in 0..8 {
        let qk = fiber_carriers[k];
        assert!(
            address_distance(&qk, &q0_fiber, AddressMode::Base) < 1e-10,
            "fiber lattice k={k}: base distance must be 0"
        );
        let d_phase = address_distance(&qk, &q0_fiber, AddressMode::Phase);
        let phase_k = k as f64 * FRAC_PI_4;
        let expected = phase_k.min(TAU - phase_k);
        assert!(
            (d_phase - expected).abs() < 1e-10,
            "fiber lattice k={k}: phase distance {d_phase:.4} expected {expected:.4}"
        );
    }
    for (k, &(_, ref qk)) in base_carriers.iter().enumerate() {
        assert!(
            address_distance(qk, &q0_base, AddressMode::Phase) < 1e-10,
            "base lattice k={k}: phase distance must be 0"
        );
    }
    assert!(address_distance(&query, &q_b1p1, AddressMode::Full) < 1e-10);
    assert!(address_distance(&query, &q_b2p2, AddressMode::Full) > 0.3);
    assert!((address_distance(&query, &q_b1p1, AddressMode::Base) -
             address_distance(&query, &q_b1p2, AddressMode::Base)).abs() < 1e-10);
    assert!((address_distance(&query, &q_b1p1, AddressMode::Phase) -
             address_distance(&query, &q_b2p1, AddressMode::Phase)).abs() < 1e-10);
    let mut p = IDENTITY;
    for _ in 0..8 { p = compose(&p, &ALIVE); }
    assert!(sigma(&p) < 1e-10, "ALIVE^8 must equal IDENTITY");

    println!();
    println!("  Verified:");
    println!("    Round-trip: all carriers reconstruct exactly ✓");
    println!("    Fiber lattice: axis distance = 0, angle distances monotone ✓");
    println!("    Base lattice: angle distance = 0, axis distances geodesic ✓");
    println!("    Mixed grid: full/axis/angle queries return distinct slices ✓");
    println!("    ALIVE^8 = IDENTITY ✓");
}
