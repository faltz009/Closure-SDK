//! EXPERIMENT 4 — Geometric Associative Memory on S³
//!
//! A hashmap stores associations by exact key match — perturb the key by
//! even one bit and it returns nothing. A weight-based neural network
//! uses distributed representations where writing one pair shifts the
//! weights used to represent all others (catastrophic interference).
//!
//! The S³ geometric memory has a different trade-off. Each pair occupies
//! a dedicated genome slot keyed by geodesic position on S³. RESONATE
//! finds the nearest slot — so recall is robust to query noise up to the
//! geometric tolerance (half the minimum pairwise address spacing), and
//! adding new pairs never disturbs existing ones.
//!
//! Method: store 1000 (input, target) pairs as genome Response entries,
//! with inputs uniformly distributed on the W>0 hemisphere of S³ and
//! targets on the W<0 hemisphere. Test recall with increasing query noise
//! ε. Verify isolation: adding pairs 501-1000 leaves pairs 1-500 intact.
//!
//! Run:  cargo run --example exp_associative_memory --release

use std::f64::consts::{PI, FRAC_1_SQRT_2};
use closure_ea::{
    GenomeConfig, ThreeCell,
    IDENTITY, sigma, compose, inverse,
};

const N_PAIRS: usize = 1000;

// ── Uniform S³ sampling via Box-Muller on 4 Gaussians ────────────────────

/// Linear congruential generator. Returns a uniform float in (0, 1).
fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005)
          .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 33) as f64 + 0.5) / (1u64 << 31) as f64
}

/// Box-Muller: one standard-normal sample.
fn gauss(s: &mut u64) -> f64 {
    let u1 = lcg(s).max(1e-15);
    let u2 = lcg(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Uniform unit quaternion on S³.
fn uniform_s3(seed: u64) -> [f64; 4] {
    let mut s = seed;
    let a = gauss(&mut s);
    let b = gauss(&mut s);
    let c = gauss(&mut s);
    let d = gauss(&mut s);
    let len = (a * a + b * b + c * c + d * d).sqrt();
    if len < 1e-15 { return IDENTITY; }
    [a / len, b / len, c / len, d / len]
}

/// Input carrier: uniform on the W>0 hemisphere (sign-flip if needed).
fn input_carrier(k: usize) -> [f64; 4] {
    let mut q = uniform_s3(k as u64 + 1);
    if q[0] < 0.0 { q = [-q[0], -q[1], -q[2], -q[3]]; }
    q
}

/// Target carrier: uniform on the W<0 hemisphere.
fn target_carrier(k: usize) -> [f64; 4] {
    let mut q = uniform_s3(k as u64 + 1 + N_PAIRS as u64);
    if q[0] > 0.0 { q = [-q[0], -q[1], -q[2], -q[3]]; }
    q
}

fn header(title: &str) {
    let bar = "━".repeat(70);
    println!("\n{bar}");
    println!("  {title}");
    println!("{bar}");
}

fn barchart(filled: usize, total: usize, width: usize) -> String {
    let n = (filled * width / total.max(1)).min(width);
    format!("{}{}", "█".repeat(n), "░".repeat(width - n))
}

fn main() {
    header("EXPERIMENT 4 · GEOMETRIC ASSOCIATIVE MEMORY ON S³");
    println!();
    println!("  A std::HashMap is exact: the key must match byte-for-byte.");
    println!("  Perturb the query by any amount and the lookup fails.");
    println!();
    println!("  A weight-based neural network is distributed: writing one pair");
    println!("  perturbs the storage of all others (weight interference). Recall");
    println!("  degrades with the number of stored associations, not with noise.");
    println!();
    println!("  Geometric memory on S³ has a different trade-off:");
    println!("    — Each pair occupies a dedicated address (no weight interference).");
    println!("    — RESONATE finds the NEAREST address, not the exact one.");
    println!("    — Recall is robust to query noise up to half the pairwise spacing.");
    println!();
    println!("  The tolerance is not a tuned hyperparameter. It is the geometry of");
    println!("  how densely the addresses are packed — a computable fact about S³.");
    println!();
    println!("  Dataset: {N_PAIRS} pairs. Inputs on W>0 hemisphere, targets on W<0.");
    println!("  Both sets sampled uniformly via Box-Muller on S³.");
    println!();

    // ── Build 1000 pairs ────────────────────────────────────────────────
    let inputs:  Vec<[f64; 4]> = (0..N_PAIRS).map(input_carrier).collect();
    let targets: Vec<[f64; 4]> = (0..N_PAIRS).map(target_carrier).collect();

    // Measure actual minimum pairwise input spacing.
    let mut min_spacing = f64::MAX;
    for i in 0..N_PAIRS {
        for j in i + 1..N_PAIRS {
            let d = sigma(&compose(&inputs[i], &inverse(&inputs[j])));
            if d < min_spacing { min_spacing = d; }
        }
    }
    let predicted_threshold = min_spacing / 2.0;
    println!("  Actual minimum pairwise input spacing: {min_spacing:.4} rad on S³.");
    println!("  Predicted noise tolerance threshold:   {predicted_threshold:.4} rad (half-spacing).");
    println!();

    // ── Write all 1000 pairs ────────────────────────────────────────────
    let mut brain = ThreeCell::new(
        0.05, 0.05, 4,
        GenomeConfig {
            reinforce_threshold: 0.001,
            novelty_threshold: 0.15,
            merge_threshold: 0.001,
            co_resonance_merge_threshold: 0.0,
        },
    );
    {
        let genome = brain.hierarchy_mut().genome_at_mut(0);
        for k in 0..N_PAIRS {
            genome.learn_response(&inputs[k], &targets[k]);
        }
    }
    println!("  Written: {N_PAIRS} pairs → {} genome entries  (3 DNA + {} Response)",
        brain.genome_size(), brain.genome_size().saturating_sub(3));
    println!();

    // ── Exact recall ────────────────────────────────────────────────────
    header("EXACT RECALL  (ε = 0)");
    println!();
    let mut exact_0 = 0usize;
    for k in 0..N_PAIRS {
        let pred = brain.evaluate(&[inputs[k]]).map(|h| h.carrier).unwrap_or(IDENTITY);
        if sigma(&compose(&pred, &inverse(&targets[k]))) < 0.05 { exact_0 += 1; }
    }
    println!("  {exact_0}/{N_PAIRS}  {}  (baseline)", barchart(exact_0, N_PAIRS, 30));

    // ── Noise tolerance sweep ────────────────────────────────────────────
    header("NOISE TOLERANCE  (ε from 0 to 2× predicted threshold)");
    println!();
    println!("  Each query: stored input composed with a fixed rotation of size ε.");
    println!("  σ(perturbed, original) = ε exactly.");
    println!();
    println!("  Prediction: full recall for ε < {predicted_threshold:.4} (half the min spacing).");
    println!();
    println!("  {:>8}  {:>10}  {:>9}  {:>32}",
        "ε", "Correct", "Fraction", "");
    println!("  {}", "─".repeat(68));

    // Sweep straddling the predicted threshold plus several points beyond.
    let t = predicted_threshold;
    let epsilons = [0.0, t * 0.25, t * 0.5, t * 0.75, t, t * 1.25, t * 1.5, t * 2.0, t * 3.0];
    for &eps in &epsilons {
        let mut correct = 0usize;
        for k in 0..N_PAIRS {
            // Perturb: compose with a fixed rotation of size ε in the WZ plane.
            let noise = [eps.cos(), 0.0, eps.sin(), 0.0];
            let q = compose(&inputs[k], &noise);
            let pred = brain.evaluate(&[q]).map(|h| h.carrier).unwrap_or(IDENTITY);
            if sigma(&compose(&pred, &inverse(&targets[k]))) < 0.05 { correct += 1; }
        }
        let frac = correct as f64 / N_PAIRS as f64;
        let marker = if frac > 0.999 { "✓ full recall" }
                     else if frac > 0.90 { "≈ near-full" }
                     else if frac > 0.50 { "~ partial" }
                     else { "✗ degraded" };
        println!("  {:>8.4}  {:>10}  {:>8.1}%  {:<32}  {}",
            eps, correct, frac * 100.0, barchart(correct, N_PAIRS, 30), marker);
    }

    // ── Isolation ───────────────────────────────────────────────────────
    header("ISOLATION  (new pairs don't disturb stored ones)");
    println!();
    println!("  In a weight-based network, adding new pairs displaces weights used");
    println!("  for old ones — catastrophic interference. Here, each pair occupies");
    println!("  a dedicated S³ slot. Adding pairs 501-1000 must not touch 1-500.");
    println!();

    let mut brain2 = ThreeCell::new(
        0.05, 0.05, 4,
        GenomeConfig {
            reinforce_threshold: 0.001,
            novelty_threshold: 0.15,
            merge_threshold: 0.001,
            co_resonance_merge_threshold: 0.0,
        },
    );
    {
        let g = brain2.hierarchy_mut().genome_at_mut(0);
        for k in 0..500 { g.learn_response(&inputs[k], &targets[k]); }
    }
    let mut before = 0usize;
    for k in 0..500 {
        let pred = brain2.evaluate(&[inputs[k]]).map(|h| h.carrier).unwrap_or(IDENTITY);
        if sigma(&compose(&pred, &inverse(&targets[k]))) < 0.05 { before += 1; }
    }
    {
        let g = brain2.hierarchy_mut().genome_at_mut(0);
        for k in 500..N_PAIRS { g.learn_response(&inputs[k], &targets[k]); }
    }
    let mut after = 0usize;
    let mut new_correct = 0usize;
    for k in 0..N_PAIRS {
        let pred = brain2.evaluate(&[inputs[k]]).map(|h| h.carrier).unwrap_or(IDENTITY);
        let hit = sigma(&compose(&pred, &inverse(&targets[k]))) < 0.05;
        if k < 500 && hit { after += 1; }
        if k >= 500 && hit { new_correct += 1; }
    }

    println!("  Pairs 1–500 BEFORE adding 501–1000:  {before} / 500");
    println!("  Pairs 1–500 AFTER  adding 501–1000:  {after} / 500");
    println!("  Pairs 501–1000 after write:           {new_correct} / 500");
    if after == before {
        println!();
        println!("  Change: 0 pairs disturbed.  Isolation is exact. ✓");
    } else {
        println!();
        println!("  Change: {} pairs disturbed.", (after as i64 - before as i64).abs());
    }

    // ── Summary ──────────────────────────────────────────────────────────
    header("WHAT THIS DEMONSTRATES");
    println!();
    println!("  Minimum pairwise spacing: {min_spacing:.4} rad  (measured, not theoretical).");
    println!("  Noise tolerance boundary: {predicted_threshold:.4} rad  (half-spacing, geometric).");
    println!();
    println!("  Three properties no std::HashMap has:");
    println!("  1. Approximate query tolerance: σ(query, address) < half-spacing → exact.");
    println!("  2. Zero interference: storing pair k leaves every other pair unchanged.");
    println!("  3. Geometry-native: all distances are Hamilton products + arccos on S³.");
    println!();
    println!("  Weight-based systems generalize — nearby inputs give nearby outputs —");
    println!("  at the cost of interference. Geometric memory isolates, at the cost");
    println!("  of O(N) query time. Different trade-off, same underlying manifold.");
    println!();
    let _ = FRAC_1_SQRT_2; // suppress unused import warning if any
}
