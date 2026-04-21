//! EXPERIMENT — Neuromodulated Learning
//!
//! The brain should not update itself the same way in every context.
//! A brain perceiving familiar, coherent content should write slowly —
//! stable attractors don't need aggressive correction.
//! A brain perceiving novel, destabilizing content should write fast —
//! the mismatch is real and must be resolved.
//!
//! This experiment demonstrates that the neuromodulatory body state
//! (arousal_tone, coherence_tone) diverges correctly between regimes.
//! The same architecture, the same genome config, the same two patterns.
//! The only difference is what the brain sees between learning episodes:
//!
//!   Regime A — Stable:  familiar, consistent input between episodes.
//!              Brain stays calm. Low arousal. Positive coherence.
//!
//!   Regime B — Disrupted: novel, high-error input between episodes.
//!              Brain is activated. High arousal. Negative coherence.
//!
//! The body state does not change what is learned. It records how
//! surprising and destabilizing recent experience has been — the signal
//! the architecture would use if it chose to modulate write rate.
//!
//! Run: cargo run --example exp_neuromodulated_learning --release

use closure_ea::{
    domain_embed, GenomeConfig, ThreeCell,
    PredictionSource, IDENTITY,
};

/// Buffer lifetime used by every brain in this experiment.
/// Alpha is derived from this: α = 1 − 1/BUFFER_LIFETIME.
const BUFFER_LIFETIME: usize = 4;
const NEUROMOD_ALPHA: f64 = 1.0 - 1.0 / BUFFER_LIFETIME as f64;

fn header(title: &str) {
    let bar = "━".repeat(70);
    println!("\n{bar}");
    println!("  {title}");
    println!("{bar}");
}

/// Build a standard brain for the experiment.
fn make_brain() -> ThreeCell {
    ThreeCell::new(
        0.15,  // cell_a_threshold
        0.20,  // hierarchy_threshold
        4,     // buffer_lifetime
        GenomeConfig {
            reinforce_threshold: 0.02,
            novelty_threshold: 0.20,
            merge_threshold: 0.05,
            co_resonance_merge_threshold: 0.0,
        },
    )
}

/// Run N rounds of: [context × context_reps] → ingest pattern → commit → evaluate.
///
/// `context_reps` context carriers are injected before each learning step.
/// More reps = stronger body-state imprint from the context.
/// Returns (final arousal_tone, final coherence_tone, total promotions).
fn run_learning_episode(
    brain: &mut ThreeCell,
    pattern: &[f64; 4],
    context: &[f64; 4],
    context_reps: usize,
    true_label: &[f64; 4],
    rounds: usize,
) -> (f64, f64, usize) {
    let mut total_promoted = 0usize;

    for _ in 0..rounds {
        // Saturate the body state with the context before learning.
        for _ in 0..context_reps {
            brain.ingest(context);
        }

        // Learning step: ingest pattern, commit a prediction, evaluate with truth.
        let steps = brain.ingest_sequence(&[*pattern]);
        let last = steps.last().unwrap();

        let predicted = last.field_read.as_ref().map(|h| h.carrier).unwrap_or(IDENTITY);
        let source = last.field_read.as_ref()
            .map(|h| PredictionSource::GenomeSlot(h.index))
            .unwrap_or(PredictionSource::GeometricFallback(predicted));
        brain.commit_prediction(predicted, source);

        if let Some(eval) = brain.evaluate_prediction(true_label) {
            total_promoted += eval
                .consolidation_reports
                .iter()
                .map(|r| r.structural.promoted_categories)
                .sum::<usize>();
        }
    }

    // Snapshot body state before the mutable force_consolidate call.
    let final_arousal   = brain.neuromod().arousal_tone;
    let final_coherence = brain.neuromod().coherence_tone;

    let promotions_from_final = brain.force_consolidate()
        .iter()
        .map(|r| r.structural.promoted_categories)
        .sum::<usize>();
    total_promoted += promotions_from_final;

    (final_arousal, final_coherence, total_promoted)
}

fn main() {
    header("EXPERIMENT · NEUROMODULATED LEARNING");
    println!();
    println!("  The brain integrates three per-step signals into a slow body state:");
    println!("    salience_sigma  — how surprising was this input to the model");
    println!("    self_free_energy — how far is Cell C from a fixed point of the genome");
    println!("    valence          — is the brain becoming more or less self-consistent");
    println!();
    println!("  NeuromodState integrates these over a window of ~{} steps (α = {:.2}).",
        (1.0 / (1.0 - NEUROMOD_ALPHA)) as usize, NEUROMOD_ALPHA);
    println!("  The two tones it produces:");
    println!();
    println!("    arousal_tone  ∈ [0, 1]   high = recently activated / surprised");
    println!("    coherence_tone ∈ [−1, 1]  positive = improving self-consistency");
    println!("                              negative = destabilizing");
    println!();
    println!("  These are observational records of recent experience.");
    println!("  The architecture can read them; the write law currently runs at baseline.");

    // ── Carriers ────────────────────────────────────────────────────────────
    let pattern     = domain_embed(b"feature:target", 0.0);
    let true_label  = domain_embed(b"label:true",     0.0);

    // Stable context: familiar, close to pattern.
    let familiar   = domain_embed(b"feature:target", 0.05);
    // Disrupted context: maximally novel, far from anything in the genome.
    let disruptive = domain_embed(b"chaos:unknown",  0.7);

    // 8 context carriers injected before each learning step.
    // With alpha=0.75: each step contributes ~25% weight; 8 steps saturate the
    // body state, making the regimes clearly distinct within 20 rounds.
    const CONTEXT_REPS: usize = 8;
    const ROUNDS: usize = 20;

    // ── Regime A: Stable familiar context ───────────────────────────────────
    header("REGIME A — STABLE, FAMILIAR CONTEXT");
    println!();
    println!("  Before each learning step: {} familiar carriers (close to pattern).", CONTEXT_REPS);
    println!("  The brain stays in a known region of S³. Low prediction error.");
    println!("  Expected: lower arousal, more positive coherence.");
    println!();

    let mut brain_a = make_brain();
    let (arousal_a, coherence_a, promoted_a) =
        run_learning_episode(&mut brain_a, &pattern, &familiar, CONTEXT_REPS, &true_label, ROUNDS);

    let arousal_label_a   = if arousal_a < 0.5 { "low" } else { "moderate" };
    let coherence_label_a = if coherence_a > 0.0 { "positive — stabilizing" } else { "near-zero" };
    println!("  After {} rounds:", ROUNDS);
    println!("    arousal_tone:    {:.4}  ({})", arousal_a, arousal_label_a);
    println!("    coherence_tone:  {:+.4}  ({})", coherence_a, coherence_label_a);
    println!("    promotions:      {}", promoted_a);

    // ── Regime B: Disrupted novel context ───────────────────────────────────
    header("REGIME B — DISRUPTED, NOVEL CONTEXT");
    println!();
    println!("  Before each learning step: {} disruptive carriers (far from genome).", CONTEXT_REPS);
    println!("  Each carrier is strongly novel. High prediction error. High SFE spike.");
    println!("  Expected: higher arousal, lower coherence.");
    println!();

    let mut brain_b = make_brain();
    let (arousal_b, coherence_b, promoted_b) =
        run_learning_episode(&mut brain_b, &pattern, &disruptive, CONTEXT_REPS, &true_label, ROUNDS);

    let arousal_label_b   = if arousal_b > arousal_a { "higher than stable regime" } else { "similar" };
    let coherence_label_b = if coherence_b < coherence_a { "lower than stable regime" } else { "similar" };
    println!("  After {} rounds:", ROUNDS);
    println!("    arousal_tone:    {:.4}  ({})", arousal_b, arousal_label_b);
    println!("    coherence_tone:  {:+.4}  ({})", coherence_b, coherence_label_b);
    println!("    promotions:      {}", promoted_b);

    // ── Direct comparison ────────────────────────────────────────────────────
    header("DIRECT COMPARISON");
    println!();
    println!("  {:>28}  {:>12}  {:>12}", "metric", "stable", "disrupted");
    println!("  {}", "─".repeat(56));
    println!("  {:>28}  {:>12.4}  {:>12.4}", "arousal_tone", arousal_a, arousal_b);
    println!("  {:>28}  {:>+12.4}  {:>+12.4}", "coherence_tone", coherence_a, coherence_b);
    println!("  {:>28}  {:>12}  {:>12}", "promotions", promoted_a, promoted_b);

    println!();
    let arousal_diff  = arousal_b - arousal_a;
    let coherence_diff = coherence_b - coherence_a;
    println!("  arousal_tone difference:   {:+.4}  (disrupted brain is more activated)", arousal_diff);
    println!("  coherence_tone difference: {:+.4}  (disrupted brain is less coherent)", coherence_diff);

    // ── What this proves ─────────────────────────────────────────────────────
    header("WHAT THIS MEANS");
    println!();
    println!("  Both brains saw the SAME learning signal (same pattern, same true_label,");
    println!("  same number of rounds). The only difference was the between-step context.");
    println!();
    println!("  The body state correctly records the character of recent experience:");
    println!("    arousal_b > arousal_a — the disrupted brain was genuinely more activated.");
    println!("    coherence_b < coherence_a — the disrupted brain was more destabilized.");
    println!();
    println!("  These are the signals the architecture needs to modulate write rate.");
    println!("  The measurement works. The runtime write law currently runs at baseline.");
    println!();
    println!("  Promotion also responds:");
    println!("    Response entries learned during destabilizing states (coherence_tone < 0)");
    println!("    are blocked from promoting to category level once coherence history exists.");
    println!("    Categories are built from convergent knowledge, not from crisis learning.");

    // Assertions: the body state correctly diverged between regimes.
    assert!(
        arousal_b > arousal_a,
        "disrupted regime must produce higher arousal_tone: {:.4} vs {:.4}",
        arousal_b, arousal_a
    );
    assert!(
        coherence_b < coherence_a,
        "disrupted regime must produce lower coherence_tone: {:.4} vs {:.4}",
        coherence_b, coherence_a
    );
    println!();
    println!("  Verified: arousal_b > arousal_a ✓   coherence_b < coherence_a ✓");
}
