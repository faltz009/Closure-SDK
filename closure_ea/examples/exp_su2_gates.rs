//! EXPERIMENT — The Carrier Language IS SU(2)
//!
//! Unit quaternions with Hamilton product = the group SU(2), the
//! mathematical structure underlying all single-qubit quantum gates.
//! This is not an analogy. It is an exact equality.
//!
//! Every carrier in this system is an SU(2) element. compose() is
//! SU(2) group multiplication. sigma() is the principal rotation angle
//! divided by two, after the q ≡ -q identification. The standard
//! quantum gate dictionary — X, Y, Z, Hadamard,
//! S, T — maps exactly to carriers. Circuit identities from quantum
//! computing hold here, verifiable by Hamilton product alone.
//!
//! Run: cargo run --example exp_su2_gates --release

use closure_ea::{compose, inverse, sigma, BKT_THRESHOLD, IDENTITY, SIGMA_BALANCE};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

fn header(title: &str) {
    let bar = "━".repeat(70);
    println!("\n{bar}");
    println!("  {title}");
    println!("{bar}");
}

fn qfmt(q: &[f64; 4]) -> String {
    format!(
        "[{:>7.4}, {:>7.4}, {:>7.4}, {:>7.4}]",
        q[0], q[1], q[2], q[3]
    )
}

/// q and r are the same rotation if they are equal or antipodal on S³.
/// (SU(2) is a double cover of SO(3): q and -q give the same rotation.)
fn same_rotation(a: &[f64; 4], b: &[f64; 4]) -> bool {
    let tol = 1e-6;
    let pos = (0..4).all(|i| (a[i] - b[i]).abs() < tol);
    let neg = (0..4).all(|i| (a[i] + b[i]).abs() < tol);
    pos || neg
}

fn check(label: &str, circuit: &str, result: &[f64; 4], want: &[f64; 4], want_name: &str) {
    let ok = same_rotation(result, want);
    // Print the canonical representative (W ≥ 0 hemisphere).
    let c = if result[0] < 0.0 {
        [-result[0], -result[1], -result[2], -result[3]]
    } else {
        *result
    };
    let mark = if ok { "✓" } else { "✗" };
    println!("  {label}");
    println!("    {circuit}");
    println!("    = {}  σ = {:.4}  →  {}  {mark}", qfmt(&c), sigma(result), want_name);
    println!();
}

fn main() {
    // ── The gate dictionary ──────────────────────────────────────────────
    let x_gate = [0.0_f64, 1.0, 0.0, 0.0];
    let y_gate = [0.0_f64, 0.0, 1.0, 0.0];
    let z_gate = [0.0_f64, 0.0, 0.0, 1.0];
    let h_gate = [0.0_f64, FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2];
    let s_gate = [FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2];
    let t_gate = [(PI / 8.0).cos(), 0.0, 0.0, (PI / 8.0).sin()];
    let alive  = [FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0, 0.0];

    header("EXPERIMENT · THE CARRIER LANGUAGE IS SU(2)");
    println!();
    println!("  A quantum computer runs programs by rotating qubits.");
    println!("  Each rotation is an element of the group SU(2) — the set of all");
    println!("  2×2 unitary matrices with determinant 1. A 'quantum gate' is");
    println!("  simply a name for a specific element of SU(2).");
    println!();
    println!("  Unit quaternions under Hamilton product are exactly SU(2).");
    println!("  Not approximately. Not analogously. The same group.");
    println!();
    println!("  Every carrier in this system is a unit quaternion.");
    println!("  Therefore every carrier is a quantum gate.");
    println!("  compose(A, B) is gate composition.");
    println!("  sigma(q) is half the principal rotation angle.");
    println!();
    println!("  The chain of exact equalities:");
    println!();
    println!("    unit quaternion  =  SU(2) element  =  single-qubit gate  =  carrier");

    // ── The gate dictionary ──────────────────────────────────────────────
    header("THE GATE DICTIONARY");
    println!();
    println!("  A rotation by angle θ around a unit axis (nx, ny, nz) is the quaternion:");
    println!("    [cos(θ/2),  nx·sin(θ/2),  ny·sin(θ/2),  nz·sin(θ/2)]");
    println!();
    println!("  sigma(q) = acos(|W|) = θ/2 on the physical rotation branch.");
    println!("  So the principal rotation angle is 2·sigma(q).");
    println!();
    println!("  {:>16}  {:>40}  {:>6}  {:>8}  {}",
        "Gate", "Carrier  [W, X, Y, Z]", "σ", "2σ = θ", "What it does");
    println!("  {}", "─".repeat(90));

    let entries: &[(&str, &[f64; 4], &str)] = &[
        ("IDENTITY",     &IDENTITY, "zero rotation — identity gate"),
        ("X  (Pauli)",   &x_gate,   "180° flip around x-axis"),
        ("Y  (Pauli)",   &y_gate,   "180° flip around y-axis"),
        ("Z  (Pauli)",   &z_gate,   "180° flip around z-axis"),
        ("H  (Hadamard)",&h_gate,   "180° flip around the (x+z)/√2 axis"),
        ("S  (√Z)",      &s_gate,   "90° rotation around z-axis"),
        ("T  (⁴√Z)",     &t_gate,   "45° rotation around z-axis"),
        ("ALIVE  (SX)",  &alive,    "90° rotation around x-axis  ← memory threshold"),
    ];

    for (name, q, desc) in entries {
        let s = sigma(q);
        println!("  {:>16}  {}  {:>6.4}  {:>6.1}°  {}",
            name, qfmt(q), s, s.to_degrees() * 2.0, desc);
    }

    println!();
    println!("  The X, Y, Z, H gates all have σ = π/2 — they are 180° half-turns,");
    println!("  sitting on the great sphere where W = 0, maximum distance from identity.");
    println!();
    println!("  S and ALIVE have σ = π/4 — they are 90° quarter-turns, sitting on the");
    println!("  Hopf equator of S³. This is the balance locus (SIGMA_BALANCE = π/4 = {:.4}).",
        SIGMA_BALANCE);
    println!("  T has σ = π/8 — a 45° rotation, halfway between identity and the equator.");

    // ── What sigma() measures ────────────────────────────────────────────
    header("WHAT sigma() MEASURES");
    println!();
    println!("  sigma(q) is the rotation metric used by the system.");
    println!("  It computes arccos(|W|), so q and -q have the same sigma.");
    println!("  That means sigma() collapses the SU(2) double cover to the");
    println!("  physical SO(3) rotation distance: σ = θ/2 on the principal branch.");
    println!();
    println!("  Reading the collapsed rotation scale:");
    println!();
    println!("  σ =  0      θ = 0°    IDENTITY — do nothing");
    println!("  σ =  π/8    θ = 45°   T gate — octant step");
    println!("  σ =  π/4    θ = 90°   S, ALIVE — quarter-turn   ← BKT threshold / Hopf equator");
    println!("  σ =  π/2    θ = 180°  X, Y, Z, H — half-turn    ← W = 0 sphere");
    println!("  σ =  0      θ = 360°  −IDENTITY in raw SU(2), same physical rotation as identity");
    println!();
    println!("  The BKT memory threshold selects entries whose coupling t ≥ {BKT_THRESHOLD}.");
    println!("  The Hopf balance condition in the Riemann zeros experiment is σ(Q) = π/4.");
    println!("  Both derive from the same geometric locus on S³.");

    // ── Circuit identities ───────────────────────────────────────────────
    header("QUANTUM CIRCUIT IDENTITIES, VERIFIED BY HAMILTON PRODUCT");
    println!();
    println!("  Quantum computing has a set of well-known algebraic identities.");
    println!("  We verify them here using only compose() — no matrices, no complex");
    println!("  numbers, no quantum simulator. Pure quaternion multiplication.");
    println!();
    println!("  Note: SU(2) is a double cover of SO(3). Quaternions q and −q represent");
    println!("  the same physical rotation. 'Same rotation' below means equal up to sign.");
    println!();

    // 1. T·T = S
    check(
        "1.  Two T gates = one S gate.",
        "T · T",
        &compose(&t_gate, &t_gate),
        &s_gate,
        "S gate",
    );

    // 2. S·S = Z
    check(
        "2.  Two S gates = Z.  Doubling 90° around z gives 180° around z.",
        "S · S",
        &compose(&s_gate, &s_gate),
        &z_gate,
        "Z gate",
    );

    // 3. Z·Z = I  (as rotation)
    check(
        "3.  Z · Z = identity rotation.  Two 180° flips cancel.",
        "Z · Z",
        &compose(&z_gate, &z_gate),
        &IDENTITY,
        "IDENTITY  (W < 0 means −I in SU(2), same rotation — see note below)",
    );

    // 4. ALIVE² = X
    check(
        "4.  ALIVE · ALIVE = X gate.  Two 90° x-rotations = one 180° x-flip.",
        "ALIVE · ALIVE",
        &compose(&alive, &alive),
        &x_gate,
        "X gate",
    );

    // 5. H · X · H = Z
    check(
        "5.  Hadamard conjugates X to Z.  The standard basis-swap identity.",
        "H · X · H",
        &compose(&compose(&h_gate, &x_gate), &h_gate),
        &z_gate,
        "Z gate",
    );

    // 6. H · Z · H = X
    check(
        "6.  Hadamard conjugates Z to X.  The reverse direction.",
        "H · Z · H",
        &compose(&compose(&h_gate, &z_gate), &h_gate),
        &x_gate,
        "X gate",
    );

    // 7. X · Y · X = −Y (anticommutation)
    check(
        "7.  X and Y anticommute: X · Y · X = −Y (same rotation as Y).",
        "X · Y · X",
        &compose(&compose(&x_gate, &y_gate), &x_gate),
        &y_gate,
        "Y gate",
    );

    // 8. Any gate composed with its inverse = identity
    let alive_inv = inverse(&alive);
    check(
        "8.  Every gate composed with its inverse = identity.",
        "ALIVE · inverse(ALIVE)",
        &compose(&alive, &alive_inv),
        &IDENTITY,
        "IDENTITY",
    );

    // ── The double cover ─────────────────────────────────────────────────
    header("THE DOUBLE COVER: WHY A 360° ROTATION IS NOT THE IDENTITY");
    println!();
    println!("  Physically, rotating an object by 360° returns it to its starting state.");
    println!("  But in SU(2) — and in this system — a 360° rotation gives −IDENTITY,");
    println!("  not +IDENTITY.");
    println!();
    println!("  You need 720° (two full turns) to return to the identity element.");
    println!("  This is called the double cover: SU(2) wraps around the rotation group");
    println!("  SO(3) exactly twice. Every physical rotation has two SU(2) representatives,");
    println!("  q and −q, antipodal on S³.");
    println!();

    let zz  = compose(&z_gate, &z_gate);
    let z4  = compose(&zz, &zz);
    println!("  Z · Z     = {}  W = {:+.4}  (360° — W component flips to −1)",
        qfmt(&zz), zz[0]);
    println!("  Z·Z · Z·Z = {}  W = {:+.4}  (720° — W component returns to +1)",
        qfmt(&z4), z4[0]);
    println!();
    println!("  sigma() uses |W|, so both register as σ = 0 (the same rotation distance).");
    println!("  The distinction lives in the sign of W: +1 is IDENTITY, −1 is its antipode.");
    println!("  The system's sigma metric already collapses this — it treats the two covers");
    println!("  as equivalent. The raw quaternion preserves the SU(2) distinction.");
    println!();
    println!("  In quantum mechanics this matters because a spin-1/2 particle acquires");
    println!("  a factor of −1 under a 360° rotation (measurable via interference).");
    println!("  The system inherits this structure from the quaternion arithmetic.");

    // ── ALIVE = SX gate ───────────────────────────────────────────────────
    header("ALIVE IS THE SX GATE");
    println!();
    println!("  ALIVE = [{:.4}, {:.4}, 0, 0]", FRAC_1_SQRT_2, FRAC_1_SQRT_2);
    println!("  σ(ALIVE) = {:.6}  (= π/4 = {:.6})", sigma(&alive), PI / 4.0);
    println!();
    println!("  In quantum computing this is the SX gate, also written √X.");
    println!("  IBM uses SX as a native physical gate on their quantum processors —");
    println!("  it is one of the primitive operations implemented directly in hardware.");
    println!();
    println!("  ALIVE is the carrier used to represent a living memory in the genome.");
    println!("  A memory 'at the BKT threshold' has ZREAD coupling ≈ {BKT_THRESHOLD},");
    println!("  and its canonical geometric address is at σ = π/4 — exactly ALIVE's location.");
    println!();
    println!("  Two ALIVE carriers composed = X gate (180° flip):");
    let alive_sq = compose(&alive, &alive);
    println!("    {}  σ = {:.4}", qfmt(&alive_sq), sigma(&alive_sq));
    println!();
    println!("  The memory consolidation boundary and a hardware quantum gate");
    println!("  are the same point on S³.");

    // ── The Hopf equator ─────────────────────────────────────────────────
    header("THE HOPF EQUATOR = ONE LOCUS, THREE EXPERIMENTS");
    println!();
    println!("  The Hopf equator of S³ is the great 2-sphere at σ = π/4.");
    println!("  In gate language: the set of all 90° rotations, around every possible axis.");
    println!();

    let ry_90 = [FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2, 0.0];
    println!("  Some gates that live on the Hopf equator (σ = {:.4}):", SIGMA_BALANCE);
    println!("    S  (z-axis)     {}  σ = {:.4}", qfmt(&s_gate), sigma(&s_gate));
    println!("    ALIVE (x-axis)  {}  σ = {:.4}", qfmt(&alive),  sigma(&alive));
    println!("    Ry(90°)(y-axis) {}  σ = {:.4}", qfmt(&ry_90),  sigma(&ry_90));
    println!();
    println!("  All three lie at σ = π/4. Same distance from identity. Same sphere.");
    println!();
    println!("  The Hopf equator appears in three separate experiments in this codebase:");
    println!();
    println!("  Exp 2 — BKT Phase Transition:");
    println!("    The memory consolidation threshold derives from the S³ Berezinskii–");
    println!("    Kosterlitz–Thouless critical coupling. Memories at the threshold have");
    println!("    geometric addresses on — or near — the Hopf equator.");
    println!();
    println!("  Exp 3 — Riemann Zeros:");
    println!("    The geometric zero condition is σ(Q(s)) = π/4 exactly. The Euler");
    println!("    product running Q(s) crosses the Hopf equator at each Riemann zero.");
    println!("    The balance error |σ(Q) − π/4| is what was minimized to find 50/50 zeros.");
    println!();
    println!("  Exp 4 — Associative Memory:");
    println!("    ALIVE = [1/√2, 1/√2, 0, 0] is the canonical 'alive' carrier,");
    println!("    used as the initial state for every stored memory pair.");
    println!("    It lives on the Hopf equator.");
    println!();
    println!("  These are not three separate uses of the number π/4.");
    println!("  They are three observations of the same geometric object:");
    println!("  the unique 2-sphere inside S³ equidistant from every identity.");

    // ── Summary ──────────────────────────────────────────────────────────
    header("SUMMARY");
    println!();
    println!("  The carrier language is a quantum gate language.");
    println!("  Not inspired by it. Not analogous to it. The same mathematical object.");
    println!();
    println!("  What quantum computing and this system share (exactly):");
    println!("    — The group SU(2) as substrate");
    println!("    — Noncommutative composition as the fundamental operation");
    println!("    — Phase accumulation along a path on S³");
    println!("    — The Hopf equator as a distinguished locus");
    println!("    — The double cover structure (q and −q as the same rotation)");
    println!();
    println!("  Where they differ:");
    println!("    — A quantum computer measures, collapsing to a classical outcome.");
    println!("      This system RESONATEs, finding the nearest genome address.");
    println!("    — A quantum computer decomposes programs into a fixed gate set.");
    println!("      This system builds carriers from data, then reads them back.");
    println!("    — A quantum computer uses superposition across the full Hilbert space.");
    println!("      This system uses composition along a sequence of carriers.");
    println!();
    println!("  The substrate is the same. The architecture built on top is different.");
    println!("  The S³ geometric computer is not a quantum computer.");
    println!("  It runs on the same geometry.");
}
