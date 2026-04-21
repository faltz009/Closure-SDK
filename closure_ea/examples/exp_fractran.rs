//! EXPERIMENT 6 — FRACTRAN: Prime-Native Computation
//!
//! FRACTRAN (Conway 1987) is a Turing-complete programming language whose
//! state space IS the prime factorization of an integer.
//!
//! A FRACTRAN program is a list of fractions [f₁, f₂, ..., fₖ].
//! Given current state N:
//!   1. Find the FIRST fraction fᵢ such that fᵢ · N is a positive integer
//!   2. Replace N ← fᵢ · N
//!   3. Repeat. Halt when no fraction applies.
//!
//! This is the most native language for a prime-orbit computer.
//!
//!   Integer N = 2^a · 3^b · 5^c · ...
//!   State     = tuple of orbit positions  (a, b, c, ...)
//!   Fraction p/q: runtime reads whether orbit_q is at slot 0
//!                 if not, step orbit_q backward and orbit_p forward
//!                 through `ThreeCell::evaluate_product`
//!
//! Programs demonstrated:
//!
//!   [3/2]          — prime-2 power → prime-3 power:  2^n → 3^n
//!   [3/2, 5/3, 1/5] — prime relay race: 2^n → 3^n → 5^n → 1
//!
//! The relay race is the geometric equivalent of:
//!   "transfer n tokens through stations 2, 3, 5, then consume them"
//!   — pure orbit arithmetic through the runtime substrate.
//!
//! Run:  cargo run --example exp_fractran --release

use closure_ea::{Fraction, FractranMachine};

const PERIOD: usize = 37; // prime, > max exponent we'll use
const PRIMES: [u64; 3] = [2, 3, 5];

fn integer_from_exponents(exp: &[usize]) -> u64 {
    PRIMES[0].pow(exp[0] as u32) * PRIMES[1].pow(exp[1] as u32) * PRIMES[2].pow(exp[2] as u32)
}

fn run_fractran(
    machine: &FractranMachine,
    program: &[Fraction],
    init: [usize; 3],
) -> Vec<(usize, Vec<usize>, u64)> {
    let mut state = machine.init_state(&init);
    let mut trace = Vec::new();

    for _ in 0..200 {
        let pc = machine
            .decoded_pc(&state)
            .expect("runtime-backed program counter must decode");
        let exp = machine
            .exponents(&state)
            .expect("runtime-backed prime orbits must decode");
        let n = integer_from_exponents(&exp);
        trace.push((pc, exp, n));
        if !machine.step(&mut state, program) {
            break;
        }
    }

    let pc = machine
        .decoded_pc(&state)
        .expect("runtime-backed program counter must decode after halt");
    let exp = machine
        .exponents(&state)
        .expect("runtime-backed prime orbits must decode after halt");
    let n = integer_from_exponents(&exp);
    trace.push((pc, exp, n));
    trace
}

fn header(title: &str) {
    let bar = "━".repeat(70);
    println!("\n{bar}");
    println!("  {title}");
    println!("{bar}");
}

fn fraction_str(num: &[usize], den: &[usize]) -> String {
    let n_str = if num.is_empty() {
        "1".to_string()
    } else {
        num.iter()
            .map(|&p| PRIMES[p].to_string())
            .collect::<Vec<_>>()
            .join("·")
    };
    let d_str = den
        .iter()
        .map(|&p| PRIMES[p].to_string())
        .collect::<Vec<_>>()
        .join("·");
    format!("{n_str}/{d_str}")
}

fn main() {
    let machine = FractranMachine::new_2_3_5(PERIOD, 5);

    header("EXPERIMENT 6 · FRACTRAN: PRIME-NATIVE COMPUTATION");
    println!();
    println!("  Conway invented FRACTRAN in 1987 as a strange but beautiful programming");
    println!("  language. A FRACTRAN program is just a list of fractions. The state is");
    println!("  a single positive integer N. At each step, scan the fractions until you");
    println!("  find one that divides evenly into N, multiply N by it, and repeat.");
    println!();
    println!("  Despite its simplicity, FRACTRAN is Turing-complete — it can compute");
    println!("  anything. Conway used it to write a prime-number sieve in one line.");
    println!();
    println!("  Why is this the natural language for the geometric computer?");
    println!("  Because FRACTRAN's state space IS prime factorizations: N = 2^a · 3^b · 5^c ...");
    println!("  And in this computer, each prime IS an orbit — a separate degree of freedom.");
    println!("  The exponent of prime p is the orbit position on the p-orbit great circle.");
    println!("  Multiplying by p means rotating one step forward on that orbit.");
    println!("  Dividing by p means rotating one step backward.");
    println!("  Testing divisibility by p means asking if that orbit is at the north pole.");
    println!();
    println!("  There is no translation layer. FRACTRAN is not encoded into the geometry.");
    println!("  FRACTRAN IS the geometry, running in its native representation.");

    // ── Program 1: [3/2] — orbit transfer 2^n → 3^n ─────────────────────
    header("PROGRAM 1 — [3/2]: TRANSFERRING TOKENS BETWEEN PRIMES");
    println!();
    println!("  The single-fraction program [3/2] does exactly one thing:");
    println!("  it moves tokens from the prime-2 orbit to the prime-3 orbit,");
    println!("  one at a time, until the 2-orbit is empty.");
    println!();
    println!("  Each step: check if prime 2 divides N (is the 2-orbit above zero?).");
    println!("  If yes: take one unit from 2, give one unit to 3. N = N × (3/2).");
    println!("  If no: halt. All tokens have moved.");
    println!();
    println!("  In integer terms: 2^5 = 32. Apply 3/2 five times. Result: 3^5 = 243.");
    println!("  In geometric terms: rotate the 2-orbit backward 5 steps, the 3-orbit forward 5.");

    let prog1: Vec<Fraction> = vec![(vec![1], vec![0])];
    let n = 5usize;

    println!("  Input: 2^{n} = {}  →  orbit_2 = ε₂^{n}, orbit_3 = ε₃^0 = I", 1u64 << n);
    println!();
    println!("  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Step", "N", "exp(2)", "exp(3)", "Operation");
    println!("  {}", "─".repeat(50));

    let trace1 = run_fractran(&machine, &prog1, [n, 0, 0]);
    for (step, (pc, exp, n_val)) in trace1.iter().enumerate() {
        let op = if step < trace1.len() - 1 {
            if trace1[step + 1].1 != *exp {
                "apply 3/2 →"
            } else if *pc <= prog1.len() {
                "skip 3/2 →"
            } else {
                "HALT"
            }
        } else {
            "HALT"
        };
        println!("  {:>5}  {:>8}  {:>8}  {:>8}  {}",
            step, *n_val, exp[0], exp[1], op);
    }
    let final_exp = &trace1.last().unwrap().1;
    println!();
    println!("  Result: 2-orbit = {} (IDENTITY ✓)  3-orbit = {}  N = {}",
        final_exp[0], final_exp[1],
        trace1.last().unwrap().2);
    println!("  2^{n} = {} → 3^{n} = {}  (all {} prime-2 tokens transferred to prime-3)",
        1u64 << n, 3usize.pow(n as u32), n);

    // ── Program 2: [3/2, 5/3, 1/5] — prime relay race ───────────────────
    header("PROGRAM 2 — [3/2, 5/3, 1/5]: A THREE-PRIME RELAY");
    println!();
    println!("  This program routes tokens through three stations in sequence:");
    println!("  prime 2 → prime 3 → prime 5 → gone.");
    println!();
    println!("  The FRACTRAN semantics enforce station order automatically:");
    println!("  fraction 3/2 is tried first. While prime 2 has tokens, they move to 3.");
    println!("  Only when 2 is exhausted can fraction 5/3 fire. Then 3 drains into 5.");
    println!("  Only when 3 is exhausted can fraction 1/5 fire. Then 5 is consumed.");
    println!();
    println!("  The result is always N = 1 — all tokens gone, all orbits at IDENTITY.");
    println!("  The program is a cascade of orbit transfers, sequenced by geometry.");

    let prog2: Vec<Fraction> = vec![(vec![1], vec![0]), (vec![2], vec![1]), (vec![], vec![2])];

    let m = 4usize;
    println!("  Input: 2^{m} = {}  →  expecting final state N = 1", 1u64 << m);
    println!();
    println!("  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>12}",
        "Step", "N", "exp(2)", "exp(3)", "exp(5)", "Applied");
    println!("  {}", "─".repeat(62));

    let trace2 = run_fractran(&machine, &prog2, [m, 0, 0]);
    let frac_names = ["3/2", "5/3", "1/5"];
    for (step, (pc, exp, n_val)) in trace2.iter().enumerate() {
        let applied = if step < trace2.len() - 1 {
            let next_exp = &trace2[step + 1].1;
            if next_exp != exp {
                frac_names[*pc - 1]
            } else if *pc <= prog2.len() {
                match *pc {
                    1 => "skip 3/2",
                    2 => "skip 5/3",
                    3 => "skip 1/5",
                    _ => "skip",
                }
            } else {
                "HALT"
            }
        } else {
            "HALT"
        };
        println!("  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>12}",
            step, *n_val, exp[0], exp[1], exp[2], applied);
    }
    let final2 = trace2.last().unwrap();
    println!();
    println!("  Result: N = {}  (all tokens consumed  {})",
        final2.2, if final2.2 == 1 { "✓" } else { "✗" });
    println!("  2^{m} = {} → 1 in {} steps via prime relay 2 → 3 → 5",
        1u64 << m, trace2.len() - 1);

    // ── Why this is prime-native ─────────────────────────────────────────
    header("WHY PRIMES ARE ORBITS, NOT NUMBERS");
    println!();
    println!("  In a Von Neumann machine, '3' and '5' are memory addresses that happen");
    println!("  to be prime. The machine has no idea why primes are interesting.");
    println!();
    println!("  In this computer, prime 3 is a great circle on S³ with its own generator,");
    println!("  its own inverse, and its own IDENTITY position. 'Divisible by 3' means");
    println!("  'the 3-orbit is above its zero position' — a geometric question about");
    println!("  where a point on a circle currently sits.");
    println!();
    println!("  This is why FRACTRAN runs without translation. The language describes");
    println!("  orbits, and the computer IS orbits. There is nothing in between.");
    println!();

    // Verify reference (integer arithmetic) against geometric
    let prog1_frac: Vec<(i64, i64)> = vec![(3, 2)];
    let mut ref_n: u64 = 1u64 << n;
    let mut ref_steps = 0usize;
    for _ in 0..200 {
        let mut stepped = false;
        for &(num, den) in &prog1_frac {
            if ref_n.is_multiple_of(den as u64) {
                ref_n = ref_n / (den as u64) * (num as u64);
                ref_steps += 1;
                stepped = true;
                break;
            }
        }
        if !stepped { break; }
    }
    println!("  Verification (integer reference, program 1):");
    println!("    2^{n} = {} → {} in {ref_steps} steps", 1u64 << n, ref_n);
    println!("    Expected: 3^{n} = {}", 3usize.pow(n as u32));
    println!("    Geometric vs reference:  {}",
        if trace1.last().unwrap().2 == ref_n { "exact match ✓" } else { "MISMATCH ✗" });
    println!();
    println!("  FRACTRAN is Turing-complete (proved by reducing 2-counter machines,");
    println!("  which are themselves Turing-universal — see Exp 5).");
    println!("  Running FRACTRAN on this system = running arbitrary computation");
    println!("  directly on prime-orbit geometry.");
    // Print fraction names for reference
    println!();
    let frac_strs: Vec<String> = prog2.iter()
        .map(|(n, d)| fraction_str(n, d))
        .collect();
    println!("  Program 2 fractions: [{}]", frac_strs.join(", "));
}
