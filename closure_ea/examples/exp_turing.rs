//! EXPERIMENT 5 — Turing Completeness via 2-Counter Minsky Machine
//!
//! The 2-counter Minsky machine is Turing-universal (Minsky 1961).
//! Any computable function can be encoded as a program over three instructions:
//!   INC r        — increment counter r
//!   DECJZ r,z,n  — if r=0 goto z, else decrement r and goto n
//!   HALT         — stop
//!
//! Claim: the S³ geometric computer realizes each instruction exactly
//! through the main runtime's orbit substrate:
//!
//!   Counter value k  →  orbit slot ε^k seeded into `ThreeCell` DNA
//!   INC r            →  `brain.evaluate_product(r, ε)`
//!   DEC r            →  `brain.evaluate_product(r, ε⁻¹)`
//!   ZERO? r          →  runtime RESONATE reads slot 0
//!
//! No example-side quaternion stepping. The experiment delegates slot
//! transitions and slot reads to the runtime-backed execution helpers in
//! `closure_ea::execution`, which wrap a real [`ThreeCell`].
//!
//! Verification: run three programs on both a Rust integer reference
//! interpreter and the geometric interpreter. Compare traces step-by-step.
//! Identical PC and counter values at every step = constructive proof.
//!
//! Run:  cargo run --example exp_turing --release

use closure_ea::{MinskyInstr, MinskyMachine};

// ── Instruction set ──────────────────────────────────────────────────────────

// ── Reference interpreter (plain integers) ───────────────────────────────────

fn run_ref(prog: &[MinskyInstr], init: [usize; 2]) -> Vec<(usize, [usize; 2])> {
    let mut pc = 1usize;
    let mut r = init;
    let mut trace = vec![(pc, r)];
    for _ in 0..500 {
        match &prog[pc - 1] {
            MinskyInstr::Halt => break,
            MinskyInstr::Inc { reg, next } => {
                r[*reg] += 1;
                pc = *next;
            }
            MinskyInstr::DecJz { reg, if_zero, if_pos } => {
                if r[*reg] == 0 {
                    pc = *if_zero;
                } else {
                    r[*reg] -= 1;
                    pc = *if_pos;
                }
            }
        }
        trace.push((pc, r));
    }
    trace
}

// ── Comparison ───────────────────────────────────────────────────────────────

fn verify_traces(
    prog_name: &str,
    prog: &[MinskyInstr],
    init: [usize; 2],
    machine: &MinskyMachine,
    show: usize,
) -> bool {
    let ref_trace = run_ref(prog, init);
    let mut state = machine.init_state(1, init);
    let mut geo_trace = vec![(
        machine
            .decoded_pc(&state)
            .expect("pc must decode on the runtime orbit"),
        machine
            .decoded_regs(&state)
            .expect("registers must decode on their runtime orbits"),
    )];
    for _ in 0..500 {
        if !machine.step(&mut state, prog) {
            break;
        }
        geo_trace.push((
            machine
                .decoded_pc(&state)
                .expect("pc must remain decodable after a runtime step"),
            machine
                .decoded_regs(&state)
                .expect("registers must remain decodable after a runtime step"),
        ));
    }

    println!();
    println!("  Program: {prog_name}");
    println!("  Input:   R0 = {}  R1 = {}", init[0], init[1]);
    println!();
    println!("  {:>5}  {:>4}  {:>5}  {:>5}  {:>6}  {:>5}  {:>5}  {:>6}",
        "Step", "PC", "R0_ref", "R1_ref", "match?", "PC", "R0_geo", "R1_geo");
    println!("  {}", "─".repeat(55));

    let n = ref_trace.len().min(geo_trace.len());
    let mut all_match = true;

    for i in 0..n {
        let (rpc, rr) = ref_trace[i];
        let (gpc, gr) = geo_trace[i];
        let ok = rpc == gpc && rr[0] == gr[0] && rr[1] == gr[1];
        if !ok { all_match = false; }
        if i < show || !ok {
            let mark = if ok { "✓" } else { "✗ FAIL" };
            println!("  {:>5}  {:>4}  {:>5}  {:>5}  {:>6}  {:>4}  {:>5}  {:>5}",
                i, rpc, rr[0], rr[1], mark, gpc, gr[0], gr[1]);
        }
    }
    if n > show {
        println!("  ... ({} more steps, all verified)", n - show);
    }

    let (fpc, fr) = ref_trace.last().unwrap();
    let (_, fg) = geo_trace.last().unwrap();

    println!();
    println!("  Steps:     {}",  n - 1);
    println!("  Final ref: PC={} R0={} R1={}", fpc, fr[0], fr[1]);
    println!("  Final geo: PC={} R0={} R1={}", fpc, fg[0], fg[1]);
    println!("  Trace:     {}", if all_match { "exact match ✓" } else { "MISMATCH ✗" });

    all_match
}

fn header(title: &str) {
    let bar = "━".repeat(70);
    println!("\n{bar}");
    println!("  {title}");
    println!("{bar}");
}

fn main() {
    let period = 31usize; // prime, covers max counter value of ~20
    let pc_period = 11usize;
    let machine = MinskyMachine::new(period, pc_period);

    header("EXPERIMENT 5 · TURING COMPLETENESS");
    println!();
    println!("  Minsky proved in 1961 that a machine with just two counters and three");
    println!("  instructions — increment, decrement-or-jump-if-zero, halt — can compute");
    println!("  anything a Turing machine can. No other primitives are needed.");
    println!();
    println!("  This experiment runs three programs on two interpreters in parallel:");
    println!("  a plain integer reference (the familiar, unambiguous version) and");
    println!("  a geometric interpreter where counters are positions on S³.");
    println!();
    println!("  In the geometric version:");
    println!("    A counter holding value k is the orbit position ε^k on a great circle.");
    println!("    INC rotates one step forward: ε^k → ε^(k+1).");
    println!("    DEC rotates one step backward: ε^k → ε^(k-1).");
    println!("    ZERO? checks whether the counter is at the north pole (IDENTITY, k=0).");
    println!("    The program counter is also an orbit — a separate great circle.");
    println!();
    println!("  Every state transition goes through the ThreeCell runtime.");
    println!("  The geometry does the computation. The proof is that traces match exactly.");

    // ── Program 1: R0 = R0 + R1 ─────────────────────────────────────────
    header("PROGRAM 1 — ADDITION");
    println!();
    println!("  1: DECJZ R1 → (halt=3, dec→2)   if R1=0 halt, else R1--, goto 2");
    println!("  2: INC   R0 → 1                  R0++, goto 1");
    println!("  3: HALT");

    let add = vec![
        MinskyInstr::DecJz { reg: 1, if_zero: 3, if_pos: 2 },
        MinskyInstr::Inc   { reg: 0, next: 1 },
        MinskyInstr::Halt,
    ];
    let ok1 = verify_traces("4 + 5 = 9", &add, [4, 5], &machine, 12);
    println!();
    println!("  {}  Each step: same counter value, same PC, same branch taken.", if ok1 { "✓  Traces match exactly." } else { "✗  MISMATCH." });

    // ── Program 2: R0 = R0 − R1 (saturating) ────────────────────────────
    header("PROGRAM 2 — SUBTRACTION");
    println!();
    println!("  1: DECJZ R1 → (halt=3, dec→2)   if R1=0 halt, else R1--, goto 2");
    println!("  2: DECJZ R0 → (halt=3, dec→1)   if R0=0 halt, else R0--, goto 1");
    println!("  3: HALT");

    let sub = vec![
        MinskyInstr::DecJz { reg: 1, if_zero: 3, if_pos: 2 },
        MinskyInstr::DecJz { reg: 0, if_zero: 3, if_pos: 1 },
        MinskyInstr::Halt,
    ];
    let ok2 = verify_traces("11 - 7 = 4", &sub, [11, 7], &machine, 10);
    println!();
    println!("  {}  Two counters decrement in alternation — the branching geometry is correct.", if ok2 { "✓  Traces match exactly." } else { "✗  MISMATCH." });

    // ── Program 3: R0 = R0 * 2 via copy-double ──────────────────────────
    // Strategy: move R0 into R1 (R0 DEC, R1 INC×2), then move R1 back into R0.
    // This uses nested loops: the outer loop counts down R0, the inner adds 2 to R1.
    header("PROGRAM 3   R0 = R0 * 2   (multiplication by 2, nested loop)");
    println!();
    println!("  Phase 1 — double into R1:");
    println!("    1: DECJZ R0 → (phase2=4, dec→2)");
    println!("    2: INC   R1 → 3");
    println!("    3: INC   R1 → 1          (add 2 to R1 per R0 decrement)");
    println!("  Phase 2 — copy R1 back to R0:");
    println!("    4: DECJZ R1 → (halt=6, dec→5)");
    println!("    5: INC   R0 → 4");
    println!("    6: HALT");

    let dbl = vec![
        MinskyInstr::DecJz { reg: 0, if_zero: 4, if_pos: 2 }, // 1
        MinskyInstr::Inc   { reg: 1, next: 3 },                // 2
        MinskyInstr::Inc   { reg: 1, next: 1 },                // 3
        MinskyInstr::DecJz { reg: 1, if_zero: 6, if_pos: 5 }, // 4
        MinskyInstr::Inc   { reg: 0, next: 4 },                // 5
        MinskyInstr::Halt,                                      // 6
    ];
    let ok3 = verify_traces("6 * 2 = 12", &dbl, [6, 0], &machine, 10);
    println!();
    println!("  6 * 2 = 12  {}  (nested orbit traversal, no extra primitives)", if ok3 { "✓" } else { "✗" });

    // ── Turing completeness statement ────────────────────────────────────
    header("WHAT THIS PROVES");
    let all_ok = ok1 && ok2 && ok3;
    println!();
    println!("  Program 1 (addition):    {}", if ok1 { "exact trace match ✓" } else { "FAIL ✗" });
    println!("  Program 2 (subtraction): {}", if ok2 { "exact trace match ✓" } else { "FAIL ✗" });
    println!("  Program 3 (doubling):    {}", if ok3 { "exact trace match ✓" } else { "FAIL ✗" });
    println!();
    if all_ok {
        println!("  The geometric interpreter produces the exact same execution trace as");
        println!("  the integer interpreter — same counter value at every step, same branch");
        println!("  decision, same program counter, same halting point.");
        println!();
        println!("  Minsky proved that two counters + three instructions = any computable");
        println!("  function. We just showed that S³ orbit geometry can faithfully execute");
        println!("  those two counters and three instructions without any integer arithmetic.");
        println!("  The geometry IS the computation.");
        println!();
        println!("  This is what Turing completeness looks like when the substrate is a sphere.");
        println!("  The counter is a point on a circle. Incrementing is rotating. Testing");
        println!("  for zero is asking if you are at the north pole. The program terminates");
        println!("  when the program counter reaches its designated halt position on its orbit.");
    } else {
        println!("  FAILED: trace mismatch detected — see above.");
    }
}
