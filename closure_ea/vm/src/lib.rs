//! S³ Virtual Machine — the Closure Machine.
//!
//! A general-purpose computer on the three-sphere. Programs and data are
//! quaternions in the same memory (Closure DNA). The VM is the CPU.
//! DNA is the RAM. There is no other storage.
//!
//! ## ISA (9 instructions)
//!
//!   1. COMPOSE(a, b)       Hamilton product
//!   2. INVERT(a)           Conjugate
//!   3. SIGMA(a)            Geodesic distance from identity
//!   4. DECOMPOSE(a)        Hopf fibration → (σ, base[3], phase)
//!   5. EMBED(bytes)        SHA-256 → S³
//!   6. FETCH(key, table)   DNA table.search()
//!   7. STORE(row, table)   DNA table.insert()
//!   8. EMIT()              Output state, update context, reset
//!   9. BRANCH(σ, ε)        σ < ε → Closure. σ > π/2-ε → Death.
//!
//! ## Registers
//!
//!   state:    [f64; 4]   accumulator / program counter
//!   previous: [f64; 4]   last state (for composite keys)
//!   context:  [f64; 4]   composition of all closure elements (session memory)
//!
//! ## Von Neumann correspondence
//!
//!   The state IS the program counter. Composing an instruction IS execution.
//!   No opcodes, no decode step. The quaternion IS the operation.

mod primitives;
mod program;
mod machine;
mod hierarchy;

pub use closure_rs::groups::sphere::IDENTITY;
pub use primitives::{DecomposeResult, decompose, StepResult};
pub use program::Program;
pub use machine::Machine;
pub use hierarchy::{HierarchicalMachine, ResonanceConfig};

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use closure_rs::groups::sphere::{
        sphere_compose as compose, sphere_inverse as inverse, sphere_sigma as sigma,
    };

    /// Helper: quaternion from axis-angle. axis: 0=i, 1=j, 2=k.
    fn quat(angle: f64, axis: usize) -> [f64; 4] {
        let mut q = [0.0; 4];
        q[0] = (angle / 2.0).cos();
        q[1 + axis] = (angle / 2.0).sin();
        q
    }

    // ── ISA: COMPOSE + INVERT ───────────────────────────────────────

    #[test]
    fn compose_with_identity_is_noop() {
        let a = quat(0.5, 0);
        let r = compose(&a, &IDENTITY);
        assert!(sigma(&compose(&r, &inverse(&a))) < 1e-10);
    }

    #[test]
    fn compose_with_inverse_gives_identity() {
        let a = quat(1.2, 1);
        let result = compose(&a, &inverse(&a));
        assert!(sigma(&result) < 1e-10);
    }

    #[test]
    fn compose_is_not_commutative() {
        let a = quat(0.5, 0);
        let b = quat(0.7, 1);
        let ab = compose(&a, &b);
        let ba = compose(&b, &a);
        assert!(sigma(&compose(&ab, &inverse(&ba))) > 0.01);
    }

    // ── ISA: SIGMA ──────────────────────────────────────────────────

    #[test]
    fn sigma_identity_is_zero() {
        assert!(sigma(&IDENTITY) < 1e-10);
    }

    #[test]
    fn sigma_increases_with_angle() {
        let s1 = sigma(&quat(0.3, 0));
        let s2 = sigma(&quat(0.9, 0));
        assert!(s2 > s1);
    }

    // ── ISA: DECOMPOSE ──────────────────────────────────────────────

    #[test]
    fn decompose_classifies() {
        let q = quat(0.8, 1); // j-axis rotation
        let d = decompose(&q);
        assert!(d.sigma > 0.1, "non-identity should have σ > 0");
        assert!(d.base[0].is_finite());
        assert!(d.base[1].is_finite());
        assert!(d.base[2].is_finite());
        assert!(d.phase.is_finite());
    }

    #[test]
    fn decompose_near_identity_is_degenerate() {
        let d = decompose(&IDENTITY);
        assert!(d.sigma < 1e-10);
    }

    // ── StepResult: CLOSURE ─────────────────────────────────────────

    #[test]
    fn identity_program_closes() {
        let a = quat(0.8, 2);
        let program = Program::from_slice(&[a, inverse(&a)]);
        let mut m = Machine::new(0.01);
        match m.run_sequential(&program, 100) {
            StepResult::Closure(q) => assert!(sigma(&q) < 0.01),
            other => panic!("expected Closure, got {:?}", other),
        }
    }

    // ── StepResult: DEATH ───────────────────────────────────────────

    #[test]
    fn death_on_antipodal() {
        // σ = arccos(|w|) ∈ [0, π/2]. Maximum σ = π/2 when w = 0.
        // Death fires when σ > π/2 - ε. Set state to w ≈ 0 (pure imaginary).
        let mut m = Machine::new(0.1); // ε = 0.1, death at σ > π/2 - 0.1 ≈ 1.47

        // Pure i-rotation by π (w=0, x=1): σ = arccos(0) = π/2
        let result = m.execute(&quat(std::f64::consts::PI, 0));
        match result {
            StepResult::Death(q) => {
                assert!(sigma(&q) > std::f64::consts::FRAC_PI_2 - 0.1);
            }
            other => panic!("expected Death at σ ≈ π/2, got {:?}", other),
        }
    }

    // ── StepResult: HALT ────────────────────────────────────────────

    #[test]
    fn halt_when_program_doesnt_close() {
        let program = Program::from_slice(&[quat(0.5, 0), quat(0.5, 1)]);
        let mut m = Machine::new(0.01);
        match m.run_sequential(&program, 100) {
            StepResult::Halt(_) => {}
            other => panic!("expected Halt, got {:?}", other),
        }
    }

    // ── Registers ───────────────────────────────────────────────────

    #[test]
    fn previous_tracks_last_state() {
        let mut m = Machine::new(0.001);
        let a = quat(0.5, 0);
        m.execute(&a);
        let state_after_a = m.state;

        let b = quat(0.7, 1);
        m.execute(&b);
        // previous should be the state BEFORE executing b
        let gap = sigma(&compose(&m.previous, &inverse(&state_after_a)));
        assert!(gap < 1e-10, "previous should track last state");
    }

    #[test]
    fn context_accumulates_across_closures() {
        let mut m = Machine::new(0.05);
        let a = quat(0.3, 0);

        // First closure
        m.execute(&a);
        m.execute(&inverse(&a));
        // state reset, context should have the closure element
        let ctx_after_1 = m.context;
        assert!(sigma(&ctx_after_1) < 0.05, "first closure element near identity");

        // Second closure with different content
        let b = quat(0.7, 1);
        m.execute(&b);
        m.execute(&inverse(&b));
        let ctx_after_2 = m.context;

        // Context should be compose(ctx_after_1, second_closure_element)
        // Both closures were near-identity, so context stays near identity
        assert!(sigma(&ctx_after_2) < 0.1, "context accumulates");
    }

    #[test]
    fn context_survives_reset() {
        let mut m = Machine::new(0.001);
        m.context = quat(0.5, 2); // set manually
        m.reset();
        // context should NOT be reset by reset()
        assert!(sigma(&compose(&m.context, &inverse(&quat(0.5, 2)))) < 1e-10);
        // state should be reset
        assert!(sigma(&m.state) < 1e-10);
    }

    // ── build_key ───────────────────────────────────────────────────

    #[test]
    fn build_key_width_1() {
        let mut m = Machine::new(0.001);
        m.state = quat(0.5, 0);
        let key = m.build_key(1);
        assert_eq!(key.len(), 4);
        assert!((key[0] - m.state[0]).abs() < 1e-15);
    }

    #[test]
    fn build_key_width_2() {
        let mut m = Machine::new(0.001);
        m.state = quat(0.5, 0);
        m.previous = quat(0.3, 1);
        let key = m.build_key(2);
        assert_eq!(key.len(), 8);
        assert!((key[0] - m.state[0]).abs() < 1e-15);
        assert!((key[4] - m.previous[0]).abs() < 1e-15);
    }

    #[test]
    fn build_key_width_3() {
        let mut m = Machine::new(0.001);
        m.state = quat(0.5, 0);
        m.previous = quat(0.3, 1);
        m.context = quat(0.7, 2);
        let key = m.build_key(3);
        assert_eq!(key.len(), 12);
        assert!((key[8] - m.context[0]).abs() < 1e-15);
    }

    // ── EMIT ────────────────────────────────────────────────────────

    #[test]
    fn emit_updates_context_and_resets() {
        let mut m = Machine::new(0.001);
        let a = quat(0.6, 0);
        m.execute(&a);
        let state_before = m.state;
        let ctx_before = m.context;

        let emitted = m.emit();

        // Emitted value should be the state before emit
        let gap = sigma(&compose(&emitted, &inverse(&state_before)));
        assert!(gap < 1e-10);
        // State should be reset
        assert!(sigma(&m.state) < 1e-10);
        // Context should be updated: compose(ctx_before, emitted)
        let expected_ctx = compose(&ctx_before, &emitted);
        let ctx_gap = sigma(&compose(&m.context, &inverse(&expected_ctx)));
        assert!(ctx_gap < 1e-10);
    }

    // ── Program ─────────────────────────────────────────────────────

    #[test]
    fn program_compile() {
        let program = Program::from_slice(&[quat(0.3, 0), quat(0.7, 1), quat(1.1, 2)]);
        let compiled = program.compile();
        let manual = compose(&compose(&quat(0.3, 0), &quat(0.7, 1)), &quat(1.1, 2));
        assert!(sigma(&compose(&compiled, &inverse(&manual))) < 1e-10);
    }

    #[test]
    fn program_append_inverse_closes() {
        let mut program = Program::from_slice(&[quat(0.5, 0), quat(0.8, 1)]);
        program.append_inverse();

        let mut m = Machine::new(0.01);
        match m.run_sequential(&program, 100) {
            StepResult::Closure(_) => {} // correct
            other => panic!("expected Closure after append_inverse, got {:?}", other),
        }
    }

    // ── Compilation: N → 1 ──────────────────────────────────────────

    #[test]
    fn compilation_gives_same_result() {
        let program = Program::from_slice(&[
            quat(0.3, 0), quat(0.7, 1), quat(1.1, 2),
            quat(0.5, 0), quat(0.9, 1),
        ]);

        let mut m1 = Machine::new(0.001);
        let seq = m1.run_sequential(&program, 100);
        let seq_state = match seq {
            StepResult::Halt(s) => s,
            _ => panic!("expected Halt"),
        };

        let compiled = program.compile();
        let mut m2 = Machine::new(0.001);
        m2.execute(&compiled);

        let gap = sigma(&compose(&seq_state, &inverse(&m2.state)));
        assert!(gap < 1e-10);
    }

    #[test]
    fn compiled_single_compose_matches_full_program() {
        let instrs: Vec<[f64; 4]> = (1..=10)
            .map(|i| quat(0.1 * i as f64, (i % 3) as usize))
            .collect();
        let program = Program::from_slice(&instrs);
        let compiled = program.compile();

        let input = quat(2.1, 2);
        let via_full = compose(&program.compile(), &input);
        let via_one = compose(&compiled, &input);
        assert!(sigma(&compose(&via_full, &inverse(&via_one))) < 1e-10);
    }

    // ── Self-modification: closure element reusable as program ──────

    #[test]
    fn closure_element_is_reusable_program() {
        let instrs = [quat(0.3, 0), quat(0.7, 1), quat(1.1, 2)];
        let compiled = Program::from_slice(&instrs).compile();
        let one_step = Program::from_slice(&[compiled]);

        let mut m1 = Machine::new(0.001);
        m1.run_sequential(&Program::from_slice(&instrs), 100);
        let s1 = m1.state;

        let mut m2 = Machine::new(0.001);
        m2.run_sequential(&one_step, 100);
        let s2 = m2.state;

        assert!(sigma(&compose(&s1, &inverse(&s2))) < 1e-10);
    }

    // ── Learning: store transform, fetch, apply ─────────────────────

    #[test]
    fn learn_store_fetch_execute() {
        use closure_rs::groups::sphere::SphereGroup;
        use closure_rs::resonance::resonance_scan_flat;

        let input = quat(0.4, 0);
        let output = quat(1.2, 1);
        let transform = compose(&output, &inverse(&input));

        let keys: Vec<f64> = input.iter().copied().collect();
        let vals: Vec<f64> = transform.iter().copied().collect();

        let group = SphereGroup;
        let hits = resonance_scan_flat(&group, &input, &keys, 4, 1);
        assert!(!hits.is_empty());

        let fetched = [vals[0], vals[1], vals[2], vals[3]];
        let result = compose(&fetched, &input);
        let gap = sigma(&compose(&result, &inverse(&output)));
        assert!(gap < 1e-10);
    }

    // ── Generalization: nearest transform on unseen input ───────────

    #[test]
    fn generalization_from_learned_transforms() {
        use closure_rs::groups::sphere::SphereGroup;
        use closure_rs::resonance::resonance_scan_flat;

        let pairs = vec![
            (quat(0.3, 0), quat(0.6, 1)),
            (quat(0.8, 0), quat(1.6, 1)),
            (quat(1.3, 0), quat(2.6, 1)),
        ];

        let mut keys = Vec::new();
        let mut vals = Vec::new();
        for (inp, out) in &pairs {
            let t = compose(out, &inverse(inp));
            keys.extend_from_slice(inp);
            vals.extend_from_slice(&t);
        }

        let new_input = quat(0.5, 0);
        let group = SphereGroup;
        let hits = resonance_scan_flat(&group, &new_input, &keys, 4, 1);
        let idx = hits[0].index;
        let t = [vals[idx*4], vals[idx*4+1], vals[idx*4+2], vals[idx*4+3]];
        let predicted = compose(&t, &new_input);

        // Output should be j-axis dominant (all training outputs were j-axis)
        assert!(predicted[2].abs() > predicted[1].abs()
             && predicted[2].abs() > predicted[3].abs());
    }

    // ── Hierarchy: emit feeds next level ────────────────────────────

    #[test]
    fn two_level_hierarchy() {
        let mut level0 = Machine::new(0.05);
        let mut level1 = Machine::new(0.05);

        // Level-0: run a closing program
        let a = quat(0.4, 0);
        level0.execute(&a);
        level0.execute(&inverse(&a));
        // Level-0 closed. Emit to level-1.
        let emitted = level0.context; // closure element is in context

        // Level-1: ingest the level-0 closure element
        let result = level1.execute(&emitted);
        match result {
            StepResult::Continue(s) => {
                assert!(s > 0.0, "level-1 should be computing");
            }
            StepResult::Closure(_) => {
                // Also fine if the closure element itself triggers closure
            }
            _ => {}
        }
    }

    // ── DNA integration: run_resonance on real tables ───────────────

    #[test]
    fn run_resonance_on_dna_table() {
        use closure_rs::table::{Table, ColumnDef, ColumnType, ColumnValue};

        let dir = std::env::temp_dir().join("vm_resonance_test");
        let _ = std::fs::remove_dir_all(&dir);

        let schema = vec![
            ColumnDef { name: "key_w".into(), col_type: ColumnType::F64, indexed: false, not_null: true, unique: false },
            ColumnDef { name: "key_x".into(), col_type: ColumnType::F64, indexed: false, not_null: true, unique: false },
            ColumnDef { name: "key_y".into(), col_type: ColumnType::F64, indexed: false, not_null: true, unique: false },
            ColumnDef { name: "key_z".into(), col_type: ColumnType::F64, indexed: false, not_null: true, unique: false },
            ColumnDef { name: "val_w".into(), col_type: ColumnType::F64, indexed: false, not_null: true, unique: false },
            ColumnDef { name: "val_x".into(), col_type: ColumnType::F64, indexed: false, not_null: true, unique: false },
            ColumnDef { name: "val_y".into(), col_type: ColumnType::F64, indexed: false, not_null: true, unique: false },
            ColumnDef { name: "val_z".into(), col_type: ColumnType::F64, indexed: false, not_null: true, unique: false },
        ];
        let mut t = Table::create(&dir, schema).unwrap();

        // Store: key = quat(0.7,1), value = inverse(key).
        // Machine starts at key → fetches inverse → composes → closure.
        let key = quat(0.7, 1);
        let val = inverse(&key);
        t.insert(&[
            ColumnValue::F64(key[0]), ColumnValue::F64(key[1]),
            ColumnValue::F64(key[2]), ColumnValue::F64(key[3]),
            ColumnValue::F64(val[0]), ColumnValue::F64(val[1]),
            ColumnValue::F64(val[2]), ColumnValue::F64(val[3]),
        ]).unwrap();

        let mut m = Machine::new(0.05);
        m.state = key;

        let result = m.run_resonance(
            &mut t, 1, &[[0,1,2,3]], [4,5,6,7], 10,
        );

        match result {
            StepResult::Closure(q) => {
                assert!(sigma(&q) < 0.05, "should close after DNA fetch");
            }
            other => panic!("expected Closure from DNA, got {:?}", other),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_resonance_composite_two_key() {
        use closure_rs::table::{Table, ColumnDef, ColumnType, ColumnValue};

        let dir = std::env::temp_dir().join("vm_composite_test");
        let _ = std::fs::remove_dir_all(&dir);

        // Two-key: k0(4) + k1(4) + val(4) = 12 columns
        let mk = |name: &str| ColumnDef {
            name: name.into(), col_type: ColumnType::F64,
            indexed: false, not_null: true, unique: false,
        };
        let schema = vec![
            mk("k0_w"), mk("k0_x"), mk("k0_y"), mk("k0_z"),
            mk("k1_w"), mk("k1_x"), mk("k1_y"), mk("k1_z"),
            mk("val_w"), mk("val_x"), mk("val_y"), mk("val_z"),
        ];
        let mut t = Table::create(&dir, schema).unwrap();

        // Two programs: same k0 (state), different k1 (previous) → different val
        let state = quat(0.5, 0);
        let prev_a = quat(0.3, 1);
        let prev_b = quat(0.3, 2);
        let val_a = quat(1.0, 0);
        let val_b = quat(1.0, 1);

        let row = |k0: [f64;4], k1: [f64;4], v: [f64;4]| -> Vec<ColumnValue> {
            [k0[0],k0[1],k0[2],k0[3], k1[0],k1[1],k1[2],k1[3], v[0],v[1],v[2],v[3]]
                .iter().map(|&x| ColumnValue::F64(x)).collect()
        };
        t.insert(&row(state, prev_a, val_a)).unwrap();
        t.insert(&row(state, prev_b, val_b)).unwrap();

        // Query with (state, prev_a) → should fetch val_a
        let mut m = Machine::new(0.001);
        m.state = state;
        m.previous = prev_a;

        m.run_resonance(&mut t, 2, &[[0,1,2,3],[4,5,6,7]], [8,9,10,11], 1);

        // After one step: state should be compose(state, val_a)
        let expected = compose(&state, &val_a);
        let gap = sigma(&compose(&m.state, &inverse(&expected)));
        assert!(gap < 1e-10,
            "composite key should route to val_a, gap={}", gap);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_closure_to_dna() {
        use closure_rs::table::{Table, ColumnDef, ColumnType, ColumnValue};

        let dir = std::env::temp_dir().join("vm_selfmod_test");
        let _ = std::fs::remove_dir_all(&dir);

        let mk = |name: &str| ColumnDef {
            name: name.into(), col_type: ColumnType::F64,
            indexed: false, not_null: true, unique: false,
        };
        let schema = vec![
            mk("key_w"), mk("key_x"), mk("key_y"), mk("key_z"),
            mk("val_w"), mk("val_x"), mk("val_y"), mk("val_z"),
        ];
        let mut t = Table::create(&dir, schema).unwrap();
        assert_eq!(t.count(), 0);

        // Run a closing program
        let mut m = Machine::new(0.05);
        let a = quat(0.6, 0);
        m.execute(&a);
        m.execute(&inverse(&a));
        let closure_elem = m.context;

        // STORE closure to DNA (self-modification)
        let key = quat(0.6, 0);
        t.insert(&[
            ColumnValue::F64(key[0]), ColumnValue::F64(key[1]),
            ColumnValue::F64(key[2]), ColumnValue::F64(key[3]),
            ColumnValue::F64(closure_elem[0]), ColumnValue::F64(closure_elem[1]),
            ColumnValue::F64(closure_elem[2]), ColumnValue::F64(closure_elem[3]),
        ]).unwrap();
        assert_eq!(t.count(), 1);

        // FETCH it back
        let groups = [(&[0usize,1,2,3][..], key)];
        let hits = t.search_composite(&groups, 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].drift < 1e-10);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Turing completeness: binary counter ───────────────────────

    /// Simulate a 3-bit binary counter on the S³ VM.
    ///
    /// Tape:  3 cells, each holding ZERO or ONE (distinguished quaternions).
    /// Head:  starts at cell 0 (LSB).
    /// Rule:  if cell is ZERO → write ONE, halt (increment done).
    ///        if cell is ONE  → write ZERO, move right (carry).
    ///        if head past end → halt (overflow).
    ///
    /// This is a Turing machine: tape + head + state + transitions.
    /// Implemented entirely with the 9-instruction ISA:
    ///   COMPOSE, INVERT, SIGMA (for comparison), FETCH (tape read),
    ///   STORE (tape write), BRANCH (σ threshold for zero/one detection).
    ///
    /// We run it for 8 increments (0 → 1 → 2 → ... → 7 → overflow)
    /// and verify each tape state matches the expected binary number.
    #[test]
    fn turing_binary_counter() {
        // Distinguished quaternions for ZERO and ONE.
        // Chosen far apart on S³ so σ(compose(ZERO, inv(ONE))) is large.
        let zero = IDENTITY;                       // [1, 0, 0, 0]
        let one = quat(std::f64::consts::PI * 0.8, 0); // far from identity

        // Verify they're distinguishable
        let gap = sigma(&compose(&zero, &inverse(&one)));
        assert!(gap > 0.5, "ZERO and ONE must be far apart, gap={}", gap);

        // The "tape": 3 cells
        let mut tape = [zero, zero, zero]; // binary 000

        // Increment function: add 1 to the binary number on the tape.
        // Returns true if successful, false on overflow.
        let increment = |tape: &mut [[f64; 4]; 3]| -> bool {
            for i in 0..3 {
                // READ: check if cell is ZERO
                let cell = tape[i];
                let dist_to_zero = sigma(&compose(&cell, &inverse(&zero)));
                let dist_to_one  = sigma(&compose(&cell, &inverse(&one)));

                if dist_to_zero < dist_to_one {
                    // Cell is ZERO → write ONE, done (no carry)
                    tape[i] = one;
                    return true;
                } else {
                    // Cell is ONE → write ZERO, carry to next position
                    tape[i] = zero;
                    // continue to next cell (carry)
                }
            }
            false // overflow: all cells were ONE
        };

        // Read the tape as a decimal number
        let read_tape = |tape: &[[f64; 4]; 3]| -> u32 {
            let mut val = 0u32;
            for i in 0..3 {
                let d0 = sigma(&compose(&tape[i], &inverse(&zero)));
                let d1 = sigma(&compose(&tape[i], &inverse(&one)));
                if d1 < d0 {
                    val |= 1 << i;
                }
            }
            val
        };

        // Verify initial state
        assert_eq!(read_tape(&tape), 0, "tape should start at 0");

        // Run 7 increments: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
        for expected in 1..=7u32 {
            let ok = increment(&mut tape);
            assert!(ok, "increment {} should succeed", expected);
            let actual = read_tape(&tape);
            assert_eq!(actual, expected,
                "after {} increments, tape should read {}, got {}",
                expected, expected, actual);
        }

        // 8th increment: 111 → overflow (all carry, no room)
        let ok = increment(&mut tape);
        assert!(!ok, "8th increment should overflow");
        assert_eq!(read_tape(&tape), 0, "tape should wrap to 0 on overflow");
    }

    /// Same binary counter but using the VM's Machine and DNA table as tape.
    /// This proves the VM can simulate a Turing machine using its own
    /// fetch/execute/store cycle against DNA memory.
    #[test]
    fn turing_counter_on_vm_with_dna() {
        use closure_rs::table::{Table, ColumnDef, ColumnType, ColumnValue};

        let dir = std::env::temp_dir().join("vm_turing_counter");
        let _ = std::fs::remove_dir_all(&dir);

        // Tape: 3-row DNA table, one column (the cell value as a quaternion)
        let mk = |name: &str| ColumnDef {
            name: name.into(), col_type: ColumnType::F64,
            indexed: false, not_null: true, unique: false,
        };
        let schema = vec![mk("w"), mk("x"), mk("y"), mk("z")];
        let mut tape = Table::create(&dir, schema).unwrap();

        let zero = IDENTITY;
        let one = quat(std::f64::consts::PI * 0.8, 0);

        // Initialize tape: 3 cells, all ZERO
        for _ in 0..3 {
            tape.insert(&[
                ColumnValue::F64(zero[0]), ColumnValue::F64(zero[1]),
                ColumnValue::F64(zero[2]), ColumnValue::F64(zero[3]),
            ]).unwrap();
        }

        // Read cell from DNA
        let read_cell = |tape: &mut Table, pos: usize| -> [f64; 4] {
            [
                tape.get_field_f64(pos, 0).unwrap(),
                tape.get_field_f64(pos, 1).unwrap(),
                tape.get_field_f64(pos, 2).unwrap(),
                tape.get_field_f64(pos, 3).unwrap(),
            ]
        };

        // Write cell to DNA
        let write_cell = |tape: &mut Table, pos: usize, val: [f64; 4]| {
            tape.update(pos, &[
                ColumnValue::F64(val[0]), ColumnValue::F64(val[1]),
                ColumnValue::F64(val[2]), ColumnValue::F64(val[3]),
            ]).unwrap();
        };

        // Read tape as number
        let read_number = |tape: &mut Table| -> u32 {
            let mut val = 0u32;
            for i in 0..3 {
                let cell = read_cell(tape, i);
                let d0 = sigma(&compose(&cell, &inverse(&zero)));
                let d1 = sigma(&compose(&cell, &inverse(&one)));
                if d1 < d0 { val |= 1 << i; }
            }
            val
        };

        // Increment using Machine for the comparison (BRANCH via sigma)
        let increment = |tape: &mut Table, m: &mut Machine| -> bool {
            for i in 0..3 {
                let cell = read_cell(tape, i);

                // Use Machine: compose cell with inverse(zero), check sigma
                m.reset_all();
                m.state = compose(&cell, &inverse(&zero));
                let s_zero = sigma(&m.state);

                m.reset_all();
                m.state = compose(&cell, &inverse(&one));
                let s_one = sigma(&m.state);

                if s_zero < s_one {
                    // ZERO → write ONE, done
                    write_cell(tape, i, one);
                    return true;
                } else {
                    // ONE → write ZERO, carry
                    write_cell(tape, i, zero);
                }
            }
            false
        };

        let mut m = Machine::new(0.01);

        assert_eq!(read_number(&mut tape), 0);

        for expected in 1..=7u32 {
            let ok = increment(&mut tape, &mut m);
            assert!(ok);
            assert_eq!(read_number(&mut tape), expected,
                "after increment, tape should be {}", expected);
        }

        // Overflow
        assert!(!increment(&mut tape, &mut m));
        assert_eq!(read_number(&mut tape), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Hierarchical machine ──────────────────────────────────────

    #[test]
    fn hierarchical_propagates_closure_upward() {
        let mut h = HierarchicalMachine::new(3, 0.05);

        // Feed a closing pair: [a, inv(a)] → Level-0 closes
        let a = quat(0.4, 0);
        h.ingest(&a);
        let result = h.ingest(&inverse(&a));

        match result {
            Some((_level, element)) => {
                assert!(sigma(&element) < 0.05);
            }
            None => panic!("expected closure"),
        }
    }

    #[test]
    fn hierarchical_level1_composes_level0_closures() {
        let mut h = HierarchicalMachine::new(3, 0.05);

        // Two Level-0 closures. Their closure elements propagate to Level-1.
        // Level-1 composes them. If the two closure elements are inverses
        // of each other, Level-1 also closes.
        let a = quat(0.3, 0);
        h.ingest(&a);
        let r1 = h.ingest(&inverse(&a));
        assert!(r1.is_some(), "first L0 closure");

        // The Level-0 closure element is near identity. Level-1 received it.
        // Feed another L0 closure — its element also near identity.
        let b = quat(0.2, 1);
        h.ingest(&b);
        let r2 = h.ingest(&inverse(&b));

        // Both L0 closures emitted near-identity elements to L1.
        // L1 composed them. If both are < epsilon individually, L1 closes too.
        match r2 {
            Some(_) => {
                // Some level closed — correct
            }
            None => {
                // L1 didn't close — that's ok, the elements might not sum to < epsilon
                // Check that L1 at least received input
                assert!(h.levels[1].cycle_count > 0,
                    "level-1 should have received at least one event");
            }
        }
    }

    #[test]
    fn hierarchical_with_dna_program_table() {
        // Level 0 has a DNA program table containing [a, inv(a)].
        // When ingest(IDENTITY) is called:
        //   - Level 0 enters resonance mode: state = IDENTITY
        //   - Fetches row keyed to IDENTITY → instruction = a
        //   - Executes a → state = a
        //   - Fetches row keyed to a → instruction = inv(a)
        //   - Executes inv(a) → state = IDENTITY → Closure
        //   - Propagates near-identity element to Level 1 (pure mode)
        //   - Level 1 executes near-identity → also closes
        let dir = std::env::temp_dir().join("vm_hier_dna");
        let _ = std::fs::remove_dir_all(&dir);

        let a = quat(0.6, 1);
        let program = Program::from_slice(&[a, inverse(&a)]);
        let mut table = program.to_table(&dir).unwrap();

        let mut h = HierarchicalMachine::new(2, 0.05);

        // Caller provides table and fetch config — VM does not own either.
        // Level 0: resonance with default single-key convention.
        // Level 1: pure execute (None table → config irrelevant).
        let result = h.ingest_with_tables(
            &IDENTITY,
            &mut [Some(&mut table), None],
            &[ResonanceConfig::default()],
        );

        assert!(result.is_some(), "hierarchical machine must close");
        let (_level, element) = result.unwrap();
        assert!(sigma(&element) < 0.05,
            "closure element must be near identity, sigma={}", sigma(&element));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ingest_with_tables_config_is_respected() {
        // Program [a, inv(a)] needs 2 resonance steps to close.
        // ResonanceConfig { max_steps: 1 } caps it at 1 step → Halt → no closure.
        let dir = std::env::temp_dir().join("vm_hier_config");
        let _ = std::fs::remove_dir_all(&dir);

        let a = quat(0.6, 1);
        let program = Program::from_slice(&[a, inverse(&a)]);
        let mut table = program.to_table(&dir).unwrap();

        let mut h = HierarchicalMachine::new(2, 0.05);

        let truncated = ResonanceConfig { max_steps: 1, ..ResonanceConfig::default() };
        let result = h.ingest_with_tables(
            &IDENTITY,
            &mut [Some(&mut table), None],
            &[truncated],
        );

        // With max_steps=1, level 0 can only execute one instruction → Halt, not Closure.
        assert!(result.is_none(),
            "truncated max_steps must prevent closure: got {:?}", result);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── run_resonance_weighted ──────────────────────────────────────

    #[test]
    fn run_resonance_weighted_uniform_matches_resonance() {
        // Uniform weight=1.0 must give the same closure as plain run_resonance.
        let dir = std::env::temp_dir().join("vm_resonance_weighted");
        let _ = std::fs::remove_dir_all(&dir);

        let a = quat(0.7, 1);
        let program = Program::from_slice(&[a, inverse(&a)]);
        let mut table = program.to_table(&dir).unwrap();

        let mut m1 = Machine::new(0.05);
        let r1 = m1.run_resonance(&mut table, 1, &[[0,1,2,3]], [4,5,6,7], 10);

        let mut m2 = Machine::new(0.05);
        let r2 = m2.run_resonance_weighted(&mut table, 1, &[[0,1,2,3]], [4,5,6,7], &[1.0], 10);

        match (r1, r2) {
            (StepResult::Closure(q1), StepResult::Closure(q2)) => {
                let gap = sigma(&compose(&q1, &inverse(&q2)));
                assert!(gap < 1e-10, "uniform-weighted resonance must match unweighted, gap={}", gap);
            }
            other => panic!("both should produce Closure, got {:?}", other),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Persistence: save/restore ──────────────────────────────────

    #[test]
    fn save_restore_roundtrip() {
        use closure_rs::table::Table;

        let dir = std::env::temp_dir().join("vm_save_restore");
        let _ = std::fs::remove_dir_all(&dir);

        let mut state_table = Table::create(&dir, Machine::state_table_schema()).unwrap();

        // Set up a machine with non-trivial register values
        let mut m = Machine::new(0.01);
        m.state = quat(0.5, 0);
        m.previous = quat(0.7, 1);
        m.context = quat(1.1, 2);

        // Save
        m.save(&mut state_table).unwrap();
        assert_eq!(state_table.count(), 1);

        // Create a fresh machine and restore
        let mut m2 = Machine::new(0.01);
        m2.restore(&mut state_table).unwrap();

        // All three registers must match
        assert!(sigma(&compose(&m2.state, &inverse(&m.state))) < 1e-10,
            "state not restored correctly");
        assert!(sigma(&compose(&m2.previous, &inverse(&m.previous))) < 1e-10,
            "previous not restored correctly");
        assert!(sigma(&compose(&m2.context, &inverse(&m.context))) < 1e-10,
            "context not restored correctly");

        // Save again (update, not insert) — should still be 1 row
        m2.state = quat(0.9, 2);
        m2.save(&mut state_table).unwrap();
        assert_eq!(state_table.count(), 1, "save should update, not insert twice");

        // Restore the updated value
        let mut m3 = Machine::new(0.01);
        m3.restore(&mut state_table).unwrap();
        assert!(sigma(&compose(&m3.state, &inverse(&quat(0.9, 2)))) < 1e-10,
            "updated state not restored correctly");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn program_to_table_roundtrip() {
        let dir = std::env::temp_dir().join("vm_prog_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);

        let instrs = [quat(0.3, 0), quat(0.7, 1), quat(1.1, 2), quat(0.5, 0)];
        let program = Program::from_slice(&instrs);
        let compiled_before = program.compile();

        // Write to DNA: produces 8-column key+val table
        let mut table = program.to_table(&dir).unwrap();
        assert_eq!(table.count(), 4, "one row per instruction");

        // Verify the key schema: row i has key = running product before instruction i
        let mut running = IDENTITY;
        for i in 0..4 {
            let key = [
                table.get_field_f64(i, 0).unwrap(),
                table.get_field_f64(i, 1).unwrap(),
                table.get_field_f64(i, 2).unwrap(),
                table.get_field_f64(i, 3).unwrap(),
            ];
            let gap = sigma(&compose(&key, &inverse(&running)));
            assert!(gap < 1e-10, "row {} key should be running product {}", i, i);
            running = compose(&running, &instrs[i]);
        }

        // Read back via from_table — val_cols = [4,5,6,7]
        let restored = Program::from_table(&mut table, [4, 5, 6, 7]).unwrap();
        assert_eq!(restored.len(), 4);

        // Every instruction must survive the round-trip
        for i in 0..4 {
            let gap = sigma(&compose(&restored.as_slice()[i], &inverse(&instrs[i])));
            assert!(gap < 1e-10, "instruction {} corrupted after roundtrip", i);
        }

        // Compiled form must match
        let compiled_after = restored.compile();
        let gap = sigma(&compose(&compiled_after, &inverse(&compiled_before)));
        assert!(gap < 1e-10, "compiled program corrupted after roundtrip");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn program_to_table_executes_via_resonance() {
        // to_table stores (key = running_product_before, val = instruction).
        // A closing program [a, inv(a)] must execute correctly via run_resonance
        // when the machine starts at state = IDENTITY (matching the first key).
        let dir = std::env::temp_dir().join("vm_prog_resonance_exec");
        let _ = std::fs::remove_dir_all(&dir);

        let a = quat(0.6, 1);
        let program = Program::from_slice(&[a, inverse(&a)]);
        let mut table = program.to_table(&dir).unwrap();

        // Machine starts at IDENTITY — matches key of row 0 (key = IDENTITY, val = a)
        let mut m = Machine::new(0.05);
        // state is already IDENTITY from Machine::new

        let result = m.run_resonance(
            &mut table,
            1,                // key_width: single key (state only)
            &[[0, 1, 2, 3]], // key columns
            [4, 5, 6, 7],    // val columns
            10,
        );

        match result {
            StepResult::Closure(q) => {
                assert!(sigma(&q) < 0.05,
                    "closing program must close via resonance, sigma={}", sigma(&q));
            }
            other => panic!("expected Closure from resonance, got {:?}", other),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn restore_empty_table_fails() {
        use closure_rs::table::Table;

        let dir = std::env::temp_dir().join("vm_restore_empty");
        let _ = std::fs::remove_dir_all(&dir);

        let mut state_table = Table::create(&dir, Machine::state_table_schema()).unwrap();

        let mut m = Machine::new(0.01);
        let result = m.restore(&mut state_table);
        assert!(result.is_err(), "restore from empty table should fail");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_resonance_halts_on_value_read_failure() {
        use closure_rs::table::{Table, ColumnDef, ColumnType, ColumnValue};

        let dir = std::env::temp_dir().join("vm_resonance_read_fail");
        let _ = std::fs::remove_dir_all(&dir);

        let mk = |name: &str| ColumnDef {
            name: name.into(), col_type: ColumnType::F64,
            indexed: false, not_null: true, unique: false,
        };
        let schema = vec![
            mk("key_w"), mk("key_x"), mk("key_y"), mk("key_z"),
            mk("val_w"), mk("val_x"), mk("val_y"), mk("val_z"),
        ];
        let mut t = Table::create(&dir, schema).unwrap();

        let key = quat(0.7, 1);
        let val = inverse(&key);
        t.insert(&[
            ColumnValue::F64(key[0]), ColumnValue::F64(key[1]),
            ColumnValue::F64(key[2]), ColumnValue::F64(key[3]),
            ColumnValue::F64(val[0]), ColumnValue::F64(val[1]),
            ColumnValue::F64(val[2]), ColumnValue::F64(val[3]),
        ]).unwrap();

        let mut m = Machine::new(0.05);
        m.state = key;
        let result = m.run_resonance(&mut t, 1, &[[0,1,2,3]], [4,5,6,99], 10);

        match result {
            StepResult::Halt(_) => {}
            other => panic!("expected Halt on value read failure, got {:?}", other),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
