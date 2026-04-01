use closure_rs::groups::sphere::{
    IDENTITY, sphere_compose as compose, sphere_sigma as sigma,
};
use crate::primitives::StepResult;
use crate::program::Program;

/// The S³ virtual machine. Three registers, one threshold.
/// Owns NO storage. Reads from and writes to DNA tables.
pub struct Machine {
    /// Current state — the accumulator / program counter.
    pub state: [f64; 4],
    /// Previous state — retained for composite key construction.
    pub previous: [f64; 4],
    /// Context — composition of all closure elements this session.
    /// Survives across closures. Session memory.
    pub context: [f64; 4],
    /// Closure threshold.
    pub epsilon: f64,
    /// Instruction pointer (sequential mode).
    pub ip: usize,
    /// Instructions executed since last reset.
    pub cycle_count: usize,
}

impl Machine {
    pub fn new(epsilon: f64) -> Self {
        Self {
            state: IDENTITY,
            previous: IDENTITY,
            context: IDENTITY,
            epsilon,
            ip: 0,
            cycle_count: 0,
        }
    }

    /// Reset state and previous to identity. Context persists.
    pub fn reset(&mut self) {
        self.state = IDENTITY;
        self.previous = IDENTITY;
        self.ip = 0;
        self.cycle_count = 0;
    }

    /// Full reset including context.
    pub fn reset_all(&mut self) {
        self.reset();
        self.context = IDENTITY;
    }

    /// ISA #9 (BRANCH) + ISA #1 (COMPOSE): one cycle.
    /// Compose instruction into state. Check sigma. Branch.
    pub fn execute(&mut self, instruction: &[f64; 4]) -> StepResult {
        self.previous = self.state;
        self.state = compose(&self.state, instruction);
        self.cycle_count += 1;
        let s = sigma(&self.state);

        if s < self.epsilon {
            let element = self.state;
            self.context = compose(&self.context, &element);
            self.state = IDENTITY;
            StepResult::Closure(element)
        } else if s > std::f64::consts::FRAC_PI_2 - self.epsilon {
            let element = self.state;
            self.state = IDENTITY;
            StepResult::Death(element)
        } else {
            StepResult::Continue(s)
        }
    }

    /// ISA #8: EMIT. Output current state, update context, reset.
    /// Used for hierarchy: emitted quaternion feeds the next level.
    pub fn emit(&mut self) -> [f64; 4] {
        let result = self.state;
        self.context = compose(&self.context, &result);
        self.state = IDENTITY;
        self.ip = 0;
        result
    }

    /// Build composite key from registers.
    /// width=1: [state]
    /// width=2: [state, previous]
    /// width=3: [state, previous, context]
    pub fn build_key(&self, width: usize) -> Vec<f64> {
        let mut key = Vec::with_capacity(width * 4);
        key.extend_from_slice(&self.state);
        if width >= 2 { key.extend_from_slice(&self.previous); }
        if width >= 3 { key.extend_from_slice(&self.context); }
        key
    }

    /// Run a program sequentially: execute each instruction in order.
    /// Stops on closure, death, or when instructions are exhausted.
    pub fn run_sequential(&mut self, program: &Program, max_steps: usize) -> StepResult {
        self.state = IDENTITY;
        self.previous = IDENTITY;
        self.ip = 0;
        self.cycle_count = 0;

        for (i, instr) in program.as_slice().iter().enumerate() {
            if i >= max_steps { break; }
            self.ip = i;
            match self.execute(instr) {
                StepResult::Closure(q) => return StepResult::Closure(q),
                StepResult::Death(q) => return StepResult::Death(q),
                StepResult::Continue(_) => continue,
                other => return other,
            }
        }
        StepResult::Halt(self.state)
    }

    /// Resonance mode: FETCH from a DNA table by composite key.
    ///
    /// Each cycle: build key from registers → search_composite → read
    /// instruction from matched row → execute → branch.
    ///
    /// key_width: 1, 2, or 3 (how many registers form the key).
    /// key_col_groups: column indices for each key component, each group is 4 indices.
    ///   e.g. width=2: &[[0,1,2,3], [4,5,6,7]] for columns k0_wxyz and k1_wxyz.
    /// val_cols: 4 column indices for the instruction quaternion.
    pub fn run_resonance(
        &mut self,
        table: &mut closure_rs::table::Table,
        key_width: usize,
        key_col_groups: &[[usize; 4]],
        val_cols: [usize; 4],
        max_steps: usize,
    ) -> StepResult {
        assert_eq!(key_col_groups.len(), key_width,
            "key_col_groups length must match key_width");

        // Do NOT reset registers — the caller sets state/previous/context
        // before calling run_resonance. The registers ARE the query.
        self.ip = 0;
        self.cycle_count = 0;

        for _ in 0..max_steps {
            // BUILD composite key from registers
            let key = self.build_key(key_width);

            // Assemble key_groups for DNA's search_composite
            let groups: Vec<(&[usize], [f64; 4])> = key_col_groups.iter()
                .enumerate()
                .map(|(i, cols)| {
                    let q = [key[i*4], key[i*4+1], key[i*4+2], key[i*4+3]];
                    (cols.as_slice(), q)
                })
                .collect();

            // FETCH: search DNA table
            let hits = match table.search_composite(&groups, 1) {
                Ok(h) => h,
                Err(_) => return StepResult::Halt(self.state),
            };
            if hits.is_empty() {
                return StepResult::Halt(self.state);
            }

            // READ instruction from matched row's value columns
            let row = hits[0].index;
            let instruction = match (
                table.get_field_f64(row, val_cols[0]),
                table.get_field_f64(row, val_cols[1]),
                table.get_field_f64(row, val_cols[2]),
                table.get_field_f64(row, val_cols[3]),
            ) {
                (Ok(w), Ok(x), Ok(y), Ok(z)) => [w, x, y, z],
                _ => return StepResult::Halt(self.state),
            };

            // EXECUTE + BRANCH
            match self.execute(&instruction) {
                StepResult::Continue(_) => continue,
                terminal => return terminal,
            }
        }
        StepResult::Halt(self.state)
    }

    /// Weighted resonance mode: identical to run_resonance but uses
    /// search_composite_weighted so the caller can bias state/previous/context
    /// contributions without changing the table schema.
    ///
    /// `weights[i]` scales the drift contribution of key group i.
    /// For uniform addressing (same result as run_resonance) use weights = &[1.0; key_width].
    pub fn run_resonance_weighted(
        &mut self,
        table: &mut closure_rs::table::Table,
        key_width: usize,
        key_col_groups: &[[usize; 4]],
        val_cols: [usize; 4],
        weights: &[f64],
        max_steps: usize,
    ) -> StepResult {
        assert_eq!(key_col_groups.len(), key_width,
            "key_col_groups length must match key_width");
        assert_eq!(weights.len(), key_width,
            "weights length must match key_width");

        self.ip = 0;
        self.cycle_count = 0;

        for _ in 0..max_steps {
            let key = self.build_key(key_width);

            let groups: Vec<(&[usize], [f64; 4])> = key_col_groups.iter()
                .enumerate()
                .map(|(i, cols)| {
                    let q = [key[i*4], key[i*4+1], key[i*4+2], key[i*4+3]];
                    (cols.as_slice(), q)
                })
                .collect();

            let hits = match table.search_composite_weighted(&groups, weights, 1) {
                Ok(h) => h,
                Err(_) => return StepResult::Halt(self.state),
            };
            if hits.is_empty() {
                return StepResult::Halt(self.state);
            }

            let row = hits[0].index;
            let instruction = match (
                table.get_field_f64(row, val_cols[0]),
                table.get_field_f64(row, val_cols[1]),
                table.get_field_f64(row, val_cols[2]),
                table.get_field_f64(row, val_cols[3]),
            ) {
                (Ok(w), Ok(x), Ok(y), Ok(z)) => [w, x, y, z],
                _ => return StepResult::Halt(self.state),
            };

            match self.execute(&instruction) {
                StepResult::Continue(_) => continue,
                terminal => return terminal,
            }
        }
        StepResult::Halt(self.state)
    }

    // ── Persistence: save/restore registers to DNA ──────────────────

    /// Save all registers to a 1-row DNA table.
    /// Inserts row 0 on first call, updates it on subsequent calls.
    pub fn save(&self, table: &mut closure_rs::table::Table) -> Result<(), std::io::Error> {
        use closure_rs::table::ColumnValue;
        let row = [
            ColumnValue::F64(self.state[0]),    ColumnValue::F64(self.state[1]),
            ColumnValue::F64(self.state[2]),    ColumnValue::F64(self.state[3]),
            ColumnValue::F64(self.previous[0]), ColumnValue::F64(self.previous[1]),
            ColumnValue::F64(self.previous[2]), ColumnValue::F64(self.previous[3]),
            ColumnValue::F64(self.context[0]),  ColumnValue::F64(self.context[1]),
            ColumnValue::F64(self.context[2]),  ColumnValue::F64(self.context[3]),
        ];
        if table.count() == 0 {
            table.insert(&row)?;
        } else {
            table.update(0, &row)?;
        }
        Ok(())
    }

    /// Canonical 12-column schema for a machine state table.
    /// Column order: state_wxyz, prev_wxyz, ctx_wxyz.
    /// Use with Machine::save / Machine::restore.
    pub fn state_table_schema() -> Vec<closure_rs::table::ColumnDef> {
        use closure_rs::table::{ColumnDef, ColumnType};
        let mk = |name: &str| ColumnDef {
            name: name.into(), col_type: ColumnType::F64,
            indexed: false, not_null: true, unique: false,
        };
        vec![
            mk("state_w"), mk("state_x"), mk("state_y"), mk("state_z"),
            mk("prev_w"),  mk("prev_x"),  mk("prev_y"),  mk("prev_z"),
            mk("ctx_w"),   mk("ctx_x"),   mk("ctx_y"),   mk("ctx_z"),
        ]
    }

    /// Restore registers from a 1-row DNA table.
    pub fn restore(&mut self, table: &mut closure_rs::table::Table) -> Result<(), std::io::Error> {
        if table.count() == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "machine state table is empty",
            ));
        }
        self.state = [
            table.get_field_f64(0, 0)?, table.get_field_f64(0, 1)?,
            table.get_field_f64(0, 2)?, table.get_field_f64(0, 3)?,
        ];
        self.previous = [
            table.get_field_f64(0, 4)?, table.get_field_f64(0, 5)?,
            table.get_field_f64(0, 6)?, table.get_field_f64(0, 7)?,
        ];
        self.context = [
            table.get_field_f64(0, 8)?,  table.get_field_f64(0, 9)?,
            table.get_field_f64(0, 10)?, table.get_field_f64(0, 11)?,
        ];
        Ok(())
    }
}
