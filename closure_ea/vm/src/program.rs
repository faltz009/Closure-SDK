use closure_rs::groups::sphere::{
    IDENTITY, sphere_compose as compose, sphere_inverse as inverse,
};

/// A sequence of quaternion instructions.
/// In-memory representation for sequential execution.
pub struct Program {
    instructions: Vec<[f64; 4]>,
}

impl Program {
    pub fn new() -> Self {
        Self { instructions: Vec::new() }
    }

    pub fn from_slice(instrs: &[[f64; 4]]) -> Self {
        Self { instructions: instrs.to_vec() }
    }

    pub fn push(&mut self, q: [f64; 4]) {
        self.instructions.push(q);
    }

    pub fn len(&self) -> usize { self.instructions.len() }
    pub fn is_empty(&self) -> bool { self.instructions.is_empty() }
    pub fn as_slice(&self) -> &[[f64; 4]] { &self.instructions }

    /// ISA compile: N instructions → 1 closure element. Algebraically exact.
    pub fn compile(&self) -> [f64; 4] {
        let mut result = IDENTITY;
        for instr in &self.instructions {
            result = compose(&result, instr);
        }
        result
    }

    /// Append the inverse of the entire program. Guarantees closure.
    pub fn append_inverse(&mut self) {
        let compiled = self.compile();
        self.instructions.push(inverse(&compiled));
    }

    /// Canonical 8-column schema for a program table.
    /// Columns: key_wxyz (running product before the instruction), val_wxyz (instruction).
    /// Use with Program::to_table / Machine::run_resonance at key_col_groups=[[0,1,2,3]], val_cols=[4,5,6,7].
    pub fn table_schema() -> Vec<closure_rs::table::ColumnDef> {
        use closure_rs::table::{ColumnDef, ColumnType};
        let mk = |name: &str| ColumnDef {
            name: name.into(), col_type: ColumnType::F64,
            indexed: false, not_null: true, unique: false,
        };
        vec![
            mk("key_w"), mk("key_x"), mk("key_y"), mk("key_z"),
            mk("val_w"), mk("val_x"), mk("val_y"), mk("val_z"),
        ]
    }

    /// Write this program to a resonance-compatible DNA table.
    /// Schema: 8 columns — key_w/x/y/z (running product before this instruction)
    ///         and val_w/x/y/z (the instruction itself).
    ///
    /// Key = machine state immediately before executing this instruction.
    /// When run_resonance is started at state = IDENTITY, it fetches instruction[0]
    /// (whose key is IDENTITY), executes it, then fetches instruction[1] by the
    /// resulting state, and so on. The program is content-addressed by its own
    /// accumulated state — the address IS the computation.
    pub fn to_table(&self, dir: &std::path::Path) -> Result<closure_rs::table::Table, std::io::Error> {
        use closure_rs::table::{Table, ColumnValue};
        let schema = Self::table_schema();
        let mut table = Table::create(dir, schema)?;
        let mut running = IDENTITY;
        for instr in &self.instructions {
            table.insert(&[
                ColumnValue::F64(running[0]), ColumnValue::F64(running[1]),
                ColumnValue::F64(running[2]), ColumnValue::F64(running[3]),
                ColumnValue::F64(instr[0]),   ColumnValue::F64(instr[1]),
                ColumnValue::F64(instr[2]),   ColumnValue::F64(instr[3]),
            ])?;
            running = compose(&running, instr);
        }
        Ok(table)
    }

    /// Load a program from a DNA table. Reads all rows, extracting instructions
    /// from the specified value columns. Works with any key+val schema.
    /// For a standard to_table() table: val_cols = [4, 5, 6, 7].
    pub fn from_table(table: &mut closure_rs::table::Table, val_cols: [usize; 4]) -> Result<Self, std::io::Error> {
        let n = table.count();
        let mut instructions = Vec::with_capacity(n);
        for i in 0..n {
            let q = [
                table.get_field_f64(i, val_cols[0])?,
                table.get_field_f64(i, val_cols[1])?,
                table.get_field_f64(i, val_cols[2])?,
                table.get_field_f64(i, val_cols[3])?,
            ];
            instructions.push(q);
        }
        Ok(Self { instructions })
    }
}
