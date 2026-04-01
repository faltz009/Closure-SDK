use closure_rs::table::Table;
use crate::primitives::StepResult;
use crate::machine::Machine;

/// Resonance fetch configuration for one level of a HierarchicalMachine.
///
/// Controls how a level fetches instructions from its DNA program table:
/// which columns form the key, which columns hold the instruction,
/// optional per-key-group drift weights, and the step budget.
///
/// Use `ResonanceConfig::default()` for the standard single-key program-table
/// convention (key_cols=0..3, val_cols=4..7, uniform weights, 1000 steps).
#[derive(Clone)]
pub struct ResonanceConfig {
    /// Column index groups for key components. Length determines key_width.
    /// Each inner array is exactly 4 column indices (one quaternion component per index).
    pub key_col_groups: Vec<[usize; 4]>,
    /// Column indices [w, x, y, z] for the instruction quaternion value.
    pub val_cols: [usize; 4],
    /// Per-key-group drift weights. `None` = uniform (calls run_resonance).
    /// `Some(w)` = weighted (calls run_resonance_weighted). Length must match key_col_groups.
    pub weights: Option<Vec<f64>>,
    /// Maximum resonance steps before returning Halt.
    pub max_steps: usize,
}

impl Default for ResonanceConfig {
    /// Standard single-key program-table convention:
    ///   key columns 0-3, val columns 4-7, uniform weights, 1000 steps.
    fn default() -> Self {
        Self {
            key_col_groups: vec![[0, 1, 2, 3]],
            val_cols: [4, 5, 6, 7],
            weights: None,
            max_steps: 1000,
        }
    }
}

/// Multiple Machines stacked by closure cadence.
/// Level-0 ingests raw events. When it closes, the closure element
/// propagates up to Level-1. And so on.
///
/// The hierarchy owns only Machine state — no Table handles.
/// Program tables and their fetch configuration are provided by the
/// caller at ingest time, so storage stays in DNA and execution stays in the VM.
pub struct HierarchicalMachine {
    pub levels: Vec<Machine>,
}

impl HierarchicalMachine {
    /// Create with n_levels levels, all in pure execute mode, sharing epsilon.
    pub fn new(n_levels: usize, epsilon: f64) -> Self {
        Self {
            levels: (0..n_levels).map(|_| Machine::new(epsilon)).collect(),
        }
    }

    /// Ingest one event in pure execute mode. All levels cascade without DNA.
    /// Returns the highest level that closed, or None.
    pub fn ingest(&mut self, event: &[f64; 4]) -> Option<(usize, [f64; 4])> {
        let mut input = *event;
        let mut highest_closure = None;
        let n = self.levels.len();

        for i in 0..n {
            match self.levels[i].execute(&input) {
                StepResult::Closure(element) => {
                    highest_closure = Some((i, element));
                    input = element;
                }
                _ => break,
            }
        }

        highest_closure
    }

    /// Ingest one event, allowing per-level DNA program tables and fetch configuration.
    ///
    /// For level i:
    ///   - `tables[i] = None` or `i >= tables.len()` → pure execute mode.
    ///   - `tables[i] = Some(&mut t)` → resonance mode using `configs[i]`
    ///     (or `ResonanceConfig::default()` if `i >= configs.len()`).
    ///
    /// The caller owns all Table handles. The VM never stores them.
    /// Returns the highest level that closed, or None.
    pub fn ingest_with_tables(
        &mut self,
        event: &[f64; 4],
        tables: &mut [Option<&mut Table>],
        configs: &[ResonanceConfig],
    ) -> Option<(usize, [f64; 4])> {
        let mut input = *event;
        let mut highest_closure = None;
        let n = self.levels.len();
        let default_config = ResonanceConfig::default();

        for i in 0..n {
            let result = if i < tables.len() {
                let config = configs.get(i).unwrap_or(&default_config);
                match &mut tables[i] {
                    Some(t) => {
                        self.levels[i].state = input;
                        let key_width = config.key_col_groups.len();
                        if let Some(ref w) = config.weights {
                            self.levels[i].run_resonance_weighted(
                                *t, key_width, &config.key_col_groups,
                                config.val_cols, w.as_slice(), config.max_steps,
                            )
                        } else {
                            self.levels[i].run_resonance(
                                *t, key_width, &config.key_col_groups,
                                config.val_cols, config.max_steps,
                            )
                        }
                    }
                    None => self.levels[i].execute(&input),
                }
            } else {
                self.levels[i].execute(&input)
            };

            match result {
                StepResult::Closure(element) => {
                    highest_closure = Some((i, element));
                    input = element;
                }
                _ => break,
            }
        }

        highest_closure
    }

    /// Reset all levels (state, previous, context, counters).
    pub fn reset_all(&mut self) {
        for m in &mut self.levels {
            m.reset_all();
        }
    }
}
