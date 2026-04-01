//! # closure_rs
//!
//! Data in → tiny summary out → compare summaries → find problems.
//!
//! The pipeline: raw bytes get SHA-256 embedded into group elements,
//! multiplied into a running product, and compared via geodesic distance.
//! The running product is a constant-size summary (8–32 bytes) regardless
//! of how many events you process.
//!
//! ## Modules
//!
//! - [`groups`]    — The geometric surfaces: Circle, Sphere, Torus, Hybrid
//! - [`path`]      — GeometricPath: running products + O(1) coherence checks
//! - [`hierarchy`] — HierarchicalClosure: O(log n) binary search localization
//! - [`embed`]     — SHA-256 embedding: raw bytes → group elements
//!
//! ## Key guarantees
//!
//! **Theorem 1**: Perturbing one element by ε changes the summary by exactly ε.
//! **Theorem 2**: Every position contributes equally — no blind spots.

// Shared core — lives in rust/src/
pub mod embed;
pub mod groups;
pub mod hopf;
pub mod hierarchy;
pub mod path;
pub mod resonance;

// DNA engine — lives in closure_dna/rust/src/
#[path = "../../closure_dna/rust/src/composition_tree.rs"]
pub mod composition_tree;
#[path = "../../closure_dna/rust/src/table.rs"]
pub mod table;

// EA engine — lives in closure_ea/rust/src/
#[path = "../../closure_ea/rust/src/trinity.rs"]
pub mod trinity;

mod pyo3_bindings;

pub use groups::circle::CircleGroup;
pub use groups::hybrid::HybridGroup;
pub use groups::sphere::SphereGroup;
pub use groups::torus::Torus;
pub use groups::LieGroup;
pub use hierarchy::{HierarchicalClosure, LocalizationResult};
pub use path::GeometricPath;
pub use resonance::{ResonanceHit, resonance_scan, resonance_scan_flat};
pub use table::Table;

use pyo3::prelude::*;

#[pymodule]
fn closure_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_bindings::register(m)
}
