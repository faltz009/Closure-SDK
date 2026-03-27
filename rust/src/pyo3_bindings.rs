//! Python bindings — the SDK surface.
//!
//! This is the boundary between Python and Rust. Everything above this
//! layer (group math, paths, embedding, localization) lives in pure Rust.
//! This module exposes it all to Python via PyO3 + numpy, so the SDK user
//! never crosses the FFI boundary for hot-path work.
//!
//! ## What lives here
//!
//! 1. **Group factories** — `circle()`, `sphere()`, `torus(k)`, `hybrid(g1, g2)`
//!    create `PyGroup` wrappers around Rust group objects.
//!
//! 2. **GeometricPath** — batch or streaming path construction, running products,
//!    O(1) coherence checks, O(log n) localization via binary search.
//!
//! 3. **StreamMonitor** — O(1)-memory monitor that ingests raw bytes one at a time.
//!    Same SHA-256 → embed → compose pipeline, but only keeps the running product.
//!
//! 4. **HierarchicalClosure** — reference-vs-test comparison with binary-search
//!    localization once both running-product paths exist.
//!
//! 5. **Full-pipeline helpers** — `path_from_raw_bytes()`, `closure_element_from_*`
//!    run the entire data → summary pipeline in Rust with zero per-element Python cost.
//!
//! ## Pipeline recap
//!
//! ```text
//! raw bytes ──SHA-256──▶ group element ──compose──▶ running product ──distance──▶ σ
//!                         (embed.rs)      (path.rs)                    (groups/)
//! ```
//!
//! ## Quick start (Python)
//!
//! ```python
//! import closure_rs
//!
//! # Pick a geometry
//! g = closure_rs.circle()                    # S¹ — content-only checks
//! g = closure_rs.sphere()                    # S³ — order-sensitive checks
//! g = closure_rs.torus(8)                    # T⁸ — per-channel diagnostics
//! g = closure_rs.hybrid(                     # combine both
//!     closure_rs.torus(8), closure_rs.sphere()
//! )
//!
//! # Build a path and check coherence
//! path = closure_rs.GeometricPath.from_elements(g, np_array)
//! sigma = path.check_global()               # σ ≈ 0 → clean
//!
//! # Localize corruption (O(log n))
//! idx, checks = ref_path.localize_against(test_path)
//!
//! # Stream raw bytes without storing the full path
//! mon = closure_rs.StreamMonitor("Sphere")
//! mon.ingest(b"record_bytes")
//! sigma = mon.sigma()
//! ```

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::groups::circle::CircleGroup;
use crate::groups::hybrid::HybridGroup;
use crate::groups::sphere::SphereGroup;
use crate::groups::torus::Torus;
use crate::groups::LieGroup;

// ── Numpy helpers ────────────────────────────────────────────────────
// These extract contiguous f64 slices from numpy arrays so Rust can
// operate on them without copying. They fail fast if the array isn't
// contiguous (e.g. a transposed view) — the user gets a clear error.

fn as_contiguous_slice_1d<'py>(
    a: &'py PyReadonlyArray1<'py, f64>,
    arg_name: &str,
) -> PyResult<&'py [f64]> {
    a.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{arg_name} must be a contiguous 1D float64 numpy array"
        ))
    })
}

fn check_dim_1d(a: &[f64], expected: usize, arg_name: &str) -> PyResult<()> {
    if a.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{arg_name} has dim {}, expected {}",
            a.len(),
            expected
        )));
    }
    Ok(())
}

fn as_contiguous_slice_2d<'py>(
    a: &'py PyReadonlyArray2<'py, f64>,
    arg_name: &str,
) -> PyResult<&'py [f64]> {
    a.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{arg_name} must be a contiguous 2D float64 numpy array"
        ))
    })
}

// ── GroupDescriptor ──────────────────────────────────────────────────
// Structured representation of which group a PyGroup / StreamMonitor uses.
// Serves two purposes:
//   1. Rebuild the same Rust group object without reparsing a string.
//   2. Route raw bytes to the correct SHA-256 embedding function.

#[derive(Clone, PartialEq, Eq)]
enum GroupDescriptor {
    Circle,
    Sphere,
    Torus(usize),
    Hybrid(Box<GroupDescriptor>, Box<GroupDescriptor>),
}

impl GroupDescriptor {
    fn canonical_name(&self) -> String {
        match self {
            GroupDescriptor::Circle => "Circle".to_string(),
            GroupDescriptor::Sphere => "Sphere".to_string(),
            GroupDescriptor::Torus(k) => format!("Torus({k})"),
            GroupDescriptor::Hybrid(left, right) => {
                format!(
                    "Hybrid({}, {})",
                    left.canonical_name(),
                    right.canonical_name()
                )
            }
        }
    }

    fn to_group(&self) -> Box<dyn LieGroup> {
        match self {
            GroupDescriptor::Circle => Box::new(CircleGroup),
            GroupDescriptor::Sphere => Box::new(SphereGroup),
            GroupDescriptor::Torus(k) => Box::new(Torus::new(*k)),
            GroupDescriptor::Hybrid(left, right) => {
                Box::new(HybridGroup::new(left.to_group(), right.to_group()))
            }
        }
    }

    /// Embed a raw byte record into a group element via SHA-256.
    /// This is the entry point to the pipeline for raw data.
    /// For Hybrid groups, embeds into each factor and concatenates.
    fn embed_record(&self, record: &[u8]) -> Vec<f64> {
        match self {
            GroupDescriptor::Circle => crate::embed::bytes_to_phase(record),
            GroupDescriptor::Sphere => crate::embed::bytes_to_sphere(record),
            GroupDescriptor::Torus(k) => crate::embed::bytes_to_torus(record, *k),
            GroupDescriptor::Hybrid(left, right) => {
                let mut out = left.embed_record(record);
                out.extend(right.embed_record(record));
                out
            }
        }
    }
}

/// Parse a group name string into a GroupDescriptor.
/// Supports recursive nesting: "Hybrid(Torus(8), Hybrid(Circle, Sphere))".
/// Uses paren-depth tracking to find the correct comma in Hybrid(G1, G2).
fn parse_group_descriptor(group_name: &str) -> PyResult<GroupDescriptor> {
    let name = group_name.trim();
    if name == "Circle" {
        return Ok(GroupDescriptor::Circle);
    }
    if name == "Sphere" {
        return Ok(GroupDescriptor::Sphere);
    }
    if let Some(inner) = name
        .strip_prefix("Torus(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let k: usize = inner
            .parse()
            .map_err(|_| PyValueError::new_err(format!("Invalid torus dimension: '{inner}'.")))?;
        if k == 0 {
            return Err(PyValueError::new_err("Torus dimension must be >= 1."));
        }
        return Ok(GroupDescriptor::Torus(k));
    }
    if let Some(inner) = name
        .strip_prefix("Hybrid(")
        .and_then(|s| s.strip_suffix(')'))
    {
        // Find the comma separating two group args, respecting nested parens.
        let mut depth = 0usize;
        let mut comma = None;
        for (i, c) in inner.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth = depth.saturating_sub(1),
                ',' if depth == 0 => {
                    comma = Some(i);
                    break;
                }
                _ => {}
            }
        }
        let comma = comma.ok_or_else(|| {
            PyValueError::new_err(format!(
                "Hybrid requires two groups: 'Hybrid(Circle, Sphere)', got '{name}'."
            ))
        })?;
        let l = parse_group_descriptor(inner[..comma].trim())?;
        let r = parse_group_descriptor(inner[comma + 1..].trim())?;
        return Ok(GroupDescriptor::Hybrid(Box::new(l), Box::new(r)));
    }
    Err(PyValueError::new_err(format!(
        "Unknown group: '{group_name}'. Use Circle, Sphere, Torus(k), or Hybrid(G1, G2)."
    )))
}

// ── PyGroup — the SDK's group handle ────────────────────────────────
// Wraps a Rust LieGroup + its descriptor. Every other Python class
// (GeometricPath, HierarchicalClosure, StreamMonitor) is built from one.

/// Python-facing group handle.
/// Exposes compose, inverse, identity, distance, and random sampling.
/// The descriptor allows child objects to clone the exact same group.
#[pyclass(name = "Group")]
pub struct PyGroup {
    inner: Box<dyn LieGroup>,
    descriptor: GroupDescriptor,
}

#[pymethods]
impl PyGroup {
    /// Group composition: a · b. Returns numpy array.
    fn compose<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let expected = self.inner.dim();
        let a_slice = as_contiguous_slice_1d(&a, "a")?;
        let b_slice = as_contiguous_slice_1d(&b, "b")?;
        check_dim_1d(a_slice, expected, "a")?;
        check_dim_1d(b_slice, expected, "b")?;
        Ok(PyArray1::from_vec(py, self.inner.compose(a_slice, b_slice)))
    }

    /// Group inverse: a⁻¹.
    fn inverse<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let expected = self.inner.dim();
        let a_slice = as_contiguous_slice_1d(&a, "a")?;
        check_dim_1d(a_slice, expected, "a")?;
        Ok(PyArray1::from_vec(py, self.inner.inverse(a_slice)))
    }

    /// Identity element.
    fn identity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.identity())
    }

    /// Geodesic distance from identity: d(a, e).
    fn distance_from_identity(&self, a: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let expected = self.inner.dim();
        let a_slice = as_contiguous_slice_1d(&a, "a")?;
        check_dim_1d(a_slice, expected, "a")?;
        Ok(self.inner.distance_from_identity(a_slice))
    }

    /// Random group element. Optional seed for reproducibility.
    #[pyo3(signature = (seed=None))]
    fn random<'py>(&self, py: Python<'py>, seed: Option<u64>) -> Bound<'py, PyArray1<f64>> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng: Box<dyn rand::RngCore> = match seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::thread_rng()),
        };
        PyArray1::from_vec(py, self.inner.random(&mut *rng))
    }

    /// Element dimension (1 for Circle, 4 for Sphere, k for Torus, etc.)
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn __repr__(&self) -> String {
        format!("closure_rs.Group('{}')", self.descriptor.canonical_name())
    }
}

// ── Group factories ──────────────────────────────────────────────────
// These are the SDK entry points. Users call closure_rs.circle(), etc.

/// Create the circle group S¹ (phases mod 2π).
/// Commutative — detects content changes, ignores ordering.
#[pyfunction]
fn circle() -> PyGroup {
    PyGroup {
        inner: Box::new(CircleGroup),
        descriptor: GroupDescriptor::Circle,
    }
}

/// Create the sphere group S³, implemented with unit quaternions.
/// Non-commutative — detects both content AND ordering changes.
#[pyfunction]
fn sphere() -> PyGroup {
    PyGroup {
        inner: Box::new(SphereGroup),
        descriptor: GroupDescriptor::Sphere,
    }
}

/// Create the k-torus T^k (k independent phase channels).
/// Each channel tracks one account/dimension independently.
/// Use channel_residuals() to see per-channel imbalances.
#[pyfunction]
#[pyo3(signature = (k))]
fn torus(k: usize) -> PyGroup {
    PyGroup {
        inner: Box::new(Torus::new(k)),
        descriptor: GroupDescriptor::Torus(k),
    }
}

/// Create a hybrid group G₁ × G₂ (direct product of two groups).
/// Elements are concatenated: [g₁ | g₂]. Distance = √(d₁² + d₂²).
/// Example: hybrid(torus(8), sphere()) checks account balances AND event ordering.
#[pyfunction]
fn hybrid(g1: &PyGroup, g2: &PyGroup) -> PyResult<PyGroup> {
    let descriptor = GroupDescriptor::Hybrid(
        Box::new(g1.descriptor.clone()),
        Box::new(g2.descriptor.clone()),
    );
    Ok(PyGroup {
        inner: descriptor.to_group(),
        descriptor,
    })
}

// ── Torus diagnostics ────────────────────────────────────────────────
// The only group-specific Python function. All other operations are
// group-agnostic and work through the LieGroup trait.

/// Per-channel signed residuals for a T^k closure element.
///
/// Returns a vector in [-π, π]^k. Each component tells you how far
/// that channel is from zero — positive = excess, negative = deficit.
/// This is the "which account is off and by how much" diagnostic.
#[pyfunction]
#[pyo3(signature = (k, a))]
fn channel_residuals<'py>(
    py: Python<'py>,
    k: usize,
    a: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let t = Torus::new(k);
    let a_slice = as_contiguous_slice_1d(&a, "a")?;
    check_dim_1d(a_slice, k, "a")?;
    Ok(PyArray1::from_vec(py, t.channel_residuals(a_slice)))
}

// ── GeometricPath ────────────────────────────────────────────────────
// The main data structure. Stores running products C_0..C_n (prefix sums
// on the group). Once built, all queries are O(1) — check_global,
// check_range, recover, closure_element. Localization is O(log n).

/// Python wrapper for GeometricPath.
/// Build from pre-embedded elements (numpy 2D array) or append one at a time.
#[pyclass(name = "GeometricPath")]
pub struct PyGeometricPath {
    inner: crate::path::GeometricPath,
}

#[pymethods]
impl PyGeometricPath {
    /// Create an empty path (identity only).
    #[new]
    fn new(group: &PyGroup) -> PyResult<Self> {
        let g = group.descriptor.to_group();
        Ok(Self {
            inner: crate::path::GeometricPath::new(g),
        })
    }

    /// Build entire path from a 2D numpy array (n × dim). All in Rust.
    /// This is the recommended constructor for batch operations.
    #[staticmethod]
    fn from_elements(group: &PyGroup, elements: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let g = group.descriptor.to_group();
        let expected_dim = group.inner.dim();
        let shape = elements.shape();
        if shape[1] != expected_dim {
            return Err(PyValueError::new_err(format!(
                "elements has dim {}, expected {}",
                shape[1], expected_dim
            )));
        }
        let data = as_contiguous_slice_2d(&elements, "elements")?;
        Ok(Self {
            inner: crate::path::GeometricPath::from_elements(g, data, expected_dim),
        })
    }

    /// Append a single element. O(1) — one compose operation.
    fn append(&mut self, g: PyReadonlyArray1<f64>) -> PyResult<()> {
        let expected = self.inner.group().dim();
        let g_slice = as_contiguous_slice_1d(&g, "g")?;
        check_dim_1d(g_slice, expected, "g")?;
        self.inner.append(g_slice);
        Ok(())
    }

    /// Get running product C_t as numpy array.
    fn running_product<'py>(
        &self,
        py: Python<'py>,
        t: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if t > self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "t={} out of range [0, {}]",
                t,
                self.inner.len()
            )));
        }
        Ok(PyArray1::from_vec(
            py,
            self.inner.running_product(t).to_vec(),
        ))
    }

    /// Recover original element g_t = C_{t-1}⁻¹ · C_t. 1-indexed.
    fn recover<'py>(&self, py: Python<'py>, t: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if t < 1 || t > self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "t={} out of range [1, {}]",
                t,
                self.inner.len()
            )));
        }
        Ok(PyArray1::from_vec(py, self.inner.recover(t)))
    }

    /// Closure scalar for sub-range [i+1..j]. O(1).
    fn check_range(&self, i: usize, j: usize) -> PyResult<f64> {
        if j > self.inner.len() || i >= j {
            return Err(PyValueError::new_err(format!(
                "invalid range [{}, {}], expected 0 <= i < j <= {}",
                i,
                j,
                self.inner.len()
            )));
        }
        Ok(self.inner.check_range(i, j))
    }

    /// Global closure scalar σ = d(C_n, identity). O(1).
    fn check_global(&self) -> f64 {
        self.inner.check_global()
    }

    /// The final running product C_n.
    fn closure_element<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.closure_element())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Compare running products at position t between two paths. O(1).
    /// Returns d(C_t^ref, C_t^test): 0 if identical up to t, > 0 if diverged.
    fn compare_at(&self, other: &PyGeometricPath, t: usize) -> PyResult<f64> {
        if self.inner.group().dim() != other.inner.group().dim() {
            return Err(PyValueError::new_err(format!(
                "path dimension mismatch: {} vs {}",
                self.inner.group().dim(),
                other.inner.group().dim()
            )));
        }
        if t > self.inner.len() || t > other.inner.len() {
            return Err(PyValueError::new_err(format!(
                "t={} out of range [0, {}] and [0, {}]",
                t,
                self.inner.len(),
                other.inner.len()
            )));
        }
        let inv = self.inner.group().inverse(self.inner.running_product(t));
        let rel = self
            .inner
            .group()
            .compose(&inv, other.inner.running_product(t));
        Ok(self.inner.group().distance_from_identity(&rel))
    }

    /// O(log n) fault localization: binary search for the first divergence point.
    ///
    /// Compares running products at midpoints to narrow down where ref and test
    /// first disagree. Returns (corrupted_index_or_none, num_comparisons).
    /// The entire search runs in Rust — no per-step Python overhead.
    #[pyo3(signature = (other, threshold = 1e-6))]
    fn localize_against(
        &self,
        other: &PyGeometricPath,
        threshold: f64,
    ) -> PyResult<(Option<usize>, usize)> {
        if self.inner.group().dim() != other.inner.group().dim() {
            return Err(PyValueError::new_err(format!(
                "path dimension mismatch: {} vs {}",
                self.inner.group().dim(),
                other.inner.group().dim()
            )));
        }
        let n_ref = self.inner.len();
        let n_test = other.inner.len();
        let n_shared = n_ref.min(n_test);
        let g = self.inner.group();

        let result = crate::hierarchy::binary_search_divergence(
            n_shared,
            n_ref,
            n_test,
            threshold,
            |t| {
                let inv = g.inverse(self.inner.running_product(t));
                let rel = g.compose(&inv, other.inner.running_product(t));
                g.distance_from_identity(&rel)
            },
        );
        Ok((result.index, result.checks))
    }
}

// ── StreamMonitor ────────────────────────────────────────────────────
// O(1) memory monitor for raw-byte streams. Same pipeline as
// path_from_raw_bytes (SHA-256 → embed → compose), but only keeps the
// running product — no path, no localization, just detection.

/// Streaming closure monitor. Ingests raw bytes one record at a time.
///
/// Keeps only the running product (constant memory). Call sigma() at any
/// point to check coherence. If σ > 0, something changed — but you can't
/// localize without the full path. Use GeometricPath for that.
#[pyclass(name = "StreamMonitor")]
pub struct PyStreamMonitor {
    kind: GroupDescriptor,
    name: String,
    group: Box<dyn LieGroup>,
    running: Vec<f64>,
    buf: Vec<f64>,
    n: usize,
}

impl PyStreamMonitor {
    fn ingest_one(&mut self, record: &[u8]) {
        let elem = self.kind.embed_record(record);
        self.group.compose_into(&self.running, &elem, &mut self.buf);
        self.running.copy_from_slice(&self.buf);
        self.n += 1;
    }
}

#[pymethods]
impl PyStreamMonitor {
    /// Create a streaming monitor for raw bytes.
    ///
    /// `group_name`: e.g. "Circle", "Sphere", "Torus(8)", "Hybrid(Torus(8), Sphere)".
    #[new]
    #[pyo3(signature = (group_name))]
    fn new(group_name: &str) -> PyResult<Self> {
        let descriptor = parse_group_descriptor(group_name)?;
        let name = descriptor.canonical_name();
        let group = descriptor.to_group();
        let running = group.identity();
        let buf = vec![0.0; group.dim()];
        Ok(Self {
            kind: descriptor,
            name,
            group,
            running,
            buf,
            n: 0,
        })
    }

    /// Ingest one raw record and update the running closure element.
    fn ingest(&mut self, record: &[u8]) {
        self.ingest_one(record);
    }

    /// Ingest many raw records.
    fn ingest_many(&mut self, records: Vec<Vec<u8>>) {
        for r in &records {
            self.ingest_one(r);
        }
    }

    /// Current closure element C_n.
    fn closure_element<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.running.clone())
    }

    /// Current sigma: geodesic distance from identity.
    fn sigma(&self) -> f64 {
        self.group.distance_from_identity(&self.running)
    }

    /// Compare two monitors of the same group: d(C_self^{-1} · C_other, e).
    fn compare_against(&self, other: &PyStreamMonitor) -> PyResult<f64> {
        if self.kind != other.kind {
            return Err(PyValueError::new_err(
                "monitor group mismatch: both monitors must use the same group name",
            ));
        }
        let inv = self.group.inverse(&self.running);
        let rel = self.group.compose(&inv, &other.running);
        Ok(self.group.distance_from_identity(&rel))
    }

    /// Reset to identity with zero ingested records.
    fn reset(&mut self) {
        self.running = self.group.identity();
        self.n = 0;
    }

    /// Number of ingested records.
    fn __len__(&self) -> usize {
        self.n
    }

    /// Group name used by this monitor.
    fn group_name(&self) -> String {
        self.name.clone()
    }

    fn __repr__(&self) -> String {
        format!("closure_rs.StreamMonitor('{}', n={})", self.name, self.n)
    }
}

// ── HierarchicalClosure ─────────────────────────────────────────────
// Reference-vs-test comparison. Stores the reference running products
// once, then localizes corruption in any test sequence with O(log n)
// comparisons. See hierarchy.rs for the algorithm.

/// Reference-based fault localization.
/// Build once from known-good elements, then check/localize any test sequence.
#[pyclass(name = "HierarchicalClosure")]
pub struct PyHierarchicalClosure {
    inner: crate::hierarchy::HierarchicalClosure,
    dim: usize,
}

#[pymethods]
impl PyHierarchicalClosure {
    /// Build from known-good elements (n × dim numpy array).
    #[new]
    #[pyo3(signature = (group, elements))]
    fn new(group: &PyGroup, elements: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let g = group.descriptor.to_group();
        let expected_dim = group.inner.dim();
        let shape = elements.shape();
        let n = shape[0];
        let dim = shape[1];
        if dim != expected_dim {
            return Err(PyValueError::new_err(format!(
                "elements has dim {}, expected {}",
                dim, expected_dim
            )));
        }

        let data = as_contiguous_slice_2d(&elements, "elements")?;
        let rows: Vec<&[f64]> = (0..n).map(|i| &data[i * dim..(i + 1) * dim]).collect();

        Ok(Self {
            inner: crate::hierarchy::HierarchicalClosure::new(g, &rows),
            dim: expected_dim,
        })
    }

    /// Check test data against reference. Returns σ (geodesic distance).
    fn check(&self, test_elements: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let shape = test_elements.shape();
        let n = shape[0];
        let dim = shape[1];
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "test_elements has dim {}, expected {}",
                dim, self.dim
            )));
        }
        let data = as_contiguous_slice_2d(&test_elements, "test_elements")?;
        let rows: Vec<&[f64]> = (0..n).map(|i| &data[i * dim..(i + 1) * dim]).collect();
        Ok(self.inner.check(&rows))
    }

    /// Localize corruption. Returns (index_or_none, checks, depth).
    #[pyo3(signature = (test_elements, threshold = 1e-6))]
    fn localize(
        &self,
        test_elements: PyReadonlyArray2<f64>,
        threshold: f64,
    ) -> PyResult<(Option<usize>, usize, usize)> {
        let shape = test_elements.shape();
        let n = shape[0];
        let dim = shape[1];
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "test_elements has dim {}, expected {}",
                dim, self.dim
            )));
        }
        let data = as_contiguous_slice_2d(&test_elements, "test_elements")?;
        let rows: Vec<&[f64]> = (0..n).map(|i| &data[i * dim..(i + 1) * dim]).collect();
        let result = self.inner.localize(&rows, threshold);
        Ok((result.index, result.checks, result.depth))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

// ── Full-pipeline helpers ────────────────────────────────────────────
// These run the entire pipeline (raw bytes → SHA-256 → embed → compose)
// in Rust. No per-element round-trip to Python.

/// Build a GeometricPath from raw byte records — full pipeline in Rust.
///
/// Each record is hashed (SHA-256), embedded into a group element, and
/// composed into the running product. Returns a full path with all
/// intermediates, so you can localize corruption afterward.
#[pyfunction]
#[pyo3(signature = (group_name, records))]
fn path_from_raw_bytes(group_name: &str, records: Vec<Vec<u8>>) -> PyResult<PyGeometricPath> {
    let descriptor = parse_group_descriptor(group_name)?;
    let mut path = crate::path::GeometricPath::new(descriptor.to_group());
    for r in &records {
        let elem = descriptor.embed_record(r);
        path.append(&elem);
    }
    Ok(PyGeometricPath { inner: path })
}

/// Compute only the closure element from raw bytes — O(1) memory.
///
/// Same pipeline as path_from_raw_bytes, but discards intermediates.
/// Use when you only need σ (detection), not localization.
#[pyfunction]
#[pyo3(signature = (group_name, records))]
fn closure_element_from_raw_bytes<'py>(
    py: Python<'py>,
    group_name: &str,
    records: Vec<Vec<u8>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let descriptor = parse_group_descriptor(group_name)?;
    let group = descriptor.to_group();
    let mut running = group.identity();
    let mut buf = vec![0.0; group.dim()];
    for r in &records {
        let elem = descriptor.embed_record(r);
        group.compose_into(&running, &elem, &mut buf);
        running.copy_from_slice(&buf);
    }
    Ok(PyArray1::from_vec(py, running))
}

/// Closure element from pre-embedded elements — O(1) memory.
///
/// Like GeometricPath.from_elements() but only returns the final running
/// product. Use when elements are already embedded (numpy array, not raw bytes).
#[pyfunction]
#[pyo3(signature = (group, elements))]
fn closure_element_from_elements<'py>(
    py: Python<'py>,
    group: &PyGroup,
    elements: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let expected_dim = group.inner.dim();
    let shape = elements.shape();
    if shape[1] != expected_dim {
        return Err(PyValueError::new_err(format!(
            "elements has dim {}, expected {}",
            shape[1], expected_dim
        )));
    }
    let data = as_contiguous_slice_2d(&elements, "elements")?;
    let result =
        crate::embed::closure_element_from_elements(group.inner.as_ref(), data, expected_dim);
    Ok(PyArray1::from_vec(py, result))
}

/// Curriculum vote collection for m-factor teaching on S³.
///
/// Takes a (V × dim) embedding table and a list of n-gram index sequences.
/// For each n-gram, composes all context words (all except the last),
/// inverts the context, and adds it as a vote for the last word's position.
///
/// Returns (vote_sums, vote_counts): a (V × dim) array of accumulated
/// inverse-context votes, and a (V,) array of how many votes each word got.
///
/// The m-factor structure (e.g. m=2+2 → dim=16) is handled by composing
/// 4 independent S³ factors, each 4 floats wide.
///
/// This replaces the Python n-gram loop that was the teaching bottleneck.
#[pyfunction]
#[pyo3(signature = (embeddings, ngrams, n_factors))]
fn curriculum_votes<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<f64>,
    ngrams: Vec<Vec<usize>>,
    n_factors: usize,
) -> PyResult<(Bound<'py, numpy::PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let emb_data = as_contiguous_slice_2d(&embeddings, "embeddings")?;
    let shape = embeddings.shape();
    let v = shape[0];
    let dim = shape[1];

    if dim != n_factors * 4 {
        return Err(PyValueError::new_err(format!(
            "embeddings dim {} != n_factors {} * 4",
            dim, n_factors
        )));
    }

    let sphere = SphereGroup;
    let mut vote_sums = vec![0.0f64; v * dim];
    let mut vote_counts = vec![0.0f64; v];

    // Scratch space for composition
    let mut context = vec![0.0f64; dim];
    let mut temp = vec![0.0f64; 4];

    for ng in &ngrams {
        if ng.len() < 2 {
            continue;
        }

        let target_idx = ng[ng.len() - 1];
        if target_idx >= v {
            continue;
        }

        // Initialize context to identity on each factor
        for f in 0..n_factors {
            let s = f * 4;
            context[s] = 1.0;
            context[s + 1] = 0.0;
            context[s + 2] = 0.0;
            context[s + 3] = 0.0;
        }

        // Compose context words (all except last)
        for &word_idx in &ng[..ng.len() - 1] {
            if word_idx >= v {
                continue;
            }
            let word_offset = word_idx * dim;
            for f in 0..n_factors {
                let s = f * 4;
                sphere.compose_into(
                    &context[s..s + 4],
                    &emb_data[word_offset + s..word_offset + s + 4],
                    &mut temp,
                );
                context[s..s + 4].copy_from_slice(&temp);
            }
        }

        // Invert context and accumulate as vote
        let vote_offset = target_idx * dim;
        for f in 0..n_factors {
            let s = f * 4;
            vote_sums[vote_offset + s] += context[s];       // w stays
            vote_sums[vote_offset + s + 1] -= context[s + 1]; // x flips
            vote_sums[vote_offset + s + 2] -= context[s + 2]; // y flips
            vote_sums[vote_offset + s + 3] -= context[s + 3]; // z flips
        }
        vote_counts[target_idx] += 1.0;
    }

    let sums = numpy::PyArray2::from_vec2(
        py,
        &vote_sums
            .chunks(dim)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>(),
    )?;
    let counts = PyArray1::from_vec(py, vote_counts);

    Ok((sums, counts))
}

/// Weighted curriculum votes — same as curriculum_votes but each context
/// word's vote is scaled by its information content weight.
///
/// Rare content words steer stronger. Common function words steer weaker.
/// This should spread positions away from the pole because the dominant
/// votes come from distinctive directions, not identity-hugging function words.
#[pyfunction]
#[pyo3(signature = (embeddings, ngrams, n_factors, weights))]
fn curriculum_votes_weighted<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<f64>,
    ngrams: Vec<Vec<usize>>,
    n_factors: usize,
    weights: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, numpy::PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let emb_data = as_contiguous_slice_2d(&embeddings, "embeddings")?;
    let w_data = as_contiguous_slice_1d(&weights, "weights")?;
    let shape = embeddings.shape();
    let v = shape[0];
    let dim = shape[1];

    if dim != n_factors * 4 {
        return Err(PyValueError::new_err(format!(
            "embeddings dim {} != n_factors {} * 4", dim, n_factors
        )));
    }
    if w_data.len() != v {
        return Err(PyValueError::new_err(format!(
            "weights len {} != vocab size {}", w_data.len(), v
        )));
    }

    let sphere = SphereGroup;
    let mut vote_sums = vec![0.0f64; v * dim];
    let mut vote_counts = vec![0.0f64; v];
    let mut context = vec![0.0f64; dim];
    let mut temp = vec![0.0f64; 4];

    for ng in &ngrams {
        if ng.len() < 2 { continue; }
        let target_idx = ng[ng.len() - 1];
        if target_idx >= v { continue; }

        // Identity init
        for f in 0..n_factors {
            let s = f * 4;
            context[s] = 1.0;
            context[s+1] = 0.0;
            context[s+2] = 0.0;
            context[s+3] = 0.0;
        }

        // Compose context words
        // Track total weight of context for normalization
        let mut total_weight = 0.0f64;
        for &word_idx in &ng[..ng.len() - 1] {
            if word_idx >= v { continue; }
            let word_offset = word_idx * dim;
            for f in 0..n_factors {
                let s = f * 4;
                sphere.compose_into(
                    &context[s..s+4],
                    &emb_data[word_offset + s..word_offset + s + 4],
                    &mut temp,
                );
                context[s..s+4].copy_from_slice(&temp);
            }
            total_weight += w_data[word_idx];
        }

        // Weight = average info content of context words
        let vote_weight = if ng.len() > 2 {
            total_weight / (ng.len() - 1) as f64
        } else {
            w_data[ng[0]]
        };

        // Invert and accumulate weighted vote
        let vote_offset = target_idx * dim;
        for f in 0..n_factors {
            let s = f * 4;
            vote_sums[vote_offset + s]     += vote_weight * context[s];
            vote_sums[vote_offset + s + 1] -= vote_weight * context[s + 1];
            vote_sums[vote_offset + s + 2] -= vote_weight * context[s + 2];
            vote_sums[vote_offset + s + 3] -= vote_weight * context[s + 3];
        }
        vote_counts[target_idx] += vote_weight;
    }

    let sums = numpy::PyArray2::from_vec2(
        py,
        &vote_sums.chunks(dim).map(|c| c.to_vec()).collect::<Vec<_>>(),
    )?;
    let counts = PyArray1::from_vec(py, vote_counts);
    Ok((sums, counts))
}

/// Refinement vote collection: progressive composition within sentences.
///
/// For each sentence [w0, w1, w2, ...], composes progressively:
///   context=[w0]         → vote for w1
///   context=[w0,w1]      → vote for w2
///   context=[w0,w1,w2]   → vote for w3
///   etc.
///
/// This avoids building millions of Python n-gram lists.
/// All composition happens in Rust.
#[pyfunction]
#[pyo3(signature = (embeddings, sentences, n_factors))]
fn refinement_votes<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<f64>,
    sentences: Vec<Vec<usize>>,
    n_factors: usize,
) -> PyResult<(Bound<'py, numpy::PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let emb_data = as_contiguous_slice_2d(&embeddings, "embeddings")?;
    let shape = embeddings.shape();
    let v = shape[0];
    let dim = shape[1];

    if dim != n_factors * 4 {
        return Err(PyValueError::new_err(format!(
            "embeddings dim {} != n_factors {} * 4",
            dim, n_factors
        )));
    }

    let sphere = SphereGroup;
    let mut vote_sums = vec![0.0f64; v * dim];
    let mut vote_counts = vec![0.0f64; v];
    let mut context = vec![0.0f64; dim];
    let mut temp = vec![0.0f64; 4];

    for sent in &sentences {
        if sent.len() < 2 {
            continue;
        }

        // Reset context to identity
        for f in 0..n_factors {
            let s = f * 4;
            context[s] = 1.0;
            context[s + 1] = 0.0;
            context[s + 2] = 0.0;
            context[s + 3] = 0.0;
        }

        // Progressive composition through the sentence
        for t in 0..sent.len() - 1 {
            let word_idx = sent[t];
            if word_idx >= v {
                continue;
            }

            // Compose word into running context
            let word_offset = word_idx * dim;
            for f in 0..n_factors {
                let s = f * 4;
                sphere.compose_into(
                    &context[s..s + 4],
                    &emb_data[word_offset + s..word_offset + s + 4],
                    &mut temp,
                );
                context[s..s + 4].copy_from_slice(&temp);
            }

            // Vote: next word should be near context⁻¹
            let target_idx = sent[t + 1];
            if target_idx >= v {
                continue;
            }
            let vote_offset = target_idx * dim;
            for f in 0..n_factors {
                let s = f * 4;
                vote_sums[vote_offset + s] += context[s];
                vote_sums[vote_offset + s + 1] -= context[s + 1];
                vote_sums[vote_offset + s + 2] -= context[s + 2];
                vote_sums[vote_offset + s + 3] -= context[s + 3];
            }
            vote_counts[target_idx] += 1.0;
        }
    }

    let sums = numpy::PyArray2::from_vec2(
        py,
        &vote_sums
            .chunks(dim)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>(),
    )?;
    let counts = PyArray1::from_vec(py, vote_counts);

    Ok((sums, counts))
}

/// Score all vocabulary words against a target quaternion.
///
/// For each word in the (V × dim) embedding table, computes the
/// average dot product across all m factors between the target
/// and the word's quaternion. Returns a (V,) array of scores.
///
/// This replaces the Python loop:
///   dots = [mf_dot(target, full[i]) for i in range(V)]
/// which is the generation/babbling bottleneck at 21K+ words.
#[pyfunction]
#[pyo3(signature = (target, embeddings, n_factors))]
fn score_vocabulary<'py>(
    py: Python<'py>,
    target: PyReadonlyArray1<f64>,
    embeddings: PyReadonlyArray2<f64>,
    n_factors: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let tgt = as_contiguous_slice_1d(&target, "target")?;
    let emb_data = as_contiguous_slice_2d(&embeddings, "embeddings")?;
    let shape = embeddings.shape();
    let v = shape[0];
    let dim = shape[1];

    if dim != n_factors * 4 {
        return Err(PyValueError::new_err(format!(
            "embeddings dim {} != n_factors {} * 4", dim, n_factors
        )));
    }
    if tgt.len() != dim {
        return Err(PyValueError::new_err(format!(
            "target len {} != dim {}", tgt.len(), dim
        )));
    }

    let mut scores = vec![0.0f64; v];

    for i in 0..v {
        let word_offset = i * dim;
        let mut total_dot = 0.0;
        for f in 0..n_factors {
            let s = f * 4;
            total_dot += tgt[s]     * emb_data[word_offset + s]
                       + tgt[s + 1] * emb_data[word_offset + s + 1]
                       + tgt[s + 2] * emb_data[word_offset + s + 2]
                       + tgt[s + 3] * emb_data[word_offset + s + 3];
        }
        scores[i] = total_dot / n_factors as f64;
    }

    Ok(PyArray1::from_vec(py, scores))
}

/// Collect follower statistics at multiple timescales in a single pass.
///
/// For each sentence, extracts what follows each word/phrase at different
/// window sizes. Returns a dict mapping window_size → {word_idx → [(follower_idx, count)]}.
///
/// Window 2 (bigrams): what single word follows this word?
/// Window 4: what word follows this 3-word phrase?
/// Window 8: what word follows this 7-word phrase?
///
/// Each Enkidu cell level uses followers at its own timescale:
/// - Word cell uses window=2 (bigram followers)
/// - Phrase cell uses window=4
/// - Sentence cell uses window=8
///
/// This enables dynamic Enkidu spawning: more context words in the prompt
/// activate higher-level cells that use wider-window followers.
#[pyfunction]
#[pyo3(signature = (sentences, vocab_size, windows, top_k = 50))]
fn collect_followers_multi<'py>(
    py: Python<'py>,
    sentences: Vec<Vec<usize>>,
    vocab_size: usize,
    windows: Vec<usize>,
    top_k: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use std::collections::HashMap;

    // For each window size, collect follower counts per word
    // Key: (window_size, context_last_word) → HashMap<follower, count>
    let mut tables: HashMap<usize, Vec<HashMap<usize, u32>>> = HashMap::new();
    for &w in &windows {
        let mut t = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            t.push(HashMap::new());
        }
        tables.insert(w, t);
    }

    for sent in &sentences {
        if sent.len() < 2 {
            continue;
        }
        for &w in &windows {
            if sent.len() < w {
                continue;
            }
            for i in 0..sent.len() - w + 1 {
                // The context is sent[i..i+w-1], the follower is sent[i+w-1]
                // Key the followers by the LAST word of the context
                // (the word just before the follower)
                let key_word = sent[i + w - 2]; // last context word
                let follower = sent[i + w - 1]; // the word that follows
                if key_word < vocab_size && follower < vocab_size {
                    if let Some(table) = tables.get_mut(&w) {
                        *table[key_word].entry(follower).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Convert to Python: dict of window → list of (word_idx → top_k followers)
    let result = pyo3::types::PyDict::new(py);
    for &w in &windows {
        if let Some(table) = tables.get(&w) {
            // For each word, sort followers by count and keep top_k
            let word_followers = pyo3::types::PyDict::new(py);
            for (word_idx, followers) in table.iter().enumerate() {
                if followers.is_empty() {
                    continue;
                }
                let mut sorted: Vec<(usize, u32)> = followers.iter()
                    .map(|(&k, &v)| (k, v))
                    .collect();
                sorted.sort_by(|a, b| b.1.cmp(&a.1));
                sorted.truncate(top_k);

                let py_list: Vec<(usize, u32)> = sorted;
                word_followers.set_item(word_idx, py_list)?;
            }
            result.set_item(w, word_followers)?;
        }
    }

    Ok(result)
}

/// Build metadata — confirms which .so Python actually loaded.
/// Returns (manifest_dir, version).
#[pyfunction]
fn build_info() -> (&'static str, &'static str) {
    (env!("CARGO_MANIFEST_DIR"), env!("CARGO_PKG_VERSION"))
}

// ── Module registration ─────────────────────────────────────────────

/// Register all Python-visible functions and classes.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(circle, m)?)?;
    m.add_function(wrap_pyfunction!(sphere, m)?)?;
    m.add_function(wrap_pyfunction!(torus, m)?)?;
    m.add_function(wrap_pyfunction!(hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(channel_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(path_from_raw_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(closure_element_from_raw_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(closure_element_from_elements, m)?)?;
    m.add_function(wrap_pyfunction!(curriculum_votes, m)?)?;
    m.add_function(wrap_pyfunction!(curriculum_votes_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(refinement_votes, m)?)?;
    m.add_function(wrap_pyfunction!(score_vocabulary, m)?)?;
    m.add_function(wrap_pyfunction!(collect_followers_multi, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_class::<PyGroup>()?;
    m.add_class::<PyGeometricPath>()?;
    m.add_class::<PyStreamMonitor>()?;
    m.add_class::<PyHierarchicalClosure>()?;
    Ok(())
}
