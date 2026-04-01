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

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

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

    /// Embed a raw byte record into a group element.
    /// For Sphere, `hashed=true` selects SHA-256 embedding and
    /// `hashed=false` selects direct geometric byte composition.
    /// For Hybrid groups, the same choice is applied recursively.
    fn embed_record(&self, record: &[u8], hashed: bool) -> Vec<f64> {
        match self {
            GroupDescriptor::Circle => crate::embed::bytes_to_phase(record),
            GroupDescriptor::Sphere => crate::embed::bytes_to_sphere(record, hashed),
            GroupDescriptor::Torus(k) => crate::embed::bytes_to_torus(record, *k),
            GroupDescriptor::Hybrid(left, right) => {
                let mut out = left.embed_record(record, hashed);
                out.extend(right.embed_record(record, hashed));
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
    hashed: bool,
    name: String,
    group: Box<dyn LieGroup>,
    running: Vec<f64>,
    buf: Vec<f64>,
    n: usize,
}

impl PyStreamMonitor {
    fn ingest_one(&mut self, record: &[u8]) {
        let elem = self.kind.embed_record(record, self.hashed);
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
    /// `hashed`: for Sphere/Hybrid(Sphere, ...), choose SHA-256 or direct geometric embedding.
    #[new]
    #[pyo3(signature = (group_name, hashed = true))]
    fn new(group_name: &str, hashed: bool) -> PyResult<Self> {
        let descriptor = parse_group_descriptor(group_name)?;
        let name = descriptor.canonical_name();
        let group = descriptor.to_group();
        let running = group.identity();
        let buf = vec![0.0; group.dim()];
        Ok(Self {
            kind: descriptor,
            hashed,
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
// These run the entire pipeline (raw bytes → embed → compose)
// in Rust. No per-element round-trip to Python.

/// Build a GeometricPath from raw byte records — full pipeline in Rust.
///
/// Each record is embedded into a group element and composed into the
/// running product. For Sphere, `hashed=true` uses SHA-256 embedding and
/// `hashed=false` uses direct geometric byte composition.
#[pyfunction]
#[pyo3(signature = (group_name, records, hashed = true))]
fn path_from_raw_bytes(
    group_name: &str,
    records: Vec<Vec<u8>>,
    hashed: bool,
) -> PyResult<PyGeometricPath> {
    let descriptor = parse_group_descriptor(group_name)?;
    let mut path = crate::path::GeometricPath::new(descriptor.to_group());
    for r in &records {
        let elem = descriptor.embed_record(r, hashed);
        path.append(&elem);
    }
    Ok(PyGeometricPath { inner: path })
}

/// Compute only the closure element from raw bytes — O(1) memory.
///
/// Same pipeline as path_from_raw_bytes, but discards intermediates.
/// Use when you only need σ (detection), not localization.
#[pyfunction]
#[pyo3(signature = (group_name, records, hashed = true))]
fn closure_element_from_raw_bytes<'py>(
    py: Python<'py>,
    group_name: &str,
    records: Vec<Vec<u8>>,
    hashed: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let descriptor = parse_group_descriptor(group_name)?;
    let group = descriptor.to_group();
    let mut running = group.identity();
    let mut buf = vec![0.0; group.dim()];
    for r in &records {
        let elem = descriptor.embed_record(r, hashed);
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
/// Train a cell classifier on S³ via algebraic correction — ENTIRELY IN RUST.
///
/// Each cell has N_FEAT feature indices into a genome table.
/// The composition of feature quaternions should land near the target quaternion.
/// For each cell: compose features, compute error via Hopf, do ideal_i correction.
///
/// Args:
///   genome: (G × 4) mutable array of quaternion positions (modified in place)
///   cells: (C × N_FEAT) array of feature indices (into genome rows)
///   targets: (C × 4) array of target quaternions
///   n_epochs: number of training passes
///   damping: learning rate
///   seed: random seed for shuffle
///
/// Returns: training accuracy (fraction correct)
#[pyfunction]
#[pyo3(signature = (genome, cells, targets, n_epochs, damping, seed=42))]
fn train_cells<'py>(
    _py: Python<'py>,
    mut genome: numpy::PyReadwriteArray2<'py, f64>,
    cells: PyReadonlyArray2<i32>,
    targets: PyReadonlyArray2<f64>,
    n_epochs: usize,
    damping: f64,
    seed: u64,
) -> PyResult<f64> {
    let cells_data = cells.as_array();
    let targets_data = targets.as_array();
    let n_cells = cells_data.shape()[0];
    let n_feat = cells_data.shape()[1];
    let mut genome_data = genome.as_array_mut();
    let n_genome = genome_data.shape()[0];

    // Inline Hamilton product (no allocation)
    #[inline(always)]
    fn ham(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
        [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
         a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
         a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
         a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]]
    }

    #[inline(always)]
    fn norm(q: &mut [f64; 4]) {
        let n = (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]).sqrt();
        if n > 1e-15 { let inv = 1.0/n; q[0]*=inv; q[1]*=inv; q[2]*=inv; q[3]*=inv; }
        else { *q = [1.0, 0.0, 0.0, 0.0]; }
    }

    #[inline(always)]
    fn inv(a: &[f64; 4]) -> [f64; 4] { [a[0], -a[1], -a[2], -a[3]] }

    #[inline(always)]
    fn compose(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
        let mut r = ham(a, b); norm(&mut r); r
    }

    // LCG for shuffling
    let mut rng = seed;
    let mut next = || -> u64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng
    };

    let mut order: Vec<usize> = (0..n_cells).collect();

    for _epoch in 0..n_epochs {
        // Shuffle
        for i in (1..n_cells).rev() {
            let j = (next() as usize) % (i + 1);
            order.swap(i, j);
        }

        for &idx in &order {
            let mut feat_idx = vec![0usize; n_feat];
            for i in 0..n_feat {
                let fi = cells_data[[idx, i]] as usize;
                feat_idx[i] = if fi < n_genome { fi } else { 0 };
            }

            let target: [f64; 4] = [targets_data[[idx,0]], targets_data[[idx,1]],
                                     targets_data[[idx,2]], targets_data[[idx,3]]];

            // Read quats from genome
            let mut quats = vec![[0.0f64; 4]; n_feat];
            for i in 0..n_feat {
                let fi = feat_idx[i];
                quats[i] = [genome_data[[fi,0]], genome_data[[fi,1]],
                            genome_data[[fi,2]], genome_data[[fi,3]]];
            }

            // Compose
            let mut c = [1.0, 0.0, 0.0, 0.0];
            for i in 0..n_feat { c = compose(&c, &quats[i]); }

            // Error
            let eq = compose(&c, &inv(&target));
            let es = eq[0].abs().min(1.0).acos();
            if es < 1e-6 { continue; }

            let w_mag = eq[0].abs();
            let rgb_mag = (eq[1]*eq[1]+eq[2]*eq[2]+eq[3]*eq[3]).sqrt();
            let tf = if w_mag > rgb_mag { 1.0 } else { 0.5 };
            let step = damping * (es / std::f64::consts::PI) * tf;

            // Left products
            let mut left = vec![[1.0, 0.0, 0.0, 0.0]; n_feat + 1];
            for i in 0..n_feat { left[i+1] = compose(&left[i], &quats[i]); }

            // Right products
            let mut right = vec![[1.0, 0.0, 0.0, 0.0]; n_feat + 1];
            for i in (0..n_feat).rev() { right[i] = compose(&quats[i], &right[i+1]); }

            // Correct each feature
            for i in 0..n_feat {
                let ideal = compose(&compose(&inv(&left[i]), &target), &inv(&right[i+1]));
                let fi = feat_idx[i];
                let pos: [f64; 4] = [genome_data[[fi,0]], genome_data[[fi,1]],
                                      genome_data[[fi,2]], genome_data[[fi,3]]];

                let mut dot = pos[0]*ideal[0]+pos[1]*ideal[1]+pos[2]*ideal[2]+pos[3]*ideal[3];
                let ideal_adj = if dot < 0.0 { dot = -dot; [-ideal[0],-ideal[1],-ideal[2],-ideal[3]] } else { ideal };

                if dot > 0.9999 {
                    for d in 0..4 { genome_data[[fi, d]] = ideal_adj[d]; }
                } else {
                    let theta = dot.min(1.0).acos();
                    let s = theta.sin();
                    if s > 1e-8 {
                        let a = ((1.0-step)*theta).sin()/s;
                        let b = (step*theta).sin()/s;
                        let mut np = [a*pos[0]+b*ideal_adj[0], a*pos[1]+b*ideal_adj[1],
                                      a*pos[2]+b*ideal_adj[2], a*pos[3]+b*ideal_adj[3]];
                        norm(&mut np);
                        for d in 0..4 { genome_data[[fi, d]] = np[d]; }
                    }
                }
            }
        }
    }

    // Training accuracy: compose each cell, dot with its target
    let mut correct = 0usize;
    for idx in 0..n_cells {
        let mut c = [1.0, 0.0, 0.0, 0.0];
        for i in 0..n_feat {
            let fi = cells_data[[idx, i]] as usize;
            let fi = if fi < n_genome { fi } else { 0 };
            let q: [f64; 4] = [genome_data[[fi,0]], genome_data[[fi,1]],
                                genome_data[[fi,2]], genome_data[[fi,3]]];
            c = compose(&c, &q);
        }
        let t: [f64; 4] = [targets_data[[idx,0]], targets_data[[idx,1]],
                            targets_data[[idx,2]], targets_data[[idx,3]]];
        let dot = (c[0]*t[0]+c[1]*t[1]+c[2]*t[2]+c[3]*t[3]).abs();
        if dot > 0.5 { correct += 1; }
    }

    Ok(correct as f64 / n_cells.max(1) as f64)
}

/// Train on continuous features via algebraic correction on S³.
///
/// Each continuous feature value becomes a quaternion rotation:
///   q = [cos(θ/2), sin(θ/2) * axis]
///   where θ = feature_value (pre-scaled by caller)
///   and axis = learned 3-vector (one per feature, stored in `axes`)
///
/// The composition of all feature quaternions should match the target.
/// The exact Lie gradient adjusts the axes to minimize prediction error.
///
/// Args:
///   features: (N × F) float64 array of continuous feature values
///   targets: (N × 4) float64 array of target quaternions
///   axes: (F × 3) mutable float64 array of axis vectors (learned, modified in place)
///   n_epochs: number of training passes
///   damping: learning rate
///   seed: random seed
///
/// Returns: (predictions: N×4 float64, accuracy: float64)
#[pyfunction]
#[pyo3(signature = (features, targets, axes, n_epochs, damping, seed=42))]
fn train_continuous<'py>(
    py: Python<'py>,
    features: PyReadonlyArray2<f64>,
    targets: PyReadonlyArray2<f64>,
    mut axes: numpy::PyReadwriteArray2<'py, f64>,
    n_epochs: usize,
    damping: f64,
    seed: u64,
) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
    let feat_data = features.as_array();
    let tgt_data = targets.as_array();
    let mut axes_data = axes.as_array_mut();
    let n_samples = feat_data.shape()[0];
    let n_feat = feat_data.shape()[1];

    #[inline(always)]
    fn ham(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
        [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
         a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
         a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
         a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]]
    }
    #[inline(always)]
    fn qnorm(q: &mut [f64; 4]) {
        let n = (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]).sqrt();
        if n > 1e-15 { let inv=1.0/n; q[0]*=inv; q[1]*=inv; q[2]*=inv; q[3]*=inv; }
        else { *q = [1.0, 0.0, 0.0, 0.0]; }
    }
    #[inline(always)]
    fn qinv(a: &[f64; 4]) -> [f64; 4] { [a[0], -a[1], -a[2], -a[3]] }
    #[inline(always)]
    fn qcomp(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] { let mut r = ham(a, b); qnorm(&mut r); r }

    // Convert feature value + axis → quaternion
    #[inline(always)]
    fn feat_to_quat(value: f64, ax: &[f64; 3]) -> [f64; 4] {
        let half = value * 0.5;
        let c = half.cos();
        let s = half.sin();
        // Normalize axis
        let an = (ax[0]*ax[0] + ax[1]*ax[1] + ax[2]*ax[2]).sqrt();
        if an < 1e-12 {
            return [1.0, 0.0, 0.0, 0.0];
        }
        let inv_n = 1.0 / an;
        let mut q = [c, s * ax[0] * inv_n, s * ax[1] * inv_n, s * ax[2] * inv_n];
        qnorm(&mut q);
        q
    }

    fn geodesic(pos: &[f64; 4], ideal: &[f64; 4], step: f64) -> [f64; 4] {
        let mut id = *ideal;
        let mut dot = pos[0]*id[0]+pos[1]*id[1]+pos[2]*id[2]+pos[3]*id[3];
        if dot < 0.0 { id = [-id[0],-id[1],-id[2],-id[3]]; dot = -dot; }
        if dot > 0.9999 { return id; }
        let theta = dot.min(1.0).acos();
        let s = theta.sin();
        if s < 1e-8 { return *pos; }
        let a = ((1.0-step)*theta).sin()/s;
        let b = (step*theta).sin()/s;
        let mut r = [a*pos[0]+b*id[0], a*pos[1]+b*id[1], a*pos[2]+b*id[2], a*pos[3]+b*id[3]];
        qnorm(&mut r);
        r
    }

    // LCG
    let mut rng = seed;
    let mut next = || -> u64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng
    };
    let mut order: Vec<usize> = (0..n_samples).collect();

    for _epoch in 0..n_epochs {
        for i in (1..n_samples).rev() {
            let j = (next() as usize) % (i + 1);
            order.swap(i, j);
        }

        for &idx in &order {
            let target: [f64; 4] = [tgt_data[[idx,0]], tgt_data[[idx,1]],
                                     tgt_data[[idx,2]], tgt_data[[idx,3]]];

            // Build quaternions from features × axes
            let mut quats = vec![[0.0f64; 4]; n_feat];
            for f in 0..n_feat {
                let ax = [axes_data[[f,0]], axes_data[[f,1]], axes_data[[f,2]]];
                quats[f] = feat_to_quat(feat_data[[idx, f]], &ax);
            }

            // Compose
            let mut c = [1.0, 0.0, 0.0, 0.0];
            for f in 0..n_feat { c = qcomp(&c, &quats[f]); }

            // Error
            let eq = qcomp(&c, &qinv(&target));
            let es = eq[0].abs().min(1.0).acos();
            if es < 1e-6 { continue; }

            let w_mag = eq[0].abs();
            let rgb_mag = (eq[1]*eq[1]+eq[2]*eq[2]+eq[3]*eq[3]).sqrt();
            let tf = if w_mag > rgb_mag { 1.0 } else { 0.5 };
            let step = damping * (es / std::f64::consts::PI) * tf;

            // Left/right products
            let mut left = vec![[1.0, 0.0, 0.0, 0.0]; n_feat + 1];
            for f in 0..n_feat { left[f+1] = qcomp(&left[f], &quats[f]); }
            let mut right = vec![[1.0, 0.0, 0.0, 0.0]; n_feat + 1];
            for f in (0..n_feat).rev() { right[f] = qcomp(&quats[f], &right[f+1]); }

            // For each feature: ideal quaternion, then back-project to axis adjustment
            for f in 0..n_feat {
                let ideal_q = qcomp(&qcomp(&qinv(&left[f]), &target), &qinv(&right[f+1]));
                let current_q = quats[f];

                // Move current_q toward ideal_q via geodesic
                let new_q = geodesic(&current_q, &ideal_q, step);

                // Back-project new_q to axis: new_q = [cos(θ/2), sin(θ/2)*axis]
                // θ is fixed (it's the feature value), so the axis must change
                let fval = feat_data[[idx, f]];
                let half = fval * 0.5;
                let sin_half = half.sin();

                if sin_half.abs() > 1e-8 {
                    // Extract new axis from new_q's vector part
                    let inv_s = 1.0 / sin_half;
                    let new_ax = [new_q[1] * inv_s, new_q[2] * inv_s, new_q[3] * inv_s];
                    // Normalize
                    let ax_n = (new_ax[0]*new_ax[0]+new_ax[1]*new_ax[1]+new_ax[2]*new_ax[2]).sqrt();
                    if ax_n > 1e-8 {
                        // Geodesic step on the axis (in R³, just lerp + normalize)
                        for d in 0..3 {
                            let cur = axes_data[[f, d]];
                            let tgt = new_ax[d] / ax_n;
                            axes_data[[f, d]] = cur + step * (tgt - cur);
                        }
                        // Re-normalize axis
                        let an = (axes_data[[f,0]].powi(2)+axes_data[[f,1]].powi(2)+axes_data[[f,2]].powi(2)).sqrt();
                        if an > 1e-8 {
                            axes_data[[f,0]] /= an;
                            axes_data[[f,1]] /= an;
                            axes_data[[f,2]] /= an;
                        }
                    }
                }
            }
        }
    }

    // Compute final predictions
    let mut preds = vec![0.0f64; n_samples * 4];
    let mut correct = 0usize;
    for idx in 0..n_samples {
        let mut c = [1.0, 0.0, 0.0, 0.0];
        for f in 0..n_feat {
            let ax = [axes_data[[f,0]], axes_data[[f,1]], axes_data[[f,2]]];
            let q = feat_to_quat(feat_data[[idx, f]], &ax);
            c = qcomp(&c, &q);
        }
        preds[idx*4] = c[0]; preds[idx*4+1] = c[1];
        preds[idx*4+2] = c[2]; preds[idx*4+3] = c[3];

        let t: [f64; 4] = [tgt_data[[idx,0]], tgt_data[[idx,1]],
                            tgt_data[[idx,2]], tgt_data[[idx,3]]];
        let dot = (c[0]*t[0]+c[1]*t[1]+c[2]*t[2]+c[3]*t[3]).abs();
        if dot > 0.5 { correct += 1; }
    }

    let acc = correct as f64 / n_samples.max(1) as f64;
    let preds_arr = numpy::PyArray1::from_vec(py, preds);
    let acc_val = pyo3::types::PyFloat::new(py, acc);
    let result = pyo3::types::PyTuple::new(py, &[preds_arr.into_any(), acc_val.into_any()])?;
    Ok(result.into())
}

// ── Resonance Query ─────────────────────────────────────────────────
// The 8th primitive. Content-addressable retrieval on S³.
// Given a query element and a GeometricPath, find the stored elements
// with lowest geodesic distance (sigma) to the query.

/// Resonance scan: find the k closest stored elements to a query.
///
/// Takes a query element (4 floats, unit quaternion on S³) and a
/// GeometricPath. Returns a list of (index, sigma, base_r, base_g,
/// base_b, phase) tuples sorted by sigma (closest first).
///
/// O(n) — scans all elements. For sub-linear retrieval, use the
/// lattice index (future).
#[pyfunction]
#[pyo3(signature = (query, path, k = 1))]
fn resonance_query<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<f64>,
    path: &PyGeometricPath,
    k: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let q = as_contiguous_slice_1d(&query, "query")?;
    if q.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "query must have 4 elements (S³ quaternion), got {}",
            q.len()
        )));
    }

    let group = crate::groups::sphere::SphereGroup;
    let results = crate::resonance::resonance_scan(&group, q, &path.inner, k);

    // Return as (k × 6) array: [index, sigma, base_r, base_g, base_b, phase]
    let n = results.len();
    let mut data = Vec::with_capacity(n * 6);
    for r in &results {
        data.push(r.index as f64);
        data.push(r.drift);
        data.push(r.base[0]);
        data.push(r.base[1]);
        data.push(r.base[2]);
        data.push(r.phase);
    }
    let arr = PyArray2::from_vec2(py, &data.chunks(6).map(|c| c.to_vec()).collect::<Vec<_>>())
        .map_err(|e| PyValueError::new_err(format!("failed to create array: {e}")))?;
    Ok(arr)
}

/// Resonance scan from raw bytes: embed the query, then scan the path.
///
/// Convenience function that handles the embed step so Python doesn't
/// need to call embed() separately.
#[pyfunction]
#[pyo3(signature = (query_bytes, path, k = 1, hashed = true))]
fn resonance_query_raw<'py>(
    py: Python<'py>,
    query_bytes: &[u8],
    path: &PyGeometricPath,
    k: usize,
    hashed: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let query = crate::embed::bytes_to_sphere(query_bytes, hashed);
    let group = crate::groups::sphere::SphereGroup;
    let results = crate::resonance::resonance_scan(&group, &query, &path.inner, k);

    let n = results.len();
    let mut data = Vec::with_capacity(n * 6);
    for r in &results {
        data.push(r.index as f64);
        data.push(r.drift);
        data.push(r.base[0]);
        data.push(r.base[1]);
        data.push(r.base[2]);
        data.push(r.phase);
    }
    let arr = PyArray2::from_vec2(py, &data.chunks(6).map(|c| c.to_vec()).collect::<Vec<_>>())
        .map_err(|e| PyValueError::new_err(format!("failed to create array: {e}")))?;
    Ok(arr)
}

// ── Table (Closure DNA) ─────────────────────────────────────────────
// Columnar database. Each field is its own element on S³.
// Schema defines columns (F64 or Bytes). Each column stored separately.
// Filter/aggregate/sort read only the relevant column — no parsing.

/// A Closure DNA database table with typed columns.
#[pyclass(name = "Table")]
pub struct PyTable {
    inner: crate::table::Table,
}

#[pymethods]
impl PyTable {
    /// Create a new table with typed columns.
    /// schema: list of (name, type, indexed) tuples. type: "f64" or "bytes".
    #[staticmethod]
    /// Create a new table with typed columns.
    /// schema: list of (name, type, indexed, not_null, unique) tuples.
    /// type: "f64", "i64", or "bytes".
    fn create(path: &str, schema: Vec<(String, String, bool, bool, bool)>) -> PyResult<Self> {
        let defs: Vec<crate::table::ColumnDef> = schema
            .into_iter()
            .map(|(name, typ, indexed, not_null, unique)| crate::table::ColumnDef {
                name,
                col_type: match typ.as_str() {
                    "f64" => crate::table::ColumnType::F64,
                    "i64" => crate::table::ColumnType::I64,
                    _ => crate::table::ColumnType::Bytes,
                },
                indexed,
                not_null,
                unique,
            })
            .collect();
        let table = crate::table::Table::create(std::path::Path::new(path), defs)
            .map_err(|e| PyValueError::new_err(format!("create failed: {e}")))?;
        Ok(Self { inner: table })
    }

    /// Open an existing table.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let table = crate::table::Table::open(std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(format!("open failed: {e}")))?;
        Ok(Self { inner: table })
    }

    /// Insert one row. values: list of (f64 or bytes) matching schema order.
    fn insert(&mut self, values: Vec<PyColumnValue>) -> PyResult<usize> {
        let vals: Vec<crate::table::ColumnValue> = values.into_iter().map(|v| v.into()).collect();
        self.inner
            .insert(&vals)
            .map_err(|e| PyValueError::new_err(format!("insert failed: {e}")))
    }

    /// Insert many rows.
    fn insert_many(&mut self, rows: Vec<Vec<PyColumnValue>>) -> PyResult<usize> {
        let converted: Vec<Vec<crate::table::ColumnValue>> = rows
            .into_iter()
            .map(|row| row.into_iter().map(|v| v.into()).collect())
            .collect();
        self.inner
            .insert_many(&converted)
            .map_err(|e| PyValueError::new_err(format!("insert_many failed: {e}")))
    }

    /// Insert many rows as typed columns. This is the native columnar ingest path.
    fn insert_columns(&mut self, columns: Vec<PyColumnBatch>) -> PyResult<usize> {
        let converted: Vec<crate::table::ColumnBatch> =
            columns.into_iter().map(|col| col.into()).collect();
        self.inner
            .insert_columns(&converted)
            .map_err(|e| PyValueError::new_err(format!("insert_columns failed: {e}")))
    }

    /// Resonance search for a typed row query.
    #[pyo3(signature = (values, k = 1))]
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        values: Vec<PyColumnValue>,
        k: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let vals: Vec<crate::table::ColumnValue> = values.into_iter().map(|v| v.into()).collect();
        let results = self
            .inner
            .search(&vals, k)
            .map_err(|e| PyValueError::new_err(format!("search failed: {e}")))?;

        let n = results.len();
        let mut data = Vec::with_capacity(n * 6);
        for r in &results {
            data.push(r.index as f64);
            data.push(r.drift);
            data.push(r.base[0]);
            data.push(r.base[1]);
            data.push(r.base[2]);
            data.push(r.phase);
        }
        let arr = PyArray2::from_vec2(py, &data.chunks(6).map(|c| c.to_vec()).collect::<Vec<_>>())
            .map_err(|e| PyValueError::new_err(format!("failed to create array: {e}")))?;
        Ok(arr)
    }

    fn build_genome(&mut self) -> PyResult<()> {
        self.inner.build_genome()
            .map_err(|e| PyValueError::new_err(format!("build_genome failed: {e}")))
    }

    #[pyo3(signature = (name = None))]
    fn snapshot(&mut self, name: Option<String>) -> PyResult<String> {
        self.inner
            .snapshot(name.as_deref())
            .map_err(|e| PyValueError::new_err(format!("snapshot failed: {e}")))
    }

    #[pyo3(signature = (limit = None))]
    fn history_json(&self, limit: Option<usize>) -> PyResult<Vec<String>> {
        let entries = self
            .inner
            .history(limit)
            .map_err(|e| PyValueError::new_err(format!("history failed: {e}")))?;
        entries
            .into_iter()
            .map(|entry| {
                serde_json::to_string(&entry)
                    .map_err(|e| PyValueError::new_err(format!("history serialize failed: {e}")))
            })
            .collect()
    }

    fn snapshots_json(&self) -> PyResult<Vec<String>> {
        let snapshots = self
            .inner
            .snapshots()
            .map_err(|e| PyValueError::new_err(format!("snapshots failed: {e}")))?;
        snapshots
            .into_iter()
            .map(|snap| {
                serde_json::to_string(&snap)
                    .map_err(|e| PyValueError::new_err(format!("snapshot serialize failed: {e}")))
            })
            .collect()
    }

    fn restore_snapshot(&mut self, name: String) -> PyResult<()> {
        self.inner
            .restore_snapshot(&name)
            .map_err(|e| PyValueError::new_err(format!("restore_snapshot failed: {e}")))
    }

    fn genome_depth(&mut self) -> PyResult<usize> {
        self.inner.genome_depth()
            .map_err(|e| PyValueError::new_err(format!("genome_depth failed: {e}")))
    }

    fn genome_codons(&mut self) -> PyResult<usize> {
        self.inner.genome_codons()
            .map_err(|e| PyValueError::new_err(format!("genome_codons failed: {e}")))
    }

    fn schema(&self) -> Vec<(String, String, bool)> {
        self.inner
            .schema_entries()
            .into_iter()
            .map(|entry| {
                let typ = match entry.col_type {
                    crate::table::ColumnType::F64 => "f64".to_string(),
                    crate::table::ColumnType::I64 => "i64".to_string(),
                    crate::table::ColumnType::Bytes => "bytes".to_string(),
                };
                (entry.name, typ, entry.indexed)
            })
            .collect()
    }

    /// Get a field value. Returns f64 or bytes depending on column type.
    fn get_f64(&mut self, row: usize, col: usize) -> PyResult<f64> {
        self.inner
            .get_field_f64(row, col)
            .map_err(|e| PyValueError::new_err(format!("get failed: {e}")))
    }

    fn get_i64(&mut self, row: usize, col: usize) -> PyResult<i64> {
        self.inner
            .get_field_i64(row, col)
            .map_err(|e| PyValueError::new_err(format!("get failed: {e}")))
    }

    fn get_bytes(&mut self, row: usize, col: usize) -> PyResult<Vec<u8>> {
        self.inner
            .get_field_bytes(row, col)
            .map_err(|e| PyValueError::new_err(format!("get failed: {e}")))
    }

    fn get_row<'py>(&mut self, py: Python<'py>, row: usize) -> PyResult<Vec<PyObject>> {
        let values = self
            .inner
            .get_row(row)
            .map_err(|e| PyValueError::new_err(format!("get_row failed: {e}")))?;
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            match value {
                crate::table::ColumnValue::Null => out.push(py.None()),
                crate::table::ColumnValue::F64(v) => out.push(v.into_pyobject(py)?.unbind().into()),
                crate::table::ColumnValue::I64(v) => out.push(v.into_pyobject(py)?.unbind().into()),
                crate::table::ColumnValue::Bytes(v) => out.push(PyBytes::new(py, &v).into()),
            }
        }
        Ok(out)
    }

    /// Column index by name.
    fn column_index(&self, name: &str) -> PyResult<usize> {
        self.inner
            .column_index(name)
            .ok_or_else(|| PyValueError::new_err(format!("no column: {name}")))
    }

    /// Filter by string column equality. Returns matching row indices.
    fn filter_equals(&mut self, col_name: &str, value: &[u8]) -> PyResult<Vec<usize>> {
        self.inner
            .filter_equals(col_name, value)
            .map_err(|e| PyValueError::new_err(format!("filter failed: {e}")))
    }

    /// Filter by numeric comparison. Returns matching row indices.
    fn filter_cmp(&mut self, col_name: &str, op: &str, value: f64) -> PyResult<Vec<usize>> {
        self.inner
            .filter_cmp(col_name, op, value)
            .map_err(|e| PyValueError::new_err(format!("filter failed: {e}")))
    }

    /// Sum a numeric column.
    fn sum(&mut self, col_name: &str) -> PyResult<f64> {
        self.inner
            .sum(col_name)
            .map_err(|e| PyValueError::new_err(format!("sum failed: {e}")))
    }

    /// Average a numeric column.
    fn avg(&mut self, col_name: &str) -> PyResult<f64> {
        self.inner
            .avg(col_name)
            .map_err(|e| PyValueError::new_err(format!("avg failed: {e}")))
    }

    /// Sort indices by numeric column.
    #[pyo3(signature = (col_name, descending = false))]
    fn argsort(&mut self, col_name: &str, descending: bool) -> PyResult<Vec<usize>> {
        self.inner
            .argsort(col_name, descending)
            .map_err(|e| PyValueError::new_err(format!("sort failed: {e}")))
    }

    /// Check table integrity. Returns drift (0 = clean).
    fn check(&self) -> f64 {
        self.inner.check()
    }

    fn check_hopf<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let view = self.inner.check_hopf();
        Ok(PyArray1::from_vec(
            py,
            vec![view.drift, view.base[0], view.base[1], view.base[2], view.phase],
        ))
    }

    fn inspect_row<'py>(&mut self, py: Python<'py>, row: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let view = self
            .inner
            .inspect_row(row)
            .map_err(|e| PyValueError::new_err(format!("inspect_row failed: {e}")))?;
        Ok(PyArray1::from_vec(
            py,
            vec![view.drift, view.base[0], view.base[1], view.base[2], view.phase],
        ))
    }

    fn audit<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let result = self
            .inner
            .audit()
            .map_err(|e| PyValueError::new_err(format!("audit failed: {e}")))?;
        Ok(PyArray1::from_vec(
            py,
            vec![
                if result.ok { 1.0 } else { 0.0 },
                result.drift,
                result.bad_row.map(|v| v as f64).unwrap_or(-1.0),
                result.hopf.base[0],
                result.hopf.base[1],
                result.hopf.base[2],
                result.hopf.phase,
            ],
        ))
    }

    fn repair(&mut self) -> PyResult<()> {
        self.inner
            .repair()
            .map_err(|e| PyValueError::new_err(format!("repair failed: {e}")))
    }

    fn update(&mut self, values: Vec<PyColumnValue>, row: usize) -> PyResult<()> {
        let vals: Vec<crate::table::ColumnValue> = values.into_iter().map(|v| v.into()).collect();
        self.inner
            .update(row, &vals)
            .map_err(|e| PyValueError::new_err(format!("update failed: {e}")))
    }

    fn delete(&mut self, row: usize) -> PyResult<()> {
        self.inner
            .delete(row)
            .map_err(|e| PyValueError::new_err(format!("delete failed: {e}")))
    }

    /// Table identity — 32-byte quaternion.
    fn identity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let id = self.inner.identity();
        PyArray1::from_vec(py, id.to_vec())
    }

    /// How many rows.
    fn count(&self) -> usize {
        self.inner.count()
    }

    fn live_row_count(&self) -> usize {
        self.inner.live_row_count()
    }

    fn is_deleted(&self, row: usize) -> PyResult<bool> {
        self.inner
            .is_deleted(row)
            .map_err(|e| PyValueError::new_err(format!("is_deleted failed: {e}")))
    }

    /// Save to disk.
    fn save(&mut self) -> PyResult<()> {
        self.inner
            .save()
            .map_err(|e| PyValueError::new_err(format!("save failed: {e}")))
    }

    fn __len__(&self) -> usize {
        self.inner.count()
    }

    fn __repr__(&self) -> String {
        format!("Table(records={}, drift={:.6})", self.inner.count(), self.inner.check())
    }
}

/// Column value passed from Python — NULL, i64, f64, or bytes.
enum PyColumnValue {
    Null,
    Int(i64),
    Float(f64),
    Bytes(Vec<u8>),
}

impl<'py> FromPyObject<'py> for PyColumnValue {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if ob.is_none() {
            return Ok(Self::Null);
        }
        if let Ok(v) = ob.extract::<i64>() {
            return Ok(Self::Int(v));
        }
        if let Ok(v) = ob.extract::<f64>() {
            return Ok(Self::Float(v));
        }
        if let Ok(v) = ob.extract::<Vec<u8>>() {
            return Ok(Self::Bytes(v));
        }
        Err(PyValueError::new_err(
            "column value must be None, int, float, or bytes",
        ))
    }
}

impl From<PyColumnValue> for crate::table::ColumnValue {
    fn from(v: PyColumnValue) -> Self {
        match v {
            PyColumnValue::Null => crate::table::ColumnValue::Null,
            PyColumnValue::Int(i) => crate::table::ColumnValue::I64(i),
            PyColumnValue::Float(f) => crate::table::ColumnValue::F64(f),
            PyColumnValue::Bytes(b) => crate::table::ColumnValue::Bytes(b),
        }
    }
}

#[derive(FromPyObject)]
enum PyColumnBatch {
    #[pyo3(transparent)]
    Int(Vec<i64>),
    #[pyo3(transparent)]
    Float(Vec<f64>),
    #[pyo3(transparent)]
    Bytes(Vec<Vec<u8>>),
}

impl From<PyColumnBatch> for crate::table::ColumnBatch {
    fn from(v: PyColumnBatch) -> Self {
        match v {
            PyColumnBatch::Int(i) => crate::table::ColumnBatch::I64(i),
            PyColumnBatch::Float(f) => crate::table::ColumnBatch::F64(f),
            PyColumnBatch::Bytes(b) => crate::table::ColumnBatch::Bytes(b),
        }
    }
}

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
    m.add_function(wrap_pyfunction!(train_cells, m)?)?;
    m.add_function(wrap_pyfunction!(train_continuous, m)?)?;
    m.add_function(wrap_pyfunction!(resonance_query, m)?)?;
    m.add_function(wrap_pyfunction!(resonance_query_raw, m)?)?;
    m.add_class::<PyGroup>()?;
    m.add_class::<PyGeometricPath>()?;
    m.add_class::<PyStreamMonitor>()?;
    m.add_class::<PyHierarchicalClosure>()?;
    m.add_class::<PyTable>()?;
    Ok(())
}
