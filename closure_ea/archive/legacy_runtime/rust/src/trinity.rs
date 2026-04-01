//! Trinity learning runtime on S³.
//!
//! The figure-8 cycle:
//!   Reality → S1 × S3 → Perception → S2 × S3 → Prediction → S1 → Reality
//!   → Response → S1 → Evaluation → Write S3
//!
//! S1 embeds reality. S2 discovers structure. S3 stores memory.
//! S3 feeds the forward half through two channels:
//!   - S3.lattice → S1: perception bias (transient)
//!   - S3.genome  → S2: T lookup for prediction (durable)
//!
//! ## The Genome (Stage 1)
//!
//! T entries are stored as (context_key, transform) pairs, both quaternions
//! on S³. No integer bins. No HashMap.
//!
//! Stage 1 key (current): compose(prev_q, ev_q) ∘ hopf_q
//!   — stable under training because genome positions are fixed and hopf_q
//!     is retrieved from a closed Hopf element, not from live S2 state.
//!   — s2_state_q is NOT included: it drifts during training (S3 error
//!     accumulation) so early-stored keys would not match eval-time queries.
//!
//! Stage 2 (future): include a stable structural-context encoding derived
//!   from S2 state that is invariant to training progression.
//! Stage 3 (future): use Hopf S¹ phase in retrieval decisions, not just S² base.
//! Stage 4 (future): back this buffer with a closure_rs::Table for persistence.
//!
//! Lookup: resonance_scan_flat on context_keys → return transforms[hit.index]
//! Upsert: if hit.drift < GENOME_MATCH_THRESHOLD → SLERP transform in place
//!         else → push new (context_key, transform) pair

use crate::hopf::decompose as hopf_decompose;
use crate::resonance::resonance_scan_flat;
use crate::groups::sphere::SphereGroup;

// ── Quaternion algebra on S³ ───────────────────────────────────────

#[inline(always)]
fn ham(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
     a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
     a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
     a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]]
}

#[inline(always)]
pub fn norm(q: &mut [f64; 4]) {
    let n = (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]).sqrt();
    if n > 1e-15 { let inv = 1.0/n; q[0]*=inv; q[1]*=inv; q[2]*=inv; q[3]*=inv; }
    else { *q = IDENTITY; }
}

pub const IDENTITY: [f64; 4] = [1.0, 0.0, 0.0, 0.0];

#[inline(always)]
pub fn compose(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    let mut r = ham(a, b); norm(&mut r); r
}

#[inline(always)]
pub fn inverse(a: &[f64; 4]) -> [f64; 4] { [a[0], -a[1], -a[2], -a[3]] }

#[inline(always)]
pub fn sigma(a: &[f64; 4]) -> f64 { a[0].abs().min(1.0).acos() }

/// Geodesic step (SLERP) on S³: interpolate from a toward b by fraction t.
pub fn slerp(a: &[f64; 4], b: &[f64; 4], t: f64) -> [f64; 4] {
    let mut dot: f64 = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    let mut target = *b;
    if dot < 0.0 { target = [-b[0], -b[1], -b[2], -b[3]]; dot = -dot; }
    if dot > 0.9999 { return target; }
    let theta = dot.min(1.0).acos();
    let sin_t = theta.sin();
    if sin_t < 1e-8 { return *a; }
    let wa = ((1.0 - t) * theta).sin() / sin_t;
    let wb = (t * theta).sin() / sin_t;
    let mut r = [wa*a[0]+wb*target[0], wa*a[1]+wb*target[1],
                 wa*a[2]+wb*target[2], wa*a[3]+wb*target[3]];
    norm(&mut r);
    r
}

// ── Kernel ─────────────────────────────────────────────────────────

pub struct Kernel {
    pub c: [f64; 4],
    pub epsilon: f64,
    pub event_count: usize,
    pub last_tick: usize,
}

impl Kernel {
    pub fn new(eps: f64) -> Self {
        Self { c: IDENTITY, epsilon: eps, event_count: 0, last_tick: 0 }
    }
    pub fn reset(&mut self) { self.c = IDENTITY; self.event_count = 0; }
    pub fn is_active(&self) -> bool { sigma(&self.c) > 0.01 }
    pub fn is_live(&self, current_tick: usize, timeout: usize) -> bool {
        self.is_active() && (current_tick - self.last_tick) < timeout
    }
    /// Compose q into running product. Returns Some(before) on closure.
    pub fn compose_and_check(&mut self, q: &[f64; 4], tick: usize) -> Option<[f64; 4]> {
        self.last_tick = tick;
        let before = self.c;
        self.c = compose(&self.c, q);
        self.event_count += 1;
        if sigma(&self.c) < self.epsilon {
            self.reset();
            if sigma(&before) > self.epsilon {
                return Some(before);
            }
            return Some(IDENTITY);
        }
        None
    }
}

// ── TreeLattice ────────────────────────────────────────────────────

pub struct TreeLattice {
    pub children: Vec<Kernel>,
    pub parents: Vec<Kernel>,
    pub base_eps: f64,
    pub max_depth: usize,
    pub closure_count: usize,
    pub tick: usize,
    pub spawn_threshold: usize,
    pub liveness_timeout: usize,
    axis_errors: [[f64; 4]; 3],
    axis_counts: [usize; 3],
}

impl TreeLattice {
    pub fn new(eps: f64, max_depth: usize) -> Self {
        Self {
            children: vec![Kernel::new(eps)],
            parents: Vec::new(),
            base_eps: eps,
            max_depth,
            closure_count: 0,
            tick: 0,
            spawn_threshold: 50,
            liveness_timeout: 100,
            axis_errors: [IDENTITY; 3],
            axis_counts: [0; 3],
        }
    }

    pub fn route(&self, q: &[f64; 4]) -> usize {
        if self.children.len() <= 1 { return 0; }
        let abs_xyz = [q[1].abs(), q[2].abs(), q[3].abs()];
        let axis = if abs_xyz[0] >= abs_xyz[1] && abs_xyz[0] >= abs_xyz[2] { 0 }
                   else if abs_xyz[1] >= abs_xyz[2] { 1 } else { 2 };
        axis.min(self.children.len() - 1)
    }

    pub fn ingest(&mut self, q: &[f64; 4]) -> Option<[f64; 4]> {
        self.tick += 1;
        let child_idx = self.route(q);
        let axis = if q[1].abs() >= q[2].abs() && q[1].abs() >= q[3].abs() { 0 }
                   else if q[2].abs() >= q[3].abs() { 1 } else { 2 };

        let closure = self.children[child_idx].compose_and_check(q, self.tick);

        if let Some(content) = closure {
            self.closure_count += 1;
            if sigma(&content) > self.base_eps {
                self.emit_to_parents(&content);
            }
            if self.children.len() == 1 && self.tick > self.spawn_threshold {
                self.maybe_spawn();
            }
            if self.tick % self.spawn_threshold == 0 {
                self.evict_stale();
            }
            return Some(content);
        }

        self.axis_errors[axis] = compose(&self.axis_errors[axis], q);
        self.axis_counts[axis] += 1;
        None
    }

    fn emit_to_parents(&mut self, content: &[f64; 4]) {
        if self.parents.is_empty() && self.max_depth > 1 {
            self.parents.push(Kernel::new(self.base_eps * 1.5));
        }
        if !self.parents.is_empty() {
            self.parent_at(0, content);
        }
    }

    fn parent_at(&mut self, lv: usize, q: &[f64; 4]) {
        if lv >= self.parents.len() || lv + 1 >= self.max_depth { return; }
        let closure = self.parents[lv].compose_and_check(q, self.tick);
        if let Some(content) = closure {
            if sigma(&content) > self.parents[lv].epsilon {
                let nl = lv + 1;
                if nl >= self.parents.len() && nl + 1 < self.max_depth {
                    self.parents.push(Kernel::new(self.base_eps * (1.5 + nl as f64 * 0.5)));
                }
                if nl < self.parents.len() {
                    self.parent_at(nl, &content);
                }
            }
        }
    }

    fn maybe_spawn(&mut self) {
        if self.children.len() >= 3 { return; }
        let mut worst_axis = 0;
        let mut worst_sigma = 0.0f64;
        for ax in 0..3 {
            if self.axis_counts[ax] > 10 {
                let sig = sigma(&self.axis_errors[ax]);
                if sig > worst_sigma { worst_sigma = sig; worst_axis = ax; }
            }
        }
        if worst_sigma > std::f64::consts::PI / 2.0 && self.children.len() <= worst_axis {
            while self.children.len() <= worst_axis {
                self.children.push(Kernel::new(self.base_eps));
            }
        }
        self.axis_errors = [IDENTITY; 3];
        self.axis_counts = [0; 3];
    }

    pub fn hierarchical_identity(&self) -> [f64; 4] {
        let mut id = IDENTITY;
        let t = self.tick;
        let to = self.liveness_timeout;
        for k in &self.children {
            if k.is_live(t, to) { id = compose(&id, &k.c); }
        }
        for k in &self.parents {
            if k.is_live(t, to) { id = compose(&id, &k.c); }
        }
        id
    }

    pub fn active_depth(&self) -> usize {
        let t = self.tick;
        let to = self.liveness_timeout;
        self.children.iter().filter(|k| k.is_live(t, to)).count()
            + self.parents.iter().filter(|k| k.is_live(t, to)).count()
    }

    fn evict_stale(&mut self) {
        for k in &mut self.children {
            if k.is_active() && !k.is_live(self.tick, self.liveness_timeout) {
                k.reset();
            }
        }
    }

    /// Dissolve transient state, preserve structure.
    pub fn reset_transient(&mut self) {
        for k in &mut self.children { k.reset(); }
        for k in &mut self.parents { k.reset(); }
        self.closure_count = 0;
        self.axis_errors = [IDENTITY; 3];
        self.axis_counts = [0; 3];
    }

    pub fn n_children(&self) -> usize { self.children.len() }
}

// ── context_quaternion ──────────────────────────────────────────────
//
// Returns the full S³ state of S2's hierarchical lattice, optionally
// enriched with a Hopf element.
//
// Not used in the main learning loop (the T key uses genome positions
// and Hopf enrichment directly). Available as a public diagnostic
// primitive — useful for inspecting S2 structural state externally.

pub fn context_quaternion(
    s2_factors: &[TreeLattice],
    hopf_enrichment: Option<&[f64; 4]>,
) -> [f64; 4] {
    let mut hi = IDENTITY;
    for s2 in s2_factors {
        hi = compose(&hi, &s2.hierarchical_identity());
    }
    if let Some(mem) = hopf_enrichment {
        hi = compose(&hi, mem);
    }
    hi
}

// ── HopfBuffer ─────────────────────────────────────────────────────
//
// Circular buffer of closed Hopf elements. Each entry stores the full S³
// element plus its Hopf decomposition: S² base (content direction, WHAT)
// and S¹ phase (fiber position, WHERE in cycle).
//
// Stage 1 retrieval: query_by_base uses only S² base distance. The S¹ phase
// is stored for future use but does not affect retrieval decisions.
// Stage 3 (future): incorporate S¹ phase to distinguish cycle positions
// with the same content direction.

pub struct HopfEntry {
    pub element: [f64; 4],
    pub base: [f64; 3],
    /// S¹ fiber position. Stored but not used in Stage 1 retrieval.
    pub phase: f64,
    pub event_key: usize,
}

pub struct HopfBuffer {
    entries: Vec<HopfEntry>,
    capacity: usize,
    write_pos: usize,
}

impl HopfBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { entries: Vec::with_capacity(capacity), capacity, write_pos: 0 }
    }

    pub fn store(&mut self, element: [f64; 4], event_key: usize) {
        let (base, phase) = hopf_decompose(&element);
        let entry = HopfEntry { element, base: [base[0], base[1], base[2]], phase, event_key };
        if self.entries.len() < self.capacity {
            self.entries.push(entry);
        } else {
            self.entries[self.write_pos % self.capacity] = entry;
        }
        self.write_pos += 1;
    }

    /// Returns the entry whose S² base is nearest to `base`, within `threshold`.
    /// S¹ phase is not considered (Stage 1). Returns None if no entry qualifies.
    pub fn query_by_base(&self, base: &[f64; 3], threshold: f64) -> Option<&HopfEntry> {
        let mut best: Option<&HopfEntry> = None;
        let mut best_dist = threshold;
        for entry in &self.entries {
            let dot = (base[0]*entry.base[0] + base[1]*entry.base[1] + base[2]*entry.base[2])
                .max(-1.0).min(1.0);
            let dist = dot.acos();
            if dist < best_dist { best_dist = dist; best = Some(entry); }
        }
        best
    }
}

// ── TGenome: S³ geometric T-storage (Stage 1) ──────────────────────
//
// Stores T entries as (context_key, transform) pairs, both quaternions
// on S³. No integer indexing. No HashMap.
//
// Stage 1 context_key = compose(prev_q, ev_q) ∘ hopf_q
//   Only fixed genome positions and a stable Hopf closure element are used.
//   Live S2 hierarchical state is excluded — see module doc for why.
//
// Context keys are immutable once stored. Only transforms are updated.
// This mirrors the genome invariant: positions (neurons) don't move,
// connections (synapses) learn.

// Practical heuristic: geodesic drift below this threshold means "same
// context, update in place" rather than "new context, push new entry."
// Not architecturally derived — tuned empirically on small test sequences.
// After vote alignment: within-class spread ~0.19 rad, cross-class ~0.22 rad.
// Threshold of 0.10 rad allows within-class SLERP while limiting cross-class
// contamination. Rebuild and retune if genome initialization changes.
const GENOME_MATCH_THRESHOLD: f64 = 0.10;

pub struct TGenome {
    /// Stored context keys: Stage 1 encoding = compose(prev_q, ev_q) ∘ hopf_q.
    context_keys: Vec<f64>,  // n × 4
    /// Stored transform quaternions: the learned T values, parallel to context_keys.
    transforms: Vec<f64>,    // n × 4
}

/// Compute softmax weights from geodesic distances.
/// Returns weights summing to 1.0, with closer (lower sigma) getting higher weight.
fn softmax_weights(distances: &[f64], temperature: f64) -> Vec<f64> {
    if distances.is_empty() { return Vec::new(); }
    if distances.len() == 1 { return vec![1.0]; }
    let scores: Vec<f64> = distances.iter().map(|d| -d / temperature).collect();
    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_s: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
    let sum: f64 = exp_s.iter().sum();
    if sum < 1e-30 { return vec![1.0 / distances.len() as f64; distances.len()]; }
    exp_s.iter().map(|e| e / sum).collect()
}

/// Weighted geodesic average of k quaternions on S³.
/// Uses iterative SLERP: stable, exact for k=1, good approximation for nearby quaternions.
fn weighted_slerp_multi(quats: &[[f64; 4]], weights: &[f64]) -> [f64; 4] {
    debug_assert_eq!(quats.len(), weights.len());
    if quats.is_empty() { return IDENTITY; }
    if quats.len() == 1 { return quats[0]; }

    let mut result = quats[0];
    let mut cum_w = weights[0];

    for i in 1..quats.len() {
        cum_w += weights[i];
        if cum_w < 1e-30 { continue; }
        let t = weights[i] / cum_w;
        result = slerp(&result, &quats[i], t);
    }
    result
}

impl TGenome {
    pub fn new() -> Self {
        Self { context_keys: Vec::new(), transforms: Vec::new() }
    }

    pub fn len(&self) -> usize { self.transforms.len() / 4 }

    pub fn is_empty(&self) -> bool { self.transforms.is_empty() }

    #[inline]
    fn transform_at(&self, index: usize) -> [f64; 4] {
        let i = index * 4;
        [self.transforms[i], self.transforms[i+1], self.transforms[i+2], self.transforms[i+3]]
    }

    /// Selective attention over the genome.
    ///
    /// Query the genome with a context quaternion. Retrieve the top-k matches
    /// by geodesic distance. Weight them via softmax(-σ/temperature). Return
    /// the weighted geodesic average of matched T transforms.
    ///
    /// k=1, temperature=any → equivalent to the old greedy lookup.
    /// k>1 → S³-native selective attention over learned memory.
    ///
    /// This IS attention:
    ///   Keys    = genome context_keys (quaternions on S³)
    ///   Query   = current running product (quaternion on S³)
    ///   Values  = genome T transforms (quaternions on S³)
    ///   Score   = exp(-σ(query, key) / temperature)
    ///   Output  = weighted geodesic average of top-k Values
    pub fn attend(&self, context_q: &[f64; 4], k: usize, temperature: f64) -> [f64; 4] {
        if self.context_keys.is_empty() { return IDENTITY; }
        let group = SphereGroup;
        let actual_k = k.min(self.len());
        let hits = resonance_scan_flat(&group, context_q, &self.context_keys, 4, actual_k);
        if hits.is_empty() { return IDENTITY; }

        // Fast path: k=1, no softmax needed
        if hits.len() == 1 {
            return self.transform_at(hits[0].index);
        }

        // Collect matched transforms and their distances
        let matched_t: Vec<[f64; 4]> = hits.iter()
            .map(|h| self.transform_at(h.index))
            .collect();
        let distances: Vec<f64> = hits.iter().map(|h| h.drift).collect();

        // Softmax over geodesic distances → attention weights
        let weights = softmax_weights(&distances, temperature);

        // Weighted geodesic average on S³
        weighted_slerp_multi(&matched_t, &weights)
    }

    /// Insert or update T.
    ///
    /// If a stored context key is within GENOME_MATCH_THRESHOLD, SLERP its
    /// transform toward t_ideal. Otherwise push a new entry — unless the
    /// genome is at capacity, in which case force-update the nearest entry.
    pub fn upsert(&mut self, context_q: &[f64; 4], t_ideal: [f64; 4], step: f64,
                  max_entries: usize) {
        if self.context_keys.is_empty() {
            self.context_keys.extend_from_slice(context_q);
            self.transforms.extend_from_slice(&t_ideal);
            return;
        }
        let group = SphereGroup;
        let hits = resonance_scan_flat(&group, context_q, &self.context_keys, 4, 1);
        let update_in_place = !hits.is_empty()
            && (hits[0].drift < GENOME_MATCH_THRESHOLD
                || self.len() >= max_entries);
        if update_in_place && !hits.is_empty() {
            let i = hits[0].index * 4;
            let t_current = [self.transforms[i], self.transforms[i+1],
                             self.transforms[i+2], self.transforms[i+3]];
            let t_updated = slerp(&t_current, &t_ideal, step);
            self.transforms[i]   = t_updated[0];
            self.transforms[i+1] = t_updated[1];
            self.transforms[i+2] = t_updated[2];
            self.transforms[i+3] = t_updated[3];
        } else {
            self.context_keys.extend_from_slice(context_q);
            self.transforms.extend_from_slice(&t_ideal);
        }
    }

    /// Flat slice of stored transforms for introspection.
    pub fn dump_transforms(&self) -> &[f64] { &self.transforms }

    /// Flat slice of stored context keys for persistence.
    pub fn dump_context_keys(&self) -> &[f64] { &self.context_keys }

    /// Load a genome from flat slices (restoring from persistent storage).
    pub fn from_flat(context_keys: &[f64], transforms: &[f64]) -> Self {
        debug_assert_eq!(context_keys.len(), transforms.len());
        debug_assert_eq!(context_keys.len() % 4, 0);
        Self {
            context_keys: context_keys.to_vec(),
            transforms: transforms.to_vec(),
        }
    }
}

// ── TrinityConfig ──────────────────────────────────────────────────

pub struct TrinityConfig {
    pub epsilon_s2: f64,
    pub epsilon_s3: f64,
    pub damping: f64,
    pub max_depth: usize,
    pub n_passes: usize,
    pub n_colors: usize,
    pub m_factors: usize,
    /// Hard cap on TGenome entries. Prevents unbounded memory growth.
    pub max_genome_entries: usize,
    /// Top-k genome matches for selective attention (1 = greedy, >1 = weighted).
    pub attend_k: usize,
    /// Softmax temperature for attention weights. Lower = sharper focus on
    /// the nearest match. Higher = more diffuse across top-k matches.
    /// 0.1 = nearly greedy. 1.0 = even spread. Default 0.1.
    pub attend_temperature: f64,
    /// Number of sequential attend layers per token. Each layer uses a separate
    /// genome. The output of layer d becomes the query modifier for layer d+1.
    /// This is the S³ equivalent of stacking transformer layers:
    ///   Layer d: query_d = compose(base_query, state_d)
    ///            T_d     = attend(genome_d, query_d)
    ///            state_{d+1} = compose(T_d, state_d)
    /// attend_depth=1 preserves existing single-genome behavior.
    pub attend_depth: usize,
}

impl Default for TrinityConfig {
    fn default() -> Self {
        Self {
            epsilon_s2: 0.15,
            epsilon_s3: 0.15,
            damping: 0.2,
            max_depth: 6,
            n_passes: 1,
            n_colors: 10,
            m_factors: 1,
            max_genome_entries: 65536,
            attend_k: 1,
            attend_temperature: 0.1,
            attend_depth: 1,
        }
    }
}

/// Generate per-factor rotation quaternions.
fn factor_rotations(m: usize) -> Vec<[f64; 4]> {
    let mut rots = Vec::with_capacity(m);
    rots.push(IDENTITY);
    let golden = 0.618033988749895 * std::f64::consts::PI;
    for i in 1..m {
        let angle = golden * i as f64;
        let axis = i % 3;
        let w = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();
        let mut q = [w, 0.0, 0.0, 0.0];
        q[1 + axis] = s;
        rots.push(q);
    }
    rots
}

// ── TrinityResult ──────────────────────────────────────────────────

pub struct TrinityResult {
    pub predictions: Vec<i32>,
    /// Number of T entries in the genome after training.
    pub genome_size: usize,
    /// Flat n×4 dump of stored T transforms (for persistence / introspection).
    pub genome_dump: Vec<f64>,
    /// Flat n×4 dump of stored context keys (for persistence — paired with genome_dump).
    pub context_keys_dump: Vec<f64>,
    /// Per-event prediction distances to each of the first n_colors genome entries.
    pub prediction_distances: Vec<f64>,
}

// ── run_trinity: the main learning loop ────────────────────────────


pub fn run_trinity(
    genome: &mut [[f64; 4]],
    events: &[i32],
    task_lengths: &[i32],
    config: &TrinityConfig,
) -> TrinityResult {
    let n_genome = genome.len();
    let n_events = events.len();

    // One genome per attend layer. Layer 0 is deepest (learns slowest),
    // layer D-1 is shallowest (learns fastest, gets full error signal).
    let depth = config.attend_depth.max(1);
    let mut t_genomes: Vec<TGenome> = (0..depth).map(|_| TGenome::new()).collect();
    let mut hopf_buf = HopfBuffer::new(1024);
    let mut predictions = vec![0i32; n_events];
    let nc = config.n_colors.min(n_genome);
    let mut prediction_distances = vec![0.0f64; n_events * nc];

    let factor_rots = factor_rotations(config.m_factors);

    for _pass in 0..config.n_passes {
        let mut offset = 0usize;

        for &task_len in task_lengths {
            let tl = task_len as usize;
            if offset + tl > n_events { break; }

            let mut s2_factors: Vec<TreeLattice> = (0..config.m_factors)
                .map(|_| TreeLattice::new(config.epsilon_s2, config.max_depth))
                .collect();
            let mut s3 = TreeLattice::new(config.epsilon_s3, config.max_depth);
            let mut last_pred: Option<[f64; 4]> = None;
            let mut last_layer_queries: Option<Vec<[f64; 4]>> = None;
            let mut last_real: Option<[f64; 4]> = None;
            let mut path_q: [f64; 4] = IDENTITY;

            for i in 0..tl {
                let ev = events[offset + i] as usize;
                if ev >= n_genome { continue; }
                let reality = genome[ev];

                // ── Return half ──
                if let (Some(pred), Some(layer_queries), Some(prev_real)) =
                    (last_pred, last_layer_queries.take(), last_real)
                {
                    let error = compose(&reality, &inverse(&pred));
                    let error_sigma = sigma(&error);
                    let w_mag = error[0].abs();
                    let rgb_mag = (error[1]*error[1]+error[2]*error[2]+error[3]*error[3]).sqrt();

                    let _ = s3.ingest(&error);
                    let t_ideal = compose(&reality, &inverse(&prev_real));

                    let base_step = config.damping * (error_sigma / std::f64::consts::PI);
                    if base_step > 1e-8 && (w_mag + rgb_mag) > 1e-8 {
                        let w_weight = w_mag / (w_mag + rgb_mag);
                        let rgb_weight = rgb_mag / (w_mag + rgb_mag);
                        let step = base_step * (w_weight * 1.0 + rgb_weight * 0.5);

                        // Update last layer with full error signal. Earlier layers
                        // learn implicitly: as the last layer improves, queries to
                        // earlier layers change (via compose(base_query, state)),
                        // causing them to specialize via the multi-pass mechanism.
                        // This avoids the credit-assignment problem of distributing
                        // a single t_ideal across layers with different roles.
                        let last = depth - 1;
                        t_genomes[last].upsert(
                            &layer_queries[last], t_ideal, step,
                            config.max_genome_entries,
                        );
                    }
                }

                // ── Forward half ──
                let s3_memory = s3.hierarchical_identity();
                let perception = compose(&reality, &inverse(&s3_memory));

                for (fi, s2) in s2_factors.iter_mut().enumerate() {
                    let rotated = compose(&factor_rots[fi], &perception);
                    let s2_closure = s2.ingest(&rotated);
                    if let Some(closure_content) = s2_closure {
                        hopf_buf.store(closure_content, ev);
                    }
                }

                path_q = compose(&path_q, &genome[ev]);
                let base_query = path_q;

                // ── Multi-layer attend ──
                // Each layer: query from base + current state → attend → compose into state.
                // This is the S³ equivalent of stacking N transformer layers.
                let mut state = reality;
                let mut layer_queries = Vec::with_capacity(depth);
                for d in 0..depth {
                    let lq = compose(&base_query, &state);
                    layer_queries.push(lq);
                    let t = t_genomes[d].attend(&lq, config.attend_k, config.attend_temperature);
                    state = compose(&t, &state);
                }
                let prediction = state;

                last_pred = Some(prediction);
                last_layer_queries = Some(layer_queries);
                last_real = Some(reality);

                let nc = config.n_colors.min(n_genome);
                let mut best = 0i32;
                let mut best_dist = f64::MAX;
                let dist_offset = (offset + i) * nc;
                for tok in 0..nc {
                    let dist = sigma(&compose(&prediction, &inverse(&genome[tok])));
                    if dist_offset + tok < prediction_distances.len() {
                        prediction_distances[dist_offset + tok] = dist;
                    }
                    if dist < best_dist { best_dist = dist; best = tok as i32; }
                }
                predictions[offset + i] = best;
            }
            offset += tl;
        }
    }

    // Dump from last layer's genome (the most-trained one).
    let genome_size: usize = t_genomes.iter().map(|g| g.len()).sum();
    let genome_dump = t_genomes.last().map_or(Vec::new(), |g| g.dump_transforms().to_vec());
    let context_keys_dump = t_genomes.last().map_or(Vec::new(), |g| g.dump_context_keys().to_vec());

    TrinityResult { predictions, genome_size, genome_dump, context_keys_dump, prediction_distances }
}
