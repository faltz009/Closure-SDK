//! Resonance query — the 8th primitive.
//!
//! Content-addressable retrieval on S³. Given a query element and a set
//! of stored elements, find the stored element(s) with lowest geodesic
//! distance (sigma) to the query. The Hopf decomposition of the gap
//! tells you HOW the match relates: base (S², direction) and phase
//! (S¹, fiber).
//!
//! This is the primitive that turns Closure from a log into a database.
//! Without it, you can verify and classify but not FIND. With it, the
//! running product becomes a searchable index: store, retrieve by
//! position (recover), retrieve by content (resonance).
//!
//! ## How it works
//!
//! For each stored element e_i:
//!   1. Compute the gap: g = compose(inverse(query), e_i)
//!   2. Measure sigma: geodesic distance from identity
//!   3. Decompose via Hopf: base (what kind of relationship) + phase (where)
//!
//! Lowest sigma = closest match. Sigma = 0 means exact match (same bytes
//! produced the same quaternion via SHA-256). Sigma > 0 means the gap
//! has structure, and the Hopf channels describe that structure.
//!
//! ## Complexity
//!
//! Brute force: O(n) — scan all elements.
//! With lattice: O(levels + block_size) — descend through index.

use crate::groups::LieGroup;
use crate::hopf::decompose as hopf_decompose;
use crate::path::GeometricPath;

/// One resonance match: which element, how close, and the Hopf channels.
#[derive(Debug, Clone)]
pub struct ResonanceHit {
    /// 0-indexed position in the table.
    pub index: usize,
    /// Geodesic distance from query to this element. 0 = exact match.
    pub drift: f64,
    /// S² base direction (R, G, B) — what kind of relationship.
    pub base: [f64; 3],
    /// S¹ fiber phase (W) — positional context in the fibration.
    pub phase: f64,
}

/// Brute-force resonance scan: query vs every element in a GeometricPath.
///
/// For each stored element: recover it from the running products, compose
/// with the query's inverse, measure drift, decompose via Hopf.
/// Returns the top-k matches sorted by drift (closest first).
///
/// Zero heap allocations in the hot loop — all scratch buffers are
/// pre-allocated. O(n) time, O(k) output memory.
pub fn resonance_scan(
    group: &dyn LieGroup,
    query: &[f64],
    path: &GeometricPath,
    k: usize,
) -> Vec<ResonanceHit> {
    let n = path.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let dim = group.dim();
    let inv_query = group.inverse(query);

    // Pre-allocate scratch buffers — zero allocations in the loop.
    let mut inv_prev = vec![0.0; dim];
    let mut element = vec![0.0; dim];
    let mut gap = vec![0.0; dim];
    let mut results: Vec<ResonanceHit> = Vec::with_capacity(n);

    for t in 1..=n {
        // Recover element: g_t = C_{t-1}⁻¹ · C_t. No allocation.
        group.inverse_into(path.running_product(t - 1), &mut inv_prev);
        group.compose_into(&inv_prev, path.running_product(t), &mut element);

        // Gap to query: inv(query) · element. No allocation.
        group.compose_into(&inv_query, &element, &mut gap);

        let drift = group.distance_from_identity(&gap);
        let gap_arr: [f64; 4] = [gap[0], gap[1], gap[2], gap[3]];
        let (base, phase) = hopf_decompose(&gap_arr);

        results.push(ResonanceHit {
            index: t - 1,
            drift,
            base,
            phase,
        });
    }

    // Sort by drift — closest matches first.
    results.sort_by(|a, b| {
        a.drift
            .partial_cmp(&b.drift)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(k);
    results
}

/// Resonance scan against a flat array of pre-embedded elements.
///
/// Same algorithm as resonance_scan, but operates on a flat slice of
/// elements (n × dim) instead of a GeometricPath. Used when elements
/// are already embedded and you don't need running products.
/// Zero heap allocations in the hot loop.
pub fn resonance_scan_flat(
    group: &dyn LieGroup,
    query: &[f64],
    elements: &[f64],
    dim: usize,
    k: usize,
) -> Vec<ResonanceHit> {
    let n = elements.len() / dim;
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let inv_query = group.inverse(query);
    let mut gap = vec![0.0; dim];
    let mut results: Vec<ResonanceHit> = Vec::with_capacity(n);

    for i in 0..n {
        let element = &elements[i * dim..(i + 1) * dim];
        group.compose_into(&inv_query, element, &mut gap);
        let drift = group.distance_from_identity(&gap);

        let gap_arr: [f64; 4] = [gap[0], gap[1], gap[2], gap[3]];
        let (base, phase) = hopf_decompose(&gap_arr);

        results.push(ResonanceHit {
            index: i,
            drift,
            base,
            phase,
        });
    }

    results.sort_by(|a, b| {
        a.drift
            .partial_cmp(&b.drift)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::bytes_to_sphere;
    use crate::groups::sphere::SphereGroup;

    /// The 10 animals — our definitive resonance test dataset.
    /// Same philosophy as the gilgamesh 100-position test:
    /// concrete, traceable, every result predictable.
    fn animals() -> Vec<Vec<f64>> {
        let names: Vec<&[u8]> = vec![
            b"cat", b"dog", b"fish", b"bird", b"frog",
            b"bear", b"wolf", b"deer", b"hawk", b"seal",
        ];
        names.iter().map(|name| bytes_to_sphere(name, false)).collect()
    }

    fn build_animal_path() -> GeometricPath {
        let g = SphereGroup;
        let elements = animals();
        let flat: Vec<f64> = elements.iter().flat_map(|e| e.iter().cloned()).collect();
        GeometricPath::from_elements(Box::new(g), &flat, 4)
    }

    /// Query 1: exact match. FIND "cat" → position 0, sigma ≈ 0.
    /// The same bytes always land on the same point on S³.
    #[test]
    fn exact_match() {
        let g = SphereGroup;
        let path = build_animal_path();
        let query = bytes_to_sphere(b"cat", false);

        let results = resonance_scan(&g, &query, &path, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0, "cat should be at position 0");
        assert!(
            results[0].drift < 1e-10,
            "exact match should have sigma ≈ 0, got {}",
            results[0].drift
        );
    }

    /// Query 2: exact match in the middle. FIND "wolf" → position 6, sigma ≈ 0.
    #[test]
    fn exact_match_middle() {
        let g = SphereGroup;
        let path = build_animal_path();
        let query = bytes_to_sphere(b"wolf", false);

        let results = resonance_scan(&g, &query, &path, 1);
        assert_eq!(results[0].index, 6, "wolf should be at position 6");
        assert!(results[0].drift < 1e-10);
    }

    /// Query 3: not stored. FIND "horse" → some position, sigma > 0.
    /// The query has no exact match, but the geometry still returns
    /// the closest element on S³. The sigma tells you it's not exact.
    #[test]
    fn no_exact_match() {
        let g = SphereGroup;
        let path = build_animal_path();
        let query = bytes_to_sphere(b"horse", false);

        let results = resonance_scan(&g, &query, &path, 1);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].drift > 0.01,
            "non-stored query should have sigma > 0, got {}",
            results[0].drift
        );
    }

    /// Query 4: top-k. FIND "cat" top 3 → cat at sigma≈0, then two others.
    /// The exact match always comes first. Others are sorted by distance.
    #[test]
    fn top_k_matches() {
        let g = SphereGroup;
        let path = build_animal_path();
        let query = bytes_to_sphere(b"cat", false);

        let results = resonance_scan(&g, &query, &path, 3);
        assert_eq!(results.len(), 3);
        // First result is the exact match
        assert_eq!(results[0].index, 0);
        assert!(results[0].drift < 1e-10);
        // Others are further away
        assert!(results[1].drift > results[0].drift);
        assert!(results[2].drift >= results[1].drift);
    }

    /// Query 5: duplicate. Store "cat" twice. FIND "cat" → both positions.
    /// The algebra treats identical bytes as identical elements.
    /// Both should have sigma ≈ 0.
    #[test]
    fn duplicate_records() {
        let g = SphereGroup;
        let mut elements = animals();
        elements.push(bytes_to_sphere(b"cat", false)); // cat again at position 10
        let flat: Vec<f64> = elements.iter().flat_map(|e| e.iter().cloned()).collect();
        let path = GeometricPath::from_elements(Box::new(g), &flat, 4);

        let query = bytes_to_sphere(b"cat", false);
        let results = resonance_scan(&SphereGroup, &query, &path, 3);

        // Both cats should be in the top results with sigma ≈ 0
        let cat_matches: Vec<&ResonanceHit> =
            results.iter().filter(|r| r.drift < 1e-10).collect();
        assert_eq!(
            cat_matches.len(),
            2,
            "should find both cats with sigma ≈ 0"
        );
        let positions: Vec<usize> = cat_matches.iter().map(|r| r.index).collect();
        assert!(positions.contains(&0), "should find cat at position 0");
        assert!(positions.contains(&10), "should find cat at position 10");
    }

    /// Determinism: same query, same path, same result. Every time.
    /// SHA-256 is deterministic. Composition is deterministic. The
    /// resonance result must be identical across runs.
    #[test]
    fn deterministic() {
        let g = SphereGroup;
        let path = build_animal_path();
        let query = bytes_to_sphere(b"bird", false);

        let r1 = resonance_scan(&g, &query, &path, 5);
        let r2 = resonance_scan(&g, &query, &path, 5);

        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.index, b.index);
            assert!((a.drift - b.drift).abs() < 1e-15);
        }
    }

    /// Sigma symmetry: resonance(A→B) has the same sigma as resonance(B→A).
    /// The geodesic distance on S³ is symmetric by the bi-invariant metric.
    #[test]
    fn sigma_symmetry() {
        let g = SphereGroup;
        let a = bytes_to_sphere(b"cat", false);
        let b = bytes_to_sphere(b"dog", false);

        // Gap A→B: compose(inverse(A), B)
        let inv_a = g.inverse(&a);
        let gap_ab = g.compose(&inv_a, &b);
        let sigma_ab = g.distance_from_identity(&gap_ab);

        // Gap B→A: compose(inverse(B), A)
        let inv_b = g.inverse(&b);
        let gap_ba = g.compose(&inv_b, &a);
        let sigma_ba = g.distance_from_identity(&gap_ba);

        assert!(
            (sigma_ab - sigma_ba).abs() < 1e-10,
            "geodesic distance should be symmetric: {} vs {}",
            sigma_ab,
            sigma_ba
        );
    }

    /// Hopf channels of exact match: sigma ≈ 0 means the gap is near
    /// identity. The base direction and phase should be near-degenerate
    /// (identity has no meaningful direction).
    #[test]
    fn hopf_channels_exact_match() {
        let g = SphereGroup;
        let path = build_animal_path();
        let query = bytes_to_sphere(b"cat", false);

        let results = resonance_scan(&g, &query, &path, 1);
        assert!(results[0].drift < 1e-10);
        // At identity, sigma ≈ 0 — the channels are near-degenerate
        // but should still be well-defined numbers (not NaN/Inf).
        assert!(results[0].phase.is_finite());
        assert!(results[0].base.iter().all(|v| v.is_finite()));
    }

    /// Every stored element is reachable. No blind spots.
    /// This is Theorem 2 (uniform detectability) applied to retrieval:
    /// every position in the path is equally findable.
    #[test]
    fn every_element_findable() {
        let g = SphereGroup;
        let path = build_animal_path();
        let animals_data = animals();
        let names = [
            "cat", "dog", "fish", "bird", "frog", "bear", "wolf", "deer", "hawk", "seal",
        ];

        for (i, element) in animals_data.iter().enumerate() {
            let results = resonance_scan(&g, element, &path, 1);
            assert_eq!(
                results[0].index, i,
                "{} should be at position {}",
                names[i], i
            );
            assert!(
                results[0].drift < 1e-10,
                "{} should have exact match (sigma ≈ 0), got {}",
                names[i],
                results[0].drift
            );
        }
    }

    /// The flat scan variant produces identical results to the path scan.
    #[test]
    fn flat_scan_matches_path_scan() {
        let g = SphereGroup;
        let elements = animals();
        let flat: Vec<f64> = elements.iter().flat_map(|e| e.iter().cloned()).collect();
        let path = GeometricPath::from_elements(Box::new(SphereGroup), &flat, 4);

        let query = bytes_to_sphere(b"deer", false);
        let path_results = resonance_scan(&g, &query, &path, 3);
        let flat_results = resonance_scan_flat(&g, &query, &flat, 4, 3);

        for (p, f) in path_results.iter().zip(flat_results.iter()) {
            assert_eq!(p.index, f.index);
            assert!((p.drift - f.drift).abs() < 1e-10);
        }
    }
}
