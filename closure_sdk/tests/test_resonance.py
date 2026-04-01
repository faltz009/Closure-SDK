"""Test the 8th primitive: resonance query.

Same dataset as the Rust tests — 10 animals, 5 queries — but through
the full Python → Rust pipeline. Every result is predictable and
traceable. Same philosophy as the gilgamesh 100-position test.

The Dataset:
    [cat, dog, fish, bird, frog, bear, wolf, deer, hawk, seal]
      0    1    2     3     4     5     6     7     8     9

The Queries:
    1. FIND "cat"   → position 0, sigma ≈ 0     (exact match)
    2. FIND "wolf"  → position 6, sigma ≈ 0     (exact match)
    3. FIND "horse" → some position, sigma > 0   (not stored)
    4. FIND "cat" top-3 → cat first, then neighbors
    5. Duplicate: store "cat" twice → both found
"""

import numpy as np
import closure_rs

# ── Setup ────────────────────────────────────────────────────────────

ANIMALS = [b"cat", b"dog", b"fish", b"bird", b"frog",
           b"bear", b"wolf", b"deer", b"hawk", b"seal"]


def build_animal_path():
    """Build a GeometricPath from the 10 animals."""
    return closure_rs.path_from_raw_bytes("Sphere", ANIMALS)


# ── Query 1: Exact match ────────────────────────────────────────────

def test_exact_match():
    """FIND "cat" → position 0, sigma ≈ 0."""
    path = build_animal_path()
    results = closure_rs.resonance_query_raw(b"cat", path, k=1)
    assert results.shape == (1, 6)
    index = int(results[0, 0])
    drift = results[0, 1]
    assert index == 0, f"cat should be at position 0, got {index}"
    assert drift < 1e-10, f"exact match should have drift ≈ 0, got {drift}"


# ── Query 2: Exact match in the middle ──────────────────────────────

def test_exact_match_middle():
    """FIND "wolf" → position 6, sigma ≈ 0."""
    path = build_animal_path()
    results = closure_rs.resonance_query_raw(b"wolf", path, k=1)
    index = int(results[0, 0])
    drift = results[0, 1]
    assert index == 6, f"wolf should be at position 6, got {index}"
    assert drift < 1e-10


# ── Query 3: Not stored ─────────────────────────────────────────────

def test_no_exact_match():
    """FIND "horse" → some position, sigma > 0."""
    path = build_animal_path()
    results = closure_rs.resonance_query_raw(b"horse", path, k=1)
    drift = results[0, 1]
    assert drift > 0.01, f"non-stored query should have drift > 0, got {drift}"


# ── Query 4: Top-k ──────────────────────────────────────────────────

def test_top_k():
    """FIND "cat" top 3 → cat at sigma≈0, then two others sorted."""
    path = build_animal_path()
    results = closure_rs.resonance_query_raw(b"cat", path, k=3)
    assert results.shape == (3, 6)

    # First result is exact match
    assert int(results[0, 0]) == 0
    assert results[0, 1] < 1e-10

    # Results are sorted by sigma (ascending)
    assert results[1, 1] > results[0, 1]
    assert results[2, 1] >= results[1, 1]


# ── Query 5: Duplicate records ──────────────────────────────────────

def test_duplicate_records():
    """Store "cat" at 0 and 10. FIND "cat" → both with sigma ≈ 0."""
    records = ANIMALS + [b"cat"]  # cat again at position 10
    path = closure_rs.path_from_raw_bytes("Sphere", records)
    results = closure_rs.resonance_query_raw(b"cat", path, k=3)

    # Both cats should have sigma ≈ 0
    exact_matches = results[results[:, 1] < 1e-10]
    assert len(exact_matches) == 2, f"should find 2 cats, found {len(exact_matches)}"
    positions = sorted(int(m[0]) for m in exact_matches)
    assert positions == [0, 10], f"cats should be at [0, 10], got {positions}"


# ── Algebraic Properties ────────────────────────────────────────────

def test_determinism():
    """Same query, same path, same result. Every time."""
    path = build_animal_path()
    r1 = closure_rs.resonance_query_raw(b"bird", path, k=5)
    r2 = closure_rs.resonance_query_raw(b"bird", path, k=5)
    np.testing.assert_array_equal(r1, r2)


def test_every_element_findable():
    """Every stored record is findable. Theorem 2: no blind spots."""
    path = build_animal_path()
    for i, animal in enumerate(ANIMALS):
        results = closure_rs.resonance_query_raw(animal, path, k=1)
        index = int(results[0, 0])
        drift = results[0, 1]
        name = animal.decode()
        assert index == i, f"{name} should be at position {i}, got {index}"
        assert drift < 1e-10, f"{name} should have drift ≈ 0, got {drift}"


def test_hopf_channels_are_finite():
    """The Hopf channels (base R,G,B and phase W) are always finite."""
    path = build_animal_path()
    results = closure_rs.resonance_query_raw(b"cat", path, k=5)
    # Columns 2,3,4 = base (R,G,B), column 5 = phase (W)
    assert np.all(np.isfinite(results[:, 2:6])), "Hopf channels must be finite"


def test_pre_embedded_query():
    """Query with a pre-embedded quaternion (not raw bytes)."""
    path = build_animal_path()
    # Embed the query manually
    query_elem = closure_rs.closure_element_from_raw_bytes("Sphere", [b"deer"])
    results = closure_rs.resonance_query(query_elem, path, k=1)
    index = int(results[0, 0])
    drift = results[0, 1]
    assert index == 7, f"deer should be at position 7, got {index}"
    assert drift < 1e-10


# ── Scale test ───────────────────────────────────────────────────────

def test_1000_records():
    """Resonance at 1000 records. Store, query, verify."""
    records = [f"record_{i:04d}".encode() for i in range(1000)]
    path = closure_rs.path_from_raw_bytes("Sphere", records)

    # Query for record at position 500
    results = closure_rs.resonance_query_raw(b"record_0500", path, k=1)
    index = int(results[0, 0])
    drift = results[0, 1]
    assert index == 500
    assert drift < 1e-10

    # Query for record at position 999
    results = closure_rs.resonance_query_raw(b"record_0999", path, k=1)
    assert int(results[0, 0]) == 999
    assert results[0, 1] < 1e-10

    # Query for non-existent record
    results = closure_rs.resonance_query_raw(b"record_9999", path, k=1)
    assert results[0, 1] > 0.01
