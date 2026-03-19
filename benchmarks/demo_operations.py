"""Demo: Algebraic operations through the Closure SDK.

Operations enabled by invertible closure summaries:
  1. Recover  — extract any embedded element from the Oracle's history
  2. Range    — coherence of any sub-range in O(1)
  3. Diff     — algebraic delta between two snapshots (constant size)
  4. Patch    — apply/reverse a delta to any snapshot in O(1)
  5. Valence  — color-channel decomposition of divergence
"""

from __future__ import annotations

import time

import numpy as np

import closure_sdk as closure

W = 92


def t(ms: float) -> str:
    if ms < 0.1:
        return f"{ms * 1000:.1f} μs"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


# ── Section 1: Element Recovery ────────────────────────────────────

def section_recovery() -> bool:
    print("=" * W)
    print("  SECTION 1: Element Recovery  (Oracle.recover)")
    print("=" * W)

    records = [f"order-{i}".encode() for i in range(10_000)]
    oracle = closure.Oracle.from_records(records)

    test_indices = [1, 100, 500, 777, 1000]

    print(f"\n  Oracle length: {len(oracle)} records")
    print(f"  Recovering elements at indices: {test_indices}")
    print()

    all_ok = True
    for idx in test_indices:
        recovered = oracle.recover(idx)
        fresh = closure.embed(records[idx - 1])
        dist = np.linalg.norm(recovered - fresh.element)
        ok = dist < 1e-12
        all_ok = all_ok and ok
        print(f"    t={idx:>4}: dist={dist:.2e}  [{'OK' if ok else 'FAIL'}]")

    print(f"\n  All recoveries exact: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ── Section 2: Range Queries ───────────────────────────────────────

def section_range_queries() -> None:
    print(f"\n\n{'=' * W}")
    print("  SECTION 2: Range Queries  (Oracle.check_range, O(1))")
    print("=" * W)

    records = [f"event-{i}".encode() for i in range(10_000)]
    oracle = closure.Oracle.from_records(records)

    ranges = [(0, 100), (0, 1000), (1000, 2000), (5000, 5500), (9000, 10000)]

    print(f"\n  Oracle length: {len(oracle)}")
    print(f"  Querying sub-range coherence (each is O(1) — one inverse + one compose):\n")

    for i, j in ranges:
        t0 = time.perf_counter()
        for _ in range(10_000):
            sigma = oracle.check_range(i, j)
        us = (time.perf_counter() - t0) * 1e6 / 10_000
        print(f"    range [{i:>5}, {j:>5}] ({j - i:>5} records): σ = {sigma:>8.4f}   time = {us:.3f} μs")

    # Tamper one record and show the targeted detection
    tamper_idx = 1500
    tampered = list(records)
    tampered[tamper_idx] = b"CORRUPTED"
    tampered_oracle = closure.Oracle.from_records(tampered)

    print(f"\n  Tampered record at index {tamper_idx}.")
    print(f"  Comparing range queries between clean and tampered:\n")

    test_ranges = [(0, 1000), (1000, 2000), (2000, 3000), (1400, 1600)]
    for i, j in test_ranges:
        sigma_clean = oracle.check_range(i, j)
        sigma_tampered = tampered_oracle.check_range(i, j)
        delta = abs(sigma_clean - sigma_tampered)
        flag = "DIFFERS" if delta > 1e-9 else "same"
        print(f"    range [{i:>5}, {j:>5}]: clean σ={sigma_clean:>8.4f}  "
              f"tampered σ={sigma_tampered:>8.4f}  Δ={delta:.4f}  [{flag}]")

    print(f"\n  Only ranges containing index {tamper_idx} show a difference.")
    print(f"  Each query: O(1). No scanning.")


# ── Section 3: Diff & Patch ────────────────────────────────────────

def section_diff_and_patch() -> bool:
    print(f"\n\n{'=' * W}")
    print("  SECTION 3: Algebraic Diff & Patch  (closure.diff / closure.compose)")
    print("=" * W)

    records_a = [f"snapshot-a-{i}".encode() for i in range(10_000)]
    records_b = list(records_a)
    # Modify 500 records
    rng = np.random.default_rng(99)
    changed = rng.choice(len(records_b), size=500, replace=False)
    for idx in changed:
        records_b[idx] = f"modified-{idx}".encode()

    state_a = closure.Oracle.from_records(records_a).state()
    state_b = closure.Oracle.from_records(records_b).state()

    delta = closure.diff(state_a, state_b)
    diff_sigma = closure.sigma(delta)

    print(f"\n  Snapshot A: 10,000 records (original)")
    print(f"  Snapshot B: 10,000 records (500 modified)")
    print(f"  σ(diff) = {diff_sigma:.6f}")

    # Apply diff to a third snapshot
    records_c = [f"snapshot-c-{i}".encode() for i in range(10_000)]
    state_c = closure.Oracle.from_records(records_c).state()

    patched = closure.compose(state_c, delta)
    unpatched = closure.compose(patched, closure.invert(delta))
    roundtrip = closure.sigma(closure.diff(unpatched, state_c))

    print(f"\n  Snapshot C (independent)")
    print(f"  C patched with diff(A→B), then unpatched")
    print(f"  Roundtrip distance: {roundtrip:.2e}")
    print(f"  Patch-unpatch exact: {'PASS' if roundtrip < 1e-12 else 'FAIL'}")

    print(f"\n  What this means:")
    print(f"    diff(A→B) is 32 bytes. It captures the net aggregate change on S³.")
    print(f"    Apply it to any snapshot in O(1). Reverse it in O(1).")
    print(f"    This is 'git diff' + 'git apply' for aggregate state.")

    return roundtrip < 1e-12


# ── Section 4: Localization Performance ────────────────────────────

def section_localization() -> bool:
    print(f"\n\n{'=' * W}")
    print("  SECTION 4: Localization Performance  (Witness vs Oracle)")
    print("=" * W)

    n = 100_000
    rng = np.random.default_rng(2026)
    records = [rng.bytes(64) for _ in range(n)]
    ci = 77_777
    corrupted = list(records)
    corrupted[ci] = b"\x00" * 64

    # Witness (hierarchical tree)
    t0 = time.perf_counter()
    witness = closure.Witness.from_records(records)
    w_build = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    w_drift = witness.check(corrupted)
    w_check = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    w_loc = witness.localize(corrupted)
    w_search = (time.perf_counter() - t0) * 1000

    # Oracle (path binary search)
    t0 = time.perf_counter()
    ref = closure.Oracle.from_records(records)
    o_build = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    test = closure.Oracle.from_records(corrupted)
    o_rebuild = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    o_loc = ref.localize_against(test)
    o_search = (time.perf_counter() - t0) * 1000

    w_ok = w_loc.index == ci
    o_ok = o_loc.index == ci

    print(f"\n  n = {n:,} records, corruption at index {ci:,}\n")

    print(f"  Witness (hierarchical tree):")
    print(f"    Build:    {t(w_build):>10}")
    print(f"    Detect:   {t(w_check):>10}   σ = {w_drift:.6f}")
    print(f"    Localize: {t(w_search):>10}   index = {w_loc.index}  checks = {w_loc.checks}  [{'OK' if w_ok else 'FAIL'}]")

    print(f"\n  Oracle (path binary search):")
    print(f"    Build:    {t(o_build):>10}")
    print(f"    Rebuild:  {t(o_rebuild):>10}")
    print(f"    Localize: {t(o_search):>10}   index = {o_loc.index}  checks = {o_loc.checks}  [{'OK' if o_ok else 'FAIL'}]")

    speedup = n / max(o_loc.checks, 1)
    print(f"\n  Linear scan would need {n:,} comparisons.")
    print(f"  Oracle used {o_loc.checks} comparisons — {speedup:,.0f}× faster.")

    return w_ok and o_ok


# ── Section 5: Valence — Color Channels ────────────────────────────

def section_valence() -> None:
    print(f"\n\n{'=' * W}")
    print("  SECTION 5: Valence — Color-Channel Decomposition")
    print("=" * W)

    print(f"\n  The Hopf fibration maps S³ → S² × S¹.")
    print(f"  This decomposes any divergence into perceptual channels:")
    print(f"    σ     — total magnitude (the thermometer)")
    print(f"    R,G,B — base coordinates on S² (chrominance = displacement type)")
    print(f"    W     — fiber phase on S¹ (luminance = existence/magnitude)")
    print()

    # Generate a divergence and expose it
    mon = closure.Seer()
    mon.ingest(b"record-a")
    mon.ingest(b"record-b")
    mon.ingest(b"record-c")

    v = closure.expose(mon.state().element)

    print(f"  After ingesting 3 records:")
    print(f"    σ = {v.sigma:.6f}")
    print(f"    R = {v.base[0]:>+.6f}   G = {v.base[1]:>+.6f}   B = {v.base[2]:>+.6f}")
    print(f"    W = {v.phase:>+.6f}")

    # Show how different corruption types produce different channel signatures
    print(f"\n  Comparing channel signatures for different corruption types:\n")

    scenarios = [
        ("Single record",    [b"a"]),
        ("Two records",      [b"a", b"b"]),
        ("Same record ×3",   [b"x", b"x", b"x"]),
        ("Long sequence",    [f"r-{i}".encode() for i in range(100)]),
    ]

    print(f"    {'Scenario':<20} {'σ':>10} {'R':>10} {'G':>10} {'B':>10} {'W':>10}")
    print(f"    {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")

    for label, recs in scenarios:
        s = closure.Seer()
        s.ingest_many(recs)
        val = closure.expose(s.state().element)
        print(f"    {label:<20} {val.sigma:>10.4f} {val.base[0]:>+10.4f} "
              f"{val.base[1]:>+10.4f} {val.base[2]:>+10.4f} {val.phase:>+10.4f}")

    print(f"\n  Each scenario lands at a different point on the ball.")
    print(f"  The channels tell you what KIND of divergence, not just how much.")


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * W)
    print("  CLOSURE SDK — ALGEBRAIC OPERATIONS DEMO")
    print("  Operations enabled by invertible composition on S³")
    print("=" * W)

    recovery_ok = section_recovery()
    section_range_queries()
    patch_ok = section_diff_and_patch()
    loc_ok = section_localization()
    section_valence()

    print(f"\n\n{'=' * W}")
    print("  SUMMARY")
    print("=" * W)
    print("  Because group elements have inverses, the SDK supports:")
    print("    1. RECOVER — extract any embedded element from the Oracle")
    print("    2. RANGE   — coherence of any sub-range in O(1)")
    print("    3. DIFF    — algebraic delta between snapshots (32 bytes)")
    print("    4. PATCH   — apply/reverse a delta in O(1)")
    print("    5. LOCALIZE — find the exact corrupted record in O(log n)")
    print("    6. VALENCE — decompose divergence into color channels")
    print()
    print("  Hashes detect. Merkle trees locate. Closure locates, measures,")
    print("  decomposes, and composes — all on a single algebraic object.")
    results = [
        ("Element recovery", recovery_ok),
        ("Diff/patch roundtrip", patch_ok),
        ("Localization", loc_ok),
    ]
    for name, ok in results:
        print(f"    {name}: {'PASS' if ok else 'FAIL'}")
    print()


if __name__ == "__main__":
    main()
