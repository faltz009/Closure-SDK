"""Demo: Stream classification and multi-fault localization.

Two capabilities unique to the SDK:
  1. localize_all  — static: find every incident between two complete sequences
  2. StreamClassifier — stream: classify incidents in real-time with bounded latency
"""

from __future__ import annotations

import closure_sdk as closure

W = 92


# ── Section 1: Multi-Fault Localization ────────────────────────────

def section_localize_all() -> bool:
    print("=" * W)
    print("  SECTION 1: Multi-Fault Localization  (localize_all)")
    print("=" * W)

    print(f"\n  Given two complete sequences, find EVERY incident — not just the first.")
    print(f"  For each incident: what record, what type (missing or reordered).\n")

    # Scenario 1: Missing records
    source = [b"tx-001", b"tx-002", b"tx-003", b"tx-004", b"tx-005"]
    target = [b"tx-001",            b"tx-003",            b"tx-005"]

    faults = closure.localize_all(source, target)

    print(f"  Scenario 1 — Missing records:")
    print(f"    Source: {[r.decode() for r in source]}")
    print(f"    Target: {[r.decode() for r in target]}")
    print(f"    Incidents found: {len(faults)}")
    for f in faults:
        print(f"      {f.incident_type:<20} record={f.record!r:<12} "
              f"source_idx={f.source_index}  target_idx={f.target_index}")

    # Scenario 2: Reordered records
    source2 = [b"a", b"b", b"c", b"d", b"e"]
    target2 = [b"a", b"c", b"b", b"d", b"e"]

    faults2 = closure.localize_all(source2, target2)

    print(f"\n  Scenario 2 — Reordered records:")
    print(f"    Source: {[r.decode() for r in source2]}")
    print(f"    Target: {[r.decode() for r in target2]}")
    print(f"    Incidents found: {len(faults2)}")
    for f in faults2:
        print(f"      {f.incident_type:<20} record={f.record!r:<12} "
              f"source_idx={f.source_index}  target_idx={f.target_index}")

    # Scenario 3: Mixed — missing + reorder
    source3 = [b"x", b"y", b"z", b"w", b"v"]
    target3 = [b"x", b"z", b"w"]

    faults3 = closure.localize_all(source3, target3)

    print(f"\n  Scenario 3 — Mixed (missing + content mismatch):")
    print(f"    Source: {[r.decode() for r in source3]}")
    print(f"    Target: {[r.decode() for r in target3]}")
    print(f"    Incidents found: {len(faults3)}")
    for f in faults3:
        print(f"      {f.incident_type:<20} record={f.record!r:<12} "
              f"source_idx={f.source_index}  target_idx={f.target_index}")

    # Scenario 4: Coherent — no incidents
    source4 = [b"ok-1", b"ok-2", b"ok-3"]
    target4 = [b"ok-1", b"ok-2", b"ok-3"]

    faults4 = closure.localize_all(source4, target4)

    print(f"\n  Scenario 4 — Coherent (identical sequences):")
    print(f"    Incidents found: {len(faults4)}")

    ok = len(faults) >= 1 and len(faults2) >= 1 and len(faults3) >= 1 and len(faults4) == 0
    print(f"\n  All scenarios correct: {'PASS' if ok else 'FAIL'}")
    return ok


# ── Section 2: Stream Classification ──────────────────────────────

def section_stream_classifier() -> bool:
    print(f"\n\n{'=' * W}")
    print("  SECTION 2: Stream Classification  (StreamClassifier)")
    print("=" * W)

    print(f"""
  The oracle problem: when a record arrives on one side but not the other,
  is it missing or just late? At arrival time, you cannot know.

  The StreamClassifier solves this with a bounded grace period:
    1. Record arrives on one side → held as unresolved
    2. Same record arrives on other side within grace → silent resolve
    3. Grace expires → promoted to "missing"
    4. Record arrives after promotion → reclassified to "reorder"

  Every step is the same binary decision. The recursion is the algorithm.
""")

    # Demo 1: Coherent stream
    print("  Demo 1 — Coherent stream (10 records, both sides):")
    c = closure.StreamClassifier()
    for i in range(10):
        rec = f"record-{i}".encode()
        c.ingest(rec, i, "source")
        c.ingest(rec, i, "target")
    incidents = c.advance_cycle()
    print(f"    Records processed: 10")
    print(f"    Incidents: {len(incidents)}")
    print(f"    Status: {'COHERENT' if len(incidents) == 0 else 'DIVERGENT'}")

    # Demo 2: Missing record
    print(f"\n  Demo 2 — Missing record (source has 'tx-99', target does not):")
    c2 = closure.StreamClassifier()
    c2.ingest(b"tx-99", 0, "source")
    missing = c2.advance_cycle()
    print(f"    After grace period: {len(missing)} incident(s)")
    if missing:
        print(f"    Type: {missing[0].incident_type}")
        print(f"    Record: {missing[0].record!r}")
        print(f"    Source index: {missing[0].source_index}")
        print(f"    Target index: {missing[0].target_index}")

    # Demo 3: Late arrival → reclassification
    print(f"\n  Demo 3 — Late arrival reclassification:")
    c3 = closure.StreamClassifier()
    c3.ingest(b"late-record", 0, "source")
    c3.advance_cycle()  # promoted to missing
    print(f"    Step 1: source gets 'late-record', grace expires → missing")
    print(f"    Step 2: target gets 'late-record' (late arrival)")
    result = c3.ingest(b"late-record", 5, "target")
    if result:
        print(f"    Reclassified to: {result.incident_type}")
        print(f"    Source index: {result.source_index} → Target index: {result.target_index}")
    print(f"    Reclassified count: {c3.reclassified_count}")

    # Demo 4: Chained errors — the recursive property
    print(f"\n  Demo 4 — Chained errors (the recursive property):")
    print(f"    After any misalignment, only two possibilities exist:")
    print(f"      a) The record arrives late → reclassify to reorder")
    print(f"      b) Another record is missing → same case, same solution")
    print(f"    The grace period handles both identically. That's the recursion.\n")

    c4 = closure.StreamClassifier()
    labels = ["A", "B", "C"]
    for i, label in enumerate(labels):
        c4.ingest(label.encode(), i, "source")

    promoted = c4.advance_cycle()
    print(f"    Source sends A, B, C — target silent.")
    print(f"    After grace: {len(promoted)} missing incidents")

    for i, label in enumerate(labels):
        r = c4.ingest(label.encode(), i + 10, "target")
        status = f"→ reclassified to {r.incident_type}" if r else "→ silent"
        print(f"    Target sends '{label}' (late) {status}")

    print(f"\n    Final state:")
    print(f"      Unresolved source: {c4.unresolved_source}")
    print(f"      Unresolved target: {c4.unresolved_target}")
    print(f"      Reclassified: {c4.reclassified_count}")

    ok = (len(incidents) == 0
          and len(missing) == 1
          and result is not None
          and c4.reclassified_count == 3
          and c4.unresolved_source == 0)

    print(f"\n  All demos correct: {'PASS' if ok else 'FAIL'}")
    return ok


# ── Section 3: Incident Valence ────────────────────────────────────

def section_incident_valence() -> None:
    print(f"\n\n{'=' * W}")
    print("  SECTION 3: Incident Valence  (expose_incident)")
    print("=" * W)

    print(f"\n  Each incident carries a color signature from the Hopf decomposition.")
    print(f"  The axis tells you the TYPE of failure (existence vs position).")
    print(f"  The channels tell you the SHAPE of the divergence.\n")

    source = [b"a", b"b", b"c", b"d"]
    target = [b"a", b"c"]  # b missing, d missing

    faults = closure.localize_all(source, target)

    mon = closure.Seer()
    mon.ingest_many(source)
    drift_elem = mon.state().element

    print(f"  Source: {[r.decode() for r in source]}")
    print(f"  Target: {[r.decode() for r in target]}")
    print(f"  Faults: {len(faults)}\n")

    print(f"    {'Record':<10} {'Type':<20} {'Axis':<12} {'σ':>8} {'R':>8} {'G':>8} {'B':>8}")
    print(f"    {'─' * 10} {'─' * 20} {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for fault in faults:
        iv = closure.expose_incident(fault, drift_elem)
        print(f"    {fault.record!r:<10} {fault.incident_type:<20} {iv.axis:<12} "
              f"{iv.sigma:>8.4f} {iv.base[0]:>+8.4f} {iv.base[1]:>+8.4f} {iv.base[2]:>+8.4f}")


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * W)
    print("  CLOSURE SDK — STREAM CLASSIFICATION & MULTI-FAULT LOCALIZATION")
    print("=" * W)

    loc_ok = section_localize_all()
    stream_ok = section_stream_classifier()
    section_incident_valence()

    print(f"\n\n{'=' * W}")
    print("  SUMMARY")
    print("=" * W)
    print("  The canon provides two tools for finding what broke:")
    print()
    print("    localize_all       Static. Two complete sequences → every incident.")
    print("                       Compose, search, classify, remove, repeat.")
    print()
    print("    StreamClassifier   Stream. Records arrive in real-time.")
    print("                       Match, wait, promote, reclassify.")
    print("                       The oracle problem → bounded binary decision.")
    print()
    print("  Both reduce to the same algebra: composition on S³,")
    print("  inversion for subtraction, sigma for measurement.")
    print()
    results = [
        ("Multi-fault localization", loc_ok),
        ("Stream classification", stream_ok),
    ]
    for name, ok in results:
        print(f"    {name}: {'PASS' if ok else 'FAIL'}")
    print()


if __name__ == "__main__":
    main()
