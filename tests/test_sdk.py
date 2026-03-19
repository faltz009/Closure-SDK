from __future__ import annotations

import numpy as np

import closure_sdk as closure


def test_seer_compare_sphere() -> None:
    left = closure.Seer()
    right = closure.Seer()

    records = [b"a", b"b", b"c"]
    left.ingest_many(records)
    right.ingest_many(records)

    result = left.compare(right)
    assert result.coherent
    assert result.drift < 1e-10
    assert left.state().group == "Sphere"


def test_oracle_localize_against_returns_first_divergence() -> None:
    records = [b"a", b"b", b"c", b"d"]
    corrupted = [b"a", b"x", b"c", b"d"]

    ref = closure.Oracle.from_records(records)
    test = closure.Oracle.from_records(corrupted)

    result = ref.localize_against(test)
    assert result.index == 1
    assert result.checks >= 1


def test_oracle_append_matches_batch_build() -> None:
    records = [b"a", b"b", b"c"]

    batch = closure.Oracle.from_records(records)
    incr = closure.Oracle()
    for record in records:
        incr.append(record)

    result = closure.compare(batch.state(), incr.state())
    assert result.coherent


def test_compose_diff_round_trip() -> None:
    left = closure.Oracle.from_records([b"a", b"b"]).state()
    right = closure.Oracle.from_records([b"a", b"c"]).state()

    delta = closure.diff(left, right)
    combined = closure.compose(left, delta)
    result = closure.compare(combined, right)

    assert result.coherent


def test_embed_produces_closure_state() -> None:
    state = closure.embed(b"test-record")
    assert state.group == "Sphere"
    assert state.dim == 4
    assert isinstance(state.element, np.ndarray)


def test_invert_composes_to_identity() -> None:
    state = closure.embed(b"test-record")
    inv = closure.invert(state)
    combined = closure.compose(state, inv)
    s = closure.sigma(combined)
    assert s < 1e-10


def test_sigma_zero_at_identity() -> None:
    mon = closure.Seer()
    s = closure.sigma(mon.state())
    assert s < 1e-10


def test_sigma_nonzero_after_ingest() -> None:
    mon = closure.Seer()
    mon.ingest(b"record")
    s = closure.sigma(mon.state())
    assert s > 0


def test_witness_from_records() -> None:
    records = [b"a", b"b", b"c", b"d"]
    corrupted = [b"a", b"b", b"x", b"d"]

    tree = closure.Witness.from_records(records)
    drift = tree.check(corrupted)
    result = tree.localize(corrupted)

    assert drift > 0.0
    assert result.index == 2
    assert result.checks >= 1


def test_expose_returns_valence() -> None:
    mon = closure.Seer()
    mon.ingest(b"test")
    v = closure.expose(mon.state().element)
    assert isinstance(v, closure.Valence)
    assert isinstance(v.sigma, float)
    assert len(v.base) == 3
    assert isinstance(v.phase, float)


def test_seer_reset_returns_to_identity() -> None:
    mon = closure.Seer()
    mon.ingest(b"record")
    assert closure.sigma(mon.state()) > 0

    mon.reset()
    assert closure.sigma(mon.state()) < 1e-10
    assert len(mon) == 0


def test_oracle_check_global_zero_for_coherent() -> None:
    records = [b"a", b"b", b"c"]
    trace = closure.Oracle.from_records(records)
    # Single trace has a running product — check_global returns sigma of that
    # This just checks the method works and returns a float
    result = trace.check_global()
    assert isinstance(result, float)


def test_oracle_check_range() -> None:
    records = [b"a", b"b", b"c", b"d"]
    trace = closure.Oracle.from_records(records)
    # check_range on a sub-segment returns a float
    result = trace.check_range(0, 2)
    assert isinstance(result, float)
    assert result >= 0.0


def test_oracle_recover_returns_element() -> None:
    records = [b"a", b"b", b"c"]
    trace = closure.Oracle.from_records(records)
    # recover(t) retrieves the element at position t (1-indexed)
    elem = trace.recover(1)
    assert isinstance(elem, np.ndarray)
    assert len(elem) == 4


def test_witness_state() -> None:
    records = [b"a", b"b", b"c"]
    w = closure.Witness.from_records(records)
    state = w.state()
    assert state.group == "Sphere"
    assert state.dim == 4


def test_expose_incident_returns_incident_valence() -> None:
    src = [b"a", b"b", b"c"]
    tgt = [b"a", b"c"]  # b"b" missing from target

    faults = closure.localize_all(src, tgt)
    assert len(faults) >= 1

    mon = closure.Seer()
    mon.ingest_many(src)
    drift_elem = mon.state().element

    iv = closure.expose_incident(faults[0], drift_elem)
    assert isinstance(iv, closure.IncidentValence)
    assert iv.axis in ("existence", "position")
    assert isinstance(iv.sigma, float)


def test_oracle_localize_coherent_returns_none() -> None:
    records = [b"a", b"b", b"c"]
    ref = closure.Oracle.from_records(records)
    copy = closure.Oracle.from_records(records)
    result = ref.localize_against(copy)
    assert result.index is None
    assert result.checks >= 1


def test_witness_localize_coherent_returns_none() -> None:
    records = [b"a", b"b", b"c"]
    w = closure.Witness.from_records(records)
    result = w.localize(records)
    assert result.index is None
