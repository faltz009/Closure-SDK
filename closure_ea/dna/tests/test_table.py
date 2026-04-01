"""Test Closure DNA columnar Table — full Python API.

Schema: name(bytes), age(f64), city(bytes), score(f64)
5 people. Filter, aggregate, sort, persistence.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
from closure_ea.dna import Table


def fresh_dir(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"cdna_{name}_"))


SCHEMA = [
    ("name", "bytes", False),
    ("age", "f64", False),
    ("city", "bytes", True),
    ("score", "f64", False),
]

PEOPLE = [
    [b"Alice", 30.0, b"Tokyo", 85.5],
    [b"Bob", 25.0, b"Paris", 92.0],
    [b"Charlie", 35.0, b"Tokyo", 78.3],
    [b"Diana", 28.0, b"Cairo", 91.0],
    [b"Edward", 40.0, b"Paris", 88.7],
]

PEOPLE_COLUMNS = [
    [b"Alice", b"Bob", b"Charlie", b"Diana", b"Edward"],
    [30.0, 25.0, 35.0, 28.0, 40.0],
    [b"Tokyo", b"Paris", b"Tokyo", b"Cairo", b"Paris"],
    [85.5, 92.0, 78.3, 91.0, 88.7],
]


def make_table(name):
    d = fresh_dir(name)
    t = Table.create(d / "test.cdna", SCHEMA)
    t.insert_many(PEOPLE)
    return d, t


# ── Create and insert ────────────────────────────────────────────

def test_create_and_count():
    d, t = make_table("create")
    assert t.count() == 5
    t.save()
    shutil.rmtree(d)


def test_get_fields():
    d, t = make_table("get")
    assert t.get_bytes(0, 0) == b"Alice"
    assert t.get_f64(0, 1) == 30.0
    assert t.get_bytes(2, 2) == b"Tokyo"
    assert t.get_f64(4, 3) == 88.7
    t.save()
    shutil.rmtree(d)


def test_column_index():
    d, t = make_table("colidx")
    assert t.column_index("name") == 0
    assert t.column_index("age") == 1
    assert t.column_index("city") == 2
    assert t.column_index("score") == 3
    t.save()
    shutil.rmtree(d)


# ── Filter ───────────────────────────────────────────────────────

def test_filter_equals():
    d, t = make_table("filter_eq")
    tokyo = t.filter_equals("city", b"Tokyo")
    assert tokyo == [0, 2]  # Alice and Charlie
    paris = t.filter_equals("city", b"Paris")
    assert paris == [1, 4]  # Bob and Edward
    t.save()
    shutil.rmtree(d)


def test_check():
    d, t = make_table("check")
    drift = t.check()
    assert isinstance(drift, float)
    assert np.isfinite(drift)
    hopf = t.check_hopf()
    assert isinstance(hopf, tuple)
    assert len(hopf) == 3
    assert np.isfinite(hopf[0])
    t.save()
    shutil.rmtree(d)


def test_identity_deterministic():
    d = fresh_dir("identity")
    t1 = Table.create(d / "a.cdna", SCHEMA)
    t2 = Table.create(d / "b.cdna", SCHEMA)
    t1.insert_many(PEOPLE)
    t2.insert_many(PEOPLE)
    np.testing.assert_allclose(t1.identity(), t2.identity(), atol=1e-12)
    t1.save()
    t2.save()
    shutil.rmtree(d)


# ── Persistence ──────────────────────────────────────────────────

def test_persistence():
    d = fresh_dir("persist")
    path = d / "test.cdna"

    with Table.create(path, SCHEMA) as t:
        t.insert_many(PEOPLE)
        id_before = t.identity().copy()

    with Table.open(path) as t:
        assert t.count() == 5
        np.testing.assert_allclose(t.identity(), id_before, atol=1e-12)
        assert t.get_bytes(0, 0) == b"Alice"
        assert t.get_f64(4, 1) == 40.0
        tokyo = t.filter_equals("city", b"Tokyo")
        assert tokyo == [0, 2]

    shutil.rmtree(d)


# ── Context manager ──────────────────────────────────────────────

def test_context_manager():
    d = fresh_dir("ctx")
    with Table.create(d / "test.cdna", SCHEMA) as t:
        t.insert_many(PEOPLE)
        assert len(t) == 5
    with Table.open(d / "test.cdna") as t:
        assert t.count() == 5
    shutil.rmtree(d)


def test_insert_columns():
    d = fresh_dir("columns")
    with Table.create(d / "test.cdna", SCHEMA) as t:
        t.insert_columns(PEOPLE_COLUMNS)
        assert t.count() == 5
        assert t.get_bytes(0, 0) == b"Alice"
        assert t.get_f64(4, 1) == 40.0
        assert t.filter_equals("city", b"Tokyo") == [0, 2]
    shutil.rmtree(d)


def test_search_and_genome():
    d = fresh_dir("search")
    with Table.create(d / "test.cdna", SCHEMA) as t:
        t.insert_columns(PEOPLE_COLUMNS)
        t.build_genome()
        assert t.genome_depth() >= 1
        assert t.genome_codons() >= 1

        hits = t.search([b"Alice", 30.0, b"Tokyo", 85.5], k=1)
        assert len(hits) == 1
        assert hits[0].position == 0
        assert hits[0].drift < 1e-10
        row_view = t.inspect_row(0)
        assert np.isfinite(row_view[0])
    shutil.rmtree(d)


def test_update_and_delete():
    d = fresh_dir("mutate")
    with Table.create(d / "test.cdna", SCHEMA) as t:
        t.insert_columns(PEOPLE_COLUMNS)
        t.update(1, [b"Beatrice", 26.0, b"Lima", 95.0])
        assert t.get_row(1) == [b"Beatrice", 26.0, b"Lima", 95.0]
        id_before = t.identity().copy()
        t.delete(1)
        # Tombstone: count stays, row quaternion is identity
        assert t.count() == 5
        # Identity changed because the composition changed
        import numpy as np
        assert not np.allclose(t.identity(), id_before)
    shutil.rmtree(d)


def test_history_and_snapshot_restore():
    d = fresh_dir("history")
    path = d / "test.cdna"
    with Table.create(path, SCHEMA) as t:
        t.insert_many(PEOPLE)
        snap = t.snapshot("baseline")
        assert snap == "baseline"

        t.update(0, [b"Alicia", 30.0, b"Tokyo", 85.5])
        assert t.get_bytes(0, 0) == b"Alicia"

        history = t.history()
        ops = [entry["op"] for entry in history]
        assert "insert_columns" in ops or "insert" in ops
        assert "snapshot" in ops
        assert "update" in ops

        snapshots = t.snapshots()
        assert any(meta["name"] == "baseline" for meta in snapshots)

        t.restore_snapshot("baseline")
        assert t.get_bytes(0, 0) == b"Alice"
        ops = [entry["op"] for entry in t.history()]
        assert "restore_snapshot" in ops

    shutil.rmtree(d)


# ── Scale ────────────────────────────────────────────────────────

def test_10k_records():
    d = fresh_dir("scale")
    schema = [
        ("id", "f64", False),
        ("name", "bytes", False),
        ("age", "f64", False),
        ("city", "bytes", True),
        ("score", "f64", False),
    ]
    cities = [b"Tokyo", b"Paris", b"Cairo", b"Lima", b"Oslo"]
    ids = [float(i) for i in range(10_000)]
    names = [f"user_{i:08}".encode() for i in range(10_000)]
    ages = [20.0 + (i % 60) for i in range(10_000)]
    city_col = [cities[i % 5] for i in range(10_000)]
    scores = [(i * 17 % 1000) / 10.0 for i in range(10_000)]
    with Table.create(d / "test.cdna", schema) as t:
        t.insert_columns([ids, names, ages, city_col, scores])
        assert t.count() == 10_000
        assert t.get_bytes(0, 1) == b"user_00000000"
        assert t.get_f64(9999, 0) == 9999.0

        # Column-level filter
        tokyo = t.filter_equals("city", b"Tokyo")
        assert len(tokyo) == 2000
        over_50 = t.filter_cmp("age", ">", 50.0)
        assert len(over_50) > 0
        avg_age = t.avg("age")
        assert avg_age > 40 and avg_age < 55
        sorted_idx = t.argsort("age")
        assert len(sorted_idx) == 10_000
        hits = t.search(
            [1234.0, b"user_00001234", 54.0, b"Oslo", 97.8],
            k=1,
        )
        assert len(hits) == 1
        assert np.isfinite(hits[0].drift)

    shutil.rmtree(d)
def test_filter_cmp():
    d, t = make_table("filter_cmp")
    over_30 = t.filter_cmp("age", ">", 30.0)
    assert over_30 == [2, 4]  # Charlie(35) and Edward(40)
    under_30 = t.filter_cmp("age", "<", 30.0)
    assert under_30 == [1, 3]  # Bob(25) and Diana(28)
    t.save()
    shutil.rmtree(d)


# ── Aggregate ────────────────────────────────────────────────────

def test_sum():
    d, t = make_table("sum")
    total = t.sum("score")
    assert abs(total - (85.5 + 92.0 + 78.3 + 91.0 + 88.7)) < 0.01
    t.save()
    shutil.rmtree(d)


def test_avg():
    d, t = make_table("avg")
    avg_age = t.avg("age")
    assert abs(avg_age - (30 + 25 + 35 + 28 + 40) / 5) < 0.01
    t.save()
    shutil.rmtree(d)


# ── Sort ─────────────────────────────────────────────────────────

def test_argsort():
    d, t = make_table("sort")
    asc = t.argsort("age")
    assert asc == [1, 3, 0, 2, 4]  # Bob(25), Diana(28), Alice(30), Charlie(35), Edward(40)
    desc = t.argsort("age", descending=True)
    assert desc == [4, 2, 0, 3, 1]
    t.save()
    shutil.rmtree(d)


def test_i64_columns_round_trip_and_numeric_ops():
    d = fresh_dir("i64")
    schema = [
        ("id", "i64", False),
        ("name", "bytes", False),
        ("score", "i64", False),
    ]
    rows = [
        [9_007_199_254_740_993, b"Alice", 10],
        [9_007_199_254_740_995, b"Bob", 30],
        [9_007_199_254_740_994, b"Carol", 20],
    ]

    with Table.create(d / "test.cdna", schema) as t:
        t.insert_many(rows)
        assert t.get_i64(0, 0) == 9_007_199_254_740_993
        assert t.get_row(1) == [9_007_199_254_740_995, b"Bob", 30]
        assert t.filter_cmp("score", ">", 15.0) == [1, 2]
        assert t.sum("score") == 60.0
        assert t.avg("score") == 20.0
        assert t.argsort("id") == [0, 2, 1]

    shutil.rmtree(d)


def test_nullable_fields_round_trip_and_skip_in_aggregates():
    d = fresh_dir("nulls")
    schema = [
        ("name", "bytes", False),
        ("age", "f64", False),
        ("score", "i64", False),
    ]

    with Table.create(d / "test.cdna", schema) as t:
        t.insert([b"Alice", 30.0, 10])
        t.insert([b"Bob", None, None])
        t.insert([b"Carol", 20.0, 30])

        assert t.get_row(1) == [b"Bob", None, None]
        assert t.sum("score") == 40.0
        assert abs(t.avg("age") - 25.0) < 1e-12
        assert t.filter_cmp("score", ">", 15.0) == [2]

        t.update(1, [b"Bob", 40.0, None])
        assert t.get_row(1) == [b"Bob", 40.0, None]
        assert abs(t.avg("age") - 30.0) < 1e-12

    shutil.rmtree(d)


# ── Integrity ────────────────────────────────────────────────────
