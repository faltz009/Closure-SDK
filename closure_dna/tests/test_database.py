import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import json
import sys

from closure_dna import Database
from closure_dna.database import _journal_path


PEOPLE_SCHEMA = [
    ("id", "f64", False),
    ("name", "bytes", False),
    ("city", "bytes", True),
]

ORDERS_SCHEMA = [
    ("order_id", "f64", False),
    ("customer", "bytes", False),
    ("amount", "f64", False),
]


def fresh_dir(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"cdna_db_{name}_"))


def test_create_open_database_and_tables():
    root = fresh_dir("create")

    with Database.create(root / "app.cdb") as db:
        people = db.create_table("people", PEOPLE_SCHEMA)
        orders = db.create_table("orders", ORDERS_SCHEMA)

        people.insert_columns([
            [0.0, 1.0],
            [b"Alice", b"Bob"],
            [b"Tokyo", b"Paris"],
        ])
        orders.insert_columns([
            [10.0, 11.0],
            [b"Alice", b"Bob"],
            [125.0, 210.0],
        ])

        assert db.tables() == ["orders", "people"]
        assert db.has_table("people")
        assert db.has_table("orders")

    with Database.open(root / "app.cdb") as db:
        assert db.tables() == ["orders", "people"]

        people = db.table("people")
        orders = db.table("orders")

        assert people.count() == 2
        assert orders.count() == 2
        assert people.get_bytes(0, 1) == b"Alice"
        assert orders.get_f64(1, 2) == 210.0

    shutil.rmtree(root)


def test_create_table_rejects_duplicate_name():
    root = fresh_dir("duplicate")

    with Database.create(root / "app.cdb") as db:
        db.create_table("people", PEOPLE_SCHEMA)
        try:
            db.create_table("people", PEOPLE_SCHEMA)
            assert False, "expected FileExistsError"
        except FileExistsError:
            pass

    shutil.rmtree(root)


def test_transaction_commit_and_rollback():
    root = fresh_dir("txn")

    with Database.create(root / "app.cdb") as db:
        people = db.create_table("people", PEOPLE_SCHEMA)
        people.insert_columns([
            [0.0, 1.0],
            [b"Alice", b"Bob"],
            [b"Tokyo", b"Paris"],
        ])

    db = Database.open(root / "app.cdb")
    with db.transaction() as tx:
        people = tx.table("people")
        people.update(1, [1.0, b"Beatrice", b"Lima"])

    with Database.open(root / "app.cdb") as db2:
        assert db2.table("people").get_row(1) == [1.0, b"Beatrice", b"Lima"]

    try:
        with Database.open(root / "app.cdb") as db3:
            with db3.transaction() as tx:
                tx.table("people").delete(0)
                raise RuntimeError("rollback")
    except RuntimeError:
        pass

    with Database.open(root / "app.cdb") as db4:
        assert db4.table("people").count() == 2
        assert db4.table("people").get_row(0) == [0.0, b"Alice", b"Tokyo"]

    shutil.rmtree(root)


def test_schema_metadata_and_relations():
    root = fresh_dir("schema")

    with Database.create(root / "app.cdb") as db:
        db.create_table(
            "customers",
            [
                {"name": "customer_id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
                {"name": "city", "type": "bytes", "indexed": True},
            ],
        )
        db.create_table(
            "orders",
            [
                {"name": "order_id", "type": "f64", "indexed": False, "primary": True},
                {"name": "customer_id", "type": "f64", "indexed": False, "references": "customers.customer_id"},
                {"name": "amount", "type": "f64", "indexed": False},
            ],
        )

        assert db.primary_key("customers") == "customer_id"
        fks = db.foreign_keys("orders")
        assert len(fks) == 1
        assert fks[0]["references"] == "customers.customer_id"

    shutil.rmtree(root)


def test_select_and_join():
    root = fresh_dir("query")

    with Database.create(root / "shop.cdb") as db:
        customers = db.create_table(
            "customers",
            [
                {"name": "customer_id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
                {"name": "city", "type": "bytes", "indexed": True},
            ],
        )
        orders = db.create_table(
            "orders",
            [
                {"name": "order_id", "type": "f64", "indexed": False, "primary": True},
                {"name": "customer_id", "type": "f64", "indexed": False, "references": "customers.customer_id"},
                {"name": "amount", "type": "f64", "indexed": False},
            ],
        )

        customers.insert_columns([
            [1.0, 2.0, 3.0],
            [b"Alice", b"Bob", b"Charlie"],
            [b"Tokyo", b"Paris", b"Tokyo"],
        ])
        orders.insert_columns([
            [10.0, 11.0, 12.0, 13.0],
            [1.0, 1.0, 2.0, 3.0],
            [125.0, 210.0, 55.0, 300.0],
        ])

        tokyo = db.select("customers", where=[("city", "=", "Tokyo")], order_by="customer_id")
        assert [row["name"] for row in tokyo] == [b"Alice", b"Charlie"]

        large_orders = db.select("orders", where=[("amount", ">", 100.0)], order_by="amount")
        assert [row["order_id"] for row in large_orders] == [10.0, 11.0, 13.0]

        joined = db.join("orders", "customers", "customer_id", where=[("customers.city", "=", b"Tokyo")])
        assert len(joined) == 3
        assert joined[0]["customers.name"] == b"Alice"

    shutil.rmtree(root)


def test_select_i64_and_null_and_left_outer_join():
    root = fresh_dir("query_i64_null")

    with Database.create(root / "shop.cdb") as db:
        customers = db.create_table(
            "customers",
            [
                {"name": "customer_id", "type": "i64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
                {"name": "city", "type": "bytes", "indexed": True},
            ],
        )
        orders = db.create_table(
            "orders",
            [
                {"name": "order_id", "type": "i64", "indexed": False, "primary": True},
                {"name": "customer_id", "type": "i64", "indexed": False, "references": "customers.customer_id"},
                {"name": "amount", "type": "f64", "indexed": False},
                {"name": "note", "type": "bytes", "indexed": False},
            ],
        )

        customers.insert_many(
            [
                [1, b"Alice", b"Tokyo"],
                [2, b"Bob", None],
                [3, b"Charlie", b"Lima"],
            ]
        )
        orders.insert_many(
            [
                [10, 1, 125.0, b"first"],
                [11, 1, None, None],
                [12, 3, 300.0, b"rush"],
            ]
        )

        high_ids = db.select("orders", where=[("order_id", ">=", 11)], order_by="order_id")
        assert [row["order_id"] for row in high_ids] == [11, 12]

        missing_notes = db.select("orders", where=[("note", "=", None)], order_by="order_id")
        assert [row["order_id"] for row in missing_notes] == [11]

        null_cities = db.select("customers", where=[("city", "is", None)], order_by="customer_id")
        assert [row["name"] for row in null_cities] == [b"Bob"]

        ordered = db.select("customers", order_by="city")
        assert [row["name"] for row in ordered] == [b"Charlie", b"Alice", b"Bob"]

        joined = db.join("customers", "orders", "customer_id", outer="left", limit=10)
        assert len(joined) == 4
        bob_row = next(row for row in joined if row["customers.customer_id"] == 2)
        assert bob_row["orders.order_id"] is None
        assert bob_row["orders.amount"] is None

    shutil.rmtree(root)


def test_select_expressions_and_subquery_in():
    root = fresh_dir("expr_subquery")

    with Database.create(root / "shop.cdb") as db:
        customers = db.create_table(
            "customers",
            [
                {"name": "customer_id", "type": "i64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
                {"name": "city", "type": "bytes", "indexed": False},
            ],
        )
        orders = db.create_table(
            "orders",
            [
                {"name": "order_id", "type": "i64", "indexed": False, "primary": True},
                {"name": "customer_id", "type": "i64", "indexed": False, "references": "customers.customer_id"},
                {"name": "amount", "type": "f64", "indexed": False},
            ],
        )

        customers.insert_many(
            [
                [1, b"Alice", b"Tokyo"],
                [2, b"Bob", b"Paris"],
                [3, b"Carol", b"Tokyo"],
            ]
        )
        orders.insert_many(
            [
                [10, 1, 125.0],
                [11, 2, 80.0],
                [12, 3, 300.0],
            ]
        )

        projected = db.select(
            "customers",
            columns=["name", "UPPER(city) AS city_upper", "customer_id + 10 AS boosted_id"],
            where=[("customer_id + 10", ">=", 12)],
            order_by="name",
        )
        assert projected == [
            {"name": b"Bob", "city_upper": b"PARIS", "boosted_id": 12},
            {"name": b"Carol", "city_upper": b"TOKYO", "boosted_id": 13},
        ]

        tokyo_ids = db.subquery("customers", "customer_id", where=[("city", "=", "Tokyo")], order_by="customer_id")
        assert tokyo_ids == [1, 3]

        tokyo_orders = db.select("orders", where=[("customer_id", "in", tokyo_ids)], order_by="order_id")
        assert [row["order_id"] for row in tokyo_orders] == [10, 12]

    shutil.rmtree(root)


def test_foreign_key_enforced_and_parent_delete_blocked():
    root = fresh_dir("fk")

    with Database.create(root / "shop.cdb") as db:
        customers = db.create_table(
            "customers",
            [
                {"name": "customer_id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False, "not_null": True},
            ],
        )
        orders = db.create_table(
            "orders",
            [
                {"name": "order_id", "type": "f64", "indexed": False, "primary": True},
                {"name": "customer_id", "type": "f64", "indexed": False, "references": "customers.customer_id"},
                {"name": "amount", "type": "f64", "indexed": False},
            ],
        )

        customers.insert([1.0, b"Alice"])
        orders.insert([10.0, 1.0, 125.0])

        try:
            orders.insert([11.0, 999.0, 50.0])
            assert False, "expected foreign key failure"
        except ValueError as exc:
            assert "foreign key failed" in str(exc)

        try:
            customers.delete(0)
            assert False, "expected parent delete rejection"
        except ValueError as exc:
            assert "still references it" in str(exc)

    shutil.rmtree(root)


def test_compact_removes_tombstones():
    root = fresh_dir("compact")

    with Database.create(root / "app.cdb") as db:
        people = db.create_table(
            "people",
            [
                {"name": "id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
                {"name": "city", "type": "bytes", "indexed": True},
            ],
        )
        people.insert_many(
            [
                [1.0, b"Alice", b"Tokyo"],
                [2.0, b"Bob", b"Paris"],
                [3.0, b"Charlie", b"Lima"],
            ]
        )
        people.delete(1)

        live = db.select("people", order_by="id")
        assert [row["name"] for row in live] == [b"Alice", b"Charlie"]

        removed = db.compact("people")
        assert removed == {"people": 1}
        assert db.table("people").count() == 2
        assert [row["name"] for row in db.select("people", order_by="id")] == [b"Alice", b"Charlie"]

    shutil.rmtree(root)


def test_add_column_rebuilds_table_and_preserves_tombstones():
    root = fresh_dir("add_column")
    db_path = root / "app.cdb"

    with Database.create(db_path) as db:
        people = db.create_table(
            "people",
            [
                {"name": "id", "type": "i64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
            ],
        )
        people.insert_many(
            [
                [1, b"Alice"],
                [2, b"Bob"],
                [3, b"Charlie"],
            ]
        )
        people.delete(1)

        db.add_column("people", {"name": "score", "type": "i64", "indexed": False}, default=7)

        schema = db.schema("people")
        assert [col["name"] for col in schema] == ["id", "name", "score"]
        assert schema[2]["type"] == "i64"

        raw = db.table("people")
        assert raw.count() == 3
        assert raw.get_row(0) == [1, b"Alice", 7]
        assert raw.get_row(1) == [2, b"Bob", 7]
        assert raw.get_row(2) == [3, b"Charlie", 7]

        live = db.select("people", order_by="id")
        assert [row["id"] for row in live] == [1, 3]
        assert [row["score"] for row in live] == [7, 7]

    shutil.rmtree(root)


def test_add_column_rejects_not_null_without_default():
    root = fresh_dir("add_column_not_null")

    with Database.create(root / "app.cdb") as db:
        people = db.create_table(
            "people",
            [
                {"name": "id", "type": "i64", "indexed": False, "primary": True},
            ],
        )
        people.insert([1])

        try:
            db.add_column("people", {"name": "score", "type": "i64", "indexed": False, "not_null": True})
            assert False, "expected add_column NOT NULL failure"
        except ValueError as exc:
            assert "NOT NULL" in str(exc)

    shutil.rmtree(root)


def test_open_recovers_interrupted_commit_from_backup():
    root = fresh_dir("recover")
    db_path = root / "app.cdb"

    with Database.create(db_path) as db:
        people = db.create_table(
            "people",
            [
                {"name": "id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
            ],
        )
        people.insert([1.0, b"Alice"])

    backup = db_path.with_name(f"{db_path.name}.bak")
    db_path.rename(backup)
    _journal_path(db_path).write_text(
        json.dumps(
            {
                "source": str(db_path),
                "backup": str(backup),
                "staged": str(root / "missing_staged.cdb"),
                "temp_root": str(root / "missing_temp"),
            }
        ),
        encoding="utf-8",
    )

    with Database.open(db_path) as db:
        assert db.table("people").get_row(0) == [1.0, b"Alice"]

    assert db_path.exists()
    assert not backup.exists()
    assert not _journal_path(db_path).exists()
    shutil.rmtree(root)


def test_audit_repair_and_info():
    root = fresh_dir("audit")
    db_path = root / "app.cdb"

    with Database.create(db_path) as db:
        people = db.create_table(
            "people",
            [
                {"name": "id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
            ],
        )
        people.insert([1.0, b"Alice"])
        people.insert([2.0, b"Bob"])

    tree_path = db_path / "people.cdna" / "tree.q"
    tree_path.unlink()

    with Database.open(db_path) as db:
        audit = db.audit("people")
        assert audit["ok"] is False
        assert audit["drift"] >= 0.0

        repaired = db.repair("people")
        assert repaired["repaired"] is True
        assert repaired["audit"]["ok"] is True

        info = db.info("people")
        assert info["rows"] == 2
        assert info["live_rows"] == 2
        assert len(info["identity"]) == 4

    shutil.rmtree(root)


def test_export_import_json_and_csv():
    root = fresh_dir("io")
    db_path = root / "app.cdb"
    json_path = root / "people.json"
    csv_path = root / "people.csv"

    with Database.create(db_path) as db:
        people = db.create_table(
            "people",
            [
                {"name": "id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
                {"name": "city", "type": "bytes", "indexed": True},
            ],
        )
        people.insert_many(
            [
                [1.0, b"Alice", b"Tokyo"],
                [2.0, b"Bob", b"Paris"],
            ]
        )

        db.export_table("people", "json", json_path)
        db.export_table("people", "csv", csv_path)
        db.import_table("people_json", "json", json_path)
        db.import_table("people_csv", "csv", csv_path)

        json_rows = db.select("people_json", order_by="id")
        csv_rows = db.select("people_csv", order_by="id")
        assert [row["name"] for row in json_rows] == [b"Alice", b"Bob"]
        assert [row["city"] for row in csv_rows] == [b"Tokyo", b"Paris"]

    shutil.rmtree(root)


def test_import_infers_i64_and_nulls():
    root = fresh_dir("import_nulls")
    db_path = root / "app.cdb"
    json_path = root / "people.json"
    csv_path = root / "people.csv"

    json_path.write_text(
        json.dumps(
            [
                {"id": 9007199254740993, "name": "Alice", "score": 10, "city": "Tokyo"},
                {"id": 9007199254740994, "name": "Bob", "score": None, "city": None},
            ]
        ),
        encoding="utf-8",
    )
    csv_path.write_text(
        "id,name,score,city\n"
        "9007199254740993,Alice,10,Tokyo\n"
        "9007199254740994,Bob,,\n",
        encoding="utf-8",
    )

    with Database.create(db_path) as db:
        db.import_table("people_json", "json", json_path)
        db.import_table("people_csv", "csv", csv_path)

        json_schema = db.schema("people_json")
        csv_schema = db.schema("people_csv")
        assert json_schema[0]["type"] == "i64"
        assert json_schema[2]["type"] == "i64"
        assert csv_schema[0]["type"] == "i64"
        assert csv_schema[2]["type"] == "i64"

        json_rows = db.select("people_json", order_by="id")
        csv_rows = db.select("people_csv", order_by="id")

        assert json_rows[0]["id"] == 9007199254740993
        assert json_rows[1]["score"] is None
        assert json_rows[1]["city"] is None
        assert csv_rows[1]["score"] is None
        assert csv_rows[1]["city"] is None

    shutil.rmtree(root)


def test_writer_waits_for_active_transaction_lock():
    root = fresh_dir("isolation")
    db_path = root / "app.cdb"

    with Database.create(db_path) as db:
        people = db.create_table(
            "people",
            [
                {"name": "id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
            ],
        )
        people.insert([1.0, b"Alice"])

    db = Database.open(db_path)
    with db.transaction() as tx:
        tx.table("people").update(0, [1.0, b"Alicia"])
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "from closure_dna import Database;"
                    f"db=Database.open(r'{db_path}');"
                    "db.table('people').insert([2.0,b'Bob']);"
                    "db.close()"
                ),
            ]
        )
        time.sleep(0.3)
        assert proc.poll() is None

    proc.wait(timeout=5)
    assert proc.returncode == 0

    with Database.open(db_path) as db2:
        assert db2.table("people").count() == 2

    shutil.rmtree(root)


def test_read_transaction_is_stable_snapshot():
    root = fresh_dir("snapshot")
    db_path = root / "app.cdb"

    with Database.create(db_path) as db:
        people = db.create_table(
            "people",
            [
                {"name": "id", "type": "f64", "indexed": False, "primary": True},
                {"name": "name", "type": "bytes", "indexed": False},
            ],
        )
        people.insert([1.0, b"Alice"])

    with Database.open(db_path) as db:
        with db.read_transaction() as snap:
            assert snap.table("people").get_row(0) == [1.0, b"Alice"]

            with db.transaction() as tx:
                tx.table("people").update(0, [1.0, b"Alicia"])
                tx.table("people").insert([2.0, b"Bob"])

            assert snap.table("people").get_row(0) == [1.0, b"Alice"]
            assert snap.table("people").count() == 1

        assert db.table("people").get_row(0) == [1.0, b"Alicia"]
        assert db.table("people").count() == 2

    shutil.rmtree(root)
