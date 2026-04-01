"""SQL acceptance tests for the current Closure DNA surface."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from closure_dna import Database
from closure_dna.sql import SQLResult, execute


def fresh_dir(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"cdna_rig_{name}_"))


def make_db(name: str) -> tuple[Path, Database]:
    root = fresh_dir(name)
    db = Database.create(root / "test.cdb")
    db.create_table(
        "people",
        [
            {"name": "id", "type": "i64", "primary": True},
            {"name": "name", "type": "bytes"},
            {"name": "age", "type": "f64"},
            {"name": "city", "type": "bytes"},
        ],
    )
    db.table("people").insert_many(
        [
            [1, b"Alice", 30.0, b"Tokyo"],
            [2, b"Bob", 25.0, b"Paris"],
            [3, b"Charlie", 35.0, b"Tokyo"],
            [4, b"Diana", 28.0, b"Cairo"],
            [5, b"Edward", 40.0, b"Paris"],
        ]
    )
    return root, db


def _close(root: Path, db: Database) -> None:
    db.close()
    shutil.rmtree(root)


def test_sql_result_shape():
    result = SQLResult(rows=[{"id": 1}], count=1, message="ok", scalar=1)
    assert result.rows == [{"id": 1}]
    assert result.count == 1
    assert result.message == "ok"
    assert result.scalar == 1


def test_sql_select_where_order_limit():
    root, db = make_db("select")
    try:
        result = execute(
            db,
            "SELECT name, age * 2 AS doubled FROM people WHERE city = 'Tokyo' ORDER BY id DESC LIMIT 1",
        )
        assert result.rows == [{"name": b"Charlie", "doubled": 70.0}]
    finally:
        _close(root, db)


def test_sql_aggregate():
    root, db = make_db("agg")
    try:
        result = execute(db, "SELECT AVG(age) FROM people WHERE city = 'Paris'")
        assert result.scalar == 32.5
        assert result.rows == [{"avg": 32.5}]
    finally:
        _close(root, db)


def test_sql_insert_update_delete():
    root, db = make_db("mutate")
    try:
        inserted = execute(db, "INSERT INTO people VALUES (6, 'Frank', 22, 'London')")
        assert inserted.count == 1
        assert db.select("people", where=[("id", "=", 6)])[0]["name"] == b"Frank"

        updated = execute(db, "UPDATE people SET city = 'Kyoto', age = 23 WHERE id = 6")
        assert updated.count == 1
        assert db.select("people", where=[("id", "=", 6)])[0]["city"] == b"Kyoto"

        deleted = execute(db, "DELETE FROM people WHERE id = 6")
        assert deleted.count == 1
        assert db.select("people", where=[("id", "=", 6)]) == []
    finally:
        _close(root, db)


def test_sql_alter_and_in():
    root, db = make_db("alter_in")
    try:
        result = execute(db, "ALTER TABLE people ADD COLUMN score i64 DEFAULT 7")
        assert "Added column" in result.message
        rows = execute(db, "SELECT * FROM people WHERE id IN (1, 3, 5) ORDER BY id").rows
        assert [row["id"] for row in rows] == [1, 3, 5]
        assert all(row["score"] == 7 for row in rows)
    finally:
        _close(root, db)
