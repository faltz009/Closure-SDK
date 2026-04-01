from __future__ import annotations

import json

from closure_ea.dna.database import Database
from closure_ea.dna.query import parse
from closure_ea.dna.repl import _dispatch


def _make_db(tmp_path):
    db = Database.create(tmp_path / "repl_test.cdb")
    db.create_table(
        "people",
        [
            {"name": "id", "type": "f64", "primary": True},
            {"name": "name", "type": "bytes", "indexed": True},
            {"name": "city", "type": "bytes", "indexed": True},
            {"name": "age", "type": "f64"},
        ],
    )
    table = db.table("people")
    table.insert([1.0, b"Alice", b"Tokyo", 31.0])
    table.insert([2.0, b"Bob", b"Tokyo", 22.0])
    table.insert([3.0, b"Carol", b"Osaka", 29.0])
    return db


def test_repl_select_natural_syntax(tmp_path, capsys):
    db = _make_db(tmp_path)
    try:
        _dispatch(db, parse('SELECT people WHERE city = Tokyo ORDER BY age DESC LIMIT 1'))
        out = capsys.readouterr().out.strip()
        rows = json.loads(out)
        assert rows == [{"id": 1.0, "name": "Alice", "city": "Tokyo", "age": 31.0}]
    finally:
        db.close()


def test_repl_select_sql_like_syntax(tmp_path, capsys):
    db = _make_db(tmp_path)
    try:
        _dispatch(db, parse('SELECT * FROM people WHERE city = Tokyo ORDER BY age DESC LIMIT 1'))
        out = capsys.readouterr().out.strip()
        rows = json.loads(out)
        assert rows == [{"id": 1.0, "name": "Alice", "city": "Tokyo", "age": 31.0}]
    finally:
        db.close()


def test_repl_update_where_natural_syntax(tmp_path, capsys):
    db = _make_db(tmp_path)
    try:
        _dispatch(db, parse('UPDATE people SET city = Kyoto WHERE name = Alice'))
        out = capsys.readouterr().out.strip()
        assert out == "1 row(s) updated"
        rows = db.select("people", where=[("name", "=", "Alice")])
        assert rows[0]["city"] == b"Kyoto"
    finally:
        db.close()


def test_repl_insert_into_sql_like_syntax(tmp_path, capsys):
    db = _make_db(tmp_path)
    try:
        _dispatch(db, parse('INSERT INTO people VALUES 4 "Davi" "Kyoto" 40'))
        out = capsys.readouterr().out.strip()
        assert out == "3"
        rows = db.select("people", where=[("name", "=", "Davi")])
        assert rows == [{"id": 4.0, "name": b"Davi", "city": b"Kyoto", "age": 40.0}]
    finally:
        db.close()


def test_repl_delete_where_natural_syntax(tmp_path, capsys):
    db = _make_db(tmp_path)
    try:
        _dispatch(db, parse('DELETE people WHERE age < 29'))
        out = capsys.readouterr().out.strip()
        assert out == "1 row(s) deleted"
        rows = db.select("people", where=[("age", "<", 29)])
        assert rows == []
    finally:
        db.close()


def test_repl_delete_from_sql_like_syntax(tmp_path, capsys):
    db = _make_db(tmp_path)
    try:
        _dispatch(db, parse('DELETE FROM people WHERE age < 29'))
        out = capsys.readouterr().out.strip()
        assert out == "1 row(s) deleted"
        rows = db.select("people", where=[("age", "<", 29)])
        assert rows == []
    finally:
        db.close()


def test_repl_group_natural_syntax(tmp_path, capsys):
    db = _make_db(tmp_path)
    try:
        _dispatch(db, parse('GROUP people BY city COUNT * AS n AVG age AS avg_age ORDER BY city'))
        out = capsys.readouterr().out.strip()
        rows = json.loads(out)
        assert rows == [
            {"city": "Osaka", "n": 1, "avg_age": 29.0},
            {"city": "Tokyo", "n": 2, "avg_age": 26.5},
        ]
    finally:
        db.close()


def test_repl_alter_table_add_column(tmp_path, capsys):
    db = _make_db(tmp_path)
    try:
        _dispatch(db, parse('ALTER TABLE people ADD COLUMN score i64 DEFAULT 7'))
        out = capsys.readouterr().out.strip()
        assert out == "people"

        rows = db.select("people", order_by="id")
        assert [row["score"] for row in rows] == [7, 7, 7]
        assert db.schema("people")[-1]["type"] == "i64"
    finally:
        db.close()


def test_repl_join_left_outer(tmp_path, capsys):
    db = Database.create(tmp_path / "repl_join.cdb")
    try:
        db.create_table(
            "customers",
            [
                {"name": "id", "type": "i64", "primary": True},
                {"name": "name", "type": "bytes"},
            ],
        )
        db.create_table(
            "orders",
            [
                {"name": "id", "type": "i64", "primary": True},
                {"name": "customer_id", "type": "i64", "references": "customers.id"},
            ],
        )
        db.table("customers").insert_many([[1, b"Alice"], [2, b"Bob"]])
        db.table("orders").insert([10, 1])

        _dispatch(db, parse("JOIN customers orders id customer_id LEFT"))
        out = capsys.readouterr().out.strip()
        rows = json.loads(out)
        assert rows[0]["customers.id"] == 1
        assert rows[0]["orders.id"] == 10
        assert rows[1]["customers.id"] == 2
        assert rows[1]["orders.id"] is None
    finally:
        db.close()
