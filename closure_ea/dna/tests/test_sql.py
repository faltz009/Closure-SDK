from __future__ import annotations

import tempfile
from pathlib import Path

from closure_ea.dna import Database
from closure_ea.dna.sql import execute


def fresh_dir(name: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"cdna_sql_{name}_"))


def _make_db(tmp_path: Path) -> Database:
    db = Database.create(tmp_path / "sql_test.cdb")
    db.create_table(
        "people",
        [
            {"name": "id", "type": "i64", "primary": True},
            {"name": "name", "type": "bytes"},
            {"name": "city", "type": "bytes"},
            {"name": "age", "type": "f64"},
        ],
    )
    people = db.table("people")
    people.insert_many(
        [
            [1, b"Alice", b"Tokyo", 31.0],
            [2, b"Bob", b"Paris", 22.0],
            [3, b"Carol", b"Osaka", 29.0],
        ]
    )
    db.create_table(
        "orders",
        [
            {"name": "id", "type": "i64", "primary": True},
            {"name": "customer_id", "type": "i64", "references": "people.id"},
            {"name": "amount", "type": "f64"},
        ],
    )
    db.table("orders").insert_many(
        [
            [10, 1, 125.0],
            [11, 1, 80.0],
            [12, 2, 55.0],
        ]
    )
    return db


def test_sql_select_insert_update_delete_and_alter(tmp_path):
    db = _make_db(tmp_path)
    try:
        result = execute(db, "SELECT * FROM people WHERE city = 'Tokyo'")
        assert result.count == 1
        assert result.rows[0]["name"] == b"Alice"

        result = execute(db, "SELECT name, age * 2 AS double_age FROM people WHERE id >= 2 ORDER BY id")
        assert result.rows == [
            {"name": b"Bob", "double_age": 44.0},
            {"name": b"Carol", "double_age": 58.0},
        ]

        result = execute(db, "SELECT COUNT(*) FROM people WHERE age >= 29")
        assert result.scalar == 2

        result = execute(db, "INSERT INTO people VALUES (4, 'Davi', 'Kyoto', 40)")
        assert result.count == 1
        assert db.select("people", where=[("id", "=", 4)])[0]["name"] == b"Davi"

        result = execute(db, "UPDATE people SET city = 'Lima', age = 41 WHERE id = 4")
        assert result.count == 1
        assert db.select("people", where=[("id", "=", 4)])[0]["city"] == b"Lima"

        result = execute(db, "DELETE FROM people WHERE id = 4")
        assert result.count == 1
        assert db.select("people", where=[("id", "=", 4)]) == []

        result = execute(db, "ALTER TABLE people ADD COLUMN score i64 DEFAULT 7")
        assert "Added column" in result.message
        rows = db.select("people", order_by="id")
        assert [row["score"] for row in rows] == [7, 7, 7]
    finally:
        db.close()


def test_database_execute_and_subquery_in(tmp_path):
    db = _make_db(tmp_path)
    try:
        tokyo = db.execute("SELECT id FROM people WHERE city = 'Tokyo'")
        ids = [row["id"] for row in tokyo.rows]
        assert ids == [1]

        result = db.execute("SELECT * FROM people WHERE id IN (1, 3) ORDER BY id DESC")
        assert [row["id"] for row in result.rows] == [3, 1]
    finally:
        db.close()


def test_sql_join_syntax(tmp_path):
    db = _make_db(tmp_path)
    try:
        result = execute(
            db,
            "SELECT people.name AS customer, orders.amount "
            "FROM people JOIN orders ON people.id = orders.customer_id "
            "WHERE orders.amount >= 80 ORDER BY orders.amount DESC",
        )
        assert result.rows == [
            {"customer": b"Alice", "orders.amount": 125.0},
            {"customer": b"Alice", "orders.amount": 80.0},
        ]
    finally:
        db.close()


def test_sql_group_by_syntax(tmp_path):
    db = _make_db(tmp_path)
    try:
        result = execute(
            db,
            "SELECT city, COUNT(*) AS n, AVG(age) AS avg_age "
            "FROM people GROUP BY city ORDER BY city",
        )
        assert result.rows == [
            {"city": b"Osaka", "n": 1, "avg_age": 29.0},
            {"city": b"Paris", "n": 1, "avg_age": 22.0},
            {"city": b"Tokyo", "n": 1, "avg_age": 31.0},
        ]
    finally:
        db.close()


def test_sql_nested_subquery_in(tmp_path):
    db = _make_db(tmp_path)
    try:
        result = execute(
            db,
            "SELECT amount FROM orders "
            "WHERE customer_id IN (SELECT id FROM people WHERE city = 'Tokyo') "
            "ORDER BY amount DESC",
        )
        assert [row["amount"] for row in result.rows] == [125.0, 80.0]
    finally:
        db.close()


def test_sql_custom_geometric_operations(tmp_path):
    db = _make_db(tmp_path)
    try:
        identity = execute(db, "SELECT IDENTITY() FROM people")
        assert identity.count == 1
        assert isinstance(identity.scalar, list)
        assert len(identity.scalar) == 4

        drift = execute(db, "SELECT DRIFT() FROM people")
        assert drift.scalar == db.table("people").check()

        decompose = execute(db, "SELECT DECOMPOSE_DRIFT() FROM people")
        assert set(decompose.scalar.keys()) == {"drift", "base", "phase"}
        assert len(decompose.scalar["base"]) == 3

        audit = execute(db, "AUDIT people")
        assert audit.scalar == db.audit("people")

        inspect = execute(db, "INSPECT ROW 0 FROM people")
        assert set(inspect.scalar.keys()) == {"drift", "base", "phase"}

        compact = execute(db, "COMPACT people")
        assert compact.scalar == {"people": 0}

        search = execute(
            db,
            "SELECT * FROM people RESONATE NEAR (1, 'Alice', 'Tokyo', 31) LIMIT 1",
        )
        assert search.count == 1
        assert search.rows[0]["name"] == b"Alice"
        assert search.rows[0]["_row"] == 0
        assert "_drift" in search.rows[0]
    finally:
        db.close()


def test_sql_parser_handles_escaped_quotes_and_parentheses(tmp_path):
    db = _make_db(tmp_path)
    try:
        execute(db, "INSERT INTO people VALUES (4, 'O''Brien', 'Tokyo', 45)")

        quoted = execute(db, "SELECT name FROM people WHERE name = 'O''Brien'")
        assert quoted.rows == [{"name": b"O'Brien"}]

        grouped = execute(
            db,
            "SELECT id FROM people WHERE (id = 1 OR id = 4) AND city = 'Tokyo' ORDER BY id",
        )
        assert [row["id"] for row in grouped.rows] == [1, 4]
    finally:
        db.close()


def test_sql_executes_multi_statement_scripts(tmp_path):
    db = _make_db(tmp_path)
    try:
        result = execute(
            db,
            """
            BEGIN;
            INSERT INTO people VALUES (4, 'Semi;Colon', 'Tokyo', 45);
            INSERT INTO people VALUES (5, 'O''Brien', 'Paris', 27);
            COMMIT;
            SELECT name FROM people WHERE name IN ('Semi;Colon', 'O''Brien') ORDER BY id;
            """,
        )
        assert [row["name"] for row in result.rows] == [b"Semi;Colon", b"O'Brien"]
        assert len(result.statements) == 5
        assert result.statements[0]["message"] == "BEGIN"
        assert result.statements[3]["message"] == "COMMIT"
    finally:
        db.close()


def test_sql_script_handles_custom_and_standard_statements(tmp_path):
    db = _make_db(tmp_path)
    try:
        result = execute(
            db,
            """
            AUDIT people;
            SELECT DRIFT() FROM people;
            SELECT name FROM people WHERE ((city = 'Tokyo' AND age >= 30) OR (city = 'Paris' AND age < 30)) ORDER BY id;
            """,
        )
        assert len(result.statements) == 3
        assert result.rows == [{"name": b"Alice"}, {"name": b"Bob"}]
    finally:
        db.close()


def test_sql_parser_handles_sql_functions_and_aliases(tmp_path):
    db = _make_db(tmp_path)
    try:
        result = execute(
            db,
            "SELECT UPPER(name) AS uname, ABS(age - 40) AS gap FROM people WHERE id = 1",
        )
        assert result.rows == [{"uname": b"ALICE", "gap": 9.0}]
    finally:
        db.close()


def test_sql_create_and_drop_table(tmp_path):
    db = _make_db(tmp_path)
    try:
        created = execute(
            db,
            "CREATE TABLE pets (id INTEGER PRIMARY KEY, owner_id INTEGER REFERENCES people(id), name TEXT NOT NULL, score REAL)",
        )
        assert "Created table pets" == created.message
        assert db.has_table("pets")
        schema = db.schema("pets")
        assert [col["name"] for col in schema] == ["id", "owner_id", "name", "score"]
        assert schema[0]["primary"] is True
        assert schema[1]["references"] == "people.id"
        assert schema[2]["not_null"] is True

        dropped = execute(db, "DROP TABLE pets")
        assert dropped.message == "Dropped table pets"
        assert not db.has_table("pets")
    finally:
        db.close()


def test_sql_distinct_having_and_like(tmp_path):
    db = _make_db(tmp_path)
    try:
        execute(db, "INSERT INTO people VALUES (4, 'Alicia', 'Tokyo', 26)")
        execute(db, "INSERT INTO people VALUES (5, 'Aaron', 'Paris', 33)")

        distinct = execute(db, "SELECT DISTINCT city FROM people ORDER BY city")
        assert distinct.rows == [{"city": b"Osaka"}, {"city": b"Paris"}, {"city": b"Tokyo"}]

        having = execute(
            db,
            "SELECT city, COUNT(*) AS n FROM people GROUP BY city HAVING COUNT(*) > 1 ORDER BY city",
        )
        assert having.rows == [
            {"city": b"Paris", "n": 2},
            {"city": b"Tokyo", "n": 2},
        ]

        like = execute(db, "SELECT name FROM people WHERE name LIKE 'A%' ORDER BY id")
        assert like.rows == [{"name": b"Alice"}, {"name": b"Alicia"}, {"name": b"Aaron"}]
    finally:
        db.close()


def test_sql_begin_commit_and_rollback(tmp_path):
    db = _make_db(tmp_path)
    try:
        begin = execute(db, "BEGIN")
        assert begin.message == "BEGIN"

        execute(db, "INSERT INTO people VALUES (4, 'Dora', 'Lima', 37)")
        inside = execute(db, "SELECT name FROM people WHERE id = 4")
        assert inside.rows == [{"name": b"Dora"}]

        rollback = execute(db, "ROLLBACK")
        assert rollback.message == "ROLLBACK"
        assert execute(db, "SELECT * FROM people WHERE id = 4").rows == []

        execute(db, "BEGIN TRANSACTION")
        execute(db, "INSERT INTO people VALUES (4, 'Dora', 'Lima', 37)")
        commit = execute(db, "COMMIT")
        assert commit.message == "COMMIT"
        assert execute(db, "SELECT name FROM people WHERE id = 4").rows == [{"name": b"Dora"}]
    finally:
        db.close()


def test_sql_between_exists_union_and_multiple_joins(tmp_path):
    db = _make_db(tmp_path)
    try:
        db.create_table(
            "cities",
            [
                {"name": "name", "type": "bytes", "primary": True},
                {"name": "country", "type": "bytes"},
            ],
        )
        db.table("cities").insert_many(
            [
                [b"Tokyo", b"Japan"],
                [b"Paris", b"France"],
                [b"Osaka", b"Japan"],
            ]
        )

        between = execute(db, "SELECT name FROM people WHERE age BETWEEN 22 AND 29 ORDER BY id")
        assert between.rows == [{"name": b"Bob"}, {"name": b"Carol"}]

        exists = execute(
            db,
            "SELECT name FROM people "
            "WHERE EXISTS (SELECT 1 FROM orders WHERE orders.customer_id = people.id AND orders.amount > 100) "
            "ORDER BY id",
        )
        assert exists.rows == [{"name": b"Alice"}]

        union = execute(
            db,
            "SELECT city FROM people WHERE id = 1 UNION SELECT city FROM people WHERE id = 2",
        )
        assert {row["city"] for row in union.rows} == {b"Tokyo", b"Paris"}

        joined = execute(
            db,
            "SELECT people.name, orders.amount, cities.country "
            "FROM people "
            "JOIN orders ON people.id = orders.customer_id "
            "JOIN cities ON people.city = cities.name "
            "WHERE cities.country = 'Japan' "
            "ORDER BY orders.amount DESC",
        )
        assert joined.rows == [
            {"people.name": b"Alice", "orders.amount": 125.0, "cities.country": b"Japan"},
            {"people.name": b"Alice", "orders.amount": 80.0, "cities.country": b"Japan"},
        ]
    finally:
        db.close()


def test_sql_right_full_join_and_index_ddl(tmp_path):
    db = _make_db(tmp_path)
    try:
        db.create_table(
            "lefts",
            [
                {"name": "id", "type": "i64", "primary": True},
                {"name": "label", "type": "bytes"},
            ],
        )
        db.create_table(
            "rights",
            [
                {"name": "id", "type": "i64", "primary": True},
                {"name": "label", "type": "bytes"},
            ],
        )
        db.table("lefts").insert_many([[1, b"L1"], [2, b"L2"]])
        db.table("rights").insert_many([[2, b"R2"], [3, b"R3"]])

        right = execute(
            db,
            "SELECT lefts.id AS left_id, rights.id AS right_id "
            "FROM lefts RIGHT JOIN rights ON lefts.id = rights.id ORDER BY rights.id",
        )
        assert right.rows == [
            {"left_id": 2, "right_id": 2},
            {"left_id": None, "right_id": 3},
        ]

        full = execute(
            db,
            "SELECT lefts.id AS left_id, rights.id AS right_id "
            "FROM lefts FULL OUTER JOIN rights ON lefts.id = rights.id",
        )
        assert full.rows == [
            {"left_id": 1, "right_id": None},
            {"left_id": 2, "right_id": 2},
            {"left_id": None, "right_id": 3},
        ]

        created = execute(db, "CREATE INDEX idx_people_city ON people(city)")
        assert created.message == "Created index idx_people_city on people(city)"
        city_col = next(col for col in db.schema("people") if col["name"] == "city")
        assert city_col["indexed"] is True
        assert city_col["index_name"] == "idx_people_city"

        dropped = execute(db, "DROP INDEX idx_people_city")
        assert dropped.message == "Dropped index idx_people_city"
        city_col = next(col for col in db.schema("people") if col["name"] == "city")
        assert city_col["indexed"] is False
        assert "index_name" not in city_col
    finally:
        db.close()
