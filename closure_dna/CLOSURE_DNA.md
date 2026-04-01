# Closure DNA

Product reference for the Closure DNA module.

Closure DNA is a standalone local database product in the Closure stack, built on the shared geometric Rust core.

The important boundary is:
- shared Rust core: `closure_rs`
- standalone database product: `closure_dna`
- sibling relationship with `closure_sdk` on the same foundation

## Architecture

```text
SQL / CLI / Web
        |
        v
closure_dna Python surface
        |
        v
closure_rs Rust extension
        |
        v
typed columnar geometric engine
```

The Rust core lives in the monorepo `rust/` crate.
That is shared implementation, not product ownership confusion.

`closure_dna` owns:
- the database surface
- SQL execution
- workbench
- demos
- database semantics

`closure_rs` owns:
- embedding
- typed table engine
- persistence primitives
- identity
- resonance
- repairable geometric state

## Storage Model

A Closure DNA database is a directory-backed `.cdb`.

Example:

```text
shop.cdb/
├── people.cdna/
├── orders.cdna/
└── ...
```

Each table is persisted as its own typed storage directory.

The engine is:
- typed
- columnar
- local
- file-backed

## Data Types

Closure DNA currently exposes:
- `i64`
- `f64`
- `bytes`

These are the real engine types.

SQL also maps standard names onto them where appropriate:
- `INTEGER` -> `i64`
- `REAL` -> `f64`
- `TEXT` -> `bytes`
- `BLOB` -> `bytes`

## Python API

Main public objects:
- `Database`
- `Transaction`
- `Table`
- `ResonanceHit`
- `SQLResult`
- `execute`

### Database

Main methods:
- `Database.create(path)`
- `Database.open(path)`
- `db.create_table(name, schema)`
- `db.drop_table(name)`
- `db.tables()`
- `db.schema(name)`
- `db.table(name)`
- `db.select(...)`
- `db.join(...)`
- `db.group_by(...)`
- `db.subquery(...)`
- `db.update_where(...)`
- `db.delete_where(...)`
- `db.add_column(...)`
- `db.compact(...)`
- `db.audit(...)`
- `db.repair(...)`
- `db.info(...)`
- `db.import_table(...)`
- `db.export_table(...)`
- `db.execute(sql)`
- `db.transaction()`
- `db.read_transaction()`

### Table

Main methods:
- `Table.create(path, schema)`
- `Table.open(path)`
- `table.insert(values)`
- `table.insert_many(rows)`
- `table.insert_columns(columns)`
- `table.get_row(row_id)`
- `table.get_f64(row_id, column)`
- `table.get_i64(row_id, column)`
- `table.get_bytes(row_id, column)`
- `table.filter_equals(column, value)`
- `table.filter_cmp(column, op, value)`
- `table.sum(column)`
- `table.avg(column)`
- `table.argsort(column, descending=False)`
- `table.search(values, k=5)`
- `table.identity()`
- `table.check()`
- `table.check_hopf()`
- `table.inspect_row(row_id)`
- `table.count()`
- `table.save()`

## SQL Surface

Standard SQL supported today:

- `CREATE TABLE`
- `DROP TABLE`
- `ALTER TABLE ... ADD COLUMN`
- `SELECT`
- `INSERT`
- `UPDATE`
- `DELETE`
- `JOIN`
- `LEFT JOIN`
- `RIGHT JOIN`
- `FULL OUTER JOIN`
- `GROUP BY`
- `HAVING`
- `DISTINCT`
- `LIKE`
- `BETWEEN`
- `EXISTS`
- `UNION`
- nested subqueries
- multi-statement scripts
- `BEGIN`
- `COMMIT`
- `ROLLBACK`

### DNA-specific SQL

- `SELECT IDENTITY() FROM table`
- `SELECT DRIFT() FROM table`
- `SELECT DECOMPOSE_DRIFT() FROM table`
- `AUDIT table`
- `COMPACT table`
- `INSPECT ROW n FROM table`
- `SELECT * FROM table RESONATE NEAR (...) LIMIT k`

These are product features, not standard SQL.
They intentionally expose the geometric capabilities of the engine.

## Web Workbench

The local web UI supports:
- opening built-in demo databases
- browsing tables
- paging rows
- editing rows
- deleting rows
- creating tables
- adding columns
- running SQL
- viewing schema details
- viewing schema relationships
- audit / repair / compact actions

Run it with:

```bash
closure-dna web
```

## CLI

Main commands include:
- `create-db`
- `create-table`
- `add-column`
- `tables`
- `schema`
- `count`
- `check`
- `audit`
- `repair`
- `info`
- `get`
- `insert`
- `update`
- `delete`
- `update-where`
- `delete-where`
- `group-by`
- `compact`
- `filter`
- `select`
- `join`
- `sum`
- `avg`
- `sort`
- `search`
- `export`
- `import`
- `sql`
- `repl`
- `web`
- `demo-databases`
- `build-demo-db`
- `web-demo`

## Demo Databases

Built demo databases:
- `browser_profile`
- `chat_app`
- `music_streaming`

They live in:

```text
closure_dna/demo/databases/
```

Source datasets live in:

```text
closure_dna/demo_data/
```

## Geometric Capabilities

What makes Closure DNA different from a normal local database:

- table identity in 32 bytes
- drift as an integrity measure
- drift decomposition
- repair from persisted column data
- resonance search over stored rows

This is why the Rust core is geometric instead of B-tree based.

## Packaging Notes

Closure DNA is packaged as its own Python project.

It should be described as:
- standalone product
- shared Rust core
- sibling product to `closure_sdk` on the same foundation

That is the correct professional release story.
