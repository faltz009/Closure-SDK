# Closure DNA

Closure DNA is a standalone local database with a geometric Rust core.

It gives you:
- typed tables
- transactions
- SQL execution
- a local web workbench
- built-in integrity identity
- resonance search over stored rows

It is part of the Closure stack.
It shares the same geometric foundation as `closure_sdk`, while standing alone as its own database product.

Its low-level engine lives in the shared monorepo Rust core in `rust/`, exposed to Python as `closure_rs`, and Closure DNA builds its database surface on top of that shared core.

## What It Is

Closure DNA is a local embedded database for structured data.

The storage engine is:
- columnar
- typed
- persisted on disk as a `.cdb` directory

The geometric layer gives each row and table a compositional identity on `S^3`.
That identity powers:
- fast integrity checks
- drift decomposition
- repair
- resonance search

## What It Supports

### Types
- `i64`
- `f64`
- `bytes`

### Core database operations
- create/open/drop database tables
- insert/update/delete
- add column with default backfill
- transactions
- read snapshots
- compact / audit / repair
- import / export
- joins
- group by + aggregates
- null support

### SQL
Closure DNA now speaks real SQL through the parser layer.

Supported SQL includes:
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
- multi-statement SQL scripts
- `BEGIN / COMMIT / ROLLBACK`

### DNA-specific SQL
- `SELECT IDENTITY() FROM table`
- `SELECT DRIFT() FROM table`
- `SELECT DECOMPOSE_DRIFT() FROM table`
- `AUDIT table`
- `COMPACT table`
- `INSPECT ROW n FROM table`
- `SELECT * FROM table RESONATE NEAR (...) LIMIT k`

## Quick Start

### Python API

```python
from closure_dna import Database

db = Database.create("shop.cdb")

db.create_table(
    "people",
    [
        {"name": "id", "type": "i64", "primary": True},
        {"name": "name", "type": "bytes"},
        {"name": "city", "type": "bytes", "indexed": True},
        {"name": "age", "type": "f64"},
    ],
)

db.table("people").insert([1, b"Alice", b"Tokyo", 31.0])
db.table("people").insert([2, b"Bob", b"Paris", 22.0])

rows = db.execute("SELECT name FROM people WHERE city = 'Tokyo'").rows
```

### CLI

```bash
closure-dna create-db shop.cdb
closure-dna create-table shop.cdb people '[{"name":"id","type":"i64","primary":true},{"name":"name","type":"bytes"}]'
closure-dna insert shop.cdb people '[1,"Alice"]'
closure-dna sql shop.cdb "SELECT * FROM people"
closure-dna web
```

### Web workbench

Run:

```bash
closure-dna web
```

The web UI includes:
- built-in demo database browser
- table browser
- row editing
- row deletion
- SQL workbench
- schema view
- schema relationship view

## Built-in Demo Databases

Built demo databases live under:

```text
closure_dna/demo/databases/
```

Current demos:
- `browser_profile.cdb`
- `chat_app.cdb`
- `music_streaming.cdb`

The web UI can open these directly from the left-hand database list.

## Package Layout

```text
closure_dna/
├── table.py        # low-level typed table wrapper over closure_rs.Table
├── database.py     # multi-table database surface
├── sql.py          # SQL parser/executor layer
├── web.py          # local web workbench
├── cli.py          # command line entrypoint
├── repl.py         # interactive REPL
├── demo/           # built demo databases + registry
├── demo_data/      # source datasets for demos
└── tests/          # module test suite
```

Shared Rust core:

```text
rust/
└── src/
    ├── table.rs
    ├── embed.rs
    ├── pyo3_bindings.rs
    └── ...
```

## Relationship To The Rust Core

Closure DNA is standalone as a product.

It sits in the same family as `closure_sdk` and shares the same low-level Rust foundation, but its public surface, database semantics, SQL layer, demos, and workbench all belong to `closure_dna`.

So the correct architecture is:
- `closure_dna` is its own package and product surface
- `closure_dna` shares the common Closure Rust core
- `closure_sdk` and `closure_dna` are sibling products on that shared foundation

## What The Geometry Adds

Traditional embedded databases give you storage and query execution.

Closure DNA also gives you:
- table identity in 32 bytes
- integrity drift checks
- drift decomposition
- repair from persisted columns
- resonance search over stored rows

Those are not wrappers around another database.
They come from the engine itself.

## Current Scope

Closure DNA is:
- a local database
- SQL-capable
- parser-backed
- geometric at the engine level

Closure DNA is not trying to be:
- a client/server database
- a distributed database
- a drop-in clone of SQLite internals

It is a different database architecture with a familiar SQL surface.

## Documentation

For the fuller product reference, see:
- [CLOSURE_DNA.md](/home/faltz009/closure-verification/CLOSURE_DNA.md)
