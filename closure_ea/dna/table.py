"""Table — columnar database on S³.

Each field is its own element on S³. Each column stored separately.
Read only the relevant column. No parsing.

    from closure_ea.dna import Table

    schema = [("name", "bytes", False), ("age", "f64", False), ("city", "bytes", True)]
    with Table.create("mydb.cdna", schema) as db:
        db.insert([b"Alice", 30.0, b"Tokyo"])
        db.insert([b"Bob", 25.0, b"Paris"])
        print(db.count())                    # 2
        print(db.check())                    # drift
        print(db.filter_equals("city", b"Tokyo"))  # [0]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import closure_rs
from .result import ResonanceHit


ColumnValue = Union[None, int, float, bytes]
ColumnBatch = Union[list[int], list[float], list[bytes]]


class Table:
    """A Closure DNA table with typed columns.

    Each field is stored in its own column file. The engine keeps the
    geometric core for identity, integrity, genome, and resonance, and
    also exposes typed table operations for full database use. All
    computation in Rust.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: closure_rs.Table) -> None:
        self._inner = inner

    @classmethod
    def create(cls, path: str | Path, schema: list[tuple]) -> "Table":
        """Create a new table with typed columns.
        schema: [(name, type, indexed, not_null, unique), ...]
        type: "f64", "i64", or "bytes"
        indexed: build quaternion sidecar for fast lookup
        not_null: reject empty bytes or NaN
        unique: reject duplicates (checked via geometric drift)
        """
        # Normalize: 3-element tuples get not_null=False, unique=False
        normalized = []
        for entry in schema:
            e = tuple(entry)
            if len(e) == 3:
                e = (e[0], e[1], e[2], False, False)
            normalized.append(e)
        return cls(closure_rs.Table.create(str(path), normalized))

    @classmethod
    def open(cls, path: str | Path) -> "Table":
        """Open an existing table."""
        return cls(closure_rs.Table.open(str(path)))

    # ── Write ────────────────────────────────────────────────────

    def insert(self, values: list[ColumnValue]) -> int:
        """Insert one row. Values must match schema order."""
        return self._inner.insert(values)

    def insert_many(self, rows: list[list[ColumnValue]]) -> int:
        """Insert many rows.

        Convenience path: rows are columnized and routed to the native
        typed column ingest path under the hood.
        """
        return self._inner.insert_many(rows)

    def insert_columns(self, columns: list[ColumnBatch]) -> int:
        """Insert many rows as typed columns. Native columnar ingest path."""
        return self._inner.insert_columns(columns)

    def search(self, values: list[ColumnValue], k: int = 1) -> list[ResonanceHit]:
        """Resonance search for a typed row query."""
        raw = self._inner.search(values, k)
        hits: list[ResonanceHit] = []
        for row in raw:
            hits.append(
                ResonanceHit(
                    position=int(row[0]),
                    drift=float(row[1]),
                    base=(float(row[2]), float(row[3]), float(row[4])),
                    phase=float(row[5]),
                )
            )
        return hits

    def build_genome(self) -> None:
        """Build the hierarchical genome from current row geometry."""
        self._inner.build_genome()

    def genome_depth(self) -> int:
        """How many hierarchy levels the genome has."""
        return self._inner.genome_depth()

    def genome_codons(self) -> int:
        """How many level-0 blocks the genome discovered."""
        return self._inner.genome_codons()

    def schema(self) -> list[tuple[str, str, bool]]:
        """Typed schema as (name, type, indexed)."""
        return list(self._inner.schema())

    # ── Read ─────────────────────────────────────────────────────

    def get_f64(self, row: int, col: int) -> float:
        """Get a numeric field value."""
        return self._inner.get_f64(row, col)

    def get_i64(self, row: int, col: int) -> int:
        """Get an exact 64-bit integer field value."""
        return int(self._inner.get_i64(row, col))

    def get_bytes(self, row: int, col: int) -> bytes:
        """Get a bytes/string field value."""
        return bytes(self._inner.get_bytes(row, col))

    def get_row(self, row: int) -> list[ColumnValue]:
        """Get a full typed row."""
        raw = self._inner.get_row(row)
        values: list[ColumnValue] = []
        for value in raw:
            if isinstance(value, (bytes, bytearray, memoryview)):
                values.append(bytes(value))
            elif value is None:
                values.append(None)
            elif isinstance(value, int):
                values.append(int(value))
            else:
                values.append(float(value))
        return values

    def column_index(self, name: str) -> int:
        """Get column index by name."""
        return self._inner.column_index(name)

    # ── Filter ───────────────────────────────────────────────────

    def filter_equals(self, col_name: str, value: bytes) -> list[int]:
        """Return row indices where a bytes column equals value. Rust-side."""
        return self._inner.filter_equals(col_name, value)

    def filter_cmp(self, col_name: str, op: str, value: float) -> list[int]:
        """Return row indices where a numeric column passes comparison. Rust-side."""
        return self._inner.filter_cmp(col_name, op, value)

    # ── Aggregate ────────────────────────────────────────────────

    def sum(self, col_name: str) -> float:
        """Sum a numeric column. Rust-side."""
        return self._inner.sum(col_name)

    def avg(self, col_name: str) -> float:
        """Average a numeric column. Rust-side."""
        return self._inner.avg(col_name)

    # ── Sort ─────────────────────────────────────────────────────

    def argsort(self, col_name: str, descending: bool = False) -> list[int]:
        """Return row indices sorted by a numeric column. Rust-side."""
        return self._inner.argsort(col_name, descending)

    # ── Integrity ────────────────────────────────────────────────

    def check(self) -> float:
        """Check table integrity. Returns drift (0 = clean)."""
        return self._inner.check()

    def check_hopf(self) -> tuple[float, tuple[float, float, float], float]:
        """Hopf view of the current table identity: drift, base, phase."""
        raw = self._inner.check_hopf()
        return float(raw[0]), (float(raw[1]), float(raw[2]), float(raw[3])), float(raw[4])

    def inspect_row(self, row: int) -> tuple[float, tuple[float, float, float], float]:
        """Hopf view of one row quaternion: drift, base, phase."""
        raw = self._inner.inspect_row(row)
        return float(raw[0]), (float(raw[1]), float(raw[2]), float(raw[3])), float(raw[4])

    def audit(self) -> dict:
        """Full table audit from disk state. No writes."""
        raw = self._inner.audit()
        bad_row = int(raw[2]) if raw[2] >= 0 else None
        return {
            "ok": bool(raw[0]),
            "drift": float(raw[1]),
            "bad_row": bad_row,
            "hopf": {
                "base": (float(raw[3]), float(raw[4]), float(raw[5])),
                "phase": float(raw[6]),
            },
        }

    def repair(self) -> None:
        """Explicitly rebuild derived geometric state from columns."""
        self._inner.repair()

    def update(self, row: int, values: list[ColumnValue]) -> None:
        """Replace one full typed row."""
        self._inner.update(values, row)

    def delete(self, row: int) -> None:
        """Delete one row. Sets its quaternion to identity (algebraic neutral).
        The row remains in the table as a tombstone — count stays the same,
        but the row contributes nothing to the composition. O(log n)."""
        self._inner.delete(row)

    def identity(self) -> np.ndarray:
        """Table identity — 32 bytes, one quaternion on S³."""
        return self._inner.identity()

    # ── Metadata ─────────────────────────────────────────────────

    def count(self) -> int:
        """How many rows."""
        return self._inner.count()

    def live_row_count(self) -> int:
        """How many rows are not tombstoned."""
        return int(self._inner.live_row_count())

    def is_deleted(self, row: int) -> bool:
        """Whether this row is a tombstone."""
        return bool(self._inner.is_deleted(row))

    # ── Persistence ──────────────────────────────────────────────

    def save(self) -> None:
        """Save to disk."""
        self._inner.save()

    def snapshot(self, name: str | None = None) -> str:
        """Create a named snapshot of the current table state."""
        return str(self._inner.snapshot(name))

    def history(self, limit: int | None = None) -> list[dict]:
        """Read append-only history entries for this table."""
        return [json.loads(line) for line in self._inner.history_json(limit)]

    def snapshots(self) -> list[dict]:
        """List available snapshots with metadata."""
        return [json.loads(line) for line in self._inner.snapshots_json()]

    def restore_snapshot(self, name: str) -> None:
        """Restore the table to a named snapshot."""
        self._inner.restore_snapshot(name)

    # ── Context manager ──────────────────────────────────────────

    def __enter__(self) -> "Table":
        return self

    def __exit__(self, *exc) -> None:
        self.save()

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return repr(self._inner)

    # ── Import / Export ──────────────────────────────────────────

    @staticmethod
    def import_csv(path: str | Path, csv_text: str, header: bool = True) -> "Table":
        """Create a table from CSV text. Infers schema from header + first row.
        Numbers become f64, everything else becomes bytes.
        """
        import csv, io
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        if not rows:
            raise ValueError("Empty CSV")

        if header:
            col_names = [c.strip() for c in rows[0]]
            rows = rows[1:]
        else:
            col_names = [f"col_{i}" for i in range(len(rows[0]))]

        if not rows:
            raise ValueError("No data rows")

        # Infer types from first row
        schema = []
        for i, name in enumerate(col_names):
            val = rows[0][i].strip() if i < len(rows[0]) else ""
            try:
                float(val)
                schema.append((name, "f64", False, False, False))
            except ValueError:
                schema.append((name, "bytes", False, False, False))

        # Build columns
        columns: list[list] = [[] for _ in col_names]
        for row in rows:
            if not any(c.strip() for c in row):
                continue
            for i in range(len(col_names)):
                val = row[i].strip() if i < len(row) else ""
                if schema[i][1] == "f64":
                    try:
                        columns[i].append(float(val))
                    except ValueError:
                        columns[i].append(0.0)
                else:
                    columns[i].append(val.encode("utf-8"))

        t = Table.create(path, schema)
        t.insert_columns(columns)
        return t

    @staticmethod
    def import_json(path: str | Path, json_text: str) -> "Table":
        """Create a table from a JSON array of objects. Infers schema from first object."""
        import json as _json
        data = _json.loads(json_text)
        if not isinstance(data, list) or not data:
            raise ValueError("Expected non-empty JSON array")

        first = data[0]
        schema = []
        col_names = list(first.keys())
        for key in col_names:
            val = first[key]
            if isinstance(val, (int, float)):
                schema.append((key, "f64", False, False, False))
            else:
                schema.append((key, "bytes", False, False, False))

        columns: list[list] = [[] for _ in col_names]
        for obj in data:
            for i, key in enumerate(col_names):
                val = obj.get(key, "")
                if schema[i][1] == "f64":
                    columns[i].append(float(val) if val != "" else 0.0)
                else:
                    columns[i].append(str(val).encode("utf-8"))

        t = Table.create(path, schema)
        t.insert_columns(columns)
        return t

    def export_csv(self) -> str:
        """Export table as CSV text."""
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        schema = self.schema()
        writer.writerow([name for name, _, _ in schema])
        for row in range(self.count()):
            record = self.get_row(row)
            writer.writerow(
                [
                    value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
                    for value in record
                ]
            )
        return buf.getvalue()

    def export_json(self) -> str:
        """Export table as a JSON array of objects."""
        import json as _json

        schema = self.schema()
        rows = []
        for row_idx in range(self.count()):
            record = {}
            for (name, _, _), value in zip(schema, self.get_row(row_idx), strict=True):
                record[name] = value.decode("utf-8") if isinstance(value, bytes) else value
            rows.append(record)
        return _json.dumps(rows, indent=2)
