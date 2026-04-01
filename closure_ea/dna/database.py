"""Database container for multiple typed Closure DNA tables.

This is a thin filesystem-level manager:

- one database = one directory
- one table = one `.cdna` subdirectory
- each table keeps its own typed geometric engine
"""

from __future__ import annotations

import ast
import csv
import fcntl
import gc
import io
import json
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .table import Table

_TXN_JOURNAL_SUFFIX = ".txn.json"
_BACKUP_SUFFIX = ".bak"


class Database:
    """A collection of named Closure DNA tables."""

    __slots__ = ("_path", "_open_tables", "_lock_handle", "_lock_depth", "_lock_mode", "_sql_txn")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._open_tables: dict[str, Table] = {}
        self._lock_handle = None
        self._lock_depth = 0
        self._lock_mode = None
        self._sql_txn = None

    @classmethod
    def create(cls, path: str | Path) -> "Database":
        root = Path(path)
        _recover_database(root)
        root.mkdir(parents=True, exist_ok=True)
        return cls(root)

    @classmethod
    def open(cls, path: str | Path) -> "Database":
        root = Path(path)
        _recover_database(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Database not found: {root}")
        return cls(root)

    @property
    def path(self) -> Path:
        return self._path

    def create_table(self, name: str, schema: list[tuple[str, str, bool]] | list[dict]) -> Table:
        table_path = self._table_path(name)
        if table_path.exists():
            raise FileExistsError(f"Table already exists: {name}")
        meta = _normalize_schema(schema)
        _validate_schema(meta, self)
        base_schema = [
            (col["name"], col["type"], col["indexed"], col["not_null"], col["unique"])
            for col in meta
        ]
        with self._write_lock():
            table = Table.create(table_path, base_schema)
            self._write_table_meta(name, meta)
            self._open_tables[name] = table
            return ManagedTable(self, name, table)

    def table(self, name: str) -> Table:
        return ManagedTable(self, name, self._raw_table(name))

    def _raw_table(self, name: str) -> Table:
        if name not in self._open_tables:
            self._open_tables[name] = Table.open(self._table_path(name))
        return self._open_tables[name]

    def _refresh_raw_table(self, name: str) -> Table:
        current = self._open_tables.get(name)
        if current is not None:
            current.save()
        self._open_tables[name] = Table.open(self._table_path(name))
        return self._open_tables[name]

    def tables(self) -> list[str]:
        with self._read_lock():
            return sorted(
                p.stem
                for p in self._path.iterdir()
                if p.is_dir() and p.suffix == ".cdna" and not p.name.startswith(".")
            )

    def has_table(self, name: str) -> bool:
        with self._read_lock():
            return self._table_path(name).is_dir()

    def schema(self, name: str) -> list[dict]:
        with self._read_lock():
            meta_path = self._table_meta_path(name)
            if meta_path.exists():
                return json.loads(meta_path.read_text(encoding="utf-8"))
            table = self.table(name)
            return _normalize_schema(table.schema())

    def primary_key(self, name: str) -> str | None:
        for col in self.schema(name):
            if col["primary"]:
                return col["name"]
        return None

    def foreign_keys(self, name: str) -> list[dict]:
        return [col for col in self.schema(name) if col["references"] is not None]

    def select(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where: list[tuple[str, str, object]] | None = None,
        order_by: str | None = None,
        descending: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict]:
        with self._read_lock():
            table = self.table(table_name)
            schema = self.schema(table_name)
            row_indices = self._live_row_indices(table_name)

            if where and isinstance(where, tuple):
                col, op, value = where
                if _schema_has_column(schema, col):
                    col_type = _column_type(schema, col)
                    if col_type == "bytes" and op == "=" and value is not None:
                        row_indices = [
                            idx
                            for idx in table.filter_equals(col, _coerce_value("bytes", value))
                            if self._is_live_row(table_name, idx)
                        ]
                    elif col_type in {"f64", "i64"} and value is not None and op in {"=", "!=", ">", "<", ">=", "<="}:
                        row_indices = [
                            idx
                            for idx in table.filter_cmp(col, op, _coerce_value(col_type, value))
                            if self._is_live_row(table_name, idx)
                        ]
            elif where and isinstance(where, list) and len(where) == 1:
                col, op, value = where[0]
                if _schema_has_column(schema, col):
                    col_type = _column_type(schema, col)
                    if col_type == "bytes" and op == "=" and value is not None:
                        row_indices = [
                            idx
                            for idx in table.filter_equals(col, _coerce_value("bytes", value))
                            if self._is_live_row(table_name, idx)
                        ]
                    elif col_type in {"f64", "i64"} and value is not None and op in {"=", "!=", ">", "<", ">=", "<="}:
                        row_indices = [
                            idx
                            for idx in table.filter_cmp(col, op, _coerce_value(col_type, value))
                            if self._is_live_row(table_name, idx)
                        ]

            records = []
            for idx in row_indices:
                record = _record_from_row(schema, table.get_row(idx))
                if where and not _match_where(record, schema, where):
                    continue
                records.append(record)

            if order_by is not None:
                records = _sort_records(records, order_by, descending)

            if offset:
                records = records[offset:]
            if limit is not None:
                records = records[:limit]

            if columns is not None:
                records = [_project_record(record, schema, columns) for record in records]

            return records

    def subquery(
        self,
        table_name: str,
        column: str,
        *,
        where: list[tuple[str, str, object]] | None = None,
        order_by: str | None = None,
        descending: bool = False,
        limit: int | None = None,
    ) -> list[object]:
        rows = self.select(
            table_name,
            columns=[column],
            where=where,
            order_by=order_by,
            descending=descending,
            limit=limit,
        )
        return [row[column] for row in rows]

    def join(
        self,
        left_table: str,
        right_table: str,
        left_on: str,
        right_on: str | None = None,
        outer: str | None = None,
        where: list[tuple[str, str, object]] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        if right_on is None:
            right_on = left_on

        with self._read_lock():
            left_schema = self.schema(left_table)
            right_schema = self.schema(right_table)
            left_rows = self.select(left_table)
            right_rows = self.select(right_table)

            index: dict[object, list[dict]] = {}
            matched_right_ids: set[int] = set()
            for idx, row in enumerate(right_rows):
                key = row[right_on]
                if key is None:
                    continue
                index.setdefault(key, []).append((idx, row))

            joined = []
            for left in left_rows:
                left_key = left[left_on]
                matches = [] if left_key is None else index.get(left_key, [])
                if not matches and outer in {"left", "full"}:
                    matches = [(None, None)]
                for match_idx, right in matches:
                    record = {}
                    for col in left_schema:
                        record[f"{left_table}.{col['name']}"] = left[col["name"]]
                    for col in right_schema:
                        record[f"{right_table}.{col['name']}"] = None if right is None else right[col["name"]]
                    if match_idx is not None:
                        matched_right_ids.add(match_idx)
                    if where and not _match_join_where(record, where):
                        continue
                    joined.append(record)
                    if limit is not None and len(joined) >= limit:
                        return joined
            if outer in {"right", "full"}:
                for idx, right in enumerate(right_rows):
                    if idx in matched_right_ids:
                        continue
                    record = {}
                    for col in left_schema:
                        record[f"{left_table}.{col['name']}"] = None
                    for col in right_schema:
                        record[f"{right_table}.{col['name']}"] = right[col["name"]]
                    if where and not _match_join_where(record, where):
                        continue
                    joined.append(record)
                    if limit is not None and len(joined) >= limit:
                        return joined
            return joined

    def save(self) -> None:
        with self._write_lock():
            for table in self._open_tables.values():
                table.save()

    def close(self) -> None:
        if self._sql_txn is not None:
            self.rollback_sql_transaction()
        self.save()
        self._open_tables.clear()

    def transaction(self) -> "Transaction":
        return Transaction(self)

    def read_transaction(self) -> "ReadTransaction":
        return ReadTransaction(self)

    def execute(self, sql: str):
        from .sql import execute

        return execute(self, sql)

    def begin_sql_transaction(self) -> None:
        if self._sql_txn is not None:
            raise ValueError("transaction already active")
        self._sql_txn = Transaction(self)

    def commit_sql_transaction(self) -> None:
        if self._sql_txn is None:
            raise ValueError("no active transaction")
        txn = self._sql_txn
        self._sql_txn = None
        txn.commit()
        self._open_tables.clear()

    def rollback_sql_transaction(self) -> None:
        if self._sql_txn is None:
            raise ValueError("no active transaction")
        txn = self._sql_txn
        self._sql_txn = None
        txn.rollback()
        self._open_tables.clear()

    def compact(self, table_name: str | None = None) -> dict[str, int]:
        names = [table_name] if table_name is not None else self.tables()
        with self._write_lock():
            return {name: self._compact_table(name) for name in names}

    def add_column(
        self,
        table_name: str,
        column: tuple[str, str, bool] | dict,
        *,
        default=None,
    ) -> Table:
        with self._write_lock():
            return self._add_column(table_name, column, default=default)

    def create_index(self, table_name: str, column_name: str, index_name: str, *, if_not_exists: bool = False) -> Table:
        with self._write_lock():
            meta = self.schema(table_name)
            target = next((col for col in meta if col["name"] == column_name), None)
            if target is None:
                raise KeyError(f"unknown column: {column_name}")
            if target.get("indexed"):
                if if_not_exists:
                    return self.table(table_name)
                raise FileExistsError(f"Index already exists on {table_name}.{column_name}")
            target["indexed"] = True
            target["index_name"] = index_name
            return self._rebuild_table_with_meta(table_name, meta)

    def drop_index(self, index_name: str, *, if_exists: bool = False) -> None:
        with self._write_lock():
            for table_name in self.tables():
                meta = self.schema(table_name)
                target = next((col for col in meta if col.get("index_name") == index_name), None)
                if target is None:
                    continue
                target["indexed"] = False
                target.pop("index_name", None)
                self._rebuild_table_with_meta(table_name, meta)
                return
            if not if_exists:
                raise FileNotFoundError(f"Index not found: {index_name}")

    def drop_table(self, name: str) -> None:
        with self._write_lock():
            if name in self._open_tables:
                del self._open_tables[name]
            shutil.rmtree(self._table_path(name))

    def audit(self, table_name: str | None = None) -> dict:
        names = [table_name] if table_name is not None else self.tables()
        results = {}
        for name in names:
            table = self.table(name)
            audit = table.audit()
            total_rows = table.count()
            live_rows = self._safe_live_row_count(name)
            tree_path = self._table_path(name) / "tree.q"
            tree_ok = tree_path.exists() and tree_path.stat().st_size > 0
            results[name] = {
                **audit,
                "ok": bool(audit["ok"] and tree_ok),
                "tree_ok": tree_ok,
                "rows": total_rows,
                "live_rows": live_rows,
                "tombstones": None if live_rows is None else total_rows - live_rows,
            }
        return results[table_name] if table_name is not None else results

    def update_where(
        self,
        table_name: str,
        set_values: dict[str, object],
        where: list[tuple[str, str, object]],
    ) -> int:
        """Recompose all live rows matching WHERE with patched values.

        set_values maps column names to new values — only listed columns
        change; the rest of the row is preserved.  Returns the number of
        rows recomposed.

        Because each row's geometric contribution is derived from its
        content, the table identity shifts automatically — no separate
        index rebuild required.
        """
        schema = self.schema(table_name)
        col_types = {col["name"]: col["type"] for col in schema}
        with self._write_lock():
            table = self._raw_table(table_name)
            matching = []
            for idx in self._live_row_indices(table_name):
                record = _record_from_row(schema, table.get_row(idx))
                if _match_where(record, schema, where):
                    matching.append((idx, record))
            for idx, record in matching:
                updated = dict(record)
                for col_name, new_val in set_values.items():
                    updated[col_name] = _coerce_value(col_types[col_name], new_val)
                values = [updated[col["name"]] for col in schema]
                self._validate_row_values(table_name, values, updating_row=idx)
                table.update(idx, values)
            if matching:
                table.save()
            return len(matching)

    def delete_where(
        self,
        table_name: str,
        where: list[tuple[str, str, object]],
    ) -> int:
        """Nullify all live rows matching WHERE (tombstone them).

        Each deleted row's quaternion is set to the identity element —
        the algebraic zero that contributes nothing to the composition.
        Returns the number of rows nullified.
        """
        schema = self.schema(table_name)
        with self._write_lock():
            table = self._raw_table(table_name)
            matching = [
                idx
                for idx in self._live_row_indices(table_name)
                if _match_where(_record_from_row(schema, table.get_row(idx)), schema, where)
            ]
            for idx in matching:
                self._validate_delete(table_name, idx)
                table.delete(idx)
            if matching:
                table.save()
            return len(matching)

    def group_by(
        self,
        table_name: str,
        column: str,
        aggregates: list[tuple[str, str, str]],
        where: list[tuple[str, str, object]] | None = None,
        order_by: str | None = None,
        descending: bool = False,
    ) -> list[dict]:
        """Partition live rows by column value and compute aggregates.

        aggregates: [(result_name, func, col), ...]
        func: "count" | "sum" | "avg" | "min" | "max"
        col: column name, or "*" for count

        Returns one dict per distinct group value, with the group key
        plus each requested aggregate.  Rows that do not satisfy WHERE
        (if given) are excluded before grouping.
        """
        schema = self.schema(table_name)
        with self._read_lock():
            table = self._raw_table(table_name)
            row_indices = self._live_row_indices(table_name)
            if where:
                row_indices = [
                    idx
                    for idx in row_indices
                    if _match_where(_record_from_row(schema, table.get_row(idx)), schema, where)
                ]
            groups: dict[object, list[dict]] = {}
            for idx in row_indices:
                record = _record_from_row(schema, table.get_row(idx))
                key = record[column]
                groups.setdefault(key, []).append(record)

            result = []
            for key, rows in groups.items():
                entry: dict[str, object] = {column: key}
                for result_name, func, agg_col in aggregates:
                    values = [r[agg_col] for r in rows if agg_col != "*" and r[agg_col] is not None]
                    if func == "count":
                        entry[result_name] = len(rows)
                    elif func == "sum":
                        entry[result_name] = sum(float(v) for v in values)
                    elif func == "avg":
                        vals = [float(v) for v in values]
                        entry[result_name] = sum(vals) / len(vals) if vals else 0.0
                    elif func == "min":
                        entry[result_name] = min(values) if values else None
                    elif func == "max":
                        entry[result_name] = max(values) if values else None
                    else:
                        raise ValueError(f"unknown aggregate function: {func!r}")
                result.append(entry)

            if order_by is not None:
                result = _sort_records(result, order_by, descending)
            return result

    def repair(self, table_name: str | None = None) -> dict:
        names = [table_name] if table_name is not None else self.tables()
        repaired = {}
        with self._write_lock():
            for name in names:
                table = self.table(name)
                table.repair()
                table.save()
                repaired[name] = {
                    "repaired": True,
                    "audit": self.audit(name),
                }
        return repaired[table_name] if table_name is not None else repaired

    def info(self, table_name: str | None = None) -> dict:
        names = [table_name] if table_name is not None else self.tables()
        tables = {}
        for name in names:
            table = self.table(name)
            total_rows = table.count()
            live_rows = self._safe_live_row_count(name)
            tables[name] = {
                "path": str(self._table_path(name)),
                "schema": self.schema(name),
                "rows": total_rows,
                "live_rows": live_rows,
                "tombstones": None if live_rows is None else total_rows - live_rows,
                "drift": table.check(),
                "identity": table.identity().tolist(),
            }
        if table_name is not None:
            return tables[table_name]
        return {"path": str(self._path), "tables": tables}

    def export_table(self, table_name: str, fmt: str, output_path: str | Path | None = None) -> str:
        rows = self.select(table_name)
        if fmt == "json":
            payload = json.dumps(_jsonable_rows(rows), indent=2)
        elif fmt == "csv":
            payload = _rows_to_csv(self.schema(table_name), rows)
        else:
            raise ValueError(f"unsupported export format: {fmt}")
        if output_path is not None:
            Path(output_path).write_text(payload, encoding="utf-8")
        return payload

    def import_table(
        self,
        name: str,
        fmt: str,
        input_path: str | Path,
        *,
        replace: bool = False,
        header: bool = True,
    ) -> Table:
        text = Path(input_path).read_text(encoding="utf-8")
        if fmt == "json":
            schema, columns = _infer_json_table(text)
        elif fmt == "csv":
            schema, columns = _infer_csv_table(text, header=header)
        else:
            raise ValueError(f"unsupported import format: {fmt}")
        if self.has_table(name):
            if not replace:
                raise FileExistsError(f"Table already exists: {name}")
            self.drop_table(name)
        table = self.create_table(name, schema)
        if columns:
            if any(any(value is None for value in column) for column in columns):
                table.insert_many([list(row) for row in zip(*columns, strict=True)])
            else:
                table.insert_columns(columns)
        table.save()
        return table

    def _table_path(self, name: str) -> Path:
        return self._path / f"{name}.cdna"

    def _table_meta_path(self, name: str) -> Path:
        return self._table_path(name) / "table_meta.json"

    def _lock_path(self) -> Path:
        return self._path.with_name(f".{self._path.name}.lock")

    @contextmanager
    def _read_lock(self):
        with self._lock("read"):
            yield

    @contextmanager
    def _write_lock(self):
        with self._lock("write"):
            yield

    @contextmanager
    def _lock(self, mode: str):
        lock_path = self._lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self._lock_depth > 0:
            if self._lock_mode == "write":
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            if self._lock_mode == mode:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            raise RuntimeError("cannot upgrade read lock to write lock within the same Database context")

        self._lock_handle = lock_path.open("a+b")
        self._lock_mode = mode
        self._lock_depth = 1
        try:
            fcntl.flock(
                self._lock_handle.fileno(),
                fcntl.LOCK_EX if mode == "write" else fcntl.LOCK_SH,
            )
            yield
        finally:
            self._lock_depth = 0
            self._lock_mode = None
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def _write_table_meta(self, name: str, schema: list[dict]) -> None:
        self._table_meta_path(name).write_text(json.dumps(schema, indent=2), encoding="utf-8")

    def _write_table_meta_at(self, table_path: Path, schema: list[dict]) -> None:
        (table_path / "table_meta.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    def _live_row_indices(self, name: str) -> list[int]:
        table = self._raw_table(name)
        return [idx for idx in range(table.count()) if self._is_live_row(name, idx)]

    def _safe_live_row_count(self, name: str) -> int | None:
        try:
            return self._raw_table(name).live_row_count()
        except Exception:
            return None

    def _is_live_row(self, name: str, row: int) -> bool:
        table = self._raw_table(name)
        return not table.is_deleted(row)

    def _find_live_matches(self, table_name: str, column: str, value) -> list[int]:
        if value is None:
            return []
        table = self._raw_table(table_name)
        column_type = _column_type(self.schema(table_name), column)
        if column_type == "bytes":
            matches = table.filter_equals(column, _coerce_value("bytes", value))
        else:
            matches = table.filter_cmp(column, "=", _coerce_value(column_type, value))
        return [idx for idx in matches if self._is_live_row(table_name, idx)]

    def _validate_row_values(
        self,
        table_name: str,
        values: list,
        *,
        updating_row: int | None = None,
    ) -> None:
        schema = self.schema(table_name)
        if len(values) != len(schema):
            raise ValueError(f"expected {len(schema)} values, got {len(values)}")

        pk = self.primary_key(table_name)
        old_row = self._raw_table(table_name).get_row(updating_row) if updating_row is not None else None

        for idx, col in enumerate(schema):
            ref = col["references"]
            if ref is None:
                continue
            if values[idx] is None:
                continue
            target_table, target_col = ref.split(".", 1)
            matches = self._find_live_matches(target_table, target_col, values[idx])
            if not matches:
                raise ValueError(
                    f"foreign key failed: {table_name}.{col['name']} -> {target_table}.{target_col} "
                    f"({values[idx]!r} not found)"
                )

        if pk is not None and updating_row is not None:
            pk_index = next(i for i, col in enumerate(schema) if col["name"] == pk)
            old_pk = old_row[pk_index]
            new_pk = values[pk_index]
            if new_pk != old_pk:
                for other_name in self.tables():
                    for col in self.foreign_keys(other_name):
                        if col["references"] == f"{table_name}.{pk}":
                            matches = self._find_live_matches(other_name, col["name"], old_pk)
                            if matches:
                                raise ValueError(
                                    f"cannot change primary key {table_name}.{pk}; "
                                    f"{other_name}.{col['name']} still references {old_pk!r}"
                                )

    def _validate_delete(self, table_name: str, row: int) -> None:
        schema = self.schema(table_name)
        pk = self.primary_key(table_name)
        if pk is None:
            return
        if not self._is_live_row(table_name, row):
            return
        record = _record_from_row(schema, self._raw_table(table_name).get_row(row))
        pk_value = record[pk]
        for other_name in self.tables():
            for col in self.foreign_keys(other_name):
                if col["references"] == f"{table_name}.{pk}":
                    matches = self._find_live_matches(other_name, col["name"], pk_value)
                    if matches:
                        raise ValueError(
                            f"cannot delete {table_name}.{pk}={pk_value!r}; "
                            f"{other_name}.{col['name']} still references it"
                        )

    def _compact_table(self, name: str) -> int:
        raw = self._raw_table(name)
        meta = self.schema(name)
        live_rows = []
        removed = 0
        for idx in range(raw.count()):
            if self._is_live_row(name, idx):
                live_rows.append(raw.get_row(idx))
            else:
                removed += 1

        if removed == 0:
            return 0

        temp_path = self._path / f".{name}.compact.cdna"
        backup_path = self._table_path(name).with_name(f"{name}.cdna.compact.bak")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        if backup_path.exists():
            shutil.rmtree(backup_path)

        rebuilt = Table.create(
            temp_path,
            [(col["name"], col["type"], col["indexed"], col["not_null"], col["unique"]) for col in meta],
        )
        if live_rows:
            rebuilt.insert_many(live_rows)
        rebuilt.save()
        self._write_table_meta_at(temp_path, meta)

        old_path = self._table_path(name)
        if name in self._open_tables:
            del self._open_tables[name]
        del raw
        gc.collect()

        old_path.rename(backup_path)
        temp_path.rename(old_path)
        shutil.rmtree(backup_path)
        return removed

    def _add_column(self, table_name: str, column: tuple[str, str, bool] | dict, *, default=None) -> Table:
        meta = self.schema(table_name)
        new_col = _normalize_schema([column])[0]
        if any(col["name"] == new_col["name"] for col in meta):
            raise ValueError(f"duplicate column name: {new_col['name']}")
        if new_col["not_null"] and default is None:
            raise ValueError(f"column {new_col['name']} is NOT NULL but no default was provided")

        extended_meta = [*meta, new_col]
        _validate_schema(extended_meta, self)

        default_value = _coerce_value(new_col["type"], default)
        if new_col["references"] is not None and default_value is not None:
            target_table, target_col = new_col["references"].split(".", 1)
            if not self._find_live_matches(target_table, target_col, default_value):
                raise ValueError(
                    f"foreign key failed: {table_name}.{new_col['name']} -> {target_table}.{target_col} "
                    f"({default_value!r} not found)"
                )

        raw = self._raw_table(table_name)
        total_rows = raw.count()
        live_flags = [self._is_live_row(table_name, idx) for idx in range(total_rows)]
        rows = [raw.get_row(idx) + [default_value] for idx in range(total_rows)]

        temp_path = self._path / f".{table_name}.alter.cdna"
        backup_path = self._table_path(table_name).with_name(f"{table_name}.cdna.alter.bak")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        if backup_path.exists():
            shutil.rmtree(backup_path)

        rebuilt = Table.create(
            temp_path,
            [(col["name"], col["type"], col["indexed"], col["not_null"], col["unique"]) for col in extended_meta],
        )
        if rows:
            rebuilt.insert_many(rows)
            for idx, is_live in enumerate(live_flags):
                if not is_live:
                    rebuilt.delete(idx)
        rebuilt.save()
        self._write_table_meta_at(temp_path, extended_meta)

        old_path = self._table_path(table_name)
        if table_name in self._open_tables:
            del self._open_tables[table_name]
        del raw
        gc.collect()

        old_path.rename(backup_path)
        temp_path.rename(old_path)
        shutil.rmtree(backup_path)

        refreshed = Table.open(old_path)
        self._open_tables[table_name] = refreshed
        return ManagedTable(self, table_name, refreshed)

    def _rebuild_table_with_meta(self, table_name: str, meta: list[dict]) -> Table:
        raw = self._raw_table(table_name)
        total_rows = raw.count()
        live_flags = [self._is_live_row(table_name, idx) for idx in range(total_rows)]
        rows = [raw.get_row(idx) for idx in range(total_rows)]

        temp_path = self._path / f".{table_name}.reindex.cdna"
        backup_path = self._table_path(table_name).with_name(f"{table_name}.cdna.reindex.bak")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        if backup_path.exists():
            shutil.rmtree(backup_path)

        rebuilt = Table.create(
            temp_path,
            [(col["name"], col["type"], col["indexed"], col["not_null"], col["unique"]) for col in meta],
        )
        if rows:
            rebuilt.insert_many(rows)
            for idx, is_live in enumerate(live_flags):
                if not is_live:
                    rebuilt.delete(idx)
        rebuilt.save()
        self._write_table_meta_at(temp_path, meta)

        old_path = self._table_path(table_name)
        if table_name in self._open_tables:
            del self._open_tables[table_name]
        del raw
        gc.collect()

        old_path.rename(backup_path)
        temp_path.rename(old_path)
        shutil.rmtree(backup_path)

        refreshed = Table.open(old_path)
        self._open_tables[table_name] = refreshed
        return ManagedTable(self, table_name, refreshed)

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Database(path={self._path!s}, tables={self.tables()!r})"


class Transaction:
    """Atomic database transaction by staging changes in a temp copy."""

    __slots__ = ("_source_path", "_temp_root", "_temp_db_path", "_db", "_committed", "_lock_handle")

    def __init__(self, db: Database) -> None:
        db.save()
        self._source_path = db.path
        self._lock_handle = db._lock_path().open("a+b")
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
        self._temp_root = Path(tempfile.mkdtemp(prefix="cdna_txn_", dir=self._source_path.parent))
        self._temp_db_path = self._temp_root / self._source_path.name
        shutil.copytree(self._source_path, self._temp_db_path)
        self._db = Database.open(self._temp_db_path)
        self._committed = False

    def create_table(self, name: str, schema: list[tuple[str, str, bool]] | list[dict]) -> Table:
        return self._db.create_table(name, schema)

    def add_column(self, table_name: str, column: tuple[str, str, bool] | dict, *, default=None) -> Table:
        return self._db.add_column(table_name, column, default=default)

    def table(self, name: str) -> Table:
        return self._db.table(name)

    def tables(self) -> list[str]:
        return self._db.tables()

    def schema(self, name: str) -> list[dict]:
        return self._db.schema(name)

    def select(self, *args, **kwargs):
        return self._db.select(*args, **kwargs)

    def join(self, *args, **kwargs):
        return self._db.join(*args, **kwargs)

    def execute(self, sql: str):
        return self._db.execute(sql)

    def commit(self) -> None:
        if self._committed:
            return
        self._db.close()
        backup = self._source_path.with_name(f"{self._source_path.name}.bak")
        journal = _journal_path(self._source_path)
        if backup.exists():
            shutil.rmtree(backup)
        journal.write_text(
            json.dumps(
                {
                    "source": str(self._source_path),
                    "backup": str(backup),
                    "staged": str(self._temp_db_path),
                    "temp_root": str(self._temp_root),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._source_path.rename(backup)
        self._temp_db_path.rename(self._source_path)
        shutil.rmtree(backup)
        shutil.rmtree(self._temp_root, ignore_errors=True)
        journal.unlink(missing_ok=True)
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()
        self._committed = True

    def rollback(self) -> None:
        if self._committed:
            return
        self._db.close()
        shutil.rmtree(self._temp_root, ignore_errors=True)
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()
        self._committed = True

    def __enter__(self) -> "Transaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


class ReadTransaction:
    """Read-only snapshot transaction.

    Copies the database under a shared lock, then releases the source lock and
    serves all reads from the frozen snapshot.
    """

    __slots__ = ("_temp_root", "_temp_db_path", "_db", "_closed")

    def __init__(self, db: Database) -> None:
        self._temp_root = Path(tempfile.mkdtemp(prefix="cdna_read_", dir=db.path.parent))
        self._temp_db_path = self._temp_root / db.path.name
        with db._read_lock():
            shutil.copytree(db.path, self._temp_db_path)
        self._db = Database.open(self._temp_db_path)
        self._closed = False

    def table(self, name: str) -> Table:
        return self._db.table(name)

    def tables(self) -> list[str]:
        return self._db.tables()

    def schema(self, name: str) -> list[dict]:
        return self._db.schema(name)

    def select(self, *args, **kwargs):
        return self._db.select(*args, **kwargs)

    def join(self, *args, **kwargs):
        return self._db.join(*args, **kwargs)

    def execute(self, sql: str):
        return self._db.execute(sql)

    def audit(self, table_name: str | None = None) -> dict:
        return self._db.audit(table_name)

    def info(self, table_name: str | None = None) -> dict:
        return self._db.info(table_name)

    def close(self) -> None:
        if self._closed:
            return
        self._db.close()
        shutil.rmtree(self._temp_root, ignore_errors=True)
        self._closed = True

    def __enter__(self) -> "ReadTransaction":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _normalize_schema(schema: list[tuple[str, str, bool]] | list[dict]) -> list[dict]:
    normalized = []
    for entry in schema:
        if isinstance(entry, dict):
            primary = bool(entry.get("primary", False))
            normalized.append(
                {
                    "name": entry["name"],
                    "type": entry["type"],
                    "indexed": bool(entry.get("indexed", False)),
                    "primary": primary,
                    "references": entry.get("references"),
                    "not_null": bool(entry.get("not_null", False) or primary),
                    "unique": bool(entry.get("unique", False) or primary),
                }
            )
        else:
            e = tuple(entry)
            if len(e) == 3:
                name, column_type, indexed = e
                not_null = False
                unique = False
            else:
                name, column_type, indexed, not_null, unique = e
            normalized.append(
                {
                    "name": name,
                    "type": column_type,
                    "indexed": bool(indexed),
                    "primary": False,
                    "references": None,
                    "not_null": bool(not_null),
                    "unique": bool(unique),
                }
            )
    return normalized


def _validate_schema(schema: list[dict], db: Database) -> None:
    seen = set()
    primary_count = 0
    for col in schema:
        name = col["name"]
        if name in seen:
            raise ValueError(f"duplicate column name: {name}")
        seen.add(name)
        if col["primary"]:
            primary_count += 1
        ref = col["references"]
        if ref is None:
            continue
        if "." not in ref:
            raise ValueError(f"invalid references target: {ref}")
        table_name, column_name = ref.split(".", 1)
        if not db.has_table(table_name):
            raise ValueError(f"referenced table does not exist: {table_name}")
        target_schema = db.schema(table_name)
        target = next((entry for entry in target_schema if entry["name"] == column_name), None)
        if target is None:
            raise ValueError(f"referenced column does not exist: {ref}")
        if target["type"] != col["type"]:
            raise ValueError(f"type mismatch for reference {col['name']} -> {ref}")
        if not (target["primary"] or target["unique"]):
            raise ValueError(f"referenced column must be primary or unique: {ref}")
    if primary_count > 1:
        raise ValueError("only one primary key is supported")


def _journal_path(root: Path) -> Path:
    return root.with_name(f"{root.name}{_TXN_JOURNAL_SUFFIX}")


def _recover_database(root: Path) -> None:
    journal = _journal_path(root)
    backup = root.with_name(f"{root.name}{_BACKUP_SUFFIX}")

    if journal.exists():
        info = json.loads(journal.read_text(encoding="utf-8"))
        source = Path(info["source"])
        backup = Path(info["backup"])
        staged = Path(info["staged"])
        temp_root = Path(info["temp_root"])

        if source.exists():
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
        elif staged.exists():
            staged.rename(source)
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
        elif backup.exists():
            backup.rename(source)

        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
        journal.unlink(missing_ok=True)
        return

    if not root.exists() and backup.exists():
        backup.rename(root)
    elif root.exists() and backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


class ManagedTable:
    """Database-aware table wrapper.

    Keeps the geometric table engine unchanged, and adds relational
    rules at the database layer: foreign keys, primary-key protection,
    and compaction.
    """

    __slots__ = ("_db", "_name", "_inner")

    def __init__(self, db: Database, name: str, inner: Table) -> None:
        self._db = db
        self._name = name
        self._inner = inner

    def insert(self, values: list) -> int:
        with self._db._write_lock():
            self._inner = self._db._refresh_raw_table(self._name)
            self._db._validate_row_values(self._name, values)
            return self._inner.insert(values)

    def insert_many(self, rows: list[list]) -> int:
        with self._db._write_lock():
            self._inner = self._db._refresh_raw_table(self._name)
            for row in rows:
                self._db._validate_row_values(self._name, row)
            return self._inner.insert_many(rows)

    def insert_columns(self, columns: list[list]) -> int:
        rows = [list(row) for row in zip(*columns, strict=True)] if columns else []
        with self._db._write_lock():
            self._inner = self._db._refresh_raw_table(self._name)
            for row in rows:
                self._db._validate_row_values(self._name, row)
            return self._inner.insert_columns(columns)

    def update(self, row: int, values: list) -> None:
        with self._db._write_lock():
            self._inner = self._db._refresh_raw_table(self._name)
            self._db._validate_row_values(self._name, values, updating_row=row)
            self._inner.update(row, values)

    def delete(self, row: int) -> None:
        with self._db._write_lock():
            self._inner = self._db._refresh_raw_table(self._name)
            self._db._validate_delete(self._name, row)
            self._inner.delete(row)

    def compact(self) -> int:
        with self._db._write_lock():
            self._inner = self._db._refresh_raw_table(self._name)
            return self._db._compact_table(self._name)

    def schema(self):
        return self._db.schema(self._name)

    def get_f64(self, row: int, col: int) -> float:
        with self._db._read_lock():
            return self._inner.get_f64(row, col)

    def get_bytes(self, row: int, col: int) -> bytes:
        with self._db._read_lock():
            return self._inner.get_bytes(row, col)

    def get_row(self, row: int) -> list:
        with self._db._read_lock():
            return self._inner.get_row(row)

    def column_index(self, name: str) -> int:
        with self._db._read_lock():
            return self._inner.column_index(name)

    def filter_equals(self, col_name: str, value: bytes) -> list[int]:
        with self._db._read_lock():
            return self._inner.filter_equals(col_name, value)

    def filter_cmp(self, col_name: str, op: str, value: float) -> list[int]:
        with self._db._read_lock():
            return self._inner.filter_cmp(col_name, op, value)

    def sum(self, col_name: str) -> float:
        with self._db._read_lock():
            return self._inner.sum(col_name)

    def avg(self, col_name: str) -> float:
        with self._db._read_lock():
            return self._inner.avg(col_name)

    def argsort(self, col_name: str, descending: bool = False) -> list[int]:
        with self._db._read_lock():
            return self._inner.argsort(col_name, descending)

    def search(self, values: list, k: int = 1):
        with self._db._read_lock():
            return self._inner.search(values, k)

    def build_genome(self) -> None:
        with self._db._write_lock():
            self._inner = self._db._refresh_raw_table(self._name)
            self._inner.build_genome()

    def genome_depth(self) -> int:
        with self._db._read_lock():
            return self._inner.genome_depth()

    def genome_codons(self) -> int:
        with self._db._read_lock():
            return self._inner.genome_codons()

    def check(self) -> float:
        with self._db._read_lock():
            return self._inner.check()

    def check_hopf(self):
        with self._db._read_lock():
            return self._inner.check_hopf()

    def inspect_row(self, row: int):
        with self._db._read_lock():
            return self._inner.inspect_row(row)

    def audit(self):
        with self._db._read_lock():
            return self._inner.audit()

    def repair(self) -> None:
        with self._db._write_lock():
            self._inner = self._db._refresh_raw_table(self._name)
            self._inner.repair()

    def identity(self):
        with self._db._read_lock():
            return self._inner.identity()

    def count(self) -> int:
        with self._db._read_lock():
            return self._inner.count()

    def save(self) -> None:
        with self._db._write_lock():
            self._inner.save()

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _column_type(schema: list[dict], column: str) -> str:
    for col in schema:
        if col["name"] == column:
            return col["type"]
    raise KeyError(column)


def _coerce_value(column_type: str, value):
    if value is None:
        return None
    if column_type == "f64":
        return float(value)
    if column_type == "i64":
        return int(value)
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")


def _coerce_like(current, value):
    if value is None:
        return None
    if isinstance(current, bytes):
        if isinstance(value, bytes):
            return value
        return str(value).encode("utf-8")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _like_match(current, pattern) -> bool:
    if current is None or pattern is None:
        return False
    current_text = current.decode("utf-8", errors="replace") if isinstance(current, bytes) else str(current)
    pattern_text = pattern.decode("utf-8", errors="replace") if isinstance(pattern, bytes) else str(pattern)
    regex = "^" + re.escape(pattern_text).replace("%", ".*").replace("_", ".") + "$"
    return re.match(regex, current_text, flags=re.DOTALL) is not None


def _record_from_row(schema: list[dict], row: list) -> dict:
    record = {}
    for col, value in zip(schema, row):
        record[col["name"]] = value
    return record


def _schema_has_column(schema: list[dict], column: str) -> bool:
    return any(col["name"] == column for col in schema)


def _project_record(record: dict, schema: list[dict], columns: list[str]) -> dict:
    projected = {}
    for column in columns:
        alias, expr = _parse_projection(column)
        if _schema_has_column(schema, expr):
            projected[alias] = record[expr]
        else:
            projected[alias] = _evaluate_expression(expr, record)
    return projected


def _match_condition(record: dict, schema: list[dict], column: str, op: str, value) -> bool:
    normalized = op.strip().lower()
    if _schema_has_column(schema, column):
        current = record[column]
        if normalized in {"in", "not in"}:
            wanted = [_coerce_value(_column_type(schema, column), item) for item in value]
        else:
            wanted = _coerce_value(_column_type(schema, column), value)
    else:
        current = _evaluate_expression(column, record)
        if normalized in {"in", "not in"}:
            wanted = [_coerce_like(current, item) for item in value]
        else:
            wanted = _coerce_like(current, value)
    if normalized == "in":
        return current in set(wanted)
    if normalized == "not in":
        return current not in set(wanted)
    if normalized in {"is", "is null"}:
        return current is None if value is None else current is wanted
    if normalized in {"is not", "is not null"}:
        return current is not None if value is None else current is not wanted
    if normalized == "like":
        return _like_match(current, wanted)
    if current is None or wanted is None:
        if op == "=":
            return current is wanted
        if op == "!=":
            return current is not wanted
        return False
    if op == "=" :  return current == wanted
    if op == "!=":  return current != wanted
    if op == ">" :  return current > wanted
    if op == "<" :  return current < wanted
    if op == ">=":  return current >= wanted
    if op == "<=":  return current <= wanted
    return True


def _match_where(record: dict, schema: list[dict], where) -> bool:
    """Evaluate a WHERE clause against a record.

    where can be:
      - a list of conditions (legacy flat AND)  [(col, op, val), ...]
      - a dict with "and" or "or" key:          {"and": [...]} / {"or": [...]}
      - a single condition tuple:               (col, op, val)
    """
    if isinstance(where, tuple):
        return _match_condition(record, schema, *where)
    if isinstance(where, dict):
        if "and" in where:
            return all(_match_where(record, schema, clause) for clause in where["and"])
        if "or" in where:
            return any(_match_where(record, schema, clause) for clause in where["or"])
        raise ValueError(f"unknown where dict key: {list(where.keys())}")
    # flat list → AND (legacy)
    return all(_match_condition(record, schema, col, op, val) for col, op, val in where)


def _match_join_where(record: dict, where) -> bool:
    """Same as _match_where but without schema coercion (join records are already typed)."""
    if isinstance(where, tuple):
        col, op, value = where
        current = record[col] if col in record else _evaluate_expression(col, record)
        normalized = op.strip().lower()
        if normalized == "in":
            return current in {_coerce_like(current, item) for item in value}
        if normalized == "not in":
            return current not in {_coerce_like(current, item) for item in value}
        if normalized in {"is", "is null"}:
            return current is None if value is None else current is value
        if normalized in {"is not", "is not null"}:
            return current is not None if value is None else current is not value
        if normalized == "like":
            return _like_match(current, value)
        if current is None or value is None:
            if op == "=":
                return current is value
            if op == "!=":
                return current is not value
            return False
        if op == "=" :  return current == value
        if op == "!=":  return current != value
        if op == ">" :  return current > value
        if op == "<" :  return current < value
        if op == ">=":  return current >= value
        if op == "<=":  return current <= value
        return True
    if isinstance(where, dict):
        if "and" in where:
            return all(_match_join_where(record, clause) for clause in where["and"])
        if "or" in where:
            return any(_match_join_where(record, clause) for clause in where["or"])
        raise ValueError(f"unknown where dict key: {list(where.keys())}")
    return all(_match_join_where(record, clause) for clause in where)


def _jsonable_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            converted[key] = value.decode("utf-8") if isinstance(value, bytes) else value
        out.append(converted)
    return out


def _rows_to_csv(schema: list[dict], rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    ordered_cols = [col["name"] for col in schema]
    writer.writerow(ordered_cols)
    for row in rows:
        writer.writerow(
            [
                row[name].decode("utf-8") if isinstance(row[name], bytes) else row[name]
                for name in ordered_cols
            ]
        )
    return buf.getvalue()


def _order_key(value):
    return (value is None, value)


def _sort_records(records: list[dict], order_by: str, descending: bool) -> list[dict]:
    present = [record for record in records if record[order_by] is not None]
    missing = [record for record in records if record[order_by] is None]
    present.sort(key=lambda record: record[order_by], reverse=descending)
    return present + missing


def _parse_projection(column: str) -> tuple[str, str]:
    upper = column.upper()
    if " AS " in upper:
        idx = upper.rfind(" AS ")
        expr = column[:idx].strip()
        alias = column[idx + 4 :].strip()
        return alias, expr
    return column, column


def _evaluate_expression(expr: str, record: dict):
    tree = ast.parse(expr, mode="eval")
    return _eval_ast(tree.body, record)


def _eval_ast(node, record: dict):
    if isinstance(node, ast.Name):
        return record[node.id]
    if isinstance(node, ast.Attribute):
        path = _attribute_path(node)
        if path is None:
            raise ValueError("unsupported attribute expression")
        return record[path]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, record)
        right = _eval_ast(node.right, record)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ValueError(f"unsupported expression operator: {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp):
        value = _eval_ast(node.operand, record)
        if value is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("unsupported expression call")
        func = node.func.id.upper()
        args = [_eval_ast(arg, record) for arg in node.args]
        if func == "UPPER":
            value = args[0]
            if value is None:
                return None
            if isinstance(value, bytes):
                return value.upper()
            return str(value).upper()
        if func == "LOWER":
            value = args[0]
            if value is None:
                return None
            if isinstance(value, bytes):
                return value.lower()
            return str(value).lower()
        if func == "ABS":
            value = args[0]
            return None if value is None else abs(value)
        raise ValueError(f"unsupported expression function: {func}")
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _attribute_path(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attribute_path(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _infer_column_type(values: list[object]) -> str:
    samples = [value for value in values if value not in (None, "")]
    if not samples:
        return "bytes"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in samples):
        return "i64"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in samples):
        return "f64"
    return "bytes"


def _parse_csv_numeric(value: str):
    text = value.strip()
    if text == "":
        return None
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _encode_import_value(column_type: str, value):
    if value in ("", None):
        return None
    if column_type == "f64":
        return float(value)
    if column_type == "i64":
        return int(value)
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")


def _infer_json_table(text: str) -> tuple[list[dict], list[list]]:
    data = json.loads(text)
    if not isinstance(data, list) or not data:
        raise ValueError("expected non-empty JSON array")
    if not isinstance(data[0], dict):
        raise ValueError("expected JSON array of objects")
    keys = list(data[0].keys())
    schema = []
    columns = [[] for _ in keys]
    column_values = [[row.get(key) for row in data] for key in keys]
    for key, values in zip(keys, column_values, strict=True):
        schema.append(
            {
                "name": key,
                "type": _infer_column_type(values),
                "indexed": False,
            }
        )
    for row in data:
        for idx, key in enumerate(keys):
            value = row.get(key, None)
            columns[idx].append(_encode_import_value(schema[idx]["type"], value))
    return schema, columns


def _infer_csv_table(text: str, *, header: bool = True) -> tuple[list[dict], list[list]]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("empty CSV")
    if header:
        names = [cell.strip() for cell in rows[0]]
        rows = rows[1:]
    else:
        names = [f"col_{idx}" for idx in range(len(rows[0]))]
    if not rows:
        raise ValueError("CSV has header but no data rows")
    schema = []
    columns = [[] for _ in names]
    parsed_rows = []
    for row in rows:
        parsed_rows.append([_parse_csv_numeric(row[idx]) if idx < len(row) else None for idx in range(len(names))])
    for idx, name in enumerate(names):
        schema.append({"name": name, "type": _infer_column_type([row[idx] for row in parsed_rows]), "indexed": False})
    for row, parsed in zip(rows, parsed_rows, strict=True):
        if not any(cell.strip() for cell in row):
            continue
        for idx in range(len(names)):
            columns[idx].append(_encode_import_value(schema[idx]["type"], parsed[idx]))
    return schema, columns
