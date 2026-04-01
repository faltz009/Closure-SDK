"""Demo dataset and database registry for Closure DNA."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from closure_ea.dna.database import Database
from closure_ea.dna.query import coerce_value


def demo_root() -> Path:
    return Path(__file__).resolve().parent


def demo_data_root() -> Path:
    return demo_root().parent / "demo_data"


def demo_database_root() -> Path:
    root = demo_root() / "databases"
    root.mkdir(parents=True, exist_ok=True)
    return root


def demo_index() -> dict:
    return json.loads((demo_data_root() / "datasets.json").read_text(encoding="utf-8"))


def demo_names() -> list[str]:
    return [entry["name"] for entry in demo_index()["datasets"]]


def demo_manifest_path(name: str) -> Path:
    for entry in demo_index()["datasets"]:
        if entry["name"] == name:
            return demo_data_root() / entry["path"]
    raise ValueError(f"unknown demo dataset: {name}")


def demo_manifest(name: str) -> dict:
    return json.loads(demo_manifest_path(name).read_text(encoding="utf-8"))


def demo_database_path(name: str) -> Path:
    return demo_database_root() / f"{name}.cdb"


def available_demos() -> list[dict]:
    demos = []
    for name in demo_names():
        manifest = demo_manifest(name)
        path = demo_database_path(name)
        demos.append(
            {
                "name": name,
                "title": manifest["title"],
                "description": manifest["description"],
                "built": path.exists(),
                "database_path": str(path),
                "counts": manifest["counts"],
            }
        )
    return demos


def _load_table_rows(manifest_path: Path, manifest: dict, table_name: str) -> list[dict]:
    data_path = manifest_path.parent / manifest["files"][table_name]
    return json.loads(data_path.read_text(encoding="utf-8"))


def _columnize(schema: list[dict], rows: list[dict]) -> list[list]:
    columns = [[] for _ in schema]
    for row in rows:
        for idx, column in enumerate(schema):
            columns[idx].append(coerce_value(column["type"], row.get(column["name"])))
    return columns


def build_demo_database(name: str, *, replace: bool = True) -> Path:
    manifest_path = demo_manifest_path(name)
    manifest = demo_manifest(name)
    path = demo_database_path(name)

    if path.exists():
        if not replace:
            return path
        shutil.rmtree(path)

    with Database.create(path) as db:
        for table_name in manifest["table_order"]:
            schema = manifest["schemas"][table_name]
            rows = _load_table_rows(manifest_path, manifest, table_name)
            table = db.create_table(table_name, schema)
            if not rows:
                continue
            columns = _columnize(schema, rows)
            if any(any(value is None for value in column) for column in columns):
                row_values = [list(row) for row in zip(*columns, strict=True)]
                table.insert_many(row_values)
            else:
                table.insert_columns(columns)
    return path


def build_all_demo_databases(*, replace: bool = True) -> list[Path]:
    return [build_demo_database(name, replace=replace) for name in demo_names()]


def ensure_demo_database(name: str) -> Path:
    path = demo_database_path(name)
    if path.exists():
        return path
    return build_demo_database(name, replace=True)
