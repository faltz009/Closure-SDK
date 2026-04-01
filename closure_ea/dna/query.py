"""Typed query parsing for Closure DNA."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    verb: str
    args: list[str]


def parse(text: str) -> Query:
    text = text.strip()
    if not text:
        return Query("", [])
    parts = shlex.split(text)
    if not parts:
        return Query("", [])
    return Query(parts[0].upper(), parts[1:])


def parse_json_arg(text: str):
    return json.loads(text)


def coerce_value(column_type: str, value):
    if value is None:
        return None
    if column_type == "f64":
        return float(value)
    if column_type == "i64":
        return int(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"cannot coerce value {value!r} to {column_type}")


def _schema_entry(entry):
    if isinstance(entry, dict):
        return entry["name"], entry["type"], bool(entry.get("indexed", False))
    return entry


def coerce_row(schema: list[tuple[str, str, bool]] | list[dict], values: list):
    if len(values) != len(schema):
        raise ValueError(f"expected {len(schema)} values, got {len(values)}")
    return [
        coerce_value(column_type, value)
        for (_, column_type, _), value in zip((_schema_entry(entry) for entry in schema), values)
    ]


def coerce_for_column(schema: list[tuple[str, str, bool]] | list[dict], column: str, value):
    for name, column_type, _ in (_schema_entry(entry) for entry in schema):
        if name == column:
            return coerce_value(column_type, value)
    raise ValueError(f"unknown column: {column}")
