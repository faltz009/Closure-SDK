"""Closure Identity — static verification via Gilgamesh.

Takes two files, composes both on S³, walks them side by side,
and reports every incident: what broke, where, and what kind.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from closure_sdk import gilgamesh_detailed, expose_incident, incident_drift

from .reader import read_file
from .formatter import format_stdout, format_report_json


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "identity",
        help="Compare two files and report every discrepancy.",
        description=(
            "Reads both files, composes each on S³, and walks them "
            "side by side. Reports missing records, reorders, and "
            "their color channels."
        ),
    )
    p.add_argument("source", type=Path, help="Source file (the reference).")
    p.add_argument("target", type=Path, help="Target file (the one to check).")
    p.add_argument("--format", choices=["jsonl", "csv", "text"], default=None,
                   help="Input format. Auto-detected from extension if omitted.")
    p.add_argument("--columns", type=str, default=None,
                   help="CSV only: comma-separated column indices (0-based).")
    p.add_argument("--header", action="store_true",
                   help="Skip the first row of each file.")
    p.add_argument("--delimiter", type=str, default=",",
                   help="CSV delimiter (default: comma).")
    p.add_argument("--max-faults", type=int, default=1000,
                   help="Stop after this many incidents (default: 1000).")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Save full report (with channels, sigma) to this JSON file.")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    source_path: Path = args.source
    target_path: Path = args.target

    if not source_path.exists():
        print(f"Error: {source_path} not found.", file=sys.stderr)
        return 1
    if not target_path.exists():
        print(f"Error: {target_path} not found.", file=sys.stderr)
        return 1

    columns = None
    if args.columns:
        columns = [int(c.strip()) for c in args.columns.split(",")]

    read_kwargs = dict(
        fmt=args.format,
        columns=columns,
        header=args.header,
        delimiter=args.delimiter,
    )

    source_records = read_file(source_path, **read_kwargs)
    target_records = read_file(target_path, **read_kwargs)

    t0 = time.perf_counter()
    result = gilgamesh_detailed(source_records, target_records, max_faults=args.max_faults)
    elapsed = time.perf_counter() - t0
    incidents = result.incidents

    # --- report file (compute channels) ---
    report_path = None
    if args.output or incidents:
        # Per-incident color: the LOCAL gap at each incident's position
        valences = [
            expose_incident(
                inc,
                incident_drift(inc, result.source_path, result.target_path),
            )
            for inc in incidents
        ]

        report_path = args.output
        if report_path is None and incidents:
            report_path = Path(f"{source_path.stem}_identity.json")

        report = format_report_json(
            incidents, valences,
            source_path.name, target_path.name,
            len(source_records), len(target_records),
            elapsed,
        )
        report_path.write_text(report, encoding="utf-8")

    # --- stdout ---
    print(format_stdout(
        incidents,
        source_path.name, target_path.name,
        len(source_records), len(target_records),
        elapsed,
        report_path=str(report_path) if report_path else None,
    ))

    return 0 if not incidents else 2
