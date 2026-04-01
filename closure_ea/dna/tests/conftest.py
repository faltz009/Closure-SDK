"""Pytest session guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

import closure_rs


def pytest_sessionstart(session) -> None:  # pragma: no cover - test bootstrap
    """Fail fast if Python imported closure_rs from outside this repository."""
    manifest_dir, version = closure_rs.build_info()
    expected = (Path(__file__).resolve().parents[3] / "rust").resolve()
    actual = Path(manifest_dir).resolve()

    if actual != expected:
        module_file = Path(getattr(closure_rs, "__file__", "<unknown>")).resolve()
        pytest.exit(
            "\n".join(
                [
                    "closure_rs import mismatch:",
                    f"  loaded module:  {module_file}",
                    f"  build manifest: {actual}",
                    f"  expected:       {expected}",
                    f"  version:        {version}",
                    "Run: python -m pip install -e '.[dev]'",
                    "Tip: set PYTHONNOUSERSITE=1 to avoid stale user-site packages.",
                ]
            ),
            returncode=2,
        )

    # Also fail fast on stale local extension builds from this same repo path.
    # This happens when an older wheel is still imported, missing newer APIs.
    missing = []
    if not hasattr(closure_rs, "StreamMonitor"):
        missing.append("StreamMonitor")

    raw_embed_ok = True
    try:
        _ = closure_rs.path_from_raw_bytes("Torus(2)", [b"probe"])
    except Exception:
        raw_embed_ok = False
        missing.append("raw embedding for torus / hybrid geometries")

    if missing:
        module_file = Path(getattr(closure_rs, "__file__", "<unknown>")).resolve()
        pytest.exit(
            "\n".join(
                [
                    "closure_rs capability mismatch (stale build):",
                    f"  loaded module:  {module_file}",
                    f"  build manifest: {actual}",
                    f"  version:        {version}",
                    f"  missing:        {', '.join(missing)}",
                    "Rebuild locally and run tests against _vendor:",
                    "  python -m pip install --no-build-isolation --no-deps --upgrade --target _vendor .",
                    "  PYTHONPATH=\"_vendor${PYTHONPATH:+:$PYTHONPATH}\" pytest tests -q",
                ]
            ),
            returncode=2,
        )
