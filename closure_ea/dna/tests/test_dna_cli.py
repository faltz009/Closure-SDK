"""Legacy pre-v0 CLI tests retired.

The current CLI targets multi-table databases, not the removed
single-table raw-record surface these tests originally exercised.
"""

from __future__ import annotations

import pytest

pytest.skip("legacy pre-v0 CLI tests retired for the current Closure DNA surface", allow_module_level=True)
