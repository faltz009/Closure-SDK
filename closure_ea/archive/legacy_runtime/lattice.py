"""
Lattice — the living lattice. A sphere of spheres.

Variable size. Grows with data. Stacks cells by closure cadence.
The hierarchy is not hardcoded — levels spawn when lower levels emit.

The Lattice IS:
    - a composition of cells at different timescales
    - a genome that accumulates what the lattice has learned
    - an identity that shapes through experience

The Lattice does NOT:
    - know what substrate it's processing
    - have fixed levels or named tiers
    - require a specific adapter (any adapter works)
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from closure_ea.kernel import Kernel, compose, inverse, sigma, identity
from closure_ea.cell import Cell, Adapter
import numpy as np


@dataclass
class LatticeCell:
    """One explicit closure object in the living lattice."""

    id: str
    level: int
    q: np.ndarray
    reason: str
    event_count: int
    child_ids: list[str]
    trigger_ids: list[str] = field(default_factory=list)
    source_keys: list[str] = field(default_factory=list)
    trigger_source_keys: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def gap(self):
        return sigma(self.q)


@dataclass(frozen=True)
class ClosureEvent:
    """Structured emission from the lattice.

    This is the stable object for observers that want to process the
    lattice's own closure stream without coupling to track-specific data.
    """

    lattice_id: str
    cell_id: str
    level: int
    content: np.ndarray
    reason: str
    event_count: int
    child_ids: tuple[str, ...]
    trigger_ids: tuple[str, ...]
    source_keys: tuple[str, ...]
    trigger_source_keys: tuple[str, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def gap(self):
        return sigma(self.content)


class Lattice:
    """The living lattice.

    Bottom cell receives events from the adapter.
    When it closes, the emission feeds the cell above.
    Levels spawn when needed. The depth matches the data.
    """

    def __init__(self, adapter, epsilon=0.15, max_depth=6, epsilon_schedule=None, on_emit=None):
        self.adapter = adapter
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.epsilon_schedule = epsilon_schedule or []
        self.on_emit = on_emit
        self._emit_listeners = []
        self.lattice_id = f"lattice|{id(self):x}"

        # The lattice: a stack of kernels wired by emission
        self.kernels = []
        self.cells = []
        self.leaf_nodes = {}
        self.pending_units = defaultdict(list)
        self._leaf_index = 0
        self._cell_indexes = defaultdict(int)
        self._build_level(0)

    @staticmethod
    def _collect_list_field(metas, field):
        """Collect values from a list-or-scalar metadata field across children."""
        values = []
        for meta in metas:
            v = meta.get(field)
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                values.extend(v)
            else:
                values.append(v)
        return sorted(set(values)) if values else []

    def _next_leaf_id(self):
        self._leaf_index += 1
        return f"leaf|{self._leaf_index:09d}"

    def _next_cell_id(self, level):
        self._cell_indexes[level] += 1
        return f"cell|l{level}|{self._cell_indexes[level]:09d}"

    def _register_leaf(self, event_key, meta=None):
        node_id = self._next_leaf_id()
        leaf_meta = dict(meta or {})
        leaf_meta.setdefault("event_key", event_key)
        self.leaf_nodes[node_id] = {
            "id": node_id,
            "level": 0,
            "kind": leaf_meta.get("kind", "event"),
            "source_key": event_key,
            "meta": leaf_meta,
        }
        self.pending_units[0].append(
            {
                "id": node_id,
                "source_key": event_key,
                "meta": leaf_meta,
            }
        )
        return node_id

    def _aggregate_meta(self, emitted_units, trailing_units, emitted_level, reason):
        """Aggregate metadata from children generically.

        For any field that has the SAME value across all children,
        propagate it.  For list fields, collect and merge.
        No substrate-specific field names — the adapter decides
        what to put in meta, the lattice just propagates.
        """
        child_meta = [unit.get("meta", {}) for unit in emitted_units]
        trigger_meta = [unit.get("meta", {}) for unit in trailing_units]

        out = {
            "level": emitted_level,
            "reason": reason,
        }

        # Propagate any field that's consistent across all children
        all_keys = set()
        for m in child_meta:
            all_keys.update(m.keys())

        for field in all_keys:
            if field in ("level", "reason", "kind", "event_key"):
                continue  # structural fields, don't propagate
            values = [m.get(field) for m in child_meta if m.get(field) is not None]
            if not values:
                continue
            # If all values are the same scalar, propagate
            if all(v == values[0] for v in values) and not isinstance(values[0], (list, tuple, set)):
                out[field] = values[0]
            # If values are lists or contain lists, collect
            else:
                collected = self._collect_list_field(child_meta, field)
                if collected:
                    out[field] = collected

        # Collect trigger metadata separately
        trigger_keys = set()
        for m in trigger_meta:
            trigger_keys.update(m.keys())
        for field in trigger_keys:
            collected = self._collect_list_field(trigger_meta, field)
            if collected:
                out[f"trigger_{field}"] = collected

        return out

    def _record_cell(self, level, content, reason, content_event_count):
        pending_before = list(self.pending_units[level])
        emitted_units = pending_before[:content_event_count]
        trailing_units = pending_before[content_event_count:]
        self.pending_units[level] = []

        if not emitted_units:
            return None

        emitted_level = level + 1
        cell_id = self._next_cell_id(emitted_level)
        meta = self._aggregate_meta(emitted_units, trailing_units, emitted_level, reason)
        cell = LatticeCell(
            id=cell_id,
            level=emitted_level,
            q=content.copy(),
            reason=reason,
            event_count=content_event_count,
            child_ids=[unit["id"] for unit in emitted_units],
            trigger_ids=[unit["id"] for unit in trailing_units],
            source_keys=[unit.get("source_key", unit["id"]) for unit in emitted_units],
            trigger_source_keys=[unit.get("source_key", unit["id"]) for unit in trailing_units],
            meta=meta,
        )
        self.cells.append(cell)
        self.pending_units[emitted_level].append(
            {
                "id": cell.id,
                "source_key": cell.id,
                "meta": cell.meta,
            }
        )
        return cell

    def _cell_to_event(self, cell):
        return ClosureEvent(
            lattice_id=self.lattice_id,
            cell_id=cell.id,
            level=cell.level,
            content=cell.q.copy(),
            reason=cell.reason,
            event_count=cell.event_count,
            child_ids=tuple(cell.child_ids),
            trigger_ids=tuple(cell.trigger_ids),
            source_keys=tuple(cell.source_keys),
            trigger_source_keys=tuple(cell.trigger_source_keys),
            meta=dict(cell.meta or {}),
        )

    def add_emit_listener(self, listener):
        """Subscribe to structured closure emissions.

        Listeners receive one `ClosureEvent`.
        """
        self._emit_listeners.append(listener)
        return listener

    def remove_emit_listener(self, listener):
        self._emit_listeners = [fn for fn in self._emit_listeners if fn is not listener]

    def _level_epsilon(self, level):
        if level < len(self.epsilon_schedule):
            return self.epsilon_schedule[level]
        return self.epsilon * (1 + level * 0.5)

    def _handle_emit(self, level, content, reason, content_event_count):
        cell = self._record_cell(level, content, reason, content_event_count)
        if cell is not None:
            event = self._cell_to_event(cell)
            if self.on_emit:
                self.on_emit(event)
            for listener in list(self._emit_listeners):
                listener(event)
        return cell

    def _build_level(self, level):
        """Add a level to the lattice."""
        if level >= self.max_depth:
            return
        if level >= len(self.kernels):
            eps = self._level_epsilon(level)
            k = Kernel(
                epsilon=eps,
                parent=None,
                on_emit=lambda content, reason, count, lvl=level: self._handle_emit(
                    lvl, content, reason, count
                ),
            )
            self.kernels.append(k)
            if level > 0:
                self.kernels[level - 1].parent = self.kernels[level]

    def ingest(self, event_key, meta=None):
        """Feed one event through the lattice.

        The adapter embeds it. The bottom kernel composes it.
        Closures propagate upward through parent wiring.
        New levels spawn when lower levels start emitting.
        The lattice checks for death (σ → π) and handles it.

        Returns: (result, gap, incident_type)
            result: 'open', 'closure', or 'death'
            gap: current σ at level 0
            incident_type: 'missing' or 'reorder'
        """
        self._register_leaf(event_key, meta=meta)
        q = self.adapter.embed(event_key)
        sigma_before = self.kernels[0].gap
        result, vote = self.kernels[0].ingest(q)
        sigma_after = self.kernels[0].gap

        # DEATH CHECK — the lattice decides, not the kernel.
        # σ → π means coherence is lost. The cell failed to close.
        # The lattice resets the cell and reports death.
        if sigma_after > math.pi * 0.95:
            self.kernels[0].reset()
            incident_type = self.adapter.classify(q)
            return 'death', 0.0, incident_type

        # Spawn higher levels if needed
        if result == 'closure' and len(self.kernels) < self.max_depth:
            next_level = 1
            while next_level < len(self.kernels):
                if self.kernels[next_level].event_count > 0:
                    next_level += 1
                else:
                    break
            if next_level >= len(self.kernels):
                self._build_level(next_level)

        incident_type = self.adapter.classify(self.kernels[0].C)
        return result, self.kernels[0].gap, incident_type

    def ingest_with_truth(self, event_key, position, meta=None):
        """Ingest with exact adapter truth."""
        self._register_leaf(event_key, meta=meta)
        self.adapter.embed_exact(event_key, position)
        result, vote = self.kernels[0].ingest(position)
        incident_type = self.adapter.classify(self.kernels[0].C)
        return result, self.kernels[0].gap, incident_type

    def ingest_stream(self, events, report_every=None):
        """Live through a stream of events."""
        import time
        t0 = time.time()
        closures = 0
        for i, event in enumerate(events):
            result, gap, incident = self.ingest(event)
            if result == 'closure':
                closures += 1
            if report_every and (i + 1) % report_every == 0:
                elapsed = time.time() - t0
                print(f"  {i+1:>9,} | {elapsed:.1f}s | {(i+1)/elapsed:.0f}/s | "
                      f"genome: {self.adapter.size} | "
                      f"levels: {self.depth} | "
                      f"closures: {self.closure_counts}")
        return closures

    @property
    def depth(self):
        """How many levels are active."""
        return sum(1 for k in self.kernels if k.event_count > 0)

    @property
    def closure_counts(self):
        """Closures at each level."""
        return [k.emission_count for k in self.kernels if k.event_count > 0]

    @property
    def predictions(self):
        """C⁻¹ at each active level — the downward signals.
        Each is the quaternion that would close the current composition
        at that level.  This IS what the lattice expects next."""
        return [k.prediction for k in self.kernels if k.event_count > 0]

    def status(self):
        """Current state of the lattice."""
        return [{
            'level': i,
            'events': k.event_count,
            'closures': k.emission_count,
            'gap': k.gap,
        } for i, k in enumerate(self.kernels) if k.event_count > 0 or i == 0]

    def reset(self):
        """Reset all kernels. Keep the genome (adapter positions)."""
        for k in self.kernels:
            k.reset()
        self.pending_units.clear()

    def flush(self, reason='boundary'):
        """Force all open scopes to emit because the adapter declared a boundary."""
        for kernel in self.kernels:
            kernel.force_emit(reason=reason)

    def clear_lattice(self):
        """Discard explicit runtime cell history. Keep adapter genome."""
        self.cells.clear()
        self.leaf_nodes.clear()
        self.pending_units.clear()
        self._leaf_index = 0
        self._cell_indexes.clear()
