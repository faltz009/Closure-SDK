"""
The Holy Trinity.

    Reality → S1 × S3 → Perception → S2 × S3 → Prediction → S1 → Reality

S1 is the boundary adapter.
S2 is the evaluation lattice. Lattice(ε=normal).
S3 is the memory lattice.   Lattice(ε=slow).

Same kernel everywhere. The difference is timing:
S2 learns fast, on local scopes.
S3 learns slowly, on consolidated scope error rather than raw per-tick error.
"""

from __future__ import annotations

import math

import numpy as np

from closure_ea.kernel import compose, inverse, sigma, identity
from closure_ea.cell import Adapter, geodesic_step
from closure_ea.lattice import Lattice


# S3 should consolidate rarely. Closure happens when sigma is SMALL,
# so memory epsilon must stay tighter than S2 while learning is slowed
# further by feeding S3 only scope-level error.
MEMORY_EPSILON = 0.8


class Trinity:
    """S1/S2/S3 learning loop.

    S1: boundary adapter.
    S2: Lattice for evaluation. Normal ε.
    S3: Lattice for memory. High ε. Same kernel.
    """

    def __init__(self, adapter: Adapter,
                 epsilon_s2: float = 0.15,
                 epsilon_s3: float = MEMORY_EPSILON,
                 max_depth: int = 6,
                 s2_schedule: list | None = None,
                 s3_schedule: list | None = None):

        self.adapter = adapter

        # S2: evaluation. Normal ε. Grows dynamically.
        self.s2 = Lattice(
            _PureAdapter(),
            epsilon=epsilon_s2,
            max_depth=max_depth,
            epsilon_schedule=s2_schedule,
        )

        # S3: memory. Same kernel, slower cadence.
        self.s3 = Lattice(
            _PureAdapter(),
            epsilon=epsilon_s3,
            max_depth=max_depth,
            epsilon_schedule=s3_schedule,
        )

        self.event_count = 0
        self._last_key = None
        self._last_prediction = None
        self._prediction_sigma_trace: list[float] = []
        self._pending_errors: list[tuple[str, np.ndarray, float]] = []

    def ingest(self, event_key, meta=None):
        """One tick through the Holy Trinity.

        Phase 1: compare reality against stored prediction → error → S3.
        Phase 2: perceive → process through S2 → predict → store.
        """
        self.event_count += 1
        self._last_key = event_key

        # S1: embed reality
        reality_q = self.adapter.embed(event_key)

        # ── PHASE 1: evaluate previous prediction ──
        if self._last_prediction is not None:
            error = compose(reality_q, inverse(self._last_prediction))
            n = np.linalg.norm(error)
            if n > 1e-8:
                error = error / n
            error_sigma = sigma(error)
            self._prediction_sigma_trace.append(error_sigma)
            self._pending_errors.append((event_key, error, error_sigma))

        # ── PHASE 2: perceive → process → predict ──

        # S1 × S3: perception = reality relative to memory
        memory_state = self.s3.kernels[0].C
        perception = compose(reality_q, inverse(memory_state))
        n = np.linalg.norm(perception)
        if n > 1e-8:
            perception = perception / n

        # S2: evaluation lattice processes perception
        perc_key = f"perc|{self.event_count}"
        self.s2.adapter.genome[perc_key] = (perception, 0)
        result_s2, gap_s2, incident_s2 = self.s2.ingest(perc_key)

        if result_s2 == 'closure':
            self._consolidate_scope(reason='s2_closure')

        # Prediction: what closes S2's scope, mapped through memory
        prediction = compose(self.s2.kernels[0].prediction, memory_state)
        n = np.linalg.norm(prediction)
        if n > 1e-8:
            prediction = prediction / n
        self._last_prediction = prediction

        incident = self.adapter.classify(perception)
        return result_s2, gap_s2, incident

    def _consolidate_scope(self, reason='boundary'):
        """Feed S3 only stabilized error from a completed S2 scope."""
        if not self._pending_errors:
            return

        scope_error = identity()
        total_sigma = 0.0
        for _event_key, error, error_sigma in self._pending_errors:
            scope_error = compose(scope_error, error)
            total_sigma += error_sigma

        n = np.linalg.norm(scope_error)
        if n > 1e-8:
            scope_error = scope_error / n

        mean_sigma = total_sigma / len(self._pending_errors)
        scope_key = f"mem|{self.event_count}|{reason}"
        self.s3.adapter.genome[scope_key] = (scope_error, 0)
        self.s3.ingest(scope_key)
        self._write_scope_positions(scope_error, mean_sigma)
        self._pending_errors.clear()

    def _write_scope_positions(self, scope_error, error_sigma):
        """Apply one delayed correction across the closed scope."""
        step_size = self.adapter.damping * (error_sigma / math.pi)
        if step_size <= 0:
            return
        seen = set()
        for event_key, _error, _error_sigma in self._pending_errors:
            if event_key in seen or event_key not in self.adapter.genome:
                continue
            seen.add(event_key)
            pos, count = self.adapter.genome[event_key]
            if count == -1:
                continue
            new_pos = geodesic_step(pos, scope_error, step_size)
            self.adapter.genome[event_key] = (new_pos, count + 1)

    @property
    def prediction(self):
        return self._last_prediction

    @property
    def memory_state(self):
        return self.s3.kernels[0].C.copy()

    @property
    def gap(self):
        return self.s2.kernels[0].gap

    @property
    def positions(self):
        return self.adapter.genome

    def flush(self, reason='boundary'):
        self.s2.flush(reason=reason)
        self._consolidate_scope(reason=reason)

    def save(self):
        return {
            'memory': self.s3.kernels[0].C.tolist(),
            'memory_events': self.s3.kernels[0].event_count,
            'positions': {
                k: (p.tolist(), c)
                for k, (p, c) in self.adapter.genome.items()
            },
            'prediction_trace': list(self._prediction_sigma_trace),
        }

    def load(self, data):
        self.s3.kernels[0].C = np.array(data['memory'])
        self.s3.kernels[0].event_count = data.get('memory_events', 0)
        for k, (q, c) in data.get('positions', {}).items():
            self.adapter.genome[k] = (np.array(q), c)
        self._prediction_sigma_trace = list(data.get('prediction_trace', []))

    def status(self):
        return {
            'events': self.event_count,
            's2': {
                'depth': self.s2.depth,
                'closures': self.s2.closure_counts,
            },
            's3': {
                'depth': self.s3.depth,
                'closures': self.s3.closure_counts,
                'sigma': sigma(self.s3.kernels[0].C),
                'events': self.s3.kernels[0].event_count,
            },
            'genome': len(self.adapter.genome),
        }

    def __repr__(self):
        s = self.status()
        return (f"Trinity(events={s['events']}, "
                f"genome={s['genome']}, "
                f"s3_σ={s['s3']['sigma']:.3f})")


class _PureAdapter(Adapter):
    """For S2 and S3 — pure S³, no substrate translation."""

    def embed(self, event_key):
        if event_key in self.genome:
            return self.genome[event_key][0].copy()
        return super().embed(event_key)
