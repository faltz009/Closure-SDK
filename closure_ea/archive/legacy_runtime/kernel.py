"""
The kernel. Pure closure recurrence. Substrate agnostic.

Knows ONLY: event (quaternion), state, gap, vote, emit.
Does NOT know: words, labels, concepts, followers, n-grams.
Does NOT decide: lifecycle, death, spawning — those belong to the lattice.

    ingest(q):
        C_before = C
        C = C · q
        vote = C_before⁻¹
        if σ(C) < ε: emit(C_before) upward, reset C
        return (result, vote)

That's the entire kernel. Everything else is an adapter.
"""

import numpy as np
import math

try:
    import closure_rs
    _G = closure_rs.sphere()
    def compose(a, b): return np.array(_G.compose(a, b))
    def inverse(a): return np.array(_G.inverse(a))
    def sigma(a): return _G.distance_from_identity(a)
    def identity(): return np.array(_G.identity())
    RUST = True
except Exception:
    def _qn(q):
        n = np.linalg.norm(q)
        return q/n if n > 1e-8 else np.array([1.,0.,0.,0.])
    def compose(a, b):
        w1,x1,y1,z1=a; w2,x2,y2,z2=b
        return _qn(np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2]))
    def inverse(a): return np.array([a[0],-a[1],-a[2],-a[3]])
    def sigma(a): return math.acos(min(abs(a[0]),1.0))
    def identity(): return np.array([1.,0.,0.,0.])
    RUST = False


class Kernel:
    """Pure closure recurrence. No substrate knowledge. No lifecycle decisions.

    The kernel composes events and reports its state.  Period.
    It does not decide to die, spawn, or change its own ε.
    Those decisions belong to the lattice (Gilgamesh).

    State: C on S³
    One operation: ingest(q) → compose, check closure, maybe emit
    Reports: gap (σ), prediction (C⁻¹), vote (C_before⁻¹)
    """

    def __init__(self, epsilon=0.15, parent=None, on_emit=None):
        self.C = identity()
        self.epsilon = epsilon
        self.parent = parent
        self.on_emit = on_emit
        self.emission_count = 0
        self.event_count = 0
        self.scope_event_count = 0

    @property
    def gap(self):
        """σ(C) — geodesic distance from identity.  How far from closure."""
        return sigma(self.C)

    @property
    def prediction(self):
        """C⁻¹ — the quaternion that would close the current composition.
        The downward signal in the gray game: what the kernel expects next."""
        return inverse(self.C)

    def ingest(self, q):
        """One perturbation. The entire kernel step.

        Args: q — a quaternion on S³ (the event)
        Returns: ('open', vote) or ('closure', vote)
            vote = C_before⁻¹ — what the event's position "should be"
                   in this context. The adapter uses this for learning.
        """
        C_before = self.C.copy()
        content_event_count = self.scope_event_count

        # COMPOSE
        self.C = compose(self.C, q)
        self.event_count += 1

        # VOTE (returned to the adapter for position learning)
        vote = inverse(C_before)

        # CHECK CLOSURE
        if sigma(self.C) < self.epsilon:
            content = C_before.copy()
            self.emission_count += 1
            self.C = identity()
            self.scope_event_count = 0

            if self.on_emit:
                self.on_emit(content, 'closure', content_event_count)

            # Emit upward ONLY if the content is meaningful
            if self.parent and sigma(content) > self.epsilon:
                self.parent.ingest(content)

            return 'closure', vote

        self.scope_event_count += 1
        return 'open', vote

    def force_emit(self, reason='boundary'):
        """Emit the current scoped content because the adapter declared a boundary."""
        content = self.C.copy()
        content_event_count = self.scope_event_count
        if content_event_count == 0:
            return None

        self.emission_count += 1
        self.C = identity()
        self.scope_event_count = 0

        if self.on_emit:
            self.on_emit(content, reason, content_event_count)

        if self.parent and sigma(content) > self.epsilon:
            self.parent.ingest(content)

        return content

    def reset(self):
        """Reset to identity. The lattice calls this, not the kernel itself."""
        self.C = identity()
        self.scope_event_count = 0
