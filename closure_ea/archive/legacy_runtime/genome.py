"""
The genome. What the lattice has learned.

Persistent structure that survives across sessions. The lattice's
internal truth — positions on S³ that represent everything the
lattice has experienced and compressed through closure.

Like DNA: compressed instructions that the lattice carries forward.
They determine how new events are processed. They change slowly
while individual cells live and die quickly.

The genome stores:
    - positions: event_key → (quaternion, count, locked)
    - hierarchy: how many levels emerged, closure counts per level
    - substrate: which adapter produced this genome

The genome does NOT store:
    - raw events (those are consumed by composition)
    - running state (that's the kernel's job, ephemeral)
    - adapter logic (that's substrate-specific)

Save/load: JSON. The genome IS the model file.
"""

import numpy as np
import json
import math
from pathlib import Path

from closure_ea.kernel import sigma, identity


class Genome:
    """What the lattice knows. Persists across sessions."""

    def __init__(self, substrate_name='unknown'):
        self.substrate = substrate_name
        self.positions = {}    # key → {'q': [4 floats], 'count': int, 'locked': bool}
        self.hierarchy = {     # what the hierarchy looked like
            'levels': 0,
            'closures_per_level': [],
            'total_events': 0,
        }

    def record_position(self, key, quaternion, count, locked=False, meta=None):
        """Store a learned position."""
        self.positions[key] = {
            'q': quaternion.tolist() if hasattr(quaternion, 'tolist') else list(quaternion),
            'count': count,
            'locked': locked,
        }
        if meta is not None:
            self.positions[key]['meta'] = meta

    def get_position(self, key):
        """Retrieve a position. Returns (quaternion, count, locked) or None."""
        if key not in self.positions:
            return None
        p = self.positions[key]
        return np.array(p['q']), p['count'], p['locked']

    def get_meta(self, key):
        """Retrieve stored metadata for a position, if present."""
        if key not in self.positions:
            return None
        return self.positions[key].get('meta')

    def record_hierarchy(self, levels, closures_per_level, total_events):
        """Store the hierarchy state."""
        self.hierarchy = {
            'levels': levels,
            'closures_per_level': list(closures_per_level),
            'total_events': total_events,
        }

    @property
    def size(self):
        """Number of learned positions."""
        return len(self.positions)

    @property
    def spread(self):
        """σ statistics of stored positions."""
        if not self.positions:
            return 0, 0, 0
        sigmas = [sigma(np.array(p['q'])) for p in self.positions.values()]
        return min(sigmas), max(sigmas), sum(sigmas) / len(sigmas)

    def save(self, path):
        """Save genome to JSON."""
        data = {
            'substrate': self.substrate,
            'positions': self.positions,
            'hierarchy': self.hierarchy,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path):
        """Load genome from JSON."""
        data = json.loads(Path(path).read_text())
        g = cls(data.get('substrate', 'unknown'))
        g.positions = data.get('positions', {})
        g.hierarchy = data.get('hierarchy', {
            'levels': 0, 'closures_per_level': [], 'total_events': 0
        })
        return g

    def __repr__(self):
        mn, mx, mean = self.spread
        return (f"Genome(substrate={self.substrate}, "
                f"positions={self.size}, "
                f"σ=[{mn:.3f}, {mx:.3f}, mean={mean:.3f}], "
                f"hierarchy={self.hierarchy})")


def snapshot_from_adapter(adapter, substrate_name='unknown'):
    """Create a genome snapshot from an adapter's current state."""
    g = Genome(substrate_name)
    for key, (pos, count) in adapter.genome.items():
        locked = count == -1
        meta = None
        if hasattr(adapter, 'position_meta_for'):
            meta = adapter.position_meta_for(key)
        elif hasattr(adapter, 'position_meta'):
            meta_store = getattr(adapter, 'position_meta')
            if isinstance(meta_store, dict):
                meta = meta_store.get(key)
        g.record_position(key, pos, abs(count), locked, meta=meta)
    return g


def snapshot_from_lattice(lattice, adapter, substrate_name='unknown'):
    """Create a genome snapshot from a Lattice + adapter."""
    g = snapshot_from_adapter(adapter, substrate_name)
    active_levels = [k for k in lattice.kernels if k.event_count > 0]
    g.record_hierarchy(
        levels=len(active_levels),
        closures_per_level=[k.emission_count for k in active_levels],
        total_events=lattice.kernels[0].event_count if lattice.kernels else 0,
    )
    return g


def load_genome_into_adapter(genome, adapter):
    """Restore a genome's positions into an adapter."""
    for key, data in genome.positions.items():
        q = np.array(data['q'])
        if data['locked']:
            adapter.embed_exact(key, q)
        else:
            adapter.genome[key] = (q, data['count'])
    return adapter
