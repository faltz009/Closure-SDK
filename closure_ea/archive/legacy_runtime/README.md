# Closure EA Runtime

Closure EA is the learning-runtime layer of the project.

It uses the same `S^3` algebra as the SDK, but for adaptive sequence learning
instead of static or streaming discrepancy diagnosis.

The core runtime is:

- `kernel.py` — Pure closure recurrence on `S^3`.
- `cell.py` — One adaptive closure cell: adapter + kernel + local genome.
- `lattice.py` — Multi-level lattice of closure cells. Every closure event becomes a first-class lattice cell with ancestry and metadata.
- `trinity.py` — The Holy Trinity. S1/S2/S3 learning loop. S1 is the adapter boundary, S2 is the evaluation lattice, S3 is the memory lattice. Same kernel everywhere, only ε differs.
- `genome.py` — Persistent learned structure saved to disk.

## How It Relates To The SDK

Closure EA is wired to the SDK at the algebra layer.

It uses the same sphere operations:

- `compose`
- `inverse`
- `sigma`
- `identity`

Those live in `kernel.py` and are backed by the Rust sphere implementation
through `closure_rs` when the Rust extension is available.

Closure EA does not use the SDK's CLI diagnosis surface for learning tracks.
It uses the shared `S^3` engine and builds adaptive runtime structure on top of
it.

## Runtime Shape

The runtime is the Holy Trinity:

    Reality → S1 × S3 → Perception → S2 × S3 → Prediction → S1 → Reality

- S1 is the boundary interface where the adapter lives.
- S2 is the evaluation lattice. It discovers structure and emits hierarchy.
- S3 is the memory lattice. It accumulates corrections across timescales.

At the lowest layer:

- an adapter embeds substrate events into `S^3`
- a kernel composes those events and measures gap from identity
- the buffer stores predictions and compares against reality next tick

At higher layers:

- emitted closure objects become new events
- the lattice records those closures as explicit lattice cells
- the same recurrence repeats at larger scopes

The hierarchy in both S2 and S3 emerges from closure cadence. No special
classes, no hardcoded tiers.

## Tracks

Tracks apply the same core to different substrates:

- `tracks/music/` — Bach corpus, polyphonic composition, hierarchical learning
- `tracks/walking/` — 3D body in PyBullet, balance, walking, navigation

Each track changes the adapter and the data, not the core recurrence.
