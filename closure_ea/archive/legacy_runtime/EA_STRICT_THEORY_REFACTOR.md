# Closure Ea — Strict Theory Refactor

## THE RULE

No patches. No backwards compatibility. No halfway architecture presented as complete.

This document describes the current strict target. Code that contradicts it is wrong.

## The Trinity

The top-level learning loop is the Holy Trinity:

    Reality → S1 × S3 → Perception → S2 × S3 → Prediction → S1 → Reality

- `S1` is the boundary interface. It is where the adapter lives.
- `S2` is the evaluation lattice. It discovers structure and emits hierarchy.
- `S3` is the memory lattice. It accumulates corrections across timescales.

The trinity structure at the top is recursive in itself. The Cell remains the basic learning unit. The Lattice remains a lattice of learning units. The same algebra recurs fractally, but the top S1/S2/S3 loop gives the directional learning cycle.

## What is implemented now

### `kernel.py`

The kernel remains the algebraic core: compose, gap, prediction, emit.

### `trinity.py`

`Trinity` is now the canonical S1/S2/S3 loop.

- `S1` uses adapter-owned embedding/decoding at the boundary.
- `S2` is a real `Lattice`.
- `S3` is a real `Lattice` fed buffered scope error.

What S3 does in the current code:
- per-tick prediction error is buffered in `_pending_errors`
- when `S2` closes a scope, those errors are composed into one `scope_error`
- that consolidated `scope_error` is what gets ingested by `S3`
- `S3` therefore learns on delayed, stabilized correction rather than raw tick noise

### Memory ownership

The learned event positions are still stored in `adapter.genome`.

`Trinity.positions` is currently a view onto that adapter-owned store, not a
separate memory object. The important architectural point is therefore not
"Trinity owns a separate genome object," but:

- the write timing is controlled by `Trinity`
- the correction path is controlled by `Trinity`
- the adapter remains the concrete storage surface for learned positions

So the intended logic is already directional, but the storage boundary is still
adapter-backed in the current implementation.

### Buffer

The learning buffer is real.

- Tick N stores a prediction.
- Tick N+1 compares reality against that stored prediction.
- The resulting per-tick error is buffered.
- When `S2` closes, buffered errors are consolidated into one scope-level correction.
- That scope-level correction is what writes into `S3` and nudges event positions.

Without that buffer there is no learning loop.

## Music track status

`train-bach` is now the canonical Trinity training path.

It:
- runs Bach through Trinity
- saves learned bar positions to `gilgamesh_bach_genome.json`
- saves Trinity state to `gilgamesh_bach_trinity.json`
- writes Trinity metrics into the Bach audit

`retrieve-bach`, `generate-bach`, `section-bach`, and `improvise-bach` still operate on the learned Bach genome. They are not yet direct decode-from-prediction paths, so the music track is improved but not yet fully generation-native under Trinity.

## Track compliance

- `music/`
  Training is now on the Trinity path. Generation still uses learned-position retrieval/beam logic and remains partially non-compliant.
- `walking/`
  Non-compliant.

## Remaining work

1. Add direct adapter decode paths so prediction can become substrate output without falling back to retrieval. The Rust `score_vocabulary` function already does this — it scores all genome positions against a target quaternion. This needs to be wired as `adapter.decode(prediction)` in Trinity.

2. Decide whether memory storage should remain adapter-backed or move into a stricter Trinity-owned structure. Right now the timing logic is in Trinity but the stored positions still live in `adapter.genome`.

3. Information-weighted scope consolidation. Trinity's `_consolidate_scope` currently gives equal weight to every tick's error. The old curriculum approach weighted rare events stronger to prevent high-frequency patterns from pulling everything toward identity (pole collapse). The same weighting should apply when composing scope error before writing to S3.

4. Multi-scale prediction consumption. The lattice hierarchy already produces predictions at every level (level 0 = local, level 1+ = larger scopes). Currently only level 0's prediction is compared against reality. Higher-level predictions should bias the scope correction or the decode selection.

5. Batch pre-training path. Trinity's online buffer-compare loop is the right architecture for learning, but for large corpora the Rust batch functions (`curriculum_votes`, `refinement_votes`) can initialize the genome orders of magnitude faster. These should become Trinity's pre-training pass — same algebra, batch-parallelized, before the online loop refines through prediction error.

6. Rewrite the walking track on Trinity.

Note: the Mirror (`mirror.py`) is NOT a separate step.  S2 IS the evaluator.  S2's hierarchy at higher levels IS the self-model — patterns in the system's own evaluations emerge from S2's closure cadence at level 1+.  A separate mirror class is redundant when S2 is a full Lattice.

## Final alignment audit

### Aligned with theory

| File | Status | Notes |
|------|--------|-------|
| `kernel.py` | **CLEAN** | Compose, check, emit, report. No lifecycle. No teaching. |
| `trinity.py` | **CLEAN** | S1=adapter, S2=Lattice(ε=normal), S3=Lattice fed scope-level buffered error. Same kernel everywhere. Buffer-compare-write cycle. |
| `lattice.py` | **CLEAN** | Composes, manages hierarchy, handles death. Does NOT teach. Teaching was removed — only Trinity handles learning. |
| `cell.py` | **CLEAN** | Cell + adapter. Does NOT teach. Teaching was removed. Cell is the cell. Trinity is the learning unit. |
| `genome.py` | **CLEAN** | Position persistence. Used by Trinity's save/load. |

### Not aligned — flagged

| File | Issue |
|------|-------|
| `mirror.py` | Dead code. S2's hierarchy IS the self-model. Mirror is not imported by any track. Kept for potential cross-substrate use but not part of the Trinity architecture. |
| `__init__.py` | **UPDATED** — Mirror removed. Exports Lattice, ClosureEvent, LatticeCell. |
| `music/` | Training path uses Trinity (Codex wired it). Generation still uses old beam search, not prediction-from-Trinity. |
| `walking/` | Not on Trinity. |

### What was removed

- `adapter.teach()` calls from `lattice.py` and `cell.py` — the old learning path bypassed the Trinity cycle. Removed. The only write path to the genome is now through Trinity's buffer-compare-write.
- MemoryLattice / MemorySphere custom classes — replaced by feeding `S3` through the same `Lattice` class used by `S2`, with slower buffered scope-error timing rather than a separate memory mechanism.
- Music-specific metadata from `lattice.py` — replaced with generic propagation.
- Death logic from `kernel.py` — moved to the Lattice (apoptosis by lattice, not by cell).
- "Gilgamesh" / "Enkidu" double-namespace with the SDK layer — renamed to Lattice / Cell in the Ea layer. The SDK classes in `closure_sdk/canon.py` are untouched.

### No gaslighting remaining in the core

The core files (kernel, cell, lattice, trinity, genome) are aligned with the current refactor target. There is one buffered write path to learned positions (Trinity's cycle). There is one kernel (47 lines). S2 and S3 run the same lattice machinery, but S3 is currently driven by consolidated scope error and still stores learned positions through the adapter surface. No backwards compatibility. No special memory class.

The tracks are NOT yet fully on Trinity. The doc says so explicitly. That is honest, not gaslighting.
