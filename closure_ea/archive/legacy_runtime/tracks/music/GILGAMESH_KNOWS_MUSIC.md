# Gilgamesh Knows Music

This document describes the music track as a technical system.

It covers:

- how symbolic music is decomposed into `S^3`
- how Gilgamesh organizes that stream as a literal closure lattice
- how the learned genome is formed and saved
- how the hierarchy is distributed across bars, segments, higher objects, and pieces
- which trained artifacts and example outputs live on disk


## Track Layout

The music track lives in:

- [gilgamesh-learns-music](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music)

Main files:

- [gilgamesh_music.py](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py)
  Canonical operational entrypoint for training, retrieval, generation,
  improvisation, classical scale-up, benchmarking, and explicit lattice export.
- [bach_generic.py](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/bach_generic.py)
  Symbolic music parser and polyphonic embedding path for Humdrum `**kern`.
- [README.md](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/README.md)
  Folder-level usage guide.
- [DATASETS.md](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/DATASETS.md)
  Corpus layout and trained data inventory.

Data and outputs:

- [corpus_cache](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/corpus_cache)
  Bach working corpus.
- [external_corpora](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/external_corpora)
  Larger symbolic corpus reservoir.
- [output](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output)
  Saved genomes, audits, renders, and lattice artifacts.
- [dev](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/dev)
  Exploratory and historical step scripts.


## Core Idea

The music track treats symbolic music as ordered structure on `S^3`.

Each bounded musical event is embedded as a quaternion.
Those quaternions are composed through time.
When a composed span resolves under the closure rule, Gilgamesh emits a larger
musical object.
That emitted object is itself a quaternion on `S^3`, so the same algebra
repeats at the next scale.

This makes the hierarchy recursive:

- note relations live on `S^3`
- slices live on `S^3`
- bars live on `S^3`
- segments live on `S^3`
- higher phrase and section objects live on `S^3`
- piece identities live on `S^3`

The lattice is therefore a sphere of spheres: each level emits stable musical
objects, and those objects become the events for the layer above.


## Strict Core Contract

The music track now sits on the stricter core surface directly.

What matters operationally:

- Gilgamesh emits structured [ClosureEvent](/home/faltz009/Closure-SDK/closure_ea/gilgamesh.py) objects, and the music recorders consume that event shape directly.
- The canonical entrypoint in [gilgamesh_music.py](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py) uses `gilgamesh.ingest(...)` for its lattice side effects and does not depend on unpacking a legacy two-value return.
- Piece, bar, composer, and related music metadata now survive through the core's generic metadata aggregation instead of a music-specific path in `gilgamesh.py`.
- Core prediction (`C^-1`) and death tracking are now available in the runtime, but the canonical music generation path still uses its explicit retrieval and beam-search logic rather than a generic core generator.

This means the music track remains valid under the strict-theory refactor without needing a music-specific compatibility shim in the core.

## Runtime Objects

The runtime has three important kinds of objects.

### 1. Leaf nodes

Every ingested event becomes a leaf node in Gilgamesh.

For the music track, those leaves are usually bar-level events, because the
parser first decomposes the score into notes and slices and then composes those
into bar identities.

Each leaf node carries:

- a runtime id
- its source key
- its level
- its metadata

### 2. Lattice cells

Every closure event becomes a first-class lattice cell in the core runtime.

The core type is [LatticeCell](/home/faltz009/Closure-SDK/closure_ea/gilgamesh.py#L28).

Each lattice cell carries:

- `id`
- `level`
- `q`
- `reason`
- `event_count`
- `child_ids`
- `trigger_ids`
- `source_keys`
- `trigger_source_keys`
- `meta`

So a closure is not a transient callback.
It is a persistent object with ancestry and scope.

### 3. Genome objects

The genome is the saved long-term memory of the track.

The music pipeline records learned positions for:

- bars
- segment cells
- higher cells
- piece cells

Each saved position contains:

- quaternion
- usage count
- lock state
- metadata

The persistence layer lives in [genome.py](/home/faltz009/Closure-SDK/closure_ea/genome.py).


## Step-By-Step Decomposition Of A Score

The operational path begins in [bach_generic.py](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/bach_generic.py) and continues through [gilgamesh_music.py](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py).

### 1. Parse the symbolic score

The parser reads Humdrum `**kern` into note events.

Each note event captures:

- voice
- MIDI pitch
- onset
- duration

This produces a time-organized note stream for each voice.

### 2. Build slices

The parser merges simultaneous voice activity into slice events.

A slice is a vertical musical moment:

- which voices are active
- which pitches are sounding
- how long the moment lasts
- where it sits in the bar

This gives the first musically meaningful stream:

> score -> note events -> slice events

### 3. Embed each note in `S^3`

Each note is embedded as a quaternion.

The embedding uses musical axes such as:

- pitch class by fifths
- pitch class by chromatic position
- octave or register
- voice placement

This places a note geometrically instead of tokenizing it.

### 4. Embed each slice in `S^3`

The slice embedding composes:

- its time-within-bar
- its duration
- all active voice-note embeddings
- rest embeddings for silent voices

So a slice quaternion carries:

- pitch content
- voicing
- silence structure
- local temporal placement

### 5. Compose a bar

A bar is the ordered product of its slices:

> `bar = slice_1 · slice_2 · ... · slice_n`

The bar quaternion is the displacement produced by the ordered inner life of
that bar.

This is an important point:

- a bar is not a bag of notes
- a bar is an ordered directional object
- the order of slices matters
- the resulting quaternion is a local musical identity

### 6. Feed bars into Gilgamesh

The music track feeds these bar identities into Gilgamesh as the first stable
musical units in the lattice runtime.

The runtime then performs repeated closure:

- bars compose into segments
- segments compose into higher objects
- higher objects compose into piece identities

The decomposition ladder is:

> score -> notes -> slices -> bars -> segments -> higher objects -> piece identity


## How The Literal Lattice Works

The runtime lives in [gilgamesh.py](/home/faltz009/Closure-SDK/closure_ea/gilgamesh.py).

Gilgamesh maintains a stack of kernels, one per active scope level.
Each kernel composes events on `S^3`.
When closure happens, the emitted span becomes a lattice cell and is queued as
input to the next level.

### Kernel recurrence

At every level, the recurrence is the same:

1. keep a running product `C`
2. compose the next quaternion into `C`
3. measure the gap `σ(C)` from identity
4. emit when closure or boundary firing occurs
5. start the next scope

That same recurrence drives every musical scale in the track.

### Boundaries

The music track uses minimal explicit structure:

- event order
- bar boundary
- piece boundary

The kernel also supports forced emission on declared boundaries through
[force_emit](/home/faltz009/Closure-SDK/closure_ea/kernel.py#L102).

This gives the lattice the minimum structure it needs to keep scopes aligned
with the symbolic substrate while still discovering the larger objects through
closure.

### Cell formation

When a closure happens:

- the runtime gathers the emitted child units
- computes aggregated metadata
- creates a `LatticeCell`
- stores ancestry links
- queues the new cell for the layer above

This is how the song becomes an explicit closure graph rather than only a
running stream.


## Musical Meaning On The Hopf Fiber

The quaternion carries musical information in both scalar and vector parts.

### Scalar part `W`

`W` measures closeness to identity.

In this track, higher `W` energy tends to correlate with:

- coherence
- return
- resolution
- near-identity cadential material

This showed up clearly in Bach, where recapitulatory or cadential spans landed
near identity more often than ordinary bars.

### Vector part `RGB`

The vector part carries displacement direction.

The working interpretation used in the music track is:

- `i` axis: fifths and harmonic displacement
- `j` axis: chromatic displacement
- `k` axis: octave, register, and temporal displacement

This means a musical object is not only “close” or “far” from identity.
It also points somewhere specific in musical space.

### Bars as displacement

Bars usually act as displacement objects rather than closure objects.

That is why they work so well for retrieval:

- they are distinctive
- they encode ordered local motion
- they can be compared directly as learned identities

### Higher closure over displacement

Once bars are treated as stable displacement objects, the higher levels can
close over those displacements:

- a span of bars can resolve into a segment
- a span of segments can resolve into a higher section
- the whole piece can resolve into a piece identity

That recursive closure over displacement is the musical hierarchy.


## Fractal Organization

The hierarchy is fractal because the same pattern repeats at each scale:

- lower events compose
- closure emits a higher object
- that object becomes a new event
- the same recurrence continues above it

In music terms:

- notes form slices
- slices form bars
- bars form segments
- segments form higher objects
- higher objects form piece identities

Each layer is more integrated than the one beneath it.


## Viewer And Song Lattice

The track includes a song viewer and explicit lattice export.

Relevant artifacts:

- [bwv772_song_viewer.html](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/bwv772_song_viewer.html)
- [inven01_music_evolved_lattice.json](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/inven01_music_evolved_lattice.json)
- [inven01_music_evolved_lattice_audit.txt](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/inven01_music_evolved_lattice_audit.txt)

The viewer presents the song as nested rings:

- outer ring: raw slice path
- next ring: bar identities
- next ring: segment identities
- center: piece identity

The evolved lattice export presents the same hierarchy as a graph:

- leaves for the lower musical units
- closure cells above them
- higher cells above those
- piece identity at the top

For `inven01`, the explicit lattice export contains:

- `335` slice nodes
- `83` segment-level emitted nodes
- `26` level-2 nodes
- `21` level-3 nodes
- `20` level-4 nodes
- `1` piece node

This gives a concrete picture of how a single piece becomes a hierarchy of
closure objects.


## Trained Genomes

### Bach genome

Saved file:

- [gilgamesh_bach_genome.json](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_bach_genome.json)

Audit:

- [gilgamesh_bach_audit.txt](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_bach_audit.txt)

Recorded statistics:

- `76` pieces
- `3861` bars
- `5237` saved objects
- hierarchy: `4` levels
- kind counts:
  - `3861` bars
  - `992` segments
  - `308` higher objects
  - `76` pieces

### Classical genome

Saved files:

- [gilgamesh_classical_genome.json](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_classical_genome.json)
- [gilgamesh_classical_manifest.json](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_classical_manifest.json)

Audits:

- [gilgamesh_classical_audit.txt](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_classical_audit.txt)
- [gilgamesh_classical_composer_audit.txt](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_classical_composer_audit.txt)

Recorded statistics:

- `2441` candidate scores scanned
- `1069` compatible pieces trained
- `54959` bars
- `386786` slices
- `74162` saved objects
- hierarchy: `4` levels
- kind counts:
  - `54959` bars
  - `13820` segments
  - `4314` higher objects
  - `1069` pieces

Cue-level benchmark results from the saved audit:

- composer-family accuracy: `56 / 56`
- exact-piece accuracy: `55 / 56`


## Demonstrated Capabilities

The track demonstrates five concrete capabilities.

### 1. Learned cue retrieval

Short cues recover learned musical identity from the genome.

### 2. Persistent hierarchy

The genome stores bars, segments, higher objects, and pieces as learned musical
objects.

### 3. Short continuation

The lattice can continue a learned musical field without exact replay.

### 4. Moderated improvisation

The track produces Bach-like variations that preserve learned identity while
altering the realized line.

Recommended listening examples:

- [inven01_solo_cue.wav](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/inven01_solo_cue.wav)
- [inven01_solo_reference.wav](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/inven01_solo_reference.wav)
- [inven01_solo_variant1.wav](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/inven01_solo_variant1.wav)
- [inven01_solo_moderated_variant1.wav](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/inven01_solo_moderated_variant1.wav)

### 5. Multi-composer classical learning

The same lattice and genome format support a larger classical run across many
pieces and multiple composer families.


## Operational Commands

Run the track through:

- [gilgamesh_music.py](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py)

Commands:

```bash
python3 closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py train-bach
python3 closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py retrieve-bach
python3 closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py generate-bach
python3 closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py section-bach
python3 closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py improvise-bach
python3 closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py train-classical
python3 closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py benchmark-classical
python3 closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py evolved-lattice
```


## Working Summary

The music track is a literal closure lattice over symbolic music:

- symbolic scores are decomposed into note events and slice events
- slices compose into bar identities on `S^3`
- bars enter Gilgamesh as the first stable musical units
- every closure event becomes a first-class lattice cell
- cells compose upward into segments, higher objects, and piece identities
- the resulting learned objects are saved as a persistent genome
- cues, continuation, improvisation, and larger-corpus training all operate on
  that same learned field

That is the operational meaning of “Gilgamesh knows music” in this folder.
