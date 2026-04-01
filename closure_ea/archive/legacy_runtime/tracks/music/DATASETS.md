# Music Datasets

This document describes the corpora used by the music track and the saved
training artifacts produced from them.


## Corpus Layers

The track uses two main corpus layers.

### Bach working corpus

Path:

- [corpus_cache](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/corpus_cache)

Inventory:

- `78` Bach `**kern` scores
- `15` inventions
- `15` sinfonias
- `48` WTC fugues

This corpus supports:

- cue retrieval
- short continuation
- section continuation
- solo variation and improvisation
- explicit song-lattice export

### Scaled symbolic corpus

Path:

- [external_corpora/humdrum-data](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/external_corpora/humdrum-data)

Source:

- `https://github.com/humdrum-tools/humdrum-data`

Observed local inventory:

- about `19,285` `.krn` files
- about `352MB` on disk

This reservoir includes keyboard music, chorales, chamber works, early music,
folk song corpora, chant, and other symbolic collections.


## Download

```bash
cd closure_ea/tracks/gilgamesh-learns-music/external_corpora/humdrum-data
make
```


## Collections Used For The First Classical Scale-Up

The first large training run selects:

- `bach`
- `beethoven`
- `chopin`
- `haydn`
- `mozart`
- `scarlatti`
- `joplin`
- `hummel`
- `corelli`
- `vivaldi`

Approximate downloaded counts for several useful collections:

- `bach`: `728`
- `beethoven`: `174`
- `chopin`: `696`
- `corelli`: `250`
- `haydn`: `259`
- `hummel`: `24`
- `joplin`: `47`
- `mozart`: `152`
- `scarlatti`: `65`
- `vivaldi`: `46`
- `songs/harmonized`: `265`
- `early-music`: `2113`
- `polish`: `1934`
- `songs/folksongs`: `8473`
- `songs/chant`: `2055`
- `songs/pop`: `105`


## Best Compatible Symbolic Families

The parser path is strongest on symbolic collections that are:

- already in `**kern`
- score-structured
- polyphonic in a manageable way
- close to keyboard or tightly notated ensemble writing

The best near-term expansion targets are:

- `bach/`
- `chopin/`
- `beethoven/piano/`
- `haydn/piano/`
- `mozart/piano/sonata/`
- `scarlatti/sonata/`
- `joplin/`


## Trained Bach Genome

Operational command:

- `gilgamesh_music.py train-bach`

Saved files:

- [gilgamesh_bach_genome.json](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_bach_genome.json)
- [gilgamesh_bach_audit.txt](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_bach_audit.txt)

Recorded result:

- `76` pieces
- `3861` bars
- `5237` saved objects
- kind counts:
  - `3861` bars
  - `992` segments
  - `308` higher objects
  - `76` pieces


## Trained Classical Genome

Operational commands:

- `gilgamesh_music.py train-classical`
- `gilgamesh_music.py benchmark-classical`

Saved files:

- [gilgamesh_classical_manifest.json](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_classical_manifest.json)
- [gilgamesh_classical_genome.json](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_classical_genome.json)
- [gilgamesh_classical_audit.txt](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_classical_audit.txt)
- [gilgamesh_classical_composer_audit.txt](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_classical_composer_audit.txt)

Recorded result:

- `2441` candidate files scanned
- `1069` compatible pieces trained
- `54959` bars
- `386786` slices
- `74162` saved genome objects
- kind counts:
  - `54959` bars
  - `13820` segments
  - `4314` higher objects
  - `1069` pieces

Compatible composer-family counts from the saved audit:

- `bach`: `524`
- `beethoven`: `4`
- `chopin`: `27`
- `corelli`: `248`
- `haydn`: `190`
- `hummel`: `9`
- `joplin`: `4`
- `mozart`: `63`

Collections with weak compatibility in this first pass are parser data:

- most `scarlatti`
- most `vivaldi`
- most of `beethoven`
- most of `chopin`
- some `haydn`
- some `mozart`


## Composer Benchmark

The classical benchmark uses composer names only for evaluation after training.

Saved audit:

- [gilgamesh_classical_composer_audit.txt](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_classical_composer_audit.txt)

Recorded results:

- piece objects: `1069`
- top-1 same-composer nearest neighbors: `343 / 1069`
- balanced cue sample: `56`
- cue composer accuracy: `56 / 56`
- cue exact-piece accuracy: `55 / 56`

This benchmark shows:

- cue-level identity separates composer families clearly
- exact piece retrieval remains strong on the balanced sample
- whole-piece nearest-neighbor geometry is looser than cue geometry


## Practical Reading Of The Data Layer

The music track has:

- a compact Bach corpus for intensive retrieval and generation work
- a larger symbolic reservoir for scaling
- a trained Bach genome
- a trained multi-composer classical genome

The next data growth step is choosing which symbolic families to ingest next and
which new parser paths to add beyond Humdrum `**kern`.
