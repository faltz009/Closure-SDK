# Gilgamesh Learns Music

This folder contains the operational music track for the Closure EA core.

Top-level files and folders:

- [gilgamesh_music.py](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/gilgamesh_music.py)
  Canonical entrypoint for training, retrieval, generation, improvisation,
  classical scale-up, benchmarking, and explicit lattice export.
- [bach_generic.py](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/bach_generic.py)
  Humdrum `**kern` parser and polyphonic embedding helper.
- [GILGAMESH_KNOWS_MUSIC.md](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/GILGAMESH_KNOWS_MUSIC.md)
  Technical documentation for the music lattice, genome, hierarchy, and saved
  artifacts.
- [DATASETS.md](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/DATASETS.md)
  Corpus inventory and trained data layout.
- [song_viewer_template.html](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/song_viewer_template.html)
  HTML viewer shell for song hierarchy and lattice artifacts.
- [corpus_cache](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/corpus_cache)
  Bach working corpus.
- [external_corpora](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/external_corpora)
  Larger symbolic corpus reservoir.
- [output](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output)
  Saved genomes, manifests, audits, renders, and lattice exports.
- [dev](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/dev)
  Exploratory and historical scripts.

## Run

From the repo root:

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

## Saved Models

- [gilgamesh_bach_genome.json](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_bach_genome.json)
- [gilgamesh_classical_genome.json](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_classical_genome.json)
- [gilgamesh_classical_manifest.json](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output/gilgamesh_classical_manifest.json)

## Start Here

1. Read [GILGAMESH_KNOWS_MUSIC.md](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/GILGAMESH_KNOWS_MUSIC.md).
2. Read [DATASETS.md](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/DATASETS.md).
3. Inspect the saved audits in [output](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output).
4. Listen to the rendered Bach examples in [output](/home/faltz009/Closure-SDK/closure_ea/tracks/gilgamesh-learns-music/output).
