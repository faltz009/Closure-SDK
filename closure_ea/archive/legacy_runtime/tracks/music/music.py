"""
Canonical music track for Gilgamesh.

This file is the durable operational surface for the music track:
    - train a Bach genome
    - retrieve from Bach cues
    - generate short Bach continuation
    - generate Bach sections
    - improvise a moderated Bach solo
    - train a larger classical genome
    - benchmark composer emergence
    - build an explicit evolved lattice for one song

Historical phase/step experiments belong in `dev/`.
This file is the real track entrypoint.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRACK_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from closure_ea.cell import Adapter  # noqa: E402
from closure_ea.genome import Genome  # noqa: E402
from closure_ea.lattice import ClosureEvent, Lattice  # noqa: E402
from closure_ea.kernel import RUST, compose, identity, inverse, sigma  # noqa: E402
from closure_ea.trinity import Trinity  # noqa: E402
from bach_generic import (  # noqa: E402
    GenericPolyphonicAdapter,
    SliceEvent,
    compose_bar,
    event_label,
    group_bars,
    parse_kern_score,
    write_polyphonic_wav,
)


CORPUS_DIR = TRACK_DIR / "corpus_cache"
OUTPUT_DIR = TRACK_DIR / "output"
EXTERNAL_CORPORA_DIR = TRACK_DIR / "external_corpora" / "humdrum-data"

BACH_GENOME_PATH = OUTPUT_DIR / "gilgamesh_bach_genome.json"
BACH_AUDIT_PATH = OUTPUT_DIR / "gilgamesh_bach_audit.txt"
BACH_RETRIEVAL_AUDIT_PATH = OUTPUT_DIR / "step5_gilgamesh_cue_retrieval_audit.txt"
BACH_GENERATION_AUDIT_PATH = OUTPUT_DIR / "step5_gilgamesh_bach_generation_audit.txt"
BACH_SECTION_AUDIT_PATH = OUTPUT_DIR / "step5_gilgamesh_bach_section_generation_audit.txt"
BACH_SOLO_AUDIT_PATH = OUTPUT_DIR / "step5_gilgamesh_bach_solo_improv_moderated_audit.txt"
BACH_TRINITY_STATE_PATH = OUTPUT_DIR / "gilgamesh_bach_trinity.json"
BACH_TRINITY_AUDIT_PATH = OUTPUT_DIR / "gilgamesh_bach_trinity_audit.txt"

CLASSICAL_MANIFEST_PATH = OUTPUT_DIR / "gilgamesh_classical_manifest.json"
CLASSICAL_GENOME_PATH = OUTPUT_DIR / "gilgamesh_classical_genome.json"
CLASSICAL_AUDIT_PATH = OUTPUT_DIR / "gilgamesh_classical_audit.txt"
CLASSICAL_COMPOSER_AUDIT_PATH = OUTPUT_DIR / "gilgamesh_classical_composer_audit.txt"

EVOLVED_LATTICE_JSON = OUTPUT_DIR / "inven01_music_evolved_lattice.json"
EVOLVED_LATTICE_AUDIT = OUTPUT_DIR / "inven01_music_evolved_lattice_audit.txt"
EVOLVED_LATTICE_SOURCE = CORPUS_DIR / "inven01.krn"

BACH_EPSILON_SCHEDULE = [1.10, 0.60, 0.45, 0.35]
EVOLVED_LATTICE_SCHEDULE = [0.90, 0.60, 0.45, 0.35]

CLASSICAL_COLLECTIONS = [
    "bach",
    "beethoven",
    "chopin",
    "haydn",
    "mozart",
    "scarlatti",
    "joplin",
    "hummel",
    "corelli",
    "vivaldi",
]
CLASSICAL_PARSE_TIMEOUT_SECONDS = 10

BACH_RETRIEVAL_CUE_LENGTH = 4
BACH_RETRIEVAL_TEST_POSITIONS = ("start", "middle", "end")

GEN_CUE_BARS = 4
GEN_CONTEXT_BARS = 3
GEN_GENERATE_BARS = 4
GEN_TOP_CONTEXT_MATCHES = 18
GEN_BEAM_WIDTH = 14
GEN_NOVELTY_REPLAY_PENALTY = 0.20
GEN_RUNNING_CLOSURE_WEIGHT = 0.08
GEN_EXAMPLE_AUDIO_LIMIT = 3

SECTION_CUE_BARS = 4
SECTION_CONTEXT_BARS = 4
SECTION_GENERATE_BARS = 8
SECTION_TOP_CONTEXT_MATCHES = 28
SECTION_BEAM_WIDTH = 22
SECTION_NOVELTY_REPLAY_PENALTY = 0.24
SECTION_RUNNING_CLOSURE_WEIGHT = 0.08
SECTION_EXAMPLE_AUDIO_LIMIT = 3

SOLO_SOURCE_SLUG = "inven01"
SOLO_CUE_START_BAR = 6
SOLO_CUE_BARS = 4
SOLO_BARS = 8
SOLO_NUM_VARIANTS = 12
SOLO_KEEP_VARIANTS = 3
SOLO_MAX_SLICE_CANDIDATES = 18
SOLO_TOP_SAMPLE = 6
SOLO_TEMPERATURE = 0.14
SOLO_NOISE_SCALE = 0.10
SOLO_MIN_BAR_SLICE_CHANGES = 1
SOLO_MAX_BAR_SLICE_CHANGES = 3
SOLO_PIECE_IDENTITY_WEIGHT = 0.10
SOLO_SOURCE_CANDIDATE_BONUS = -0.10
SOLO_NONSOURCE_CANDIDATE_PENALTY = 0.16

CLASSICAL_BENCHMARK_CUE_LENGTH = 4
CLASSICAL_MAX_PIECES_PER_COMPOSER = 8
CLASSICAL_BENCHMARK_SEED = 7


@dataclass(frozen=True)
class PieceData:
    slug: str
    title: str
    bars: list[list[SliceEvent]]
    raw_quats: list[np.ndarray]
    learned_quats: list[np.ndarray]

    @property
    def n_bars(self) -> int:
        return len(self.bars)


@dataclass
class Candidate:
    source_slug: str
    bar_num: int
    q: np.ndarray
    context_score: float

    @property
    def key(self) -> str:
        return f"{self.source_slug}:{self.bar_num}"


@dataclass
class BeamState:
    generated_keys: list[str]
    generated_quats: list[np.ndarray]
    generated_bars: list[list[SliceEvent]]
    score: float
    traces: list[str]


@dataclass(frozen=True)
class SliceCandidate:
    source_slug: str
    source_bar: int
    q: np.ndarray
    voices: tuple[tuple[str, tuple[int, ...]], ...]
    duration: Fraction
    occupancy: tuple[int, ...]


@dataclass
class Variant:
    seed: int
    varied_solo_bars: list[list[SliceEvent]]
    changed_slices: int
    changed_bars: int
    retrieved_piece: str
    retrieved_start_bar: int
    margin: float
    source_distance: float
    per_bar_changes: list[int]
    trace: list[str]


@dataclass
class SliceDescriptor:
    key: str
    bar: int
    start: float
    duration: float
    label: str


class ParseTimeout(RuntimeError):
    pass


@contextmanager
def time_limit(seconds: int):
    def _handler(signum, frame):
        raise ParseTimeout(f"parse timeout after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class BarLevelAdapter(Adapter):
    """Adapter for pre-composed bar quaternions."""

    def __init__(self):
        super().__init__(damping=0.05)

    def embed_bar(self, piece_id: str, bar_num: int, bar_q: np.ndarray) -> str:
        key = f"{piece_id}:{bar_num}"
        if key not in self.genome:
            self.genome[key] = (bar_q.copy(), 0)
        return key

    def embed(self, event_key):
        if event_key in self.genome:
            return self.genome[event_key][0].copy()
        return super().embed(event_key)


class HierarchicalRecorder:
    """Capture emitted higher-level objects as the lattice runs."""

    def __init__(self, genome: Genome):
        self.genome = genome
        self.global_indexes = defaultdict(int)
        self.current_piece = ""
        self.current_title = ""
        self.current_composer = ""
        self.current_relpath = ""
        self.pending_units = defaultdict(list)
        self.current_piece_emits = []
        self.current_piece_boundary_emits = []

    def start_piece(self, piece: dict) -> None:
        self.current_piece = piece["piece_id"]
        self.current_title = piece.get("title", piece["piece_id"])
        self.current_composer = piece.get("composer", "")
        self.current_relpath = piece.get("relpath", "")
        self.pending_units.clear()
        self.current_piece_emits = []
        self.current_piece_boundary_emits = []

    def note_bar(self, key: str, bar_num: int) -> None:
        self.pending_units[0].append({"key": key, "bars": [bar_num]})

    @staticmethod
    def _flatten_bars(units: list[dict]) -> list[int]:
        bars = []
        for unit in units:
            bars.extend(unit.get("bars", []))
        return sorted(set(bars))

    def on_emit(self, event: ClosureEvent) -> None:
        level = event.level - 1
        content = event.content
        reason = event.reason
        content_event_count = event.event_count
        pending_before = list(self.pending_units[level])
        emitted_units = pending_before[:content_event_count]
        trailing_units = pending_before[content_event_count:]
        self.pending_units[level] = []
        if not emitted_units:
            return

        emitted_level = level + 1
        self.global_indexes[emitted_level] += 1
        key = f"l{emitted_level}|{self.current_piece}|{self.global_indexes[emitted_level]:05d}"
        bars = self._flatten_bars(emitted_units)
        trigger_bars = self._flatten_bars(trailing_units)
        kind = "segment" if emitted_level == 1 else "higher"
        meta = {
            "kind": kind,
            "piece": self.current_piece,
            "title": self.current_title,
            "composer": self.current_composer,
            "relpath": self.current_relpath,
            "level": emitted_level,
            "reason": reason,
            "bars": bars,
            "child_keys": [unit["key"] for unit in emitted_units],
            "content_event_count": content_event_count,
        }
        if bars:
            meta["bar_start"] = bars[0]
            meta["bar_end"] = bars[-1]
        if trigger_bars:
            meta["trigger_bars"] = trigger_bars
        trigger_keys = [unit["key"] for unit in trailing_units if unit.get("key")]
        if trigger_keys:
            meta["trigger_keys"] = trigger_keys

        self.genome.record_position(key, content, 1, meta=meta)
        desc = {"key": key, "bars": bars}
        self.pending_units[emitted_level].append(desc)
        self.current_piece_emits.append((emitted_level, key, bars, reason))
        if reason != "closure":
            self.current_piece_boundary_emits.append((emitted_level, key, bars))

    def finish_piece(self) -> str | None:
        if self.current_piece_boundary_emits:
            level, source_key, bars = max(self.current_piece_boundary_emits, key=lambda item: item[0])
        elif self.current_piece_emits:
            level, source_key, bars, _ = max(self.current_piece_emits, key=lambda item: item[0])
        else:
            return None

        stored = self.genome.get_position(source_key)
        if stored is None:
            return None

        q, count, locked = stored
        self.genome.record_position(
            f"piece|{self.current_piece}",
            q,
            count,
            locked=locked,
            meta={
                "kind": "piece",
                "piece": self.current_piece,
                "title": self.current_title,
                "composer": self.current_composer,
                "relpath": self.current_relpath,
                "source_key": source_key,
                "source_level": level,
                "bars": bars,
            },
        )
        return source_key


class SliceLevelAdapter(Adapter):
    """Adapter that feeds slice quaternions into Gilgamesh."""

    def __init__(self, poly_adapter: GenericPolyphonicAdapter):
        super().__init__(damping=0.03)
        self.poly_adapter = poly_adapter
        self.slice_meta: dict[str, dict] = {}

    def register_slice(self, key: str, slice_event: SliceEvent) -> None:
        q = self.poly_adapter.embed_slice(slice_event)
        if key not in self.genome:
            self.genome[key] = (q.copy(), 0)
        self.slice_meta[key] = {
            "kind": "slice",
            "bar": slice_event.bar,
            "start": float(slice_event.start),
            "duration": float(slice_event.duration),
            "label": event_label(slice_event),
        }

    def embed(self, event_key):
        if event_key in self.genome:
            return self.genome[event_key][0].copy()
        return super().embed(event_key)


class EvolvedLatticeRecorder:
    """Explicit closure graph recorder for one song."""

    def __init__(self, piece_slug: str):
        self.piece_slug = piece_slug
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.pending_units = defaultdict(list)
        self.level_indexes = defaultdict(int)
        self.closures_per_level = Counter()
        self.current_piece_emits: list[tuple[int, str, list[int], str]] = []

    def note_slice(self, desc: SliceDescriptor) -> None:
        self.nodes[desc.key] = {
            "id": desc.key,
            "kind": "slice",
            "level": 0,
            "piece": self.piece_slug,
            "bar_start": desc.bar,
            "bar_end": desc.bar,
            "bars": [desc.bar],
            "time_start": desc.start,
            "duration": desc.duration,
            "label": desc.label,
        }
        self.pending_units[0].append({"key": desc.key, "bars": [desc.bar]})

    @staticmethod
    def _flatten_bars(units: list[dict]) -> list[int]:
        bars = []
        for unit in units:
            bars.extend(unit.get("bars", []))
        return sorted(set(bars))

    def on_emit(self, event: ClosureEvent) -> None:
        level = event.level - 1
        content = event.content
        reason = event.reason
        content_event_count = event.event_count
        pending_before = list(self.pending_units[level])
        emitted_units = pending_before[:content_event_count]
        trailing_units = pending_before[content_event_count:]
        self.pending_units[level] = []
        if not emitted_units:
            return

        emitted_level = level + 1
        self.level_indexes[emitted_level] += 1
        node_id = f"{self.piece_slug}|l{emitted_level}|{self.level_indexes[emitted_level]:05d}"
        bars = self._flatten_bars(emitted_units)
        trigger_bars = self._flatten_bars(trailing_units)
        node_kind = "segment" if emitted_level == 1 else "higher"

        self.nodes[node_id] = {
            "id": node_id,
            "kind": node_kind,
            "level": emitted_level,
            "piece": self.piece_slug,
            "reason": reason,
            "bar_start": bars[0] if bars else None,
            "bar_end": bars[-1] if bars else None,
            "bars": bars,
            "trigger_bars": trigger_bars,
            "content_event_count": content_event_count,
            "sigma": float(sigma(content)),
            "q": content.tolist(),
            "child_keys": [unit["key"] for unit in emitted_units],
        }
        for child in emitted_units:
            self.edges.append({"parent": node_id, "child": child["key"]})

        self.pending_units[emitted_level].append({"key": node_id, "bars": bars})
        self.current_piece_emits.append((emitted_level, node_id, bars, reason))
        self.closures_per_level[emitted_level] += 1

    def finalize_piece(self) -> str | None:
        if not self.current_piece_emits:
            return None
        level, source_key, bars, _reason = max(self.current_piece_emits, key=lambda item: item[0])
        piece_id = f"piece|{self.piece_slug}"
        source = self.nodes[source_key]
        self.nodes[piece_id] = {
            "id": piece_id,
            "kind": "piece",
            "level": level + 1,
            "piece": self.piece_slug,
            "reason": "piece_end",
            "bar_start": bars[0] if bars else None,
            "bar_end": bars[-1] if bars else None,
            "bars": bars,
            "sigma": source["sigma"],
            "q": source["q"],
            "source_key": source_key,
        }
        self.edges.append({"parent": piece_id, "child": source_key})
        return piece_id


def read_title(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("!!!OTL:"):
            return line.split(":", 1)[1].strip()
    return path.stem


def flatten_bars(bars: list[list[SliceEvent]]) -> list[SliceEvent]:
    return [event for bar in bars for event in bar]


def compose_sequence(quats: list[np.ndarray]) -> np.ndarray:
    q = identity()
    for part in quats:
        q = compose(q, part)
    return q / np.linalg.norm(q)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    return identity() if norm == 0 else q / norm


def mean_window_sigma(lhs: list[np.ndarray], rhs: list[np.ndarray]) -> float:
    vals = [sigma(compose(a, inverse(b))) for a, b in zip(lhs, rhs)]
    return sum(vals) / len(vals)


def count_diff(before: list[int], after: list[int]) -> list[int]:
    width = max(len(before), len(after))
    before_pad = before + [0] * (width - len(before))
    after_pad = after + [0] * (width - len(after))
    return [a - b for a, b in zip(after_pad, before_pad)]


def parse_bar_key(key: str) -> tuple[str, int]:
    slug, bar_num = key.rsplit(":", 1)
    return slug, int(bar_num)


def choose_cue_start(n_bars: int, cue_len: int, position: str) -> int | None:
    if n_bars < cue_len:
        return None
    if position == "start":
        return 0
    if position == "middle":
        return max(0, (n_bars - cue_len) // 2)
    if position == "end":
        return n_bars - cue_len
    raise ValueError(f"Unknown cue position: {position}")


def load_bach_piece(path: Path, genome: Genome | None = None) -> PieceData | None:
    try:
        slices, _, voice_order = parse_kern_score(path)
        bars = group_bars(slices)
        if not bars:
            return None
        adapter = GenericPolyphonicAdapter(voice_order)
        raw_quats = [compose_bar(adapter, bar_events) for bar_events in bars]
        learned_quats = []
        slug = path.stem
        for bar_num, raw_q in enumerate(raw_quats, start=1):
            if genome is None:
                learned_quats.append(raw_q)
            else:
                stored = genome.get_position(f"{slug}:{bar_num}")
                learned_quats.append(stored[0] if stored is not None else raw_q)
        return PieceData(
            slug=slug,
            title=read_title(path),
            bars=bars,
            raw_quats=raw_quats,
            learned_quats=learned_quats,
        )
    except Exception:
        return None


def load_bach_training_piece(path: Path) -> dict | None:
    piece = load_bach_piece(path, None)
    if piece is None:
        return None
    slices = flatten_bars(piece.bars)
    return {
        "piece_id": piece.slug,
        "slug": piece.slug,
        "title": piece.title,
        "n_bars": piece.n_bars,
        "n_slices": len(slices),
        "bar_quats": piece.raw_quats,
    }


def build_exact_windows(pieces: list[PieceData], window_len: int) -> set[tuple[str, ...]]:
    windows = set()
    for piece in pieces:
        keys = [f"{piece.slug}:{i}" for i in range(1, piece.n_bars + 1)]
        for start in range(0, piece.n_bars - window_len + 1):
            windows.add(tuple(keys[start:start + window_len]))
    return windows


def retrieve_window_piece(
    query_quats: list[np.ndarray],
    pieces: list[PieceData],
) -> tuple[str, float, int, list[tuple[str, float, int]]]:
    rankings = []
    window_len = len(query_quats)
    for piece in pieces:
        if piece.n_bars < window_len:
            continue
        best_score = math.inf
        best_start = -1
        for start in range(0, piece.n_bars - window_len + 1):
            score = mean_window_sigma(query_quats, piece.learned_quats[start:start + window_len])
            if score < best_score:
                best_score = score
                best_start = start
        rankings.append((piece.slug, best_score, best_start))
    rankings.sort(key=lambda item: item[1])
    best_slug, best_score, best_start = rankings[0]
    return best_slug, best_score, best_start, rankings[:5]


def context_candidates(
    context_quats: list[np.ndarray],
    pieces: list[PieceData],
    actual_next_key: str | None,
    top_matches: int,
    novelty_penalty: float,
) -> list[Candidate]:
    matches = []
    context_len = len(context_quats)
    for piece in pieces:
        if piece.n_bars <= context_len:
            continue
        for start in range(0, piece.n_bars - context_len):
            score = mean_window_sigma(context_quats, piece.learned_quats[start:start + context_len])
            successor_bar_num = start + context_len + 1
            matches.append((score, piece.slug, successor_bar_num))

    matches.sort(key=lambda item: item[0])
    top = matches[:top_matches]
    by_key: dict[tuple[str, int], float] = {}
    for score, slug, bar_num in top:
        key = (slug, bar_num)
        if key not in by_key or score < by_key[key]:
            by_key[key] = score

    piece_map = {piece.slug: piece for piece in pieces}
    out = []
    for (slug, bar_num), best_score in by_key.items():
        score = best_score
        if actual_next_key is not None and f"{slug}:{bar_num}" == actual_next_key:
            score += novelty_penalty
        out.append(Candidate(source_slug=slug, bar_num=bar_num, q=piece_map[slug].learned_quats[bar_num - 1], context_score=score))
    out.sort(key=lambda item: item.context_score)
    return out


def infer_piece_identity(path: Path) -> tuple[str, str, str]:
    rel = path.relative_to(EXTERNAL_CORPORA_DIR)
    rel_no_suffix = rel.with_suffix("")
    composer = rel.parts[0]
    piece_id = "__".join(rel_no_suffix.parts)
    return composer, rel.as_posix(), piece_id


def iter_candidate_paths() -> list[Path]:
    files = []
    for root_name in CLASSICAL_COLLECTIONS:
        root = EXTERNAL_CORPORA_DIR / root_name
        if root.exists():
            files.extend(sorted(root.rglob("*.krn")))
    return files


def load_classical_piece(path: Path) -> tuple[dict | None, dict]:
    composer, relpath, piece_id = infer_piece_identity(path)
    manifest = {"piece_id": piece_id, "composer": composer, "relpath": relpath}
    try:
        with time_limit(CLASSICAL_PARSE_TIMEOUT_SECONDS):
            slices, _, voice_order = parse_kern_score(path)
        bars = group_bars(slices)
        if not bars:
            manifest["status"] = "empty"
            return None, manifest
        music_adapter = GenericPolyphonicAdapter(voice_order)
        bar_quats = [compose_bar(music_adapter, bar_events) for bar_events in bars]
        piece = {
            "piece_id": piece_id,
            "title": piece_id,
            "composer": composer,
            "relpath": relpath,
            "n_bars": len(bar_quats),
            "n_slices": len(slices),
            "voice_order": list(voice_order),
            "bar_quats": bar_quats,
        }
        manifest.update({"status": "ok", "n_bars": piece["n_bars"], "n_slices": piece["n_slices"], "n_voices": len(voice_order)})
        return piece, manifest
    except ParseTimeout as exc:
        manifest["status"] = "timeout"
        manifest["error"] = str(exc)
        return None, manifest
    except Exception as exc:  # noqa: BLE001
        manifest["status"] = "error"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        return None, manifest


def build_stored_bars_by_piece(genome: Genome, required_kind: str = "bar") -> dict[str, list[np.ndarray]]:
    bars_by_piece = defaultdict(list)
    for key, payload in genome.positions.items():
        meta = payload.get("meta", {})
        if meta.get("kind") != required_kind:
            continue
        piece = meta["piece"]
        bar_num = meta.get("bar_num")
        q = np.array(payload["q"], dtype=np.float64)
        if bar_num is None:
            continue
        bars_by_piece[piece].append((bar_num, q))
    stored = {}
    for piece, items in bars_by_piece.items():
        items.sort(key=lambda item: item[0])
        stored[piece] = [q for _, q in items]
    return stored


def cell_meta_payload(cell) -> dict:
    meta = dict(cell.meta)
    meta["child_ids"] = list(cell.child_ids)
    meta["child_keys"] = list(cell.source_keys)
    meta["trigger_ids"] = list(cell.trigger_ids)
    if cell.trigger_source_keys:
        meta["trigger_keys"] = list(cell.trigger_source_keys)
    meta["content_event_count"] = cell.event_count
    return meta


def select_piece_identity_cell(cells) -> object | None:
    if not cells:
        return None
    boundary_cells = [cell for cell in cells if cell.reason != "closure"]
    source_pool = boundary_cells if boundary_cells else cells
    return max(source_pool, key=lambda cell: cell.level)


def retrieve_piece_from_cue(
    cue_quats: list[np.ndarray],
    stored_bars_by_piece: dict[str, list[np.ndarray]],
) -> tuple[str, float, int, list[tuple[str, float, int]]]:
    rankings = []
    cue_len = len(cue_quats)
    for slug, bars in stored_bars_by_piece.items():
        if len(bars) < cue_len:
            continue
        best_score = math.inf
        best_start = -1
        for start in range(len(bars) - cue_len + 1):
            score = mean_window_sigma(cue_quats, bars[start:start + cue_len])
            if score < best_score:
                best_score = score
                best_start = start
        rankings.append((slug, best_score, best_start))
    rankings.sort(key=lambda item: item[1])
    best_slug, best_score, best_start = rankings[0]
    return best_slug, best_score, best_start, rankings[:5]


def train_bach() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_krn = sorted(CORPUS_DIR.glob("*.krn"))
    pieces = []
    total_bars = 0
    total_slices = 0

    print("GILGAMESH LEARNS BACH")
    print(f"Corpus: {len(all_krn)} Kern files in {CORPUS_DIR.name}/")
    print(f"Rust: {RUST}")
    print("=" * 70)

    print("\n1. PARSE — BAR-scoped bar quaternions")
    for path in all_krn:
        piece = load_bach_training_piece(path)
        if piece is None:
            continue
        pieces.append(piece)
        total_bars += piece["n_bars"]
        total_slices += piece["n_slices"]
    print(f"   Pieces: {len(pieces)}")
    print(f"   Total bars: {total_bars}")
    print(f"   Total slices: {total_slices}")

    genome = Genome("bach_music")
    bar_adapter = BarLevelAdapter()
    trinity = Trinity(
        bar_adapter,
        epsilon_s2=BACH_EPSILON_SCHEDULE[0],
        max_depth=4,
        s2_schedule=BACH_EPSILON_SCHEDULE,
    )

    print("\n2. TRINITY — canonical S1/S2/S3 learning loop")
    print(f"   S2 epsilon schedule: {BACH_EPSILON_SCHEDULE}")
    print("   Memory cadence: 4")

    bar_meta = {}
    s2_gaps = []
    piece_records = []
    t0 = time.time()
    print(f"\n3. LIVE — process {len(pieces)} pieces through continuous memory")

    for piece in pieces:
        piece_gap_start = len(s2_gaps)
        for bar_num, bar_q in enumerate(piece["bar_quats"], start=1):
            key = bar_adapter.embed_bar(piece["piece_id"], bar_num, bar_q)
            bar_meta[key] = {
                "kind": "bar",
                "piece": piece["piece_id"],
                "title": piece["title"],
                "bars": [bar_num],
                "bar_num": bar_num,
            }
            trinity.ingest(key)
            s2_gaps.append(trinity.gap)

        piece_trace = s2_gaps[piece_gap_start:]
        piece_records.append({
            'slug': piece['piece_id'],
            'title': piece['title'],
            'n_bars': piece['n_bars'],
            'mean_s2_sigma': (sum(piece_trace) / len(piece_trace)) if piece_trace else 0.0,
        })
        print(
            f"   {piece['piece_id']:<12} {piece['n_bars']:>3} bars  "
            f"mean S2 σ: {piece_records[-1]['mean_s2_sigma']:.4f}  genome(bar): {len(trinity.positions)}"
        )

    elapsed = time.time() - t0
    print(f"\n   Done: {elapsed:.2f}s  ({total_bars / elapsed:.0f} bars/s)")

    print("\n4. GENOME — persist learned bar positions from Trinity")
    for key, (q, count) in trinity.positions.items():
        locked = count == -1
        genome.record_position(key, q, abs(count), locked=locked, meta=bar_meta.get(key))

    genome.record_hierarchy(
        levels=trinity.s2.depth,
        closures_per_level=trinity.s2.closure_counts,
        total_events=trinity.s2.kernels[0].event_count if trinity.s2.kernels else 0,
    )

    kind_counts = defaultdict(int)
    for payload in genome.positions.values():
        kind_counts[payload.get('meta', {}).get('kind', 'unknown')] += 1

    mn, mx, mean = genome.spread
    pred_trace = trinity._prediction_sigma_trace
    early_s2 = sum(s2_gaps[:50]) / min(50, len(s2_gaps)) if s2_gaps else 0.0
    late_s2 = sum(s2_gaps[-50:]) / min(50, len(s2_gaps)) if s2_gaps else 0.0
    early_pred = sum(pred_trace[:50]) / min(50, len(pred_trace)) if pred_trace else 0.0
    late_pred = sum(pred_trace[-50:]) / min(50, len(pred_trace)) if pred_trace else 0.0

    print(f"   Positions: {genome.size}")
    print(f"   σ spread: min={mn:.4f}  max={mx:.4f}  mean={mean:.4f}")
    print(f"   Hierarchy: {genome.hierarchy}")
    print(f"   S2 early mean σ: {early_s2:.4f}")
    print(f"   S2 late mean σ: {late_s2:.4f}")
    print(f"   S2 delta: {early_s2 - late_s2:+.4f}")
    print(f"   Prediction early mean σ: {early_pred:.4f}")
    print(f"   Prediction late mean σ: {late_pred:.4f}")
    print(f"   Prediction delta: {early_pred - late_pred:+.4f}")

    genome.save(BACH_GENOME_PATH)
    BACH_TRINITY_STATE_PATH.write_text(json.dumps(trinity.save(), indent=2))
    print("\n5. SAVE")
    print(f"   Genome saved: {BACH_GENOME_PATH.name}")
    print(f"   Trinity state saved: {BACH_TRINITY_STATE_PATH.name}")

    print("\n" + "=" * 70)
    print("GILGAMESH LEARNS BACH")
    print("=" * 70)
    print(f"  Pieces: {len(pieces)}")
    print(f"  Bars: {total_bars}")
    print(f"  Genome: {genome.size} positions")
    print(f"  Bars stored: {kind_counts['bar']}")
    print(f"  Hierarchy: {genome.hierarchy}")
    print(f"  S2 delta: {early_s2 - late_s2:+.4f}")
    print(f"  Prediction delta: {early_pred - late_pred:+.4f}")
    print(f"  Trinity: {trinity.status()}")
    print(f"  Genome file: {BACH_GENOME_PATH.name} ({BACH_GENOME_PATH.stat().st_size / 1024:.1f} KB)")
    print("=" * 70)

    with BACH_AUDIT_PATH.open("w") as f:
        f.write("Gilgamesh Learns Bach\n\n")
        f.write(f"Pieces: {len(pieces)}\n")
        f.write(f"Bars: {total_bars}\n")
        f.write(f"Genome positions: {genome.size}\n")
        f.write(f"Genome σ: min={mn:.4f} max={mx:.4f} mean={mean:.4f}\n")
        f.write(f"Hierarchy: {genome.hierarchy}\n")
        f.write(f"Kind counts: {dict(kind_counts)}\n")
        f.write(f"S2 early mean sigma: {early_s2:.6f}\n")
        f.write(f"S2 late mean sigma: {late_s2:.6f}\n")
        f.write(f"S2 delta: {early_s2 - late_s2:.6f}\n")
        f.write(f"Prediction early mean sigma: {early_pred:.6f}\n")
        f.write(f"Prediction late mean sigma: {late_pred:.6f}\n")
        f.write(f"Prediction delta: {early_pred - late_pred:.6f}\n")
        f.write(f"Trinity status: {trinity.status()}\n\n")
        f.write("Per-piece:\n")
        for rec in piece_records:
            f.write(
                f"  {rec['slug']} ({rec['title']}): {rec['n_bars']} bars, "
                f"mean_s2_sigma={rec['mean_s2_sigma']:.6f}\n"
            )

def retrieve_bach() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not BACH_GENOME_PATH.exists():
        raise SystemExit(f"Missing genome file: {BACH_GENOME_PATH}\nRun `train-bach` first.")

    genome = Genome.load(BACH_GENOME_PATH)
    stored_bars_by_piece = build_stored_bars_by_piece(genome)
    pieces = []
    for path in sorted(CORPUS_DIR.glob("*.krn")):
        piece = load_bach_piece(path, None)
        if piece is not None and piece.slug in stored_bars_by_piece:
            pieces.append(piece)

    print("GILGAMESH BACH RETRIEVAL")
    print(f"Genome: {BACH_GENOME_PATH.name}")
    print(f"Rust: {RUST}")
    print("=" * 70)
    print(f"Loaded genome positions: {genome.size}")
    print(f"Corpus pieces available for retrieval: {len(pieces)}")
    print(f"Cue length: {BACH_RETRIEVAL_CUE_LENGTH} bars")

    results = []
    print("\n1. RETRIEVAL — start/middle/end cues")
    for piece in pieces:
        for position in BACH_RETRIEVAL_TEST_POSITIONS:
            start = choose_cue_start(piece.n_bars, BACH_RETRIEVAL_CUE_LENGTH, position)
            if start is None:
                continue
            cue_quats = piece.raw_quats[start:start + BACH_RETRIEVAL_CUE_LENGTH]
            best_slug, best_score, best_start, top5 = retrieve_piece_from_cue(cue_quats, stored_bars_by_piece)
            ok = best_slug == piece.slug
            results.append(
                {
                    "piece": piece.slug,
                    "title": piece.title,
                    "position": position,
                    "cue_start_bar": start + 1,
                    "cue_end_bar": start + BACH_RETRIEVAL_CUE_LENGTH,
                    "retrieved_piece": best_slug,
                    "retrieved_start_bar": best_start + 1,
                    "score": best_score,
                    "ok": ok,
                    "top5": top5,
                }
            )

    correct = sum(1 for r in results if r["ok"])
    print(f"   Accuracy: {correct}/{len(results)} ({correct / len(results) * 100:.1f}%)")

    print("\n2. EXAMPLES — top matches from the learned genome")
    for slug in ["inven01", "sinfo01", "wtc1f01"]:
        piece = next((p for p in pieces if p.slug == slug), None)
        if piece is None or piece.n_bars < BACH_RETRIEVAL_CUE_LENGTH:
            continue
        start = choose_cue_start(piece.n_bars, BACH_RETRIEVAL_CUE_LENGTH, "middle")
        cue_quats = piece.raw_quats[start:start + BACH_RETRIEVAL_CUE_LENGTH]
        best_slug, best_score, best_start, top5 = retrieve_piece_from_cue(cue_quats, stored_bars_by_piece)
        print(
            f"   {slug} middle cue bars {start + 1}-{start + BACH_RETRIEVAL_CUE_LENGTH} "
            f"→ {best_slug} at bars {best_start + 1}-{best_start + BACH_RETRIEVAL_CUE_LENGTH} "
            f"σ={best_score:.4f}"
        )
        for rank, (cand_slug, score, cand_start) in enumerate(top5, start=1):
            print(f"      {rank}. {cand_slug:<8} bars {cand_start + 1}-{cand_start + BACH_RETRIEVAL_CUE_LENGTH} σ={score:.4f}")

    print("\n3. MINIMUM CUE")
    min_lengths = []
    for piece in pieces:
        found = None
        for cue_len in range(1, min(6, piece.n_bars) + 1):
            cue_quats = piece.raw_quats[:cue_len]
            best_slug, _best_score, _best_start, _top5 = retrieve_piece_from_cue(cue_quats, stored_bars_by_piece)
            if best_slug == piece.slug:
                found = cue_len
                break
        if found is not None:
            min_lengths.append(found)
    if min_lengths:
        print(f"   Minimum cue lengths: min={min(min_lengths)}  max={max(min_lengths)}  mean={sum(min_lengths) / len(min_lengths):.2f}")

    summary = {
        "genome_positions": genome.size,
        "pieces": len(pieces),
        "cue_length": BACH_RETRIEVAL_CUE_LENGTH,
        "tested_cues": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "minimum_cue_lengths": {
            "count": len(min_lengths),
            "min": min(min_lengths) if min_lengths else None,
            "max": max(min_lengths) if min_lengths else None,
            "mean": (sum(min_lengths) / len(min_lengths)) if min_lengths else None,
        },
    }
    with BACH_RETRIEVAL_AUDIT_PATH.open("w") as f:
        f.write("Gilgamesh Bach Retrieval\n\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n\nDetailed cues:\n")
        for r in results:
            verdict = "PASS" if r["ok"] else "FAIL"
            f.write(
                f"  {verdict} {r['piece']} {r['position']} bars {r['cue_start_bar']}-{r['cue_end_bar']} "
                f"→ {r['retrieved_piece']} bars {r['retrieved_start_bar']}-{r['retrieved_start_bar'] + BACH_RETRIEVAL_CUE_LENGTH - 1} "
                f"σ={r['score']:.6f}\n"
            )
            for rank, (cand_slug, score, cand_start) in enumerate(r["top5"], start=1):
                f.write(f"    {rank}. {cand_slug} bars {cand_start + 1}-{cand_start + BACH_RETRIEVAL_CUE_LENGTH} σ={score:.6f}\n")


def generate_bach() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not BACH_GENOME_PATH.exists():
        raise SystemExit(f"Missing genome: {BACH_GENOME_PATH}")

    genome = Genome.load(BACH_GENOME_PATH)
    pieces = [load_bach_piece(path, genome) for path in sorted(CORPUS_DIR.glob("inven*.krn"))]
    pieces = [piece for piece in pieces if piece is not None]
    exact_windows = build_exact_windows(pieces, GEN_GENERATE_BARS)

    print("GILGAMESH BACH CONTINUATION")
    print(f"Corpus: {len(pieces)} inventions")
    print(f"Genome: {BACH_GENOME_PATH.name}")
    print(f"Cue bars: {GEN_CUE_BARS}  Generate: {GEN_GENERATE_BARS}  Context: {GEN_CONTEXT_BARS}")
    print(f"Rust: {RUST}")
    print("=" * 70)

    def generate_continuation(source_piece: PieceData, cue_start: int) -> dict:
        cue_quats = source_piece.learned_quats[cue_start:cue_start + GEN_CUE_BARS]
        beams = [BeamState(generated_keys=[], generated_quats=[], generated_bars=[], score=0.0, traces=[])]
        actual_next_keys = []
        for i in range(GEN_GENERATE_BARS):
            idx = cue_start + GEN_CUE_BARS + i
            actual_next_keys.append(f"{source_piece.slug}:{idx + 1}" if idx < source_piece.n_bars else None)

        for step in range(GEN_GENERATE_BARS):
            next_beams = []
            for beam in beams:
                combined_quats = cue_quats + beam.generated_quats
                context_quats = combined_quats[-GEN_CONTEXT_BARS:]
                candidates = context_candidates(
                    context_quats=context_quats,
                    pieces=pieces,
                    actual_next_key=actual_next_keys[step],
                    top_matches=GEN_TOP_CONTEXT_MATCHES,
                    novelty_penalty=GEN_NOVELTY_REPLAY_PENALTY,
                )
                for cand in candidates:
                    if cand.key in beam.generated_keys:
                        continue
                    new_keys = beam.generated_keys + [cand.key]
                    running_q = compose_sequence(beam.generated_quats + [cand.q])
                    running_penalty = GEN_RUNNING_CLOSURE_WEIGHT * sigma(running_q)
                    source_piece_data = next(piece for piece in pieces if piece.slug == cand.source_slug)
                    next_beams.append(
                        BeamState(
                            generated_keys=new_keys,
                            generated_quats=beam.generated_quats + [cand.q],
                            generated_bars=beam.generated_bars + [source_piece_data.bars[cand.bar_num - 1]],
                            score=beam.score + cand.context_score + running_penalty,
                            traces=beam.traces + [f"step={step + 1} candidate={cand.key} context={cand.context_score:.4f} running_sigma={sigma(running_q):.4f}"],
                        )
                    )
            next_beams.sort(key=lambda item: item.score)
            beams = next_beams[:GEN_BEAM_WIDTH]

        evaluations = []
        for beam in beams:
            is_exact_copy = tuple(beam.generated_keys) in exact_windows
            matches_source_actual = tuple(beam.generated_keys) == tuple(k for k in actual_next_keys if k is not None)
            source_overlap = sum(
                1
                for idx, key in enumerate(beam.generated_keys)
                if idx < len(actual_next_keys) and actual_next_keys[idx] is not None and key == actual_next_keys[idx]
            )
            combined_window = cue_quats + beam.generated_quats
            best_slug, best_score, best_start, top5 = retrieve_window_piece(combined_window, pieces)
            source_piece_score = next(
                score
                for slug, score, _start in sorted(
                    [
                        (
                            piece.slug,
                            mean_window_sigma(combined_window, piece.learned_quats[start:start + len(combined_window)]),
                            start,
                        )
                        for piece in pieces if piece.n_bars >= len(combined_window)
                        for start in range(0, piece.n_bars - len(combined_window) + 1)
                    ],
                    key=lambda item: item[1],
                )
                if slug == source_piece.slug
            )
            top5_other = [item for item in top5 if item[0] != source_piece.slug]
            nearest_other_score = top5_other[0][1] if top5_other else math.inf
            margin = nearest_other_score - source_piece_score
            evaluations.append(
                {
                    "beam": beam,
                    "is_exact_copy": is_exact_copy,
                    "matches_source_actual": matches_source_actual,
                    "source_overlap": source_overlap,
                    "retrieved_piece": best_slug,
                    "retrieved_score": best_score,
                    "retrieved_start": best_start + 1,
                    "source_score": source_piece_score,
                    "margin": margin,
                    "top5": top5,
                }
            )

        def eval_key(item: dict) -> tuple:
            return (
                0 if not item["is_exact_copy"] else 1,
                0 if item["retrieved_piece"] == source_piece.slug else 1,
                0 if not item["matches_source_actual"] else 1,
                item["source_overlap"],
                -item["margin"],
                item["beam"].score,
            )

        best = min(evaluations, key=eval_key)
        return {
            "cue_start_bar": cue_start + 1,
            "cue_end_bar": cue_start + GEN_CUE_BARS,
            "generated_keys": best["beam"].generated_keys,
            "generated_quats": best["beam"].generated_quats,
            "generated_bars": best["beam"].generated_bars,
            "score": best["beam"].score,
            "retrieved_piece": best["retrieved_piece"],
            "retrieved_start_bar": best["retrieved_start"],
            "source_score": best["source_score"],
            "margin": best["margin"],
            "exact_copy": best["is_exact_copy"],
            "exact_source_continuation": best["matches_source_actual"],
            "source_overlap": best["source_overlap"],
            "top5": best["top5"],
            "trace": best["beam"].traces,
        }

    results = []
    audio_examples = 0
    for piece in pieces:
        if piece.n_bars < GEN_CUE_BARS + GEN_GENERATE_BARS:
            continue
        cue_start = max(0, (piece.n_bars // 2) - (GEN_CUE_BARS // 2))
        if cue_start + GEN_CUE_BARS + GEN_GENERATE_BARS > piece.n_bars:
            cue_start = piece.n_bars - (GEN_CUE_BARS + GEN_GENERATE_BARS)
        result = generate_continuation(piece, cue_start)
        result["piece"] = piece.slug
        result["title"] = piece.title
        results.append(result)
        success = (not result["exact_copy"] and not result["exact_source_continuation"] and result["retrieved_piece"] == piece.slug)
        print(
            f"  {piece.slug}: cue {result['cue_start_bar']}-{result['cue_end_bar']} "
            f"-> {','.join(result['generated_keys'])}  source={'YES' if result['retrieved_piece'] == piece.slug else 'NO'}  "
            f"novel={'YES' if (not result['exact_copy'] and not result['exact_source_continuation']) else 'NO'}  "
            f"overlap={result['source_overlap']}/{GEN_GENERATE_BARS}  margin={result['margin']:.4f}  {'PASS' if success else 'FAIL'}"
        )
        if audio_examples < GEN_EXAMPLE_AUDIO_LIMIT:
            cue_bars = piece.bars[cue_start:cue_start + GEN_CUE_BARS]
            ref_bars = piece.bars[cue_start + GEN_CUE_BARS: cue_start + GEN_CUE_BARS + GEN_GENERATE_BARS]
            gen_full = cue_bars + result["generated_bars"]
            ref_full = cue_bars + ref_bars
            prefix = OUTPUT_DIR / f"{piece.slug}_generation"
            write_polyphonic_wav(flatten_bars(cue_bars), prefix.with_name(prefix.name + "_cue.wav"))
            write_polyphonic_wav(flatten_bars(gen_full), prefix.with_name(prefix.name + "_generated.wav"))
            write_polyphonic_wav(flatten_bars(ref_full), prefix.with_name(prefix.name + "_reference.wav"))
            audio_examples += 1

    total = len(results)
    source_kept = sum(1 for r in results if r["retrieved_piece"] == r["piece"])
    novel = sum(1 for r in results if not r["exact_copy"] and not r["exact_source_continuation"])
    strong_novel = sum(1 for r in results if r["source_overlap"] <= 2)
    strict_pass = sum(1 for r in results if (r["retrieved_piece"] == r["piece"] and not r["exact_copy"] and not r["exact_source_continuation"]))
    strong_pass = sum(
        1
        for r in results
        if (r["retrieved_piece"] == r["piece"] and not r["exact_copy"] and not r["exact_source_continuation"] and r["source_overlap"] <= 2)
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Pieces tested: {total}")
    print(f"  Source identity retained: {source_kept}/{total}")
    print(f"  Novel 4-bar continuations: {novel}/{total}")
    print(f"  Strict pass (novel + source retained): {strict_pass}/{total}")
    print(f"  Strong novelty (<=2 source bars reused): {strong_novel}/{total}")
    print(f"  Strong pass: {strong_pass}/{total}")
    if results:
        margins = [r["margin"] for r in results]
        print(f"  Identity margin: min={min(margins):.4f}  max={max(margins):.4f}  mean={sum(margins) / len(margins):.4f}")

    summary = {
        "pieces_tested": total,
        "cue_bars": GEN_CUE_BARS,
        "generate_bars": GEN_GENERATE_BARS,
        "context_bars": GEN_CONTEXT_BARS,
        "source_identity_retained": source_kept,
        "novel_4bar_continuations": novel,
        "strict_pass": strict_pass,
        "strong_novelty": strong_novel,
        "strong_pass": strong_pass,
        "results": [
            {
                "piece": r["piece"],
                "title": r["title"],
                "cue_start_bar": r["cue_start_bar"],
                "cue_end_bar": r["cue_end_bar"],
                "generated_keys": r["generated_keys"],
                "retrieved_piece": r["retrieved_piece"],
                "retrieved_start_bar": r["retrieved_start_bar"],
                "source_score": r["source_score"],
                "margin": r["margin"],
                "exact_copy": r["exact_copy"],
                "exact_source_continuation": r["exact_source_continuation"],
                "source_overlap": r["source_overlap"],
                "trace": r["trace"],
                "top5": r["top5"],
            }
            for r in results
        ],
    }
    with BACH_GENERATION_AUDIT_PATH.open("w") as f:
        f.write("Gilgamesh Bach Continuation\n\n")
        f.write(json.dumps(summary, indent=2, default=float))
        f.write("\n")


def section_bach() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not BACH_GENOME_PATH.exists():
        raise SystemExit(f"Missing genome: {BACH_GENOME_PATH}")

    genome = Genome.load(BACH_GENOME_PATH)
    pieces = [load_bach_piece(path, genome) for path in sorted(CORPUS_DIR.glob("inven*.krn"))]
    pieces = [piece for piece in pieces if piece is not None and piece.n_bars >= SECTION_CUE_BARS + SECTION_GENERATE_BARS]
    exact_windows = build_exact_windows(pieces, SECTION_GENERATE_BARS)

    print("GILGAMESH BACH SECTION GENERATION")
    print(f"Corpus: {len(pieces)} inventions")
    print(f"Genome: {BACH_GENOME_PATH.name}")
    print(f"Cue bars: {SECTION_CUE_BARS}  Generate: {SECTION_GENERATE_BARS}  Context: {SECTION_CONTEXT_BARS}")
    print(f"Rust: {RUST}")
    print("=" * 70)

    def score_source_piece(query_quats: list[np.ndarray], source_piece: PieceData) -> float:
        scores = []
        for start in range(0, source_piece.n_bars - len(query_quats) + 1):
            scores.append(mean_window_sigma(query_quats, source_piece.learned_quats[start:start + len(query_quats)]))
        return min(scores)

    def generate_section(source_piece: PieceData, cue_start: int) -> dict:
        cue_quats = source_piece.learned_quats[cue_start:cue_start + SECTION_CUE_BARS]
        beams = [BeamState(generated_keys=[], generated_quats=[], generated_bars=[], score=0.0, traces=[])]
        actual_next_keys = []
        for i in range(SECTION_GENERATE_BARS):
            idx = cue_start + SECTION_CUE_BARS + i
            actual_next_keys.append(f"{source_piece.slug}:{idx + 1}" if idx < source_piece.n_bars else None)
        piece_map = {piece.slug: piece for piece in pieces}

        for step in range(SECTION_GENERATE_BARS):
            next_beams = []
            for beam in beams:
                combined_quats = cue_quats + beam.generated_quats
                context_quats = combined_quats[-SECTION_CONTEXT_BARS:]
                candidates = context_candidates(
                    context_quats=context_quats,
                    pieces=pieces,
                    actual_next_key=actual_next_keys[step],
                    top_matches=SECTION_TOP_CONTEXT_MATCHES,
                    novelty_penalty=SECTION_NOVELTY_REPLAY_PENALTY,
                )
                for cand in candidates:
                    if cand.key in beam.generated_keys:
                        continue
                    new_keys = beam.generated_keys + [cand.key]
                    new_quats = beam.generated_quats + [cand.q]
                    running_q = compose_sequence(new_quats)
                    running_penalty = SECTION_RUNNING_CLOSURE_WEIGHT * sigma(running_q)
                    cross_piece_penalty = 0.10 if step >= 4 and cand.source_slug != source_piece.slug else 0.0
                    next_beams.append(
                        BeamState(
                            generated_keys=new_keys,
                            generated_quats=new_quats,
                            generated_bars=beam.generated_bars + [piece_map[cand.source_slug].bars[cand.bar_num - 1]],
                            score=beam.score + cand.context_score + running_penalty + cross_piece_penalty,
                            traces=beam.traces + [
                                f"step={step + 1} candidate={cand.key} context={cand.context_score:.4f} running_sigma={sigma(running_q):.4f} cross_piece_penalty={cross_piece_penalty:.2f}"
                            ],
                        )
                    )
            next_beams.sort(key=lambda item: item.score)
            beams = next_beams[:SECTION_BEAM_WIDTH]

        evaluations = []
        for beam in beams:
            is_exact_copy = tuple(beam.generated_keys) in exact_windows
            exact_source_continuation = tuple(beam.generated_keys) == tuple(k for k in actual_next_keys if k is not None)
            source_overlap = sum(
                1
                for idx, key in enumerate(beam.generated_keys)
                if idx < len(actual_next_keys) and actual_next_keys[idx] is not None and key == actual_next_keys[idx]
            )
            source_bars_used = sum(1 for key in beam.generated_keys if key.startswith(f"{source_piece.slug}:"))
            combined_window = cue_quats + beam.generated_quats
            best_slug, best_score, best_start, top5 = retrieve_window_piece(combined_window, pieces)
            source_score = score_source_piece(combined_window, source_piece)
            top5_other = [item for item in top5 if item[0] != source_piece.slug]
            nearest_other_score = top5_other[0][1] if top5_other else math.inf
            margin = nearest_other_score - source_score
            evaluations.append(
                {
                    "beam": beam,
                    "is_exact_copy": is_exact_copy,
                    "exact_source_continuation": exact_source_continuation,
                    "source_overlap": source_overlap,
                    "source_bars_used": source_bars_used,
                    "retrieved_piece": best_slug,
                    "retrieved_score": best_score,
                    "retrieved_start": best_start + 1,
                    "source_score": source_score,
                    "margin": margin,
                    "top5": top5,
                }
            )

        def eval_key(item: dict) -> tuple:
            return (
                0 if not item["is_exact_copy"] else 1,
                0 if item["retrieved_piece"] == source_piece.slug else 1,
                0 if not item["exact_source_continuation"] else 1,
                item["source_overlap"],
                -item["source_bars_used"],
                -item["margin"],
                item["beam"].score,
            )

        best = min(evaluations, key=eval_key)
        return {
            "cue_start_bar": cue_start + 1,
            "cue_end_bar": cue_start + SECTION_CUE_BARS,
            "generated_keys": best["beam"].generated_keys,
            "generated_bars": best["beam"].generated_bars,
            "retrieved_piece": best["retrieved_piece"],
            "retrieved_start_bar": best["retrieved_start"],
            "source_score": best["source_score"],
            "margin": best["margin"],
            "exact_copy": best["is_exact_copy"],
            "exact_source_continuation": best["exact_source_continuation"],
            "source_overlap": best["source_overlap"],
            "source_bars_used": best["source_bars_used"],
            "trace": best["beam"].traces,
            "top5": best["top5"],
        }

    results = []
    audio_examples = 0
    for piece in pieces:
        cue_start = max(0, (piece.n_bars // 2) - (SECTION_CUE_BARS // 2))
        if cue_start + SECTION_CUE_BARS + SECTION_GENERATE_BARS > piece.n_bars:
            cue_start = piece.n_bars - (SECTION_CUE_BARS + SECTION_GENERATE_BARS)
        result = generate_section(piece, cue_start)
        result["piece"] = piece.slug
        result["title"] = piece.title
        results.append(result)
        success = result["retrieved_piece"] == piece.slug and not result["exact_copy"] and not result["exact_source_continuation"]
        print(
            f"  {piece.slug}: cue {result['cue_start_bar']}-{result['cue_end_bar']} "
            f"-> {','.join(result['generated_keys'])}  source={'YES' if result['retrieved_piece'] == piece.slug else 'NO'}  "
            f"overlap={result['source_overlap']}/{SECTION_GENERATE_BARS}  ownbars={result['source_bars_used']}/{SECTION_GENERATE_BARS}  "
            f"margin={result['margin']:.4f}  {'PASS' if success else 'FAIL'}"
        )
        if audio_examples < SECTION_EXAMPLE_AUDIO_LIMIT:
            cue_bars = piece.bars[cue_start:cue_start + SECTION_CUE_BARS]
            ref_bars = piece.bars[cue_start + SECTION_CUE_BARS: cue_start + SECTION_CUE_BARS + SECTION_GENERATE_BARS]
            gen_full = cue_bars + result["generated_bars"]
            ref_full = cue_bars + ref_bars
            prefix = OUTPUT_DIR / f"{piece.slug}_section_generation"
            write_polyphonic_wav(flatten_bars(cue_bars), prefix.with_name(prefix.name + "_cue.wav"))
            write_polyphonic_wav(flatten_bars(gen_full), prefix.with_name(prefix.name + "_generated.wav"))
            write_polyphonic_wav(flatten_bars(ref_full), prefix.with_name(prefix.name + "_reference.wav"))
            audio_examples += 1

    total = len(results)
    kept = sum(1 for r in results if r["retrieved_piece"] == r["piece"])
    novel = sum(1 for r in results if not r["exact_copy"] and not r["exact_source_continuation"])
    strong_novel = sum(1 for r in results if r["source_overlap"] <= 4)
    strong_pass = sum(
        1
        for r in results
        if r["retrieved_piece"] == r["piece"]
        and not r["exact_copy"]
        and not r["exact_source_continuation"]
        and r["source_overlap"] <= 4
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Pieces tested: {total}")
    print(f"  Source identity retained: {kept}/{total}")
    print(f"  Novel 8-bar sections: {novel}/{total}")
    print(f"  Strong novelty (<=4 source bars reused): {strong_novel}/{total}")
    print(f"  Strong pass: {strong_pass}/{total}")
    if results:
        margins = [r["margin"] for r in results]
        ownbars = [r["source_bars_used"] for r in results]
        print(f"  Identity margin: min={min(margins):.4f}  max={max(margins):.4f}  mean={sum(margins) / len(margins):.4f}")
        print(f"  Source bars used: min={min(ownbars)}  max={max(ownbars)}  mean={sum(ownbars) / len(ownbars):.2f}")

    summary = {
        "pieces_tested": total,
        "cue_bars": SECTION_CUE_BARS,
        "generate_bars": SECTION_GENERATE_BARS,
        "context_bars": SECTION_CONTEXT_BARS,
        "source_identity_retained": kept,
        "novel_8bar_sections": novel,
        "strong_novelty": strong_novel,
        "strong_pass": strong_pass,
        "results": [
            {
                "piece": r["piece"],
                "title": r["title"],
                "cue_start_bar": r["cue_start_bar"],
                "cue_end_bar": r["cue_end_bar"],
                "generated_keys": r["generated_keys"],
                "retrieved_piece": r["retrieved_piece"],
                "retrieved_start_bar": r["retrieved_start_bar"],
                "source_score": r["source_score"],
                "margin": r["margin"],
                "exact_copy": r["exact_copy"],
                "exact_source_continuation": r["exact_source_continuation"],
                "source_overlap": r["source_overlap"],
                "source_bars_used": r["source_bars_used"],
                "trace": r["trace"],
                "top5": r["top5"],
            }
            for r in results
        ],
    }
    with BACH_SECTION_AUDIT_PATH.open("w") as f:
        f.write("Gilgamesh Bach Section Generation\n\n")
        f.write(json.dumps(summary, indent=2, default=float))
        f.write("\n")


def improvise_bach() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not BACH_GENOME_PATH.exists():
        raise SystemExit(f"Missing genome: {BACH_GENOME_PATH}")

    def occupancy_signature(event: SliceEvent) -> tuple[int, ...]:
        return tuple(len(notes) for _, notes in event.voices)

    def anchored_event(anchor: SliceEvent, candidate: SliceCandidate) -> SliceEvent:
        return SliceEvent(bar=anchor.bar, start=anchor.start, duration=anchor.duration, voices=candidate.voices)

    def mean_pitch(notes: tuple[int, ...]) -> float | None:
        return None if not notes else sum(notes) / len(notes)

    def event_voice_means(event: SliceEvent) -> dict[str, float | None]:
        return {voice: mean_pitch(notes) for voice, notes in event.voices}

    def consonance_penalty(event: SliceEvent) -> float:
        active = [notes[0] for _, notes in event.voices if notes]
        if len(active) < 2:
            return 0.0
        interval = abs(active[1] - active[0]) % 12
        return 0.0 if interval in {0, 3, 4, 5, 7, 8, 9} else 0.18

    def voice_leading_penalty(previous_event: SliceEvent | None, new_event: SliceEvent, anchor: SliceEvent) -> float:
        penalty = 0.0
        prev_means = event_voice_means(previous_event) if previous_event is not None else {}
        new_means = event_voice_means(new_event)
        anchor_means = event_voice_means(anchor)
        for voice, new_mean in new_means.items():
            if new_mean is None:
                continue
            prev_mean = prev_means.get(voice)
            anchor_mean = anchor_means.get(voice)
            if prev_mean is not None:
                leap = abs(new_mean - prev_mean)
                if leap > 7:
                    penalty += 0.05 * (leap - 7)
            if anchor_mean is not None:
                dev = abs(new_mean - anchor_mean)
                if dev > 5:
                    penalty += 0.04 * (dev - 5)
        return penalty

    def noisy_target(anchor_q: np.ndarray, similar: list[np.ndarray], rng: random.Random) -> np.ndarray:
        mean_q = normalize_quat(np.sum(np.stack(similar), axis=0)) if similar else anchor_q
        noise = np.array([rng.gauss(0.0, 1.0) for _ in range(4)], dtype=np.float64)
        noise = normalize_quat(noise)
        mixed = 0.74 * anchor_q + 0.16 * mean_q + SOLO_NOISE_SCALE * noise
        return normalize_quat(mixed)

    def softmax_pick(options: list[tuple[float, SliceCandidate]], rng: random.Random) -> SliceCandidate:
        weights = [math.exp(-score / SOLO_TEMPERATURE) for score, _ in options]
        total = sum(weights)
        pick = rng.random() * total
        acc = 0.0
        for weight, (_, candidate) in zip(weights, options):
            acc += weight
            if acc >= pick:
                return candidate
        return options[-1][1]

    def build_slice_library() -> tuple[dict[tuple[Fraction, tuple[int, ...]], list[SliceCandidate]], dict[str, GenericPolyphonicAdapter]]:
        library = {}
        adapters = {}
        for path in sorted(CORPUS_DIR.glob("inven*.krn")):
            slices, _, voice_order = parse_kern_score(path)
            bars = group_bars(slices)
            adapter = GenericPolyphonicAdapter(voice_order)
            adapters[path.stem] = adapter
            for bar_idx, bar in enumerate(bars, start=1):
                for event in bar:
                    q = adapter.embed_slice(event)
                    sig = (event.duration, occupancy_signature(event))
                    library.setdefault(sig, []).append(
                        SliceCandidate(
                            source_slug=path.stem,
                            source_bar=bar_idx,
                            q=q,
                            voices=event.voices,
                            duration=event.duration,
                            occupancy=occupancy_signature(event),
                        )
                    )
        return library, adapters

    def shortlist_candidates(
        piece_adapter: GenericPolyphonicAdapter,
        library: dict[tuple[Fraction, tuple[int, ...]], list[SliceCandidate]],
        anchor: SliceEvent,
        source_slug: str,
    ) -> list[SliceCandidate]:
        sig = (anchor.duration, occupancy_signature(anchor))
        pool = list(library.get(sig, []))
        if not pool:
            pool = [item for items in library.values() for item in items if item.duration == anchor.duration]
        anchor_q = piece_adapter.embed_slice(anchor)
        source_pool = [cand for cand in pool if cand.source_slug == source_slug]
        source_pool.sort(key=lambda cand: sigma(compose(anchor_q, inverse(cand.q))))
        if len(source_pool) >= SOLO_TOP_SAMPLE:
            pool = source_pool + [cand for cand in pool if cand.source_slug != source_slug]
        pool.sort(key=lambda cand: sigma(compose(anchor_q, inverse(cand.q))))
        return pool[:SOLO_MAX_SLICE_CANDIDATES]

    def choose_change_indices(rng: random.Random, bar_len: int) -> set[int]:
        n_changes = rng.randint(SOLO_MIN_BAR_SLICE_CHANGES, min(SOLO_MAX_BAR_SLICE_CHANGES, bar_len))
        preferred = list(range(1, max(1, bar_len - 1)))
        if len(preferred) < n_changes:
            preferred = list(range(bar_len))
        rng.shuffle(preferred)
        return set(preferred[:n_changes])

    def vary_bar(
        rng: random.Random,
        piece_adapter: GenericPolyphonicAdapter,
        library: dict[tuple[Fraction, tuple[int, ...]], list[SliceCandidate]],
        source_slug: str,
        source_bar: list[SliceEvent],
        target_bar_q: np.ndarray,
        source_piece_q: np.ndarray,
    ) -> tuple[list[SliceEvent], int, list[str]]:
        change_indices = choose_change_indices(rng, len(source_bar))
        varied = []
        q_running = identity()
        changed = 0
        trace = [f"change_indices={sorted(change_indices)}"]

        for idx, anchor in enumerate(source_bar):
            previous_event = varied[-1] if varied else None
            anchor_q = piece_adapter.embed_slice(anchor)
            if idx not in change_indices:
                varied.append(anchor)
                q_running = compose(q_running, anchor_q)
                continue

            candidates = shortlist_candidates(piece_adapter, library, anchor, source_slug)
            similar_quats = [cand.q for cand in candidates[:SOLO_TOP_SAMPLE]]
            target_q = noisy_target(anchor_q, similar_quats, rng)
            scored = []
            for cand in candidates[:SOLO_TOP_SAMPLE]:
                new_event = anchored_event(anchor, cand)
                q_after = compose(q_running, cand.q)
                residual = sigma(compose(q_after, inverse(target_bar_q)))
                local = sigma(compose(target_q, inverse(cand.q)))
                piece_bias = SOLO_PIECE_IDENTITY_WEIGHT * sigma(compose(q_after, inverse(source_piece_q)))
                source_bonus = SOLO_SOURCE_CANDIDATE_BONUS if cand.source_slug == source_slug else SOLO_NONSOURCE_CANDIDATE_PENALTY
                same_penalty = 0.08 if new_event == anchor else 0.0
                voice_pen = voice_leading_penalty(previous_event, new_event, anchor)
                consonance = consonance_penalty(new_event)
                score = residual + 0.50 * local + 0.05 * piece_bias + source_bonus + same_penalty + voice_pen + consonance
                scored.append((score, cand))

            scored.sort(key=lambda item: item[0])
            selectable = [item for item in scored[: min(6, len(scored))] if anchored_event(anchor, item[1]) != anchor]
            if not selectable:
                selectable = scored[: min(6, len(scored))]
            chosen = softmax_pick(selectable, rng)
            new_event = anchored_event(anchor, chosen)
            if new_event != anchor:
                changed += 1
            varied.append(new_event)
            q_running = compose(q_running, chosen.q)
            trace.append(f"slice={idx + 1} chose={chosen.source_slug}:{chosen.source_bar} top={[ (round(s,4), c.source_slug+':'+str(c.source_bar)) for s,c in scored[:4] ]}")

        return varied, changed, trace

    def piece_identity_score(query_quats: list[np.ndarray], source_piece: PieceData) -> float:
        vals = []
        for start in range(0, source_piece.n_bars - len(query_quats) + 1):
            vals.append(sum(sigma(compose(a, inverse(b))) for a, b in zip(query_quats, source_piece.learned_quats[start:start + len(query_quats)])) / len(query_quats))
        return min(vals)

    genome = Genome.load(BACH_GENOME_PATH)
    pieces = [load_bach_piece(path, genome) for path in sorted(CORPUS_DIR.glob("inven*.krn"))]
    pieces = [piece for piece in pieces if piece is not None]
    source_piece = next((piece for piece in pieces if piece.slug == SOLO_SOURCE_SLUG), None)
    if source_piece is None:
        raise SystemExit(f"Missing source piece: {SOLO_SOURCE_SLUG}")

    library, adapters = build_slice_library()
    piece_adapter = adapters[SOLO_SOURCE_SLUG]
    cue_start = SOLO_CUE_START_BAR - 1
    solo_start = cue_start + SOLO_CUE_BARS
    solo_end = solo_start + SOLO_BARS
    cue_bars = source_piece.bars[cue_start:solo_start]
    source_solo_bars = source_piece.bars[solo_start:solo_end]
    source_solo_quats = [compose_bar(piece_adapter, bar) for bar in source_solo_bars]
    source_piece_q = compose_sequence(source_piece.learned_quats)

    variants = []
    for seed in range(SOLO_NUM_VARIANTS):
        rng = random.Random(seed)
        varied_solo_bars = []
        per_bar_changes = []
        trace = []
        for bar_num, (bar, target_q) in enumerate(zip(source_solo_bars, source_solo_quats), start=solo_start + 1):
            varied_bar, changed, bar_trace = vary_bar(
                rng=rng,
                piece_adapter=piece_adapter,
                library=library,
                source_slug=SOLO_SOURCE_SLUG,
                source_bar=bar,
                target_bar_q=target_q,
                source_piece_q=source_piece_q,
            )
            varied_solo_bars.append(varied_bar)
            per_bar_changes.append(changed)
            trace.append(f"bar={bar_num} changed_slices={changed}")
            trace.extend(bar_trace)

        combined_bars = cue_bars + varied_solo_bars
        combined_quats = [compose_bar(piece_adapter, bar) for bar in combined_bars]
        retrieved_piece, _best_score, retrieved_start, top5 = retrieve_window_piece(combined_quats, pieces)
        source_score = piece_identity_score(combined_quats, source_piece)
        nearest_other = min((score for slug, score, _ in top5 if slug != SOLO_SOURCE_SLUG), default=source_score)
        margin = nearest_other - source_score
        variants.append(
            Variant(
                seed=seed,
                varied_solo_bars=varied_solo_bars,
                changed_slices=sum(per_bar_changes),
                changed_bars=sum(1 for x in per_bar_changes if x > 0),
                retrieved_piece=retrieved_piece,
                retrieved_start_bar=retrieved_start + 1,
                margin=margin,
                source_distance=source_score,
                per_bar_changes=per_bar_changes,
                trace=trace,
            )
        )

    variants.sort(key=lambda v: (0 if v.retrieved_piece == SOLO_SOURCE_SLUG else 1, -v.margin, v.changed_slices))
    best = variants[:SOLO_KEEP_VARIANTS]

    print("GILGAMESH BACH SOLO IMPROVISATION")
    print(f"Source: {SOLO_SOURCE_SLUG}")
    print(f"Cue bars: {SOLO_CUE_START_BAR}-{SOLO_CUE_START_BAR + SOLO_CUE_BARS - 1}")
    print(f"Solo bars: {SOLO_CUE_START_BAR + SOLO_CUE_BARS}-{SOLO_CUE_START_BAR + SOLO_CUE_BARS + SOLO_BARS - 1}")
    print(f"Variants tried: {SOLO_NUM_VARIANTS}")
    print(f"Rust: {RUST}")
    print("=" * 70)

    prefix = OUTPUT_DIR / f"{SOLO_SOURCE_SLUG}_solo_moderated"
    write_polyphonic_wav(flatten_bars(cue_bars), prefix.with_name(prefix.name + "_cue.wav"))
    write_polyphonic_wav(flatten_bars(cue_bars + source_solo_bars), prefix.with_name(prefix.name + "_reference.wav"))
    for idx, variant in enumerate(best, start=1):
        print(
            f"  variant {idx}: seed={variant.seed} piece={variant.retrieved_piece} "
            f"changed_slices={variant.changed_slices} changed_bars={variant.changed_bars} margin={variant.margin:.4f}"
        )
        write_polyphonic_wav(flatten_bars(cue_bars + variant.varied_solo_bars), prefix.with_name(prefix.name + f"_variant{idx}.wav"))

    summary = {
        "source_piece": SOLO_SOURCE_SLUG,
        "cue_start_bar": SOLO_CUE_START_BAR,
        "cue_end_bar": SOLO_CUE_START_BAR + SOLO_CUE_BARS - 1,
        "solo_start_bar": SOLO_CUE_START_BAR + SOLO_CUE_BARS,
        "solo_end_bar": SOLO_CUE_START_BAR + SOLO_CUE_BARS + SOLO_BARS - 1,
        "variants_tried": SOLO_NUM_VARIANTS,
        "best_variants": [
            {
                "seed": v.seed,
                "retrieved_piece": v.retrieved_piece,
                "retrieved_start_bar": v.retrieved_start_bar,
                "changed_slices": v.changed_slices,
                "changed_bars": v.changed_bars,
                "per_bar_changes": v.per_bar_changes,
                "margin": v.margin,
                "source_distance": v.source_distance,
                "trace": v.trace,
            }
            for v in best
        ],
    }
    with BACH_SOLO_AUDIT_PATH.open("w") as f:
        f.write("Gilgamesh Bach Solo Improvisation\n\n")
        f.write(json.dumps(summary, indent=2, default=float))
        f.write("\n")


def train_classical() -> None:
    if not EXTERNAL_CORPORA_DIR.exists():
        raise SystemExit(f"Missing corpus root: {EXTERNAL_CORPORA_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    candidate_paths = iter_candidate_paths()
    pieces = []
    manifest_entries = []
    total_bars = 0
    total_slices = 0
    composer_counts = defaultdict(int)

    print("GILGAMESH LEARNS CLASSICAL")
    print(f"Corpus root: {EXTERNAL_CORPORA_DIR}")
    print(f"Selected collections: {', '.join(CLASSICAL_COLLECTIONS)}")
    print(f"Candidate files: {len(candidate_paths)}")
    print(f"Rust: {RUST}")
    print("=" * 70)

    print("\n1. DISCOVER — build permanent manifest from compatible scores")
    t0 = time.time()
    for idx, path in enumerate(candidate_paths, start=1):
        piece, manifest = load_classical_piece(path)
        manifest_entries.append(manifest)
        if piece is None:
            continue
        pieces.append(piece)
        total_bars += piece["n_bars"]
        total_slices += piece["n_slices"]
        composer_counts[piece["composer"]] += 1
        if idx % 250 == 0:
            print(f"   scanned {idx:>4}/{len(candidate_paths)}  ok={len(pieces)}  bars={total_bars}")

    manifest = {
        "corpus_root": str(EXTERNAL_CORPORA_DIR),
        "selected_collections": CLASSICAL_COLLECTIONS,
        "parse_timeout_seconds": CLASSICAL_PARSE_TIMEOUT_SECONDS,
        "candidate_files": len(candidate_paths),
        "compatible_pieces": len(pieces),
        "total_bars": total_bars,
        "total_slices": total_slices,
        "composer_counts": dict(sorted(composer_counts.items())),
        "pieces": manifest_entries,
    }
    CLASSICAL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"   Compatible pieces: {len(pieces)}")
    print(f"   Total bars: {total_bars}")
    print(f"   Total slices: {total_slices}")
    print(f"   Manifest saved: {CLASSICAL_MANIFEST_PATH.name}")
    for composer, count in sorted(composer_counts.items()):
        print(f"      {composer:<10} {count:>4} pieces")

    genome = Genome("classical_music")
    bar_adapter = BarLevelAdapter()
    gilgamesh = Lattice(
        bar_adapter,
        epsilon=BACH_EPSILON_SCHEDULE[0],
        max_depth=4,
        epsilon_schedule=BACH_EPSILON_SCHEDULE,
    )

    print("\n2. LIVE — train the lattice on the classical corpus")
    print(f"   Epsilon schedule: {BACH_EPSILON_SCHEDULE}")
    bar_meta = {}
    piece_records = []
    train_t0 = time.time()

    for idx, piece in enumerate(pieces, start=1):
        cells_before = len(gilgamesh.cells)
        closures_before = gilgamesh.closure_counts[:]
        for bar_num, bar_q in enumerate(piece["bar_quats"], start=1):
            key = bar_adapter.embed_bar(piece["piece_id"], bar_num, bar_q)
            bar_meta[key] = {
                "kind": "bar",
                "piece": piece["piece_id"],
                "composer": piece["composer"],
                "relpath": piece["relpath"],
                "title": piece["title"],
                "bars": [bar_num],
                "bar_num": bar_num,
            }
            gilgamesh.ingest(key, meta=bar_meta[key])
        gilgamesh.flush(reason="piece_end")
        piece_cells = gilgamesh.cells[cells_before:]
        for cell in piece_cells:
            genome.record_position(cell.id, cell.q, 1, meta=cell_meta_payload(cell))
        piece_identity_cell = select_piece_identity_cell(piece_cells)
        piece_identity_source = None
        if piece_identity_cell is not None:
            piece_identity_source = piece_identity_cell.id
            genome.record_position(
                f"piece|{piece['piece_id']}",
                piece_identity_cell.q,
                1,
                meta={
                    "kind": "piece",
                    "piece": piece["piece_id"],
                    "title": piece["title"],
                    "composer": piece["composer"],
                    "relpath": piece["relpath"],
                    "source_key": piece_identity_cell.id,
                    "source_level": piece_identity_cell.level,
                    "bars": piece_identity_cell.meta.get("bars", []),
                },
            )
        closures_after = gilgamesh.closure_counts[:]
        new_closures = count_diff(closures_before, closures_after)
        level1_count = sum(1 for cell in piece_cells if cell.meta.get("kind") == "segment")
        higher_count = sum(1 for cell in piece_cells if cell.meta.get("kind") == "higher")
        piece_records.append(
            {
                "piece": piece["piece_id"],
                "composer": piece["composer"],
                "relpath": piece["relpath"],
                "n_bars": piece["n_bars"],
                "closures_per_level": new_closures,
                "segments": level1_count,
                "higher": higher_count,
                "piece_identity_source": piece_identity_source,
            }
        )
        if idx % 100 == 0 or idx == len(pieces):
            print(f"   trained {idx:>4}/{len(pieces)}  composer={piece['composer']:<10}  bars={piece['n_bars']:>4}  genome(bar)={bar_adapter.size}")
        gilgamesh.reset()

    elapsed = time.time() - train_t0
    print(f"   Done: {elapsed:.2f}s  ({total_bars / elapsed:.0f} bars/s)")

    print("\n3. SAVE — persist bars + higher identities")
    for key, (q, count) in bar_adapter.genome.items():
        locked = count == -1
        genome.record_position(key, q, abs(count), locked=locked, meta=bar_meta.get(key))
    active_levels = [k for k in gilgamesh.kernels if k.event_count > 0 or k.emission_count > 0]
    genome.record_hierarchy(
        levels=len(active_levels),
        closures_per_level=[k.emission_count for k in active_levels],
        total_events=gilgamesh.kernels[0].event_count if gilgamesh.kernels else 0,
    )
    genome.save(CLASSICAL_GENOME_PATH)

    kind_counts = defaultdict(int)
    for payload in genome.positions.values():
        kind_counts[payload.get("meta", {}).get("kind", "unknown")] += 1
    mn, mx, mean = genome.spread
    print(f"   Genome saved: {CLASSICAL_GENOME_PATH.name}")
    print(f"   Genome positions: {genome.size}")
    print(f"   σ spread: min={mn:.4f}  max={mx:.4f}  mean={mean:.4f}")
    print(f"   Hierarchy: {genome.hierarchy}")
    for kind in sorted(kind_counts):
        print(f"   {kind:<8}: {kind_counts[kind]}")

    with CLASSICAL_AUDIT_PATH.open("w") as f:
        f.write("Gilgamesh Learns Classical\n\n")
        f.write(f"Selected collections: {CLASSICAL_COLLECTIONS}\n")
        f.write(f"Candidate files: {len(candidate_paths)}\n")
        f.write(f"Compatible pieces: {len(pieces)}\n")
        f.write(f"Total bars: {total_bars}\n")
        f.write(f"Total slices: {total_slices}\n")
        f.write(f"Parse+manifest time: {time.time() - t0:.2f}s\n")
        f.write(f"Training time: {elapsed:.2f}s\n")
        f.write(f"Genome positions: {genome.size}\n")
        f.write(f"Genome σ: min={mn:.4f} max={mx:.4f} mean={mean:.4f}\n")
        f.write(f"Hierarchy: {genome.hierarchy}\n")
        f.write(f"Kind counts: {dict(kind_counts)}\n")
        f.write(f"Composer counts: {dict(sorted(composer_counts.items()))}\n\n")
        f.write("Per-piece summary:\n")
        for rec in piece_records:
            f.write(
                f"  {rec['composer']:<10} {rec['piece']} bars={rec['n_bars']} "
                f"closures={rec['closures_per_level']} segments={rec['segments']} "
                f"higher={rec['higher']} piece_identity={rec['piece_identity_source']}\n"
            )

    print("\n4. COMPLETE")
    print(f"   Manifest: {CLASSICAL_MANIFEST_PATH.name}")
    print(f"   Genome:   {CLASSICAL_GENOME_PATH.name}")
    print(f"   Audit:    {CLASSICAL_AUDIT_PATH.name}")


def benchmark_classical() -> None:
    if not CLASSICAL_GENOME_PATH.exists():
        raise SystemExit(f"Missing genome file: {CLASSICAL_GENOME_PATH}")
    if not CLASSICAL_MANIFEST_PATH.exists():
        raise SystemExit(f"Missing manifest file: {CLASSICAL_MANIFEST_PATH}")

    def build_piece_objects(genome: Genome):
        pieces = []
        for key, payload in genome.positions.items():
            meta = payload.get("meta", {})
            if meta.get("kind") != "piece":
                continue
            pieces.append(
                {
                    "key": key,
                    "piece": meta["piece"],
                    "composer": meta["composer"],
                    "relpath": meta["relpath"],
                    "q": np.array(payload["q"], dtype=np.float64),
                }
            )
        return pieces

    def nearest_piece_neighbors(pieces):
        rows = []
        for i, src in enumerate(pieces):
            best = None
            for j, dst in enumerate(pieces):
                if i == j:
                    continue
                sep = sigma(compose(src["q"], inverse(dst["q"])))
                if best is None or sep < best[0]:
                    best = (sep, dst)
            if best is not None:
                sep, dst = best
                rows.append(
                    {
                        "piece": src["piece"],
                        "composer": src["composer"],
                        "neighbor_piece": dst["piece"],
                        "neighbor_composer": dst["composer"],
                        "sigma": sep,
                        "same_composer": src["composer"] == dst["composer"],
                    }
                )
        return rows

    def composer_mean_separation(pieces):
        by_comp = defaultdict(list)
        for p in pieces:
            by_comp[p["composer"]].append(p)
        rows = []
        for composer, members in sorted(by_comp.items()):
            within = []
            between = []
            for i, src in enumerate(members):
                for j, dst in enumerate(members):
                    if i >= j:
                        continue
                    within.append(sigma(compose(src["q"], inverse(dst["q"]))))
                for other_composer, others in by_comp.items():
                    if other_composer == composer:
                        continue
                    for dst in others[: min(5, len(others))]:
                        between.append(sigma(compose(src["q"], inverse(dst["q"]))))
            rows.append(
                {
                    "composer": composer,
                    "within_mean": sum(within) / len(within) if within else None,
                    "between_mean": sum(between) / len(between) if between else None,
                }
            )
        return rows

    def build_stored_piece_windows(genome: Genome):
        bars_by_piece = defaultdict(list)
        for _key, payload in genome.positions.items():
            meta = payload.get("meta", {})
            if meta.get("kind") != "bar":
                continue
            piece = meta["piece"]
            bar_num = meta["bar_num"]
            q = np.array(payload["q"], dtype=np.float64)
            composer = meta["composer"]
            bars_by_piece[piece].append((bar_num, q, composer))
        stored = {}
        for piece, items in bars_by_piece.items():
            items.sort(key=lambda item: item[0])
            stored[piece] = {"composer": items[0][2], "bars": [q for _, q, _ in items]}
        return stored

    def retrieve_piece_from_cue_classical(cue_quats, stored_bars_by_piece):
        rankings = []
        cue_len = len(cue_quats)
        for piece, payload in stored_bars_by_piece.items():
            bars = payload["bars"]
            if len(bars) < cue_len:
                continue
            best_score = math.inf
            for start in range(len(bars) - cue_len + 1):
                score = mean_window_sigma(cue_quats, bars[start:start + cue_len])
                if score < best_score:
                    best_score = score
            rankings.append((piece, payload["composer"], best_score))
        rankings.sort(key=lambda item: item[2])
        return rankings

    def balanced_sample(manifest):
        rng = random.Random(CLASSICAL_BENCHMARK_SEED)
        by_comp = defaultdict(list)
        for piece in manifest["pieces"]:
            if piece.get("status") == "ok":
                by_comp[piece["composer"]].append(piece)
        sample = []
        for _composer, items in sorted(by_comp.items()):
            rng.shuffle(items)
            sample.extend(items[:CLASSICAL_MAX_PIECES_PER_COMPOSER])
        return sample

    def load_piece_quats(relpath: str):
        path = EXTERNAL_CORPORA_DIR / relpath
        slices, _, voice_order = parse_kern_score(path)
        bars = group_bars(slices)
        adapter = GenericPolyphonicAdapter(voice_order)
        return [compose_bar(adapter, bar_events) for bar_events in bars]

    genome = Genome.load(CLASSICAL_GENOME_PATH)
    manifest = json.loads(CLASSICAL_MANIFEST_PATH.read_text())

    print("GILGAMESH CLASSICAL COMPOSER BENCHMARK")
    print(f"Genome: {CLASSICAL_GENOME_PATH.name}")
    print(f"Manifest: {CLASSICAL_MANIFEST_PATH.name}")
    print(f"Rust: {RUST}")
    print("=" * 70)

    pieces = build_piece_objects(genome)
    nn_rows = nearest_piece_neighbors(pieces)
    same_top1 = sum(1 for row in nn_rows if row["same_composer"])
    print("\n1. PIECE NEIGHBORS — does composer structure emerge?")
    print(f"   Piece objects: {len(pieces)}")
    print(f"   Top-1 same-composer neighbors: {same_top1}/{len(nn_rows)} ({same_top1/len(nn_rows)*100:.1f}%)")

    sample = balanced_sample(manifest)
    stored_bars = build_stored_piece_windows(genome)
    cue_rows = []
    print("\n2. CUE TO COMPOSER — balanced retrieval sample")
    print(f"   Cue length: {CLASSICAL_BENCHMARK_CUE_LENGTH} bars")
    print(f"   Sample pieces: {len(sample)}")

    for idx, piece in enumerate(sample, start=1):
        bar_quats = load_piece_quats(piece["relpath"])
        if len(bar_quats) < CLASSICAL_BENCHMARK_CUE_LENGTH:
            continue
        cue_quats = bar_quats[:CLASSICAL_BENCHMARK_CUE_LENGTH]
        rankings = retrieve_piece_from_cue_classical(cue_quats, stored_bars)
        best_piece, best_composer, best_score = rankings[0]
        cue_rows.append(
            {
                "piece": piece["piece_id"],
                "composer": piece["composer"],
                "retrieved_piece": best_piece,
                "retrieved_composer": best_composer,
                "sigma": best_score,
                "same_composer": piece["composer"] == best_composer,
                "same_piece": piece["piece_id"] == best_piece,
                "top5": rankings[:5],
            }
        )
        if idx % 25 == 0 or idx == len(sample):
            print(f"   processed {idx:>3}/{len(sample)}")

    cue_same_composer = sum(1 for row in cue_rows if row["same_composer"])
    cue_same_piece = sum(1 for row in cue_rows if row["same_piece"])
    print(f"   Composer accuracy: {cue_same_composer}/{len(cue_rows)} ({cue_same_composer/len(cue_rows)*100:.1f}%)")
    print(f"   Exact piece accuracy: {cue_same_piece}/{len(cue_rows)} ({cue_same_piece/len(cue_rows)*100:.1f}%)")

    sep_rows = composer_mean_separation(pieces)
    with CLASSICAL_COMPOSER_AUDIT_PATH.open("w") as f:
        f.write("Gilgamesh Classical Composer Benchmark\n\n")
        f.write(f"Piece objects: {len(pieces)}\n")
        f.write(f"Top-1 same-composer neighbors: {same_top1}/{len(nn_rows)}\n")
        f.write(f"Balanced cue sample: {len(cue_rows)}\n")
        f.write(f"Cue composer accuracy: {cue_same_composer}/{len(cue_rows)}\n")
        f.write(f"Cue exact-piece accuracy: {cue_same_piece}/{len(cue_rows)}\n\n")
        f.write("Nearest-neighbor piece rows:\n")
        for row in nn_rows:
            verdict = "SAME" if row["same_composer"] else "DIFF"
            f.write(
                f"  {verdict} {row['composer']}:{row['piece']} -> "
                f"{row['neighbor_composer']}:{row['neighbor_piece']} σ={row['sigma']:.6f}\n"
            )
        f.write("\nComposer separation:\n")
        for row in sep_rows:
            f.write(f"  {row['composer']}: within_mean={row['within_mean']:.6f} between_mean={row['between_mean']:.6f}\n")
        f.write("\nCue rows:\n")
        for row in cue_rows:
            verdict = "PASS" if row["same_composer"] else "FAIL"
            f.write(
                f"  {verdict} {row['composer']}:{row['piece']} -> "
                f"{row['retrieved_composer']}:{row['retrieved_piece']} σ={row['sigma']:.6f}\n"
            )


def evolved_lattice() -> None:
    if not EVOLVED_LATTICE_SOURCE.exists():
        raise SystemExit(f"Missing source score: {EVOLVED_LATTICE_SOURCE}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    slices, _, voice_order = parse_kern_score(EVOLVED_LATTICE_SOURCE)
    poly_adapter = GenericPolyphonicAdapter(voice_order)
    adapter = SliceLevelAdapter(poly_adapter)
    gilgamesh = Lattice(
        adapter,
        epsilon=EVOLVED_LATTICE_SCHEDULE[0],
        max_depth=6,
        epsilon_schedule=EVOLVED_LATTICE_SCHEDULE,
    )

    print("MUSIC EVOLVED LATTICE")
    print(f"Source: {EVOLVED_LATTICE_SOURCE.name}")
    print(f"Slices: {len(slices)}")
    print(f"Voices: {voice_order}")
    print(f"Epsilon schedule: {EVOLVED_LATTICE_SCHEDULE}")
    print(f"Rust: {RUST}")
    print("=" * 70)

    last_bar = None
    for idx, slice_event in enumerate(slices):
        key = f"{EVOLVED_LATTICE_SOURCE.stem}:slice:{idx:04d}"
        adapter.register_slice(key, slice_event)
        slice_meta = {
            "kind": "slice",
            "piece": EVOLVED_LATTICE_SOURCE.stem,
            "bar": slice_event.bar,
            "start": float(slice_event.start),
            "duration": float(slice_event.duration),
            "label": event_label(slice_event),
            "bars": [slice_event.bar],
        }
        if last_bar is not None and slice_event.bar != last_bar:
            gilgamesh.flush(reason="bar_end")
        gilgamesh.ingest(key, meta=slice_meta)
        last_bar = slice_event.bar

    gilgamesh.flush(reason="piece_end")
    piece_source = select_piece_identity_cell(gilgamesh.cells)

    nodes = []
    for leaf in gilgamesh.leaf_nodes.values():
        meta = leaf.get("meta", {})
        bars = meta.get("bars", [meta.get("bar")] if meta.get("bar") is not None else [])
        nodes.append(
            {
                "id": leaf["id"],
                "kind": meta.get("kind", "event"),
                "level": 0,
                "piece": meta.get("piece", EVOLVED_LATTICE_SOURCE.stem),
                "bar_start": bars[0] if bars else None,
                "bar_end": bars[-1] if bars else None,
                "bars": bars,
                "time_start": meta.get("start"),
                "duration": meta.get("duration"),
                "label": meta.get("label"),
                "source_key": meta.get("event_key"),
            }
        )

    edges = []
    for cell in gilgamesh.cells:
        cell_node = {
            "id": cell.id,
            "kind": cell.meta.get("kind", "segment" if cell.level == 1 else "higher"),
            "level": cell.level,
            "piece": cell.meta.get("piece", EVOLVED_LATTICE_SOURCE.stem),
            "reason": cell.reason,
            "bar_start": cell.meta.get("bar_start"),
            "bar_end": cell.meta.get("bar_end"),
            "bars": cell.meta.get("bars", []),
            "trigger_bars": cell.meta.get("trigger_bars", []),
            "content_event_count": cell.event_count,
            "sigma": float(cell.gap),
            "q": cell.q.tolist(),
            "child_keys": list(cell.source_keys),
        }
        nodes.append(cell_node)
        for child_id in cell.child_ids:
            edges.append({"parent": cell.id, "child": child_id})

    piece_id = None
    if piece_source is not None:
        piece_id = f"piece|{EVOLVED_LATTICE_SOURCE.stem}"
        nodes.append(
            {
                "id": piece_id,
                "kind": "piece",
                "level": piece_source.level + 1,
                "piece": EVOLVED_LATTICE_SOURCE.stem,
                "reason": "piece_end",
                "bar_start": piece_source.meta.get("bar_start"),
                "bar_end": piece_source.meta.get("bar_end"),
                "bars": piece_source.meta.get("bars", []),
                "sigma": float(piece_source.gap),
                "q": piece_source.q.tolist(),
                "source_key": piece_source.id,
            }
        )
        edges.append({"parent": piece_id, "child": piece_source.id})

    nodes.sort(key=lambda item: (item["level"], item.get("bar_start") or 0, item["id"]))
    payload = {
        "piece": EVOLVED_LATTICE_SOURCE.stem,
        "source_path": str(EVOLVED_LATTICE_SOURCE),
        "voices": list(voice_order),
        "slice_count": len(slices),
        "epsilon_schedule": EVOLVED_LATTICE_SCHEDULE,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "piece_id": piece_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "closures_per_level": dict(sorted(Counter(cell.level for cell in gilgamesh.cells).items())),
            "gilgamesh_status": gilgamesh.status(),
        },
    }
    EVOLVED_LATTICE_JSON.write_text(json.dumps(payload, indent=2))

    level_counts = Counter(node["level"] for node in nodes)
    kind_counts = Counter(node["kind"] for node in nodes)
    with EVOLVED_LATTICE_AUDIT.open("w") as f:
        f.write("Music Evolved Lattice\n\n")
        f.write(f"Source: {EVOLVED_LATTICE_SOURCE}\n")
        f.write(f"Slices: {len(slices)}\n")
        f.write(f"Voices: {voice_order}\n")
        f.write(f"Epsilon schedule: {EVOLVED_LATTICE_SCHEDULE}\n")
        f.write(f"Node count: {len(nodes)}\n")
        f.write(f"Edge count: {len(edges)}\n")
        f.write(f"Level counts: {dict(sorted(level_counts.items()))}\n")
        f.write(f"Kind counts: {dict(sorted(kind_counts.items()))}\n")
        f.write(f"Closures per level: {dict(sorted(Counter(cell.level for cell in gilgamesh.cells).items()))}\n")
        f.write(f"Piece node: {piece_id}\n\n")
        f.write("Top-level status:\n")
        for row in gilgamesh.status():
            f.write(f"  level={row['level']} events={row['events']} closures={row['closures']} gap={row['gap']:.6f}\n")

    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print(f"Level counts: {dict(sorted(level_counts.items()))}")
    print(f"Kind counts: {dict(sorted(kind_counts.items()))}")
    print(f"Closures per level: {dict(sorted(Counter(cell.level for cell in gilgamesh.cells).items()))}")
    print(f"Saved JSON: {EVOLVED_LATTICE_JSON.name}")
    print(f"Saved audit: {EVOLVED_LATTICE_AUDIT.name}")


COMMANDS = {
    "train-bach": train_bach,
    "retrieve-bach": retrieve_bach,
    "generate-bach": generate_bach,
    "section-bach": section_bach,
    "improvise-bach": improvise_bach,
    "train-classical": train_classical,
    "benchmark-classical": benchmark_classical,
    "evolved-lattice": evolved_lattice,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gilgamesh_music.py", description="Canonical operational file for the Gilgamesh music track.")
    parser.add_argument("command", choices=sorted(COMMANDS), help="Operation to run.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
