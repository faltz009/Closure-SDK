"""
Generic Humdrum `**kern` parsing and polyphonic S^3 embedding.

This generalizes the two-voice invention harness to arbitrary fixed spine
counts so larger Bach corpora can be loaded without rewriting the adapter for
each collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable
import math
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from closure_ea.cell import Adapter
from closure_ea.kernel import compose, identity


FIFTHS = ["C", "G", "D", "A", "E", "B", "Gb", "Db", "Ab", "Eb", "Bb", "F"]
CHROMATIC = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
FIFTHS_IDX = {n: i for i, n in enumerate(FIFTHS)}
CHROM_IDX = {n: i for i, n in enumerate(CHROMATIC)}


@dataclass(frozen=True)
class NoteEvent:
    voice: str
    midi: int
    start: Fraction
    duration: Fraction

    @property
    def end(self) -> Fraction:
        return self.start + self.duration


@dataclass(frozen=True)
class SliceEvent:
    bar: int
    start: Fraction
    duration: Fraction
    voices: tuple[tuple[str, tuple[int, ...]], ...]

    def notes_for_voice(self, voice: str) -> tuple[int, ...]:
        for name, notes in self.voices:
            if name == voice:
                return notes
        return ()


def parse_duration(token: str) -> Fraction:
    i = 0
    while i < len(token) and not token[i].isdigit():
        i += 1
    if i >= len(token):
        raise ValueError(f"No duration in token: {token!r}")
    digits = []
    while i < len(token) and token[i].isdigit():
        digits.append(token[i])
        i += 1
    denom = int("".join(digits))
    base = Fraction(4, denom)
    dots = 0
    while i < len(token) and token[i] == ".":
        dots += 1
        i += 1
    dur = base
    add = base
    for _ in range(dots):
        add /= 2
        dur += add
    return dur


def token_pitch_to_midi(token: str) -> int:
    cleaned = "".join(ch for ch in token if ch not in "[]_(){};/\\'`~^vLMWQPOq")
    letters = "".join(ch for ch in cleaned if ch.isalpha() and ch.lower() in "abcdefg")
    if not letters:
        raise ValueError(f"No pitch letters in token: {token!r}")

    base_letter = letters[0]
    repeats = len(letters)
    pc_map = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
    midi = pc_map[base_letter.lower()]

    if base_letter.islower():
        octave = 4 + (repeats - 1)
    else:
        octave = 3 - (repeats - 1)

    accidental = cleaned.count("#") - cleaned.count("-")
    if "n" in cleaned:
        accidental = 0

    midi += accidental + 12 * (octave + 1)
    return midi


def parse_spine_token(token: str, voice: str, now: Fraction) -> tuple[list[NoteEvent], Fraction] | None:
    if token == ".":
        return None

    subtokens = token.split()
    duration = parse_duration(subtokens[0])
    events: list[NoteEvent] = []
    for subtoken in subtokens:
        if "r" in subtoken:
            continue
        midi = token_pitch_to_midi(subtoken)
        events.append(NoteEvent(voice=voice, midi=midi, start=now, duration=duration))
    return events, duration


def parse_kern_score(path: Path) -> tuple[list[SliceEvent], list[str], tuple[str, ...]]:
    lines = path.read_text().splitlines()
    now = Fraction(0, 1)
    bar = 0
    notes: list[NoteEvent] = []
    voice_order: tuple[str, ...] | None = None
    active_end: dict[str, Fraction] = {}
    bar_markers: list[Fraction] = []

    for raw in lines:
        if raw.startswith("!"):
            continue
        if raw.startswith("**"):
            spines = raw.split("\t")
            voice_order = tuple(f"v{i+1}" for i in range(len(spines)))
            active_end = {voice: Fraction(0, 1) for voice in voice_order}
            continue
        if raw.startswith("*"):
            continue
        if raw.startswith("="):
            parts = raw.split("\t")
            label = parts[0].lstrip("=")
            digits = "".join(ch for ch in label if ch.isdigit())
            if digits:
                bar = int(digits)
            else:
                bar += 1
            bar_markers.append(now)
            continue
        if "\t" not in raw:
            continue
        if voice_order is None:
            raise ValueError(f"No **kern spine header before data in {path}")

        parts = raw.split("\t")
        if len(parts) < len(voice_order):
            parts = parts + ["."] * (len(voice_order) - len(parts))

        for voice, token in zip(voice_order, parts[: len(voice_order)]):
            parsed = parse_spine_token(token.strip(), voice, now)
            if parsed is not None:
                new_events, duration = parsed
                notes.extend(new_events)
                active_end[voice] = now + duration

        deltas = [end - now for end in active_end.values() if end > now]
        if deltas:
            now += min(deltas)

    total_duration = max((note.end for note in notes), default=Fraction(0, 1))
    boundaries = sorted(set([Fraction(0, 1), total_duration] + [n.start for n in notes] + [n.end for n in notes]))
    slices: list[SliceEvent] = []

    bar_starts = list(bar_markers)
    if not bar_starts or bar_starts[0] != Fraction(0, 1):
        bar_starts = [Fraction(0, 1)] + bar_starts

    bar_index = 1
    next_bar_ptr = 1
    next_bar_start = bar_starts[next_bar_ptr] if next_bar_ptr < len(bar_starts) else None

    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        while next_bar_start is not None and start >= next_bar_start:
            bar_index += 1
            next_bar_ptr += 1
            next_bar_start = bar_starts[next_bar_ptr] if next_bar_ptr < len(bar_starts) else None

        voice_map: dict[str, list[int]] = {voice: [] for voice in voice_order}
        for note in notes:
            if note.start <= start < note.end:
                voice_map[note.voice].append(note.midi)

        if not any(voice_map.values()):
            continue

        voices = tuple((voice, tuple(sorted(voice_map[voice]))) for voice in voice_order)
        slices.append(SliceEvent(bar=bar_index, start=start, duration=end - start, voices=voices))

    raw_data_lines = [raw for raw in lines if raw and not raw.startswith("!") and not raw.startswith("*")]
    return slices, raw_data_lines, voice_order


def group_bars(slices: Iterable[SliceEvent]) -> list[list[SliceEvent]]:
    bars: dict[int, list[SliceEvent]] = {}
    for event in slices:
        bars.setdefault(event.bar, []).append(event)
    return [bars[idx] for idx in sorted(bars)]


def midi_to_label(midi: int) -> str:
    pc = CHROMATIC[midi % 12]
    octave = (midi // 12) - 1
    return f"{pc}{octave}"


def event_label(event: SliceEvent) -> str:
    parts = []
    for voice, notes in event.voices:
        rendered = ",".join(midi_to_label(m) for m in notes) if notes else "rest"
        parts.append(f"{voice}:{rendered}")
    return " | ".join(parts)


class GenericPolyphonicAdapter(Adapter):
    """Voice-aware note embedding on S^3 for an arbitrary fixed spine count."""

    def __init__(self, voice_order: tuple[str, ...]):
        super().__init__(damping=0.03)
        self.voice_order = voice_order
        self.voice_index = {voice: idx for idx, voice in enumerate(voice_order)}

    def _voice_angle(self, voice: str) -> float:
        idx = self.voice_index[voice]
        return -0.27 + idx * 0.18

    def embed_pitch(self, midi: int, voice: str) -> np.ndarray:
        key = f"{voice}:{midi}"
        if key in self.genome:
            return self.genome[key][0].copy()

        pitch_class = CHROMATIC[midi % 12]
        octave = (midi // 12) - 1

        theta_h = (FIFTHS_IDX[pitch_class] / 12) * 2 * math.pi
        theta_c = (CHROM_IDX[pitch_class] / 12) * 2 * math.pi
        theta_o = ((octave - 2) / 6) * math.pi
        theta_v = self._voice_angle(voice)

        q_h = np.array([math.cos(theta_h / 2), math.sin(theta_h / 2), 0.0, 0.0])
        q_c = np.array([math.cos(theta_c / 2), 0.0, math.sin(theta_c / 2), 0.0])
        q_o = np.array([math.cos(theta_o / 2), 0.0, 0.0, math.sin(theta_o / 2)])
        q_v = np.array([math.cos(theta_v / 2), math.sin(theta_v / 2), 0.0, math.sin(theta_v / 2)])

        q = compose(compose(q_h, q_c), compose(q_o, q_v))
        q /= np.linalg.norm(q)
        self.embed_exact(key, q)
        return q.copy()

    def embed_rest(self, voice: str, duration: Fraction) -> np.ndarray:
        beats = float(duration)
        angle = 0.07 + min(beats, 4.0) * 0.03
        voice_shift = self._voice_angle(voice)
        q = np.array([math.cos(angle), 0.0, math.sin(angle + voice_shift), 0.0])
        q /= np.linalg.norm(q)
        return q

    def embed_slice(self, event: SliceEvent) -> np.ndarray:
        q = identity()
        beat_pos = float(event.start % 4)
        theta_t = (beat_pos / 4.0) * (math.pi / 2.0)
        q_time = np.array([math.cos(theta_t / 2), 0.0, 0.0, math.sin(theta_t / 2)])
        q = compose(q, q_time)

        theta_d = min(float(event.duration), 4.0) * 0.11
        q_dur = np.array([math.cos(theta_d / 2), 0.0, math.sin(theta_d / 2), math.sin(theta_d / 2) * 0.25])
        q = compose(q, q_dur)

        for voice in self.voice_order:
            notes = event.notes_for_voice(voice)
            if notes:
                for midi in notes:
                    q = compose(q, self.embed_pitch(midi, voice))
            else:
                q = compose(q, self.embed_rest(voice, event.duration))
        return q / np.linalg.norm(q)


def compose_bar(adapter: GenericPolyphonicAdapter, events: list[SliceEvent]) -> np.ndarray:
    q = identity()
    for event in events:
        q = compose(q, adapter.embed_slice(event))
    return q / np.linalg.norm(q)


def write_polyphonic_wav(
    slices: list[SliceEvent],
    filename: Path,
    tempo_qpm: float = 59.3,
    sr: int = 22050,
) -> float:
    quarter_seconds = 60.0 / tempo_qpm
    samples: list[int] = []

    for event in slices:
        dur_seconds = float(event.duration) * quarter_seconds
        ns = max(1, int(dur_seconds * sr))
        active_midis = [m for _, notes in event.voices for m in notes]
        if not active_midis:
            samples.extend([0] * ns)
            continue

        freqs = [440.0 * (2 ** ((m - 69) / 12)) for m in active_midis]
        voice_gain = 0.22 / max(1, len(freqs))
        for i in range(ns):
            t = i / sr
            attack = min(1.0, i / (sr * 0.004))
            decay = math.exp(-t * 2.7)
            value = 0.0
            for freq in freqs:
                value += voice_gain * (
                    math.sin(2 * math.pi * freq * t)
                    + 0.25 * math.sin(2 * math.pi * freq * 2 * t)
                    + 0.08 * math.sin(2 * math.pi * freq * 3 * t)
                )
            samples.append(int(max(-1.0, min(1.0, attack * decay * value)) * 32767))

    with filename.open("wb") as f:
        n = len(samples)
        f.write(b"RIFF" + struct.pack("<I", 36 + n * 2) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", 16) + struct.pack("<HHIIHH", 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data" + struct.pack("<I", n * 2))
        for sample in samples:
            f.write(struct.pack("<h", sample))
    return len(samples) / sr
