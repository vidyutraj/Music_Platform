"""Chord tracks: which chord is playing when, and how to name it.

`ChordTrack` is the value the rest of the package works on. `detect_chords` fills one in
with the lv-chordia model (Jiang et al., ISMIR 2019), which knows 7ths, half-diminished,
diminished 7ths, sus chords, 9/11/13 and common inversions. Like `detect_beats`, it
imports the model lazily so everything else stays light and testable without it.

Chords are stored as pitch classes (0 = C ... 11 = B) rather than names, so transposing is
plain arithmetic and spelling (B♭ vs A♯) is decided only when a name is printed.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from music_platform.beats import BeatGrid

NO_CHORD = "N"

# Model quality (Harte syntax) -> what musicians write after the root.
QUALITY_SUFFIX = {
    "maj": "",
    "min": "m",
    "dim": "°",
    "aug": "+",
    "7": "7",
    "maj7": "maj7",
    "min7": "m7",
    "dim7": "°7",
    "hdim7": "m7♭5",
    "minmaj7": "m(maj7)",
    "maj6": "6",
    "min6": "m6",
    "9": "9",
    "maj9": "maj9",
    "min9": "m9",
    "11": "11",
    "13": "13",
    "sus2": "sus2",
    "sus4": "sus4",
    "sus4(b7)": "7sus4",
}

# Chord tones as semitones above the root; used to guess the key.
QUALITY_TONES = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
    "minmaj7": (0, 3, 7, 11),
    "maj6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
    "9": (0, 4, 7, 10, 2),
    "maj9": (0, 4, 7, 11, 2),
    "min9": (0, 3, 7, 10, 2),
    "11": (0, 7, 10, 2, 5),
    "13": (0, 4, 7, 10, 2, 9),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "sus4(b7)": (0, 5, 7, 10),
}

# Harte bass degrees ("G:7/3" = G7 with its 3rd, B, in the bass) -> semitones above root.
DEGREE_SEMITONES = {
    "1": 0, "b2": 1, "2": 2, "#2": 3, "b3": 3, "3": 4, "4": 5, "#4": 6, "b5": 6, "5": 7,
    "#5": 8, "b6": 8, "6": 9, "bb7": 9, "b7": 10, "7": 11, "b9": 1, "9": 2, "#9": 3,
    "11": 5, "#11": 6, "b13": 8, "13": 9,
}  # fmt: skip

LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_NAMES = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")
FLAT_NAMES = ("C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B")
# Major keys written with flats: F, B♭, E♭, A♭, D♭ (G♭/F♯ is spelled with sharps).
FLAT_MAJOR_TONICS = frozenset({5, 10, 3, 8, 1})

# Krumhansl-Kessler key profiles, index 0 = tonic.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_ROOT_RE = re.compile(r"^([A-G])([#b]*)$")


def note_pc(name: str) -> int:
    """Pitch class of a note name like "C", "F#", "Bb" (0 = C)."""
    match = _ROOT_RE.match(name)
    if match is None:
        raise ValueError(f"not a note name: {name!r}")
    letter, accidentals = match.groups()
    return (LETTER_PC[letter] + accidentals.count("#") - accidentals.count("b")) % 12


def spell(pc: int, prefer_flats: bool) -> str:
    return (FLAT_NAMES if prefer_flats else SHARP_NAMES)[pc % 12]


@dataclass(frozen=True)
class Chord:
    """One chord over a span of time. `root` is None for "no chord" (silence, drums only)."""

    start: float
    end: float
    root: int | None
    quality: str = "maj"
    bass: int | None = None  # only set when it differs from the root (an inversion)

    @property
    def is_chord(self) -> bool:
        return self.root is not None

    def same_harmony(self, other: Chord) -> bool:
        return (self.root, self.quality, self.bass) == (other.root, other.quality, other.bass)

    def transposed(self, semitones: int) -> Chord:
        if self.root is None:
            return self
        bass = None if self.bass is None else (self.bass + semitones) % 12
        return replace(self, root=(self.root + semitones) % 12, bass=bass)


def parse_harte(label: str, start: float, end: float) -> Chord:
    """A Harte label from the model ("C:maj7", "G:7/3", "Bb:hdim7", "N") as a Chord."""
    label = label.strip()
    if label in (NO_CHORD, "X", ""):
        return Chord(start, end, None, NO_CHORD)
    body, _, bass_degree = label.partition("/")
    root_name, _, quality = body.partition(":")
    root = note_pc(root_name)
    quality = quality or "maj"
    bass = None
    if bass_degree:
        if bass_degree not in DEGREE_SEMITONES:
            raise ValueError(f"unknown bass degree in {label!r}")
        bass = (root + DEGREE_SEMITONES[bass_degree]) % 12
        if bass == root:
            bass = None
    return Chord(start, end, root, quality, bass)


def chord_name(chord: Chord, prefer_flats: bool = False) -> str:
    """How a musician would write it: "Cmaj7", "Bm7♭5", "G/B", "N.C." for no chord."""
    if chord.root is None:
        return "N.C."
    suffix = QUALITY_SUFFIX.get(chord.quality, chord.quality)
    name = spell(chord.root, prefer_flats) + suffix
    if chord.bass is not None:
        name += "/" + spell(chord.bass, prefer_flats)
    return name


@dataclass(frozen=True)
class Key:
    tonic: int
    mode: str  # "major" or "minor"

    @property
    def prefer_flats(self) -> bool:
        relative_major = self.tonic if self.mode == "major" else (self.tonic + 3) % 12
        return relative_major in FLAT_MAJOR_TONICS

    @property
    def name(self) -> str:
        return f"{spell(self.tonic, self.prefer_flats)} {self.mode}"

    def transposed(self, semitones: int) -> Key:
        return Key((self.tonic + semitones) % 12, self.mode)


@dataclass(frozen=True)
class ChordTrack:
    """Chords in time order, each starting where the previous one ends."""

    chords: tuple[Chord, ...]

    @classmethod
    def from_labels(cls, segments: list[tuple[float, float, str]]) -> ChordTrack:
        return cls(tuple(parse_harte(label, start, end) for start, end, label in segments))

    def to_labels(self) -> list[tuple[float, float, str]]:
        return [(c.start, c.end, _to_harte(c)) for c in self.chords]

    def to_json(self, path: Path) -> None:
        payload = [{"start": s, "end": e, "chord": label} for s, e, label in self.to_labels()]
        path.write_text(json.dumps(payload))

    @classmethod
    def from_json(cls, path: Path) -> ChordTrack:
        payload = json.loads(path.read_text())
        return cls.from_labels([(p["start"], p["end"], p["chord"]) for p in payload])

    def scaled(self, speed: float) -> ChordTrack:
        """The track for the same song played at `speed` (1.25 = 25% faster)."""
        if speed <= 0:
            raise ValueError("speed must be positive")
        return ChordTrack(
            tuple(replace(c, start=c.start / speed, end=c.end / speed) for c in self.chords)
        )

    def transposed(self, semitones: int) -> ChordTrack:
        return ChordTrack(tuple(c.transposed(semitones) for c in self.chords))

    def chord_at(self, time: float) -> Chord | None:
        for chord in self.chords:
            if chord.start <= time < chord.end:
                return chord
        return None


def _to_harte(chord: Chord) -> str:
    if chord.root is None:
        return NO_CHORD
    label = f"{SHARP_NAMES[chord.root].replace('♯', '#')}:{chord.quality}"
    if chord.bass is not None:
        interval = (chord.bass - chord.root) % 12
        label += "/" + _BASS_DEGREE_FOR_INTERVAL[interval]
    return label


_BASS_DEGREE_FOR_INTERVAL = {
    0: "1", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "b5", 7: "5", 8: "b6", 9: "6",
    10: "b7", 11: "7",
}  # fmt: skip


def estimate_key(track: ChordTrack) -> Key:
    """Best-fitting major or minor key, from how long each pitch class sounds in the chords.

    Chord tones (plus the bass) are weighted by duration and correlated against the
    Krumhansl-Kessler key profiles, the standard approach for key finding.
    """
    histogram = np.zeros(12)
    for chord in track.chords:
        if chord.root is None:
            continue
        duration = chord.end - chord.start
        for interval in QUALITY_TONES.get(chord.quality, (0,)):
            histogram[(chord.root + interval) % 12] += duration
        histogram[chord.root] += duration  # the root carries the harmony
        if chord.bass is not None:
            histogram[chord.bass] += duration
    if not histogram.any():
        return Key(0, "major")

    best = Key(0, "major")
    best_score = -np.inf
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for tonic in range(12):
            score = float(np.corrcoef(histogram, np.roll(profile, tonic))[0, 1])
            if score > best_score:
                best, best_score = Key(tonic, mode), score
    return best


def snap_to_beats(
    track: ChordTrack,
    grid: BeatGrid,
    max_shift_beats: float = 1 / 3,
    min_length_beats: float = 0.5,
) -> ChordTrack:
    """Line chord changes up with the beats, the way a lead sheet would write them.

    A change within `max_shift_beats` of a beat moves onto it. Afterwards, any chord
    shorter than `min_length_beats` (usually the model wavering at a transition) is
    absorbed by the chord before it, and repeated chords are merged.
    """
    if len(grid.beats) < 2 or len(track.chords) < 2:
        return track
    period = 60.0 / grid.bpm
    beats = grid.beats

    def snap(t: float) -> float:
        nearest = float(beats[np.argmin(np.abs(beats - t))])
        return nearest if abs(nearest - t) <= max_shift_beats * period else t

    chords = list(track.chords)
    bounds = [chords[0].start] + [snap(c.start) for c in chords[1:]] + [chords[-1].end]
    for i in range(1, len(bounds)):
        bounds[i] = max(bounds[i], bounds[i - 1])  # keep time order if two snap to one beat
    moved = [replace(c, start=bounds[i], end=bounds[i + 1]) for i, c in enumerate(chords)]

    kept: list[Chord] = []
    for chord in moved:
        too_short = chord.end - chord.start < min_length_beats * period
        if kept and (too_short or kept[-1].same_harmony(chord)):
            kept[-1] = replace(kept[-1], end=chord.end)
        elif kept and kept[-1].end - kept[-1].start < min_length_beats * period:
            # A blip at the very start: let the next chord take its time.
            kept[-1] = replace(chord, start=kept[-1].start)
        else:
            kept.append(chord)
    return ChordTrack(tuple(kept))


def detect_chords(audio_path: Path) -> ChordTrack:
    """Run the lv-chordia large-vocabulary model (CPU; about 15 s for a 4-minute song)."""
    from lv_chordia import chord_recognition

    # The model prints a progress line per ensemble member; keep the terminal clean.
    with contextlib.redirect_stderr(io.StringIO()):
        segments = chord_recognition(str(audio_path), chord_dict_name="submission")
    return ChordTrack.from_labels(
        [(float(s["start_time"]), float(s["end_time"]), str(s["chord"])) for s in segments]
    )


def bar_chart(track: ChordTrack, grid: BeatGrid, prefer_flats: bool = False) -> list[str]:
    """A plain-text chord chart, one bar per line, one column per beat.

    Each bar names its opening chord, and after that a name appears only where the chord
    changes ("." = keep playing the same chord). A pickup before the first downbeat is bar 0.
    """
    if len(grid.beats) == 0:
        return []
    starts = list(grid.downbeats)
    if not starts or grid.beats[0] < starts[0] - 0.05:
        starts.insert(0, float(grid.beats[0]))
        first_bar = 0
    else:
        first_bar = 1

    lines = []
    for i, bar_start in enumerate(starts):
        bar_end = starts[i + 1] if i + 1 < len(starts) else np.inf
        bar_beats = [b for b in grid.beats if bar_start - 0.05 <= b < bar_end - 0.05]
        cells, previous = [], None
        for beat in bar_beats:
            chord = track.chord_at(beat + 0.05)  # just after the beat, past any wobble
            name = chord_name(chord, prefer_flats) if chord else "N.C."
            cells.append(f"{name if name != previous else '.':<8}")
            previous = name
        lines.append(f"{first_bar + i:>4} | {''.join(cells).rstrip()}")
    return lines
