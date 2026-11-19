import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_platform.beats import BeatGrid
from music_platform.chords import (
    Chord,
    ChordTrack,
    bar_chart,
    chord_name,
    detect_chords,
    estimate_key,
    parse_harte,
    snap_to_beats,
)


def name(label: str, flats: bool = False) -> str:
    return chord_name(parse_harte(label, 0, 1), flats)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("C:maj", "C"),
        ("C", "C"),
        ("C:maj7", "Cmaj7"),
        ("A:min7", "Am7"),
        ("G:7/3", "G7/B"),
        ("B:hdim7", "Bm7♭5"),
        ("C#:dim7", "C♯°7"),
        ("F:maj/5", "F/C"),
        ("A:min/b7", "Am/G"),
        ("D:sus4(b7)", "D7sus4"),
        ("E:min9", "Em9"),
        ("N", "N.C."),
        ("X", "N.C."),
    ],
)
def test_harte_labels_read_as_musicians_write_them(label: str, expected: str) -> None:
    assert name(label) == expected


def test_flat_roots_parse_and_spell_with_flats() -> None:
    assert name("Bb:maj7", flats=True) == "B♭maj7"
    assert name("Bb:maj7", flats=False) == "A♯maj7"


def test_bass_equal_to_root_is_not_an_inversion() -> None:
    assert parse_harte("C:maj/1", 0, 1).bass is None


def test_rejects_garbage_labels() -> None:
    with pytest.raises(ValueError):
        parse_harte("H:maj", 0, 1)


def test_transpose_moves_root_and_bass_but_not_quality() -> None:
    assert chord_name(parse_harte("C:maj7", 0, 1).transposed(2)) == "Dmaj7"
    assert chord_name(parse_harte("G:7/3", 0, 1).transposed(-2)) == "F7/A"
    assert parse_harte("N", 0, 1).transposed(5).root is None


def track(*labels: str, length: float = 2.0) -> ChordTrack:
    return ChordTrack.from_labels(
        [(i * length, (i + 1) * length, label) for i, label in enumerate(labels)]
    )


def test_key_of_a_bb_major_turnaround_uses_flats() -> None:
    key = estimate_key(track("Bb:maj7", "G:min7", "C:min7", "F:7"))
    assert key.name == "B♭ major"
    assert key.prefer_flats


def test_key_of_a_minor_ii_v_i() -> None:
    key = estimate_key(track("B:hdim7", "E:7", "A:min", "A:min", "D:min7", "E:7", "A:min"))
    assert key.name == "A minor"
    assert not key.prefer_flats


def test_key_follows_transposition() -> None:
    song = track("C:maj", "F:maj", "G:7", "C:maj")
    assert estimate_key(song).name == "C major"
    assert estimate_key(song.transposed(5)).name == "F major"
    assert estimate_key(song.transposed(5)).prefer_flats


def grid_120bpm(bars: int = 4) -> BeatGrid:
    beats = [i * 0.5 for i in range(bars * 4)]
    return BeatGrid.from_lists(beats, beats[::4])


def test_snap_moves_changes_onto_nearby_beats() -> None:
    raw = ChordTrack.from_labels([(0.0, 2.04, "C:maj"), (2.04, 4.0, "A:min")])
    snapped = snap_to_beats(raw, grid_120bpm())
    assert [c.start for c in snapped.chords] == [0.0, 2.0]


def test_snap_leaves_changes_far_from_beats_alone() -> None:
    raw = ChordTrack.from_labels([(0.0, 2.25, "C:maj"), (2.25, 4.0, "A:min")])
    assert [c.start for c in snap_to_beats(raw, grid_120bpm()).chords] == [0.0, 2.25]


def test_snap_absorbs_short_blips_and_merges_repeats() -> None:
    raw = ChordTrack.from_labels(
        [(0.0, 0.02, "N"), (0.02, 2.0, "C:maj"), (2.0, 2.1, "E:min"), (2.1, 4.0, "C:maj")]
    )
    snapped = snap_to_beats(raw, grid_120bpm())
    assert len(snapped.chords) == 1
    assert snapped.chords[0].start == 0.0
    assert snapped.chords[0].end == 4.0
    assert chord_name(snapped.chords[0]) == "C"


def test_json_round_trip_keeps_inversions(tmp_path: Path) -> None:
    original = track("C:maj7", "G:7/3", "A:min/b7", "N")
    original.to_json(tmp_path / "chords.json")
    loaded = ChordTrack.from_json(tmp_path / "chords.json")
    assert loaded == original


def test_scaled_matches_scaled_beats() -> None:
    slower = track("C:maj", "F:maj").scaled(0.5)
    assert [c.start for c in slower.chords] == [0.0, 4.0]


def test_chord_at() -> None:
    song = track("C:maj", "F:maj")
    assert chord_name(song.chord_at(2.5) or Chord(0, 0, None)) == "F"
    assert song.chord_at(10) is None


def test_bar_chart_marks_changes_inside_a_bar() -> None:
    song = ChordTrack.from_labels([(0.0, 3.0, "C:maj7"), (3.0, 4.0, "G:7/3")])
    assert bar_chart(song, grid_120bpm(bars=2)) == [
        "   1 | Cmaj7   .       .       .",
        "   2 | Cmaj7   .       G7/B    .",
    ]


# ---- the real model, on a synthesized progression with known answers ----

SR = 22050
PROGRESSION = [  # (expected name, bass MIDI note, chord tones)
    ("Cmaj7", 36, [60, 64, 67, 71]),
    ("Am7", 45, [57, 60, 64, 67]),
    ("Dm7", 38, [62, 65, 69, 72]),
    ("E7", 40, [56, 59, 62, 64]),
    ("Bm7♭5", 47, [59, 62, 65, 69]),
    ("Fmaj7", 41, [60, 64, 65, 69]),
]


def write_progression(path: Path) -> None:
    def note(midi: int, seconds: float) -> np.ndarray:
        t = np.arange(int(SR * seconds)) / SR
        f = 440 * 2 ** ((midi - 69) / 12)
        return sum(np.sin(2 * np.pi * f * k * t) / k**1.5 for k in range(1, 6)) * np.exp(-1.5 * t)

    bars = []
    for _, bass, tones in PROGRESSION:
        bar = np.zeros(SR * 2)
        for beat in range(4):
            start = beat * SR // 2
            bar[start : start + SR // 2] += sum(note(m, 0.5) for m in [bass, *tones])
        bars.append(bar)
    audio = np.concatenate(bars)
    sf.write(str(path), 0.8 * audio / np.abs(audio).max(), SR)


@pytest.mark.skipif(importlib.util.find_spec("lv_chordia") is None, reason="lv-chordia missing")
def test_model_recognizes_seventh_chords(tmp_path: Path) -> None:
    song = tmp_path / "progression.wav"
    write_progression(song)
    detected = detect_chords(song)
    for bar, (expected, _, _) in enumerate(PROGRESSION):
        chord = detected.chord_at(bar * 2.0 + 1.0)  # middle of each 2-second bar
        assert chord is not None
        assert chord_name(chord) == expected
