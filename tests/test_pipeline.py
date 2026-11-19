import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_platform.beats import BeatGrid
from music_platform.chords import ChordTrack
from music_platform.pipeline import beat_grid_for, chord_track_for, render_metronome_tracks

SR = 22050


class FakeDetector:
    """Stands in for the Beat This! model and counts how often it's called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, path: Path) -> BeatGrid:
        self.calls += 1
        return BeatGrid.from_lists([0.0, 0.5, 1.0, 1.5], [0.0, 1.0])


def write_song(path: Path, seconds: float = 2.0) -> Path:
    t = np.arange(int(seconds * SR)) / SR
    sf.write(str(path), 0.3 * np.sin(2 * np.pi * 220 * t), SR)
    return path


def test_beat_grid_is_cached_by_file_contents(tmp_path: Path) -> None:
    song = write_song(tmp_path / "song.wav")
    detector = FakeDetector()

    beat_grid_for(song, tmp_path / "cache", detector=detector)
    beat_grid_for(song, tmp_path / "cache", detector=detector)
    assert detector.calls == 1

    beat_grid_for(song, tmp_path / "cache", refresh=True, detector=detector)
    assert detector.calls == 2


def test_renders_one_track_per_speed(tmp_path: Path) -> None:
    song = write_song(tmp_path / "song.wav")
    tracks = render_metronome_tracks(
        song, [1.0], tmp_path / "out", tmp_path / "cache", detector=FakeDetector()
    )

    assert len(tracks) == 1
    assert tracks[0].path.name == "song_metronome_100pct.wav"
    assert tracks[0].bpm == 120.0
    audio, rate = sf.read(str(tracks[0].path))
    assert rate == SR
    assert len(audio) == 2 * SR


@pytest.mark.skipif(shutil.which("rubberband") is None, reason="rubberband CLI not installed")
def test_transposed_track_is_tagged_and_keeps_length(tmp_path: Path) -> None:
    song = write_song(tmp_path / "song.wav")
    [track] = render_metronome_tracks(
        song, [1.0], tmp_path / "out", tmp_path / "cache", detector=FakeDetector(), semitones=-2
    )

    assert track.path.name == "song_metronome_100pct_-2st.wav"
    assert track.semitones == -2
    audio, _ = sf.read(str(track.path))
    assert len(audio) == 2 * SR


class FakeChordDetector:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, path: Path) -> ChordTrack:
        self.calls += 1
        return ChordTrack.from_labels([(0.0, 1.0, "C:maj7"), (1.0, 2.0, "G:7/3")])


def test_chord_track_is_cached_by_file_contents(tmp_path: Path) -> None:
    song = write_song(tmp_path / "song.wav")
    detector = FakeChordDetector()

    first = chord_track_for(song, tmp_path / "cache", detector=detector)
    second = chord_track_for(song, tmp_path / "cache", detector=detector)
    assert detector.calls == 1
    assert first == second

    chord_track_for(song, tmp_path / "cache", refresh=True, detector=detector)
    assert detector.calls == 2
