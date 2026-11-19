"""End-to-end jobs built from the smaller pieces.

Beat detection is the slow step, so its result is cached by the song's contents:
rendering the same song at five speeds runs the model once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from music_platform.audio import load_mono, save_wav
from music_platform.beats import BeatGrid, detect_beats
from music_platform.chords import ChordTrack, detect_chords
from music_platform.metronome import overlay
from music_platform.tempo import shift_pitch, stretch

BeatDetector = Callable[[Path], BeatGrid]
ChordDetector = Callable[[Path], ChordTrack]


@dataclass(frozen=True)
class RenderedTrack:
    path: Path
    speed: float
    bpm: float
    semitones: float = 0.0


def file_digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def beat_grid_for(
    song_path: Path,
    cache_dir: Path,
    refresh: bool = False,
    detector: BeatDetector = detect_beats,
) -> BeatGrid:
    """The song's beat grid, from the cache when we've seen this exact file before.

    The model's raw output is what's cached; the cleanup (`BeatGrid.regularized`) runs on
    every read, so improving it never needs the model to run again.
    """
    cached = cache_dir / f"{file_digest(song_path)}.beats.json"
    if cached.exists() and not refresh:
        return BeatGrid.from_json(cached).regularized()
    grid = detector(song_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    grid.to_json(cached)
    return grid.regularized()


def chord_track_for(
    song_path: Path,
    cache_dir: Path,
    refresh: bool = False,
    detector: ChordDetector = detect_chords,
) -> ChordTrack:
    """The model's raw chord track, from the cache when we've seen this exact file before.

    It's cached unsnapped, so beat alignment (`chords.snap_to_beats`) can change freely.
    """
    cached = cache_dir / f"{file_digest(song_path)}.chords.json"
    if cached.exists() and not refresh:
        return ChordTrack.from_json(cached)
    track = detector(song_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    track.to_json(cached)
    return track


def render_metronome_tracks(
    song_path: Path,
    speeds: list[float],
    out_dir: Path,
    cache_dir: Path,
    refresh_beats: bool = False,
    detector: BeatDetector = detect_beats,
    semitones: float = 0.0,
) -> list[RenderedTrack]:
    """One wav per speed: the song, time-stretched, with clicks that stay on its beats.

    `semitones` transposes the song first (-2 = a whole step lower). Transposing keeps the
    timing, so the same beat grid is used either way.
    """
    grid = beat_grid_for(song_path, cache_dir, refresh_beats, detector)
    song, sample_rate = load_mono(song_path)
    song = shift_pitch(song, sample_rate, semitones)
    key_tag = f"_{semitones:+g}st" if semitones else ""

    tracks = []
    for speed in speeds:
        stretched = stretch(song, sample_rate, speed)
        mixed = overlay(stretched, grid.scaled(speed), sample_rate)
        name = f"{song_path.stem}_metronome_{round(speed * 100)}pct{key_tag}.wav"
        out_path = out_dir / name
        save_wav(out_path, mixed, sample_rate)
        tracks.append(
            RenderedTrack(path=out_path, speed=speed, bpm=grid.bpm * speed, semitones=semitones)
        )
    return tracks
