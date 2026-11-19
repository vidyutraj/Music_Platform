import numpy as np

from music_platform.beats import BeatGrid
from music_platform.metronome import normalize, overlay, render_clicks

SR = 22050


def test_clicks_land_on_the_right_samples() -> None:
    grid = BeatGrid.from_lists([0.0, 0.5, 1.0, 1.5], [0.0, 1.0])
    beat, downbeat = render_clicks(grid, SR, length=2 * SR)

    # Downbeat clicks at 0.0 s and 1.0 s; ordinary beat clicks only at 0.5 s and 1.5 s
    for t in (0.0, 1.0):
        assert np.abs(downbeat[int(t * SR) : int(t * SR) + 200]).max() > 0.1
        assert np.abs(beat[int(t * SR) : int(t * SR) + 200]).max() < 1e-6
    for t in (0.5, 1.5):
        assert np.abs(beat[int(t * SR) : int(t * SR) + 200]).max() > 0.1


def test_no_downbeats_gives_silent_downbeat_track() -> None:
    grid = BeatGrid.from_lists([0.0, 0.5], [])
    _, downbeat = render_clicks(grid, SR, length=SR)
    assert not downbeat.any()


def test_overlay_keeps_length_and_never_clips() -> None:
    loud_song = np.ones(SR, dtype=np.float64)
    grid = BeatGrid.from_lists([0.0, 0.5], [0.0])
    mixed = overlay(loud_song, grid, SR)
    assert len(mixed) == SR
    assert np.abs(mixed).max() <= 1.0 + 1e-9


def test_normalize_leaves_quiet_audio_alone() -> None:
    quiet = np.full(10, 0.2)
    np.testing.assert_array_equal(normalize(quiet), quiet)
