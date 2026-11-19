from pathlib import Path

import numpy as np
import pytest

from music_platform.beats import BeatGrid


def grid_120bpm(bars: int = 4) -> BeatGrid:
    """A clean 4/4 grid at 120 BPM: a beat every 0.5 s, a downbeat every 2 s."""
    beats = [i * 0.5 for i in range(bars * 4)]
    return BeatGrid.from_lists(beats, beats[::4])


def test_bpm_from_median_beat_gap() -> None:
    assert grid_120bpm().bpm == pytest.approx(120.0)


def test_non_downbeats_excludes_downbeats() -> None:
    grid = grid_120bpm(bars=2)
    assert grid.non_downbeats().tolist() == [0.5, 1.0, 1.5, 2.5, 3.0, 3.5]


def test_non_downbeats_tolerates_tiny_timing_mismatch() -> None:
    grid = BeatGrid.from_lists([0.0, 0.5, 1.0], [0.005])
    assert grid.non_downbeats().tolist() == [0.5, 1.0]


def test_scaled_slower_spreads_beats_out() -> None:
    slower = grid_120bpm().scaled(0.5)
    assert slower.bpm == pytest.approx(60.0)
    assert slower.downbeats[1] == pytest.approx(4.0)


@pytest.mark.parametrize("speed", [0.0, -1.0])
def test_scaled_rejects_non_positive_speed(speed: float) -> None:
    with pytest.raises(ValueError):
        grid_120bpm().scaled(speed)


def test_rejects_unsorted_times() -> None:
    with pytest.raises(ValueError):
        BeatGrid.from_lists([1.0, 0.5], [])


def test_json_round_trip(tmp_path: Path) -> None:
    original = grid_120bpm()
    path = tmp_path / "grid.json"
    original.to_json(path)
    restored = BeatGrid.from_json(path)
    np.testing.assert_array_equal(restored.beats, original.beats)
    np.testing.assert_array_equal(restored.downbeats, original.downbeats)
