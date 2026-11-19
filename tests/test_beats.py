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


def assert_same_grid(a: BeatGrid, b: BeatGrid) -> None:
    np.testing.assert_allclose(a.beats, b.beats)
    np.testing.assert_allclose(a.downbeats, b.downbeats)


def test_regularized_leaves_a_clean_grid_alone() -> None:
    clean = grid_120bpm(bars=8)
    assert_same_grid(clean.regularized(), clean)


def test_regularized_drops_a_double_detection() -> None:
    clean = grid_120bpm(bars=4)
    beats = sorted([*clean.beats.tolist(), 3.08])  # one strum heard as two beats
    assert_same_grid(BeatGrid.from_lists(beats, clean.downbeats.tolist()).regularized(), clean)


def test_regularized_keeps_the_better_placed_beat_of_a_pair() -> None:
    clean = grid_120bpm(bars=4)
    beats = [b for b in clean.beats.tolist() if b != 3.0] + [2.92, 3.02]
    out = BeatGrid.from_lists(sorted(beats), clean.downbeats.tolist()).regularized()
    assert 3.02 in out.beats.tolist()
    assert 2.92 not in out.beats.tolist()


def test_regularized_fills_a_missed_beat() -> None:
    clean = grid_120bpm(bars=4)
    beats = [b for b in clean.beats.tolist() if b != 5.5]
    assert_same_grid(BeatGrid.from_lists(beats, clean.downbeats.tolist()).regularized(), clean)


def test_regularized_leaves_long_breaks_unfilled() -> None:
    beats = [0.0, 0.5, 1.0, 1.5, 6.0, 6.5, 7.0, 7.5]  # a 4.5 s break, ~9 beats long
    out = BeatGrid.from_lists(beats, [0.0, 6.0]).regularized()
    assert out.beats.tolist() == beats


def test_regularized_removes_a_stray_downbeat_and_restores_a_missed_one() -> None:
    clean = grid_120bpm(bars=6)
    downbeats = [0.0, 2.0, 3.0, 4.0, 8.0, 10.0]  # 3.0 is mid-bar, 6.0 is missing
    out = BeatGrid.from_lists(clean.beats.tolist(), downbeats).regularized()
    assert_same_grid(out, clean)


def test_regularized_keeps_a_real_two_beat_bar() -> None:
    # Two bars of 4, one bar of 2, then 4s again: every later downbeat shifts by 2 beats.
    beats = [i * 0.5 for i in range(22)]
    downbeats = [0.0, 2.0, 4.0, 5.0, 7.0, 9.0]
    out = BeatGrid.from_lists(beats, downbeats).regularized()
    assert out.downbeats.tolist() == downbeats
