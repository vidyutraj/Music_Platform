"""Click tracks drawn from a beat grid and mixed over a song."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from music_platform.beats import BeatGrid, FloatArray


@dataclass(frozen=True)
class ClickStyle:
    """Pitch of the clicks. The downbeat ("1" of the bar) is higher so it stands out."""

    beat_hz: float = 1000.0
    downbeat_hz: float = 1500.0


@dataclass(frozen=True)
class MixLevels:
    """Relative volume of each layer before the mix is normalized."""

    song: float = 0.8
    beat: float = 0.5
    downbeat: float = 0.7


DEFAULT_STYLE = ClickStyle()
DEFAULT_LEVELS = MixLevels()


def render_clicks(
    grid: BeatGrid, sample_rate: int, length: int, style: ClickStyle = DEFAULT_STYLE
) -> tuple[FloatArray, FloatArray]:
    """Two click tracks of `length` samples: ordinary beats, and downbeats."""
    return (
        _clicks(grid.non_downbeats(), sample_rate, length, style.beat_hz),
        _clicks(grid.downbeats, sample_rate, length, style.downbeat_hz),
    )


def overlay(
    song: FloatArray,
    grid: BeatGrid,
    sample_rate: int,
    levels: MixLevels = DEFAULT_LEVELS,
    style: ClickStyle = DEFAULT_STYLE,
) -> FloatArray:
    """The song with the metronome on top, normalized so it never clips."""
    beat, downbeat = render_clicks(grid, sample_rate, len(song), style)
    mix = levels.song * song + levels.beat * beat + levels.downbeat * downbeat
    return normalize(mix)


def normalize(audio: FloatArray) -> FloatArray:
    """Scale down only if the peak would clip; quieter audio is left alone."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio / max(1.0, peak)


def _clicks(times: FloatArray, sample_rate: int, length: int, freq_hz: float) -> FloatArray:
    if len(times) == 0:
        return np.zeros(length, dtype=np.float64)
    clicks = librosa.clicks(times=times, sr=sample_rate, click_freq=freq_hz, length=length)
    return np.asarray(clicks, dtype=np.float64)
