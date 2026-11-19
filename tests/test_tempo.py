import shutil

import numpy as np
import pytest

from music_platform.tempo import shift_pitch, stretch

SR = 22050
needs_rubberband = pytest.mark.skipif(
    shutil.which("rubberband") is None, reason="rubberband CLI not installed"
)


def test_full_speed_returns_an_unchanged_copy() -> None:
    audio = np.linspace(-1, 1, SR)
    out = stretch(audio, SR, 1.0)
    np.testing.assert_array_equal(out, audio)
    assert out is not audio


@pytest.mark.parametrize("speed", [0.0, -0.5])
def test_rejects_non_positive_speed(speed: float) -> None:
    with pytest.raises(ValueError):
        stretch(np.zeros(SR), SR, speed)


@needs_rubberband
def test_half_speed_doubles_length() -> None:
    audio = np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
    out = stretch(audio, SR, 0.5)
    assert len(out) == pytest.approx(2 * SR, rel=0.01)


def test_no_transpose_returns_an_unchanged_copy() -> None:
    audio = np.linspace(-1, 1, SR)
    out = shift_pitch(audio, SR, 0)
    np.testing.assert_array_equal(out, audio)
    assert out is not audio


@pytest.mark.parametrize("semitones", [-13, 12.5])
def test_rejects_transpose_beyond_an_octave(semitones: float) -> None:
    with pytest.raises(ValueError):
        shift_pitch(np.zeros(SR), SR, semitones)


def dominant_hz(audio: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return float(np.fft.rfftfreq(len(audio), 1 / sr)[np.argmax(spectrum)])


@needs_rubberband
def test_transpose_moves_pitch_and_keeps_length() -> None:
    audio = np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
    out = shift_pitch(audio, SR, 12)
    assert len(out) == len(audio)
    assert dominant_hz(out, SR) == pytest.approx(880, rel=0.02)
