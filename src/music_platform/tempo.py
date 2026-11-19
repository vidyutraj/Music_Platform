"""Speed changes that keep the pitch, and pitch changes (transposing) that keep the speed."""

from __future__ import annotations

import shutil

import numpy as np

from music_platform.beats import FloatArray


class RubberBandMissingError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "Speed changes need the Rubber Band command-line tool. Install it with "
            "`brew install rubberband` (macOS) or `sudo apt install rubberband-cli` (Linux)."
        )


def stretch(audio: FloatArray, sample_rate: int, speed: float) -> FloatArray:
    """Play `audio` at `speed` (0.75 = three-quarter speed) keeping the same pitch.

    Uses Rubber Band, which sounds far cleaner on piano than a plain phase vocoder.
    """
    if speed <= 0:
        raise ValueError("speed must be positive")
    if speed == 1.0:
        return np.array(audio, dtype=np.float64)
    if shutil.which("rubberband") is None:
        raise RubberBandMissingError()

    import pyrubberband as pyrb

    return np.asarray(pyrb.time_stretch(audio, sample_rate, speed), dtype=np.float64)


MAX_SEMITONES = 12.0


def shift_pitch(audio: FloatArray, sample_rate: int, semitones: float) -> FloatArray:
    """Move `audio` up or down by `semitones` (-2 = a whole step lower), same length and tempo.

    The timing doesn't change, so a song's beat grid still lines up after the shift.
    """
    if abs(semitones) > MAX_SEMITONES:
        raise ValueError(f"semitones must be between -{MAX_SEMITONES:g} and {MAX_SEMITONES:g}")
    if semitones == 0:
        return np.array(audio, dtype=np.float64)
    if shutil.which("rubberband") is None:
        raise RubberBandMissingError()

    import pyrubberband as pyrb

    shifted = np.asarray(pyrb.pitch_shift(audio, sample_rate, semitones), dtype=np.float64)
    # Rubber Band can be off by a few samples; pin the length so beat times stay exact.
    if len(shifted) >= len(audio):
        return shifted[: len(audio)]
    return np.pad(shifted, (0, len(audio) - len(shifted)))
