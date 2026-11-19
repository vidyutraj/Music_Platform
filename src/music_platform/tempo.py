"""Changing a song's speed without changing its pitch."""

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
