"""Loading and saving audio. Everything inside the package is mono float64."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from music_platform.beats import FloatArray


def load_mono(path: Path, sample_rate: int | None = None) -> tuple[FloatArray, int]:
    """Load any common format (wav, mp3, m4a via ffmpeg) as mono.

    `sample_rate=None` keeps the file's own rate.
    """
    audio, rate = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float64), int(rate)


def save_wav(path: Path, audio: FloatArray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)
