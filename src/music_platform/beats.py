"""Beat grids: where the beats and downbeats of a song fall, in seconds.

`BeatGrid` is the value the rest of the package works on. `detect_beats` fills one in
using the Beat This! model; it imports torch lazily so everything else stays light and
testable without the model installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DOWNBEAT_MATCH_TOLERANCE_S = 0.02


@dataclass(frozen=True)
class BeatGrid:
    """Beat and downbeat times in seconds. Every downbeat is also a beat."""

    beats: FloatArray
    downbeats: FloatArray

    def __post_init__(self) -> None:
        for name in ("beats", "downbeats"):
            times = getattr(self, name)
            if times.ndim != 1:
                raise ValueError(f"{name} must be a 1-D array of times")
            if np.any(times < 0):
                raise ValueError(f"{name} must not contain negative times")
            if np.any(np.diff(times) < 0):
                raise ValueError(f"{name} must be sorted in time order")

    @classmethod
    def from_lists(cls, beats: list[float], downbeats: list[float]) -> BeatGrid:
        return cls(np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64))

    @property
    def bpm(self) -> float:
        """Tempo from the median gap between beats, robust to a few odd beats."""
        if len(self.beats) < 2:
            raise ValueError("need at least two beats to estimate tempo")
        return float(60.0 / np.median(np.diff(self.beats)))

    def non_downbeats(self, tolerance: float = DOWNBEAT_MATCH_TOLERANCE_S) -> FloatArray:
        """Beats that are not also downbeats, so each moment gets exactly one click."""
        if len(self.downbeats) == 0:
            return np.array(self.beats, dtype=np.float64)
        is_down = np.isclose(self.beats[:, None], self.downbeats[None, :], atol=tolerance).any(
            axis=1
        )
        return self.beats[~is_down]

    def scaled(self, speed: float) -> BeatGrid:
        """The grid for the same song played at `speed` (1.25 = 25% faster)."""
        if speed <= 0:
            raise ValueError("speed must be positive")
        return BeatGrid(self.beats / speed, self.downbeats / speed)

    def to_json(self, path: Path) -> None:
        payload = {"beats": self.beats.tolist(), "downbeats": self.downbeats.tolist()}
        path.write_text(json.dumps(payload))

    @classmethod
    def from_json(cls, path: Path) -> BeatGrid:
        payload = json.loads(path.read_text())
        return cls.from_lists(payload["beats"], payload["downbeats"])


def detect_beats(
    audio_path: Path, checkpoint: str = "final0", device: str | None = None
) -> BeatGrid:
    """Run Beat This! on an audio file. Uses the GPU when one is available."""
    import torch
    from beat_this.inference import File2Beats

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    file2beats = File2Beats(checkpoint_path=checkpoint, device=device, dbn=False)
    beats, downbeats = file2beats(str(audio_path))
    return BeatGrid(np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64))
