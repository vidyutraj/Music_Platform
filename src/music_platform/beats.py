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

# Cleanup thresholds, as fractions of the local beat period.
EXTRA_BEAT_RATIO = 0.6  # closer than this to the previous beat = a double detection
MISSING_BEAT_RATIO = 1.5  # further than this = a beat was missed...
MAX_FILL_RATIO = 4.5  # ...unless it's a long break, which is left alone
DOWNBEAT_SNAP_RATIO = 0.35  # how far a downbeat may sit from the beat it belongs to


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

    def regularized(self) -> BeatGrid:
        """The grid with the model's obvious slips fixed, for a steadier metronome.

        Without drums (solo guitar, piano) the model sometimes hears one strum as two beats,
        skips a beat, or marks a downbeat mid-bar. This drops beats that come far too soon,
        fills single missed beats, and keeps downbeats a consistent number of beats apart
        (still allowing a genuine odd-length bar). A clean grid comes back unchanged.
        """
        if len(self.beats) < 4:
            return self
        beats = np.asarray(_fill_missing_beats(_drop_extra_beats(list(self.beats))))
        return BeatGrid(beats, _consistent_downbeats(beats, self.downbeats))

    def to_json(self, path: Path) -> None:
        payload = {"beats": self.beats.tolist(), "downbeats": self.downbeats.tolist()}
        path.write_text(json.dumps(payload))

    @classmethod
    def from_json(cls, path: Path) -> BeatGrid:
        payload = json.loads(path.read_text())
        return cls.from_lists(payload["beats"], payload["downbeats"])


def _local_period(gaps: FloatArray, i: int) -> float:
    """Typical beat length around gap `i`; a median, so one bad gap doesn't skew it."""
    return float(np.median(gaps[max(0, i - 4) : i + 5]))


def _drop_extra_beats(beats: list[float]) -> list[float]:
    while True:
        gaps = np.diff(beats)
        for i, gap in enumerate(gaps):
            period = _local_period(gaps, i)
            if gap < EXTRA_BEAT_RATIO * period:
                # Keep whichever of the pair sits where the neighbours say a beat belongs.
                guesses = []
                if i > 0:
                    guesses.append(beats[i - 1] + period)
                if i + 2 < len(beats):
                    guesses.append(beats[i + 2] - period)
                expected = float(np.mean(guesses)) if guesses else beats[i]
                first_is_worse = abs(beats[i] - expected) > abs(beats[i + 1] - expected)
                del beats[i if first_is_worse else i + 1]
                break
        else:
            return beats


def _fill_missing_beats(beats: list[float]) -> list[float]:
    gaps = np.diff(beats)
    out = [beats[0]]
    for i, gap in enumerate(gaps):
        period = _local_period(gaps, i)
        if MISSING_BEAT_RATIO * period < gap <= MAX_FILL_RATIO * period:
            n = round(gap / period)
            out.extend(beats[i] + gap * k / n for k in range(1, n))
        out.append(beats[i + 1])
    return out


def _consistent_downbeats(beats: FloatArray, downbeats: FloatArray) -> FloatArray:
    if len(downbeats) == 0:
        return downbeats
    period = float(np.median(np.diff(beats)))
    found: list[int] = []
    for d in downbeats:
        j = int(np.argmin(np.abs(beats - d)))
        if abs(beats[j] - d) <= DOWNBEAT_SNAP_RATIO * period and (not found or j > found[-1]):
            found.append(j)
    if len(found) < 3:
        return beats[found]

    meter = _beats_per_bar(np.diff(found))
    if meter is None:
        return beats[found]
    kept = [found[0]]
    for k, j in enumerate(found[1:], start=1):
        if (j - kept[-1]) % meter == 0:
            kept.append(j)
            continue
        # A real odd-length bar shifts every later downbeat; a stray one doesn't.
        following = found[k + 1 : k + 3]
        if following and all((f - j) % meter == 0 for f in following):
            kept.append(j)

    filled: list[int] = []
    for a, b in zip(kept[:-1], kept[1:], strict=True):
        filled.extend(range(a, b, meter))  # also restores downbeats the model missed
    filled.extend(range(kept[-1], len(beats), meter))
    return beats[filled]


def _beats_per_bar(bar_lengths: NDArray[np.int64]) -> int | None:
    """The song's bar length in beats, read from the bars the model found.

    Picks the largest length (2-7) that most of the song's beats fit into evenly. A stray
    mid-bar downbeat splits a 4 into 2 + 2, and a real odd bar is a one-off, so neither
    outweighs the bars around them; and preferring the largest stops 4/4 reading as 2/4.
    """
    total = int(bar_lengths.sum())
    for meter in range(7, 1, -1):
        covered = int(bar_lengths[bar_lengths % meter == 0].sum())
        if covered >= 0.6 * total:
            return meter
    return None


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
