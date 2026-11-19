# music-platform

Audio tools for music practice. Right now: detect a song's beats and chords, overlay a
metronome, and render practice versions at slower or faster speeds (or in another key) with
the clicks still on the beat.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and the two system tools:

   ```bash
   # macOS
   brew install uv rubberband ffmpeg
   # Linux
   sudo apt install rubberband-cli ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install the project (creates `.venv/`):

   ```bash
   uv sync
   ```

The first beat detection downloads the Beat This! model (about 80 MB) and caches it.

## Usage

```bash
# Detect beats and print tempo + first downbeats
uv run music-platform beats data/songs/my_song.mp3

# Chord chart, bar by bar, with the estimated key (add --transpose -2 to see it a step lower)
uv run music-platform chords data/songs/my_song.mp3

# Song with a metronome on top, at full speed
uv run music-platform metronome data/songs/my_song.mp3

# Practice versions at 60%, 80% and full speed, all in one go
uv run music-platform metronome data/songs/my_song.mp3 --speed 0.6 0.8 1.0

# Transpose into an easier key (here a whole step lower); beats and tempo stay put
uv run music-platform metronome data/songs/my_song.mp3 --transpose -2
```

Output files land in `out/`. Beat grids are cached in `.music-cache/` by the song's
contents, so rendering more speeds doesn't rerun the model. Use `--refresh` to force it.

For a browser UI with a tempo slider, key controls and a live chord readout:
`uv run python web/server.py`, then open http://127.0.0.1:8000.

Put your songs under `data/` — it's gitignored, so audio never ends up in the repo.

## Development

```bash
uv run pytest           # tests (the model and Rubber Band are faked or skipped)
uv run ruff check .     # lint
uv run mypy src         # strict type checks
```

## Layout

```
src/music_platform/
├── beats.py       BeatGrid (beat + downbeat times) and Beat This! detection
├── chords.py      ChordTrack, chord naming/transposing, key estimate, lv-chordia detection
├── audio.py       load any format as mono, save wav
├── metronome.py   click tracks and the normalized mix
├── tempo.py       speed and pitch changes, each without the other (Rubber Band)
├── pipeline.py    end-to-end jobs, with beat-grid caching
└── cli.py         the music-platform command
```

Each module does one job and only `pipeline.py` wires them together, so the next pieces
(MIDI alignment, chroma, the chord checker) slot in as new modules beside these.

## Chord recognition

Chords come from [lv-chordia](https://pypi.org/project/lv-chordia/), a packaging of the
large-vocabulary model by Jiang et al. (ISMIR 2019). It recognizes major, minor, dim, aug,
sus2/sus4/7sus4, 7, maj7, m7, dim7, m7♭5, 9, maj9, m9, 11 and 13 chords, plus common
inversions (C/E, Am/G). It has no 6th or minor-major-7th chords, and inversions only for
triads, so a G7/B shows up as G/B. Chord changes are snapped to the Beat This! beats.

**Licensing:** the code is MIT, but the pretrained weights were trained on research chord
datasets (Isophonics, Billboard and others) whose terms may not allow commercial use. Fine
for personal use; check before shipping anything commercial.
