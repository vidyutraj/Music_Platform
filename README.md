# music-platform

Audio tools for music practice. Right now: detect a song's beats, overlay a metronome, and
render practice versions at slower or faster speeds with the clicks still on the beat.

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

# Song with a metronome on top, at full speed
uv run music-platform metronome data/songs/my_song.mp3

# Practice versions at 60%, 80% and full speed, all in one go
uv run music-platform metronome data/songs/my_song.mp3 --speed 0.6 0.8 1.0
```

Output files land in `out/`. Beat grids are cached in `.music-cache/` by the song's
contents, so rendering more speeds doesn't rerun the model. Use `--refresh` to force it.

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
├── audio.py       load any format as mono, save wav
├── metronome.py   click tracks and the normalized mix
├── tempo.py       speed changes without pitch change (Rubber Band)
├── pipeline.py    end-to-end jobs, with beat-grid caching
└── cli.py         the music-platform command
```

Each module does one job and only `pipeline.py` wires them together, so the next pieces
(MIDI alignment, chroma, the chord checker) slot in as new modules beside these.
