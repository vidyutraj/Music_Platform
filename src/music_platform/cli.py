"""Command-line entry point: `music-platform beats | chords | metronome ...`."""

from __future__ import annotations

import argparse
from pathlib import Path

from music_platform.chords import bar_chart, estimate_key, snap_to_beats
from music_platform.pipeline import beat_grid_for, chord_track_for, render_metronome_tracks

DEFAULT_CACHE_DIR = Path(".music-cache")
DEFAULT_OUT_DIR = Path("out")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="music-platform", description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="where beat grids are cached"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    beats = commands.add_parser("beats", help="detect beats and print a summary")
    beats.add_argument("song", type=Path)
    beats.add_argument("--refresh", action="store_true", help="ignore the cached beat grid")

    chords = commands.add_parser("chords", help="detect chords and print a bar-by-bar chart")
    chords.add_argument("song", type=Path)
    chords.add_argument(
        "--transpose", type=int, default=0, metavar="SEMITONES", help="show the chords shifted"
    )
    chords.add_argument("--refresh", action="store_true", help="ignore cached beats and chords")

    metronome = commands.add_parser("metronome", help="overlay clicks, optionally at new speeds")
    metronome.add_argument("song", type=Path)
    metronome.add_argument(
        "--speed",
        type=float,
        nargs="+",
        default=[1.0],
        help="one or more speeds, e.g. --speed 0.6 0.8 1.0",
    )
    metronome.add_argument(
        "--transpose",
        type=float,
        default=0.0,
        metavar="SEMITONES",
        help="shift the key, e.g. --transpose -2 for a whole step lower (beats unchanged)",
    )
    metronome.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    metronome.add_argument("--refresh", action="store_true", help="ignore the cached beat grid")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.song.exists():
        print(f"error: {args.song} not found")
        return 1

    if args.command == "beats":
        grid = beat_grid_for(args.song, args.cache_dir, refresh=args.refresh)
        print(
            f"{len(grid.beats)} beats, {len(grid.downbeats)} downbeats, ~{grid.bpm:.1f} BPM\n"
            f"first downbeats (s): {', '.join(f'{t:.2f}' for t in grid.downbeats[:8])}"
        )
        return 0

    if args.command == "chords":
        grid = beat_grid_for(args.song, args.cache_dir, refresh=args.refresh)
        raw = chord_track_for(args.song, args.cache_dir, refresh=args.refresh)
        chart = snap_to_beats(raw, grid).transposed(args.transpose)
        song_key = estimate_key(chart)
        print(f'key: {song_key.name}   ~{grid.bpm:.1f} BPM   ("." = same chord continues)')
        print("\n".join(bar_chart(chart, grid, song_key.prefer_flats)))
        return 0

    tracks = render_metronome_tracks(
        args.song,
        args.speed,
        args.out_dir,
        args.cache_dir,
        refresh_beats=args.refresh,
        semitones=args.transpose,
    )
    for track in tracks:
        key = f"  {track.semitones:+g} st" if track.semitones else ""
        print(f"{track.speed:>5.0%}  ~{track.bpm:5.1f} BPM{key}  ->  {track.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
