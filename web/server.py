"""Tiny local web UI: upload a song, preview it at any tempo or key, download only what you keep.

Run with `uv run python web/server.py`, then open http://127.0.0.1:8000.
Standard library only; it calls the same pipeline the CLI uses.

Nothing piles up on disk: uploads live in a temp dir for the length of one request.
The only thing kept is the small beat-grid cache in `.music-cache/`, so a song you've
analyzed before comes back instantly.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from music_platform.chords import QUALITY_SUFFIX, estimate_key, snap_to_beats  # noqa: E402
from music_platform.pipeline import (  # noqa: E402
    beat_grid_for,
    chord_track_for,
    render_metronome_tracks,
)

PAGE = Path(__file__).with_name("index.html")
CACHE_DIR = ROOT / ".music-cache"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MIN_SPEED, MAX_SPEED = 0.25, 2.0


def safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "song"
    return stem[:100]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if urlparse(self.path).path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path not in ("/api/beats", "/api/chords", "/api/render"):
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_UPLOAD_BYTES:
                raise ValueError("upload is empty or too large")

            with tempfile.TemporaryDirectory() as tmp:
                song = Path(tmp) / safe_name(query.get("name", "song"))
                song.write_bytes(self.rfile.read(length))

                if url.path == "/api/beats":
                    grid = beat_grid_for(song, CACHE_DIR)
                    body = {
                        "bpm": grid.bpm,
                        "beats": grid.beats.tolist(),
                        "downbeats": grid.downbeats.tolist(),
                    }
                    self._send(200, json.dumps(body).encode(), "application/json")
                    return

                if url.path == "/api/chords":
                    grid = beat_grid_for(song, CACHE_DIR)
                    track = snap_to_beats(chord_track_for(song, CACHE_DIR), grid)
                    key = estimate_key(track)
                    body = {
                        "key": {"tonic": key.tonic, "mode": key.mode},
                        "chords": [
                            {
                                "start": c.start,
                                "end": c.end,
                                "root": c.root,
                                "bass": c.bass,
                                "suffix": QUALITY_SUFFIX.get(c.quality, c.quality),
                            }
                            for c in track.chords
                        ],
                    }
                    self._send(200, json.dumps(body).encode(), "application/json")
                    return

                speed = float(query.get("speed", "1"))
                if not MIN_SPEED <= speed <= MAX_SPEED:
                    raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
                semitones = float(query.get("semitones", "0"))
                [track] = render_metronome_tracks(
                    song, [speed], Path(tmp) / "out", CACHE_DIR, semitones=semitones
                )
                key = f"_{semitones:+g}st" if semitones else ""
                filename = f"{song.stem}_{round(track.bpm)}bpm{key}.wav"
                self._send(
                    200,
                    track.path.read_bytes(),
                    "audio/wav",
                    {"Content-Disposition": f'attachment; filename="{filename}"'},
                )
        except Exception as exc:  # surface any pipeline error to the page
            self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")

    def _send(
        self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"open http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
