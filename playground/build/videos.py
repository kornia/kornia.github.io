"""Prepare the short sample clips the playground can run operators on, frame by frame.

Reads ``samples/videos/sources.json`` (clip id, source file, start second, credit, licence), cuts a
4-second 256x256 centre crop with ffmpeg as MP4 (H.264) and WebM (VP9) plus a poster JPEG, and
writes ``samples/videos/videos.json`` for ``build.py`` to copy into ``registry.json``. The source
downloads are not committed (see .gitignore); the outputs under ``playground/videos/`` are.

    python playground/build/videos.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIZE = 256
SECONDS = 4
OUT = ROOT / "videos"


def main() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required to prepare the sample clips")
    sources = json.loads((HERE / "samples" / "videos" / "sources.json").read_text())
    OUT.mkdir(exist_ok=True)
    entries = []
    for src in sources:
        infile = HERE / "samples" / "videos" / src["file"]
        vf = f"crop='min(iw,ih)':'min(iw,ih)',scale={SIZE}:{SIZE}:flags=lanczos,fps=24"
        common = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(src["start"]), "-t", str(SECONDS), "-i", str(infile), "-an", "-vf", vf]
        subprocess.run(common + ["-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / f"{src['id']}.mp4")], check=True)
        subprocess.run(common + ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "34", str(OUT / f"{src['id']}.webm")], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(src["start"]), "-i", str(infile), "-frames:v", "1", "-vf", f"crop='min(iw,ih)':'min(iw,ih)',scale=96:96:flags=lanczos", "-q:v", "4", str(OUT / f"{src['id']}_thumb.jpg")], check=True)
        entries.append({
            "id": src["id"], "label": src["label"], "kind": "video",
            "mp4": f"videos/{src['id']}.mp4", "webm": f"videos/{src['id']}.webm", "thumb": f"videos/{src['id']}_thumb.jpg",
            "seconds": SECONDS, "credit": src["credit"], "license": src["license"], "source": src["source"],
        })
        sizes = {ext: (OUT / f"{src['id']}.{ext}").stat().st_size // 1024 for ext in ("mp4", "webm")}
        print(f"{src['id']}: {sizes['mp4']} KB mp4, {sizes['webm']} KB webm")
    (HERE / "samples" / "videos" / "videos.json").write_text(json.dumps(entries, indent=1) + "\n")


if __name__ == "__main__":
    main()
