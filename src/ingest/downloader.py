"""Download audio from a URL with yt-dlp and cut a 48 kHz stereo WAV with ffmpeg."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.net import use_system_trust_store

log = logging.getLogger(__name__)

SAMPLE_RATE = 48_000
DEFAULT_WINDOW_SEC = 30.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = PROJECT_ROOT / "output" / "audio"


class DownloadError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    """FFMPEG_PATH, then PATH, then the binary bundled with the imageio-ffmpeg package."""
    env = os.environ.get("FFMPEG_PATH")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise DownloadError(
            "ffmpeg not found. Install it (winget install ffmpeg), set FFMPEG_PATH, or pip install imageio-ffmpeg"
        ) from e


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "audio"


def _run_ffmpeg(args: list[str]) -> None:
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise DownloadError(f"ffmpeg failed: {proc.stderr.strip()[-500:]}")


def to_wav_48k(src: Path, dst: Path, start_sec: float | None = None, duration_sec: float | None = None) -> Path:
    """Convert any audio file to 48 kHz stereo 24-bit WAV, optionally cutting a segment."""
    args: list[str] = []
    if start_sec is not None:
        args += ["-ss", f"{start_sec:.3f}"]
    args += ["-i", str(src)]
    if duration_sec is not None:
        args += ["-t", f"{duration_sec:.3f}"]
    args += ["-ar", str(SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s24le", str(dst)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(args)
    return dst


def find_energetic_window(wav: Path, window_sec: float = DEFAULT_WINDOW_SEC) -> float:
    """Start time (s) of the loudest `window_sec` stretch of the file (hop = 1 s)."""
    import soundfile as sf

    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    total = len(mono) / sr
    if total <= window_sec:
        return 0.0
    sec = len(mono) // sr
    per_sec = np.array([np.mean(mono[i * sr:(i + 1) * sr] ** 2) for i in range(sec)])
    w = int(window_sec)
    sums = np.convolve(per_sec, np.ones(w), mode="valid")
    return float(np.argmax(sums))


def download_audio(
    url: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    name: str | None = None,
) -> Path:
    """Download `url` and write a 48 kHz stereo WAV of the requested slice.

    start_sec/end_sec default to the most energetic 30 s of the track. Returns the WAV path.
    """
    use_system_trust_store()
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise DownloadError("yt-dlp is not installed") from e

    out = Path(out_dir) if out_dir else DEFAULT_WORK_DIR
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ntm_dl_") as tmp:
        opts = {
            "format": "bestaudio/best", "outtmpl": str(Path(tmp) / "src.%(ext)s"),
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "noprogress": True, "logtostderr": True,  # stdout is the MCP transport; never write to it
            "ffmpeg_location": str(Path(find_ffmpeg()).parent),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as e:
            raise DownloadError(f"yt-dlp could not download {url}: {e}") from e
        files = [p for p in Path(tmp).iterdir() if p.stem == "src"]
        if not files:
            raise DownloadError("yt-dlp produced no audio file")
        title = name or _slug(str(info.get("title") or info.get("id") or "audio"))

        full = to_wav_48k(files[0], Path(tmp) / "full.wav")
        if start_sec is None:
            start = find_energetic_window(full) if end_sec is None else max(0.0, end_sec - DEFAULT_WINDOW_SEC)
            end = end_sec if end_sec is not None else start + DEFAULT_WINDOW_SEC
        else:
            start = float(start_sec)
            end = end_sec if end_sec is not None else start + DEFAULT_WINDOW_SEC
        if end <= start:
            raise DownloadError(f"end_sec ({end}) must be after start_sec ({start})")
        dst = out / f"{title}_{int(start)}-{int(end)}s.wav"
        to_wav_48k(full, dst, start, end - start)
    log.info("saved %s", dst)
    return dst
