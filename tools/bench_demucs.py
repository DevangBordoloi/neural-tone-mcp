"""Time Demucs htdemucs_6s guitar separation on one device.

  python tools/bench_demucs.py cpu|cuda [clip.wav] [--seconds 25] [--runs 3]

Run it with the interpreter whose torch build you want to measure (CPU venv vs CUDA venv).
The clip is tiled/cut to --seconds because separation time scales with duration, not content.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from src.ingest.separator import separate_guitar  # noqa: E402

DEFAULT_CLIP = Path(__file__).resolve().parents[1] / "output" / "audio" / "Me_at_the_zoo_0-18s.wav"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("device", choices=["cpu", "cuda"])
    ap.add_argument("clip", nargs="?", type=Path, default=DEFAULT_CLIP)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()

    if a.device == "cuda" and not torch.cuda.is_available():
        print(f"CUDA not available in this torch build ({torch.__version__})")
        return 2
    data, sr = sf.read(str(a.clip), dtype="float32", always_2d=True)
    n = int(a.seconds * sr)
    data = np.tile(data, (n // len(data) + 1, 1))[:n]
    tmp = Path(tempfile.mkdtemp(prefix="bench_"))
    wav = tmp / "clip.wav"
    sf.write(str(wav), data, sr, subtype="PCM_24")

    print(f"torch {torch.__version__} | device {a.device}" + (f" ({torch.cuda.get_device_name(0)})" if a.device == "cuda" else "")
          + f" | clip {a.seconds:.0f}s @ {sr} Hz")
    t = time.perf_counter()
    separate_guitar(wav, out_dir=tmp, device=a.device)  # warm-up: model load + weights + kernel init
    print(f"first call (model load + warm-up): {time.perf_counter() - t:.1f}s")

    times = []
    for i in range(a.runs):
        if a.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t = time.perf_counter()
        separate_guitar(wav, out_dir=tmp, device=a.device)
        if a.device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        print(f"  run {i + 1}: {times[-1]:.1f}s")
    print(f"RESULT {a.device}: median {statistics.median(times):.1f}s for {a.seconds:.0f}s of audio "
          f"({a.seconds / statistics.median(times):.2f}x realtime)"
          + (f", peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB" if a.device == "cuda" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
