"""Isolate the guitar stem with Demucs, then pick rhythm (Side) or lead (Mid) content.

Strategies (see "Guitar Stem Separation Tools.pdf", sections 4.1 and 7.1):
  single   htdemucs_6s on the full mix (one pass; the model with an explicit guitar output).
  cascade  a 4-stem model first removes vocals, drums and bass; htdemucs_6s then separates the guitar from the
           remaining "other" stem, so it never has to decide whether a sharp transient is a snare or a pick attack.
           Costs a second model pass, and anything stage 1 wrongly assigned to the drums (pick attacks) is gone.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np

from src.models import Focus  # noqa: F401  (re-exported for callers)
from src.net import use_system_trust_store

log = logging.getLogger(__name__)

MODEL_NAME = "htdemucs_6s"
STAGE1_MODEL = "htdemucs"  # 4 stems: vocals, drums, bass, other
STRATEGIES = ("single", "cascade")
DEFAULT_STRATEGY = "cascade"
LOSSY_EXTENSIONS = {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".wma"}
GATE_DBFS = -45.0
GATE_FRAME_SEC = 0.05
_MODEL_CACHE: dict[str, object] = {}


class SeparationError(RuntimeError):
    pass


def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g, axis=-1).astype(np.float32)


def resolve_strategy(strategy: str | None = None) -> str:
    """Explicit argument, else NEURAL_TONE_SEPARATION, else the default."""
    chosen = (strategy or os.environ.get("NEURAL_TONE_SEPARATION") or DEFAULT_STRATEGY).lower()
    if chosen not in STRATEGIES:
        raise SeparationError(f"separation strategy must be one of {STRATEGIES}, not {chosen!r}")
    return chosen


def _load_model(name: str, device: str):
    from demucs.pretrained import get_model

    key = f"{name}@{device}"
    if key not in _MODEL_CACHE:
        log.info("loading %s on %s", name, device)
        _MODEL_CACHE[key] = get_model(name).eval()
    return _MODEL_CACHE[key]


def _run_model(model, wav: np.ndarray, sr: int, device: str, shifts: int, overlap: float) -> np.ndarray:
    """Separate a (2, n) float32 signal at `sr`; returns (sources, 2, n) at the model's sample rate.

    The signal is normalised to unit variance for the network and scaled back afterwards, as the Demucs CLI does.
    """
    import torch
    from demucs.apply import apply_model

    x = torch.from_numpy(_resample(wav[:2], sr, model.samplerate))
    ref = x.mean(0)
    x = (x - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, x[None], device=device, shifts=shifts, split=True, overlap=overlap, progress=False)[0]
    return (out * (ref.std() + 1e-8) + ref.mean()).cpu().numpy()


def _stem(model, sources: np.ndarray, name: str) -> np.ndarray:
    if name not in model.sources:
        raise SeparationError(f"{model.sources} has no {name!r} stem")
    return sources[model.sources.index(name)]


def separate_guitar(
    audio_path: str | Path,
    out_dir: str | Path | None = None,
    device: str | None = None,
    model_name: str = MODEL_NAME,
    strategy: str | None = None,
    shifts: int | None = None,
    overlap: float | None = None,
    stage1_model: str = STAGE1_MODEL,
) -> Path:
    """Run Demucs and write `<out_dir>/<stem>_guitar.wav` (input sample rate, stereo). Returns that path.

    strategy: "single" or "cascade" (see module docstring); defaults to NEURAL_TONE_SEPARATION, else "cascade".
    shifts:   test-time augmentation passes (each adds a full model pass; default NEURAL_TONE_SHIFTS or 0).
    overlap:  chunk overlap for smoother joins (default NEURAL_TONE_OVERLAP or 0.25).
    Lossless input matters: MP3/Opus/AAC sources lose the fine phase detail the network relies on.
    """
    import soundfile as sf

    use_system_trust_store()  # model weights download over HTTPS on first use
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise SeparationError("demucs/torch are not installed") from e

    audio_path = Path(audio_path)
    strategy = resolve_strategy(strategy)
    shifts = int(os.environ.get("NEURAL_TONE_SHIFTS", 0)) if shifts is None else shifts
    overlap = float(os.environ.get("NEURAL_TONE_OVERLAP", 0.25)) if overlap is None else overlap
    if audio_path.suffix.lower() in LOSSY_EXTENSIONS:
        log.warning("%s is a lossy format; separation is noticeably better from a lossless source (WAV/FLAC)", audio_path.name)
    out = Path(out_dir) if out_dir else audio_path.parent / "stems"
    out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)  # (n, ch)
    out_sr = sr
    wav = data.T
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    wav = wav[:2]

    guitar_model = _load_model(model_name, device)
    if "guitar" not in guitar_model.sources:
        raise SeparationError(f"{model_name} has no guitar stem (sources: {guitar_model.sources})")
    if strategy == "cascade":
        stage1 = _load_model(stage1_model, device)
        log.info("cascade stage 1: %s -> 'other' (vocals, drums, bass removed)", stage1_model)
        other = _stem(stage1, _run_model(stage1, wav, sr, device, shifts, overlap), "other")
        wav, sr = other, stage1.samplerate  # stage 2 listens to the cleaned mix
    guitar = _stem(guitar_model, _run_model(guitar_model, wav, sr, device, shifts, overlap), "guitar")
    guitar = _resample(guitar, guitar_model.samplerate, out_sr)

    dst = out / f"{audio_path.stem}_guitar.wav"
    sf.write(str(dst), guitar.T, out_sr, subtype="PCM_24")
    for warning in assess_stem(audio_path, dst)["warnings"]:
        log.warning("stem check: %s", warning)
    return dst


def assess_stem(mix_path: str | Path, stem_path: str | Path, active_dbfs: float = -50.0) -> dict:
    """Sanity-check a separated stem against its mix. No ground truth exists, so this only catches obvious failures.

    Returns rms levels (dBFS), the stem-to-mix level in dB, the fraction of 50 ms frames above `active_dbfs`, and a list
    of warnings (an empty stem means tone matching would measure noise, not a guitar).
    """
    import soundfile as sf

    def mono(path):
        d, r = sf.read(str(path), dtype="float32", always_2d=True)
        return d.mean(axis=1), r

    def db(x):
        return float(20 * np.log10(max(float(np.sqrt(np.mean(x**2))) if x.size else 0.0, 1e-9)))

    mix, _ = mono(mix_path)
    stem, sr = mono(stem_path)
    n = max(1, int(sr * GATE_FRAME_SEC))
    frames = len(stem) // n
    blocks = stem[: frames * n].reshape(frames, n) if frames else np.zeros((1, n), np.float32)
    active = float(np.mean(20 * np.log10(np.sqrt(np.mean(blocks**2, axis=1)) + 1e-12) > active_dbfs))
    report = {"mix_rms_db": db(mix), "stem_rms_db": db(stem), "stem_to_mix_db": db(stem) - db(mix),
              "active_fraction": active, "warnings": []}
    if report["stem_rms_db"] < -60:
        report["warnings"].append(f"guitar stem is almost silent ({report['stem_rms_db']:.0f} dBFS): is there a guitar in this section?")
    elif report["stem_to_mix_db"] < -25:
        report["warnings"].append(f"guitar stem is {-report['stem_to_mix_db']:.0f} dB below the mix: probably little or no guitar here")
    if active < 0.25:
        report["warnings"].append(f"only {active:.0%} of the stem is above {active_dbfs:.0f} dBFS: mostly silent, features may be unreliable")
    return report


def _gate(x: np.ndarray, sr: int, threshold_dbfs: float = GATE_DBFS) -> np.ndarray:
    """Drop frames whose RMS is below the threshold (segments are concatenated, not zeroed)."""
    n = max(1, int(sr * GATE_FRAME_SEC))
    frames = len(x) // n
    if frames == 0:
        return x
    blocks = x[: frames * n].reshape(frames, n)
    rms = np.sqrt(np.mean(blocks**2, axis=1) + 1e-20)
    keep = 20 * np.log10(rms) > threshold_dbfs
    return blocks[keep].reshape(-1)


def extract_part_focus(
    audio_path: str | Path,
    focus: Literal["rhythm", "lead", "full"] = "full",
    gate: bool = True,
) -> tuple[np.ndarray, int]:
    """Return (mono signal, sample rate) emphasising the requested guitar part.

    rhythm -> Side channel S = (L - R) / 2   (double-tracked, hard-panned rhythm guitars),
              silent segments below -45 dBFS removed
    lead   -> Mid channel  M = (L + R) / 2
    full   -> mono sum (same as Mid)
    """
    import soundfile as sf

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    left, right = data[:, 0], data[:, 1]
    if focus == "rhythm":
        sig = (left - right) / 2
        gated = _gate(sig, sr) if gate else sig
        if gate and len(gated) < sr * 0.5:
            log.warning("Side channel is nearly silent (mono-ish stem?); using Mid instead")
            sig = (left + right) / 2
            gated = _gate(sig, sr)
        sig = gated
    elif focus in ("lead", "full"):
        sig = (left + right) / 2
    else:
        raise SeparationError(f"focus must be rhythm, lead or full, not {focus!r}")
    return sig.astype(np.float32), sr
