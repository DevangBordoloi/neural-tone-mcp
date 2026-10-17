"""Isolate the guitar stem with Demucs htdemucs_6s, then pick rhythm (Side) or lead (Mid) content."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

from src.models import Focus  # noqa: F401  (re-exported for callers)
from src.net import use_system_trust_store

log = logging.getLogger(__name__)

MODEL_NAME = "htdemucs_6s"
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


def separate_guitar(
    audio_path: str | Path,
    out_dir: str | Path | None = None,
    device: str | None = None,
    model_name: str = MODEL_NAME,
) -> Path:
    """Run Demucs and write `<out_dir>/<stem>_guitar.wav` (48 kHz stereo). Returns that path."""
    import soundfile as sf

    use_system_trust_store()  # model weights download over HTTPS on first use
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as e:  # pragma: no cover
        raise SeparationError("demucs/torch are not installed") from e

    audio_path = Path(audio_path)
    out = Path(out_dir) if out_dir else audio_path.parent / "stems"
    out.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_name not in _MODEL_CACHE:
        log.info("loading %s on %s", model_name, device)
        _MODEL_CACHE[model_name] = get_model(model_name).eval()
    model = _MODEL_CACHE[model_name]
    if "guitar" not in model.sources:  # type: ignore[attr-defined]
        raise SeparationError(f"{model_name} has no guitar stem (sources: {model.sources})")  # type: ignore[attr-defined]

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)  # (n, ch)
    wav = data.T
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    wav = _resample(wav[:2], sr, model.samplerate)  # type: ignore[attr-defined]

    t = torch.from_numpy(wav)
    ref = t.mean(0)
    t = (t - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, t[None], device=device, split=True, overlap=0.25, progress=False)[0]
    sources = sources * (ref.std() + 1e-8) + ref.mean()
    guitar = sources[model.sources.index("guitar")].cpu().numpy()  # type: ignore[attr-defined]
    guitar = _resample(guitar, model.samplerate, sr)  # type: ignore[attr-defined]

    dst = out / f"{audio_path.stem}_guitar.wav"
    sf.write(str(dst), guitar.T, sr, subtype="PCM_24")
    return dst


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
