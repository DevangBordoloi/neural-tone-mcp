"""Headless verification: render a preset through the real VST3 with pedalboard, then audit it against the reference.

`render_preset_preview` plays a dry DI through the plugin with the given parameters. `compute_band_spectral_deltas`
compares the render with the reference stem across the six guitar bands used everywhere else in the project.
This measures tonal colour, not the notes: the DI riff and the reference play different material, so a small error
means the spectral balance is close, not that the two sound identical.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.dsp.extractor import _band_ratios
from src.dsp.vst_inspector import InspectorError, load_plugin
from src.models import BAND_NAMES

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DI_PATH = "assets/default_di_riff.wav"
SILENCE_DBFS = -90.0


class ClosedLoopError(RuntimeError):
    pass


def resolve_project_path(path: str | os.PathLike[str]) -> Path:
    """Relative paths are tried against the working directory first, then the project root (MCP hosts vary the cwd)."""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# default DI asset
# ---------------------------------------------------------------------------
def generate_default_di(path: str | os.PathLike[str], seconds: float = 4.0, sr: int = 48_000) -> Path:
    """Write a clean synthetic palm-muted DI: a low-passed sawtooth on Drop D (D2, 73.4 Hz) with exponential decay."""
    from scipy.signal import butter, sosfilt

    f0 = 73.42
    t = np.arange(int(seconds * sr)) / sr
    # band-limited sawtooth (sum of harmonics with 1/k amplitude), then a gentle pickup-like low-pass
    saw = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, int(6000 / f0) + 1))
    saw = sosfilt(butter(2, 3000, btype="low", fs=sr, output="sos"), saw)

    env = np.zeros_like(t)
    for onset in (0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 3.5):  # syncopated palm-muted chugs
        if onset >= seconds:
            continue
        tt = t - onset
        hit = np.where(tt >= 0, np.minimum(tt / 0.002, 1.0) * np.exp(-np.maximum(tt, 0) / 0.08), 0.0)
        env = np.maximum(env, hit)
    y = saw * env
    y = (y / np.abs(y).max() * 0.5).astype(np.float32)  # peak -6 dBFS, headroom for the amp's input stage

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y, sr, subtype="PCM_16")
    return out


def ensure_default_di(path: str | os.PathLike[str] = DEFAULT_DI_PATH) -> Path:
    """Return the DI path, generating `assets/default_di_riff.wav` first if it is the missing default."""
    p = resolve_project_path(path)
    if not p.exists():
        if Path(path).as_posix() != DEFAULT_DI_PATH:
            raise ClosedLoopError(f"DI file not found: {path}")
        log.info("generating default DI riff at %s", p)
        generate_default_di(p)
    return p


# ---------------------------------------------------------------------------
# render + audit
# ---------------------------------------------------------------------------
def render_preset_preview(
    vst3_path: str,
    vst_parameters: dict[str, float],
    di_wav_path: str,
    output_wav_path: str,
    plugin: Any | None = None,
) -> None:
    """Run `di_wav_path` through the VST3 at `vst3_path` with `vst_parameters` applied; write `output_wav_path`.

    pedalboard's `raw_value` is already the plugin's normalized 0-1 position, so no further range mapping is needed:
    values are clamped to 0-1 and unknown parameter names are skipped with a warning. Pass an already-loaded
    `plugin` to skip the slow (20 s+) load when rendering more than once.
    """
    from pedalboard.io import AudioFile

    plugin = plugin if plugin is not None else load_plugin(vst3_path)
    params = plugin.parameters
    skipped = []
    for name, value in vst_parameters.items():
        if name not in params:
            skipped.append(name)
            continue
        params[name].raw_value = min(1.0, max(0.0, float(value)))
    if skipped:
        log.warning("plugin has no parameters named %s (skipped)", skipped[:8])

    with AudioFile(str(di_wav_path)) as f:
        audio, sr = f.read(f.frames), int(f.samplerate)
    if audio.shape[0] == 1:  # amp sims expect a stereo bus
        audio = np.repeat(audio, 2, axis=0)
    out = plugin(audio, sr, reset=True)

    out_path = Path(output_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with AudioFile(str(out_path), "w", sr, out.shape[0]) as f:
        f.write(out)


def _load_mono(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


def compute_band_spectral_deltas(reference_wav: str, rendered_wav: str) -> tuple[float, dict[str, float]]:
    """Per-band dB error (rendered minus reference) over Sub, Body, Low-Mid, Mid, High-Mid and Presence.

    Band energy is measured relative to the 60-10000 Hz total, so overall loudness does not matter. Returns
    (mean absolute error in dB, {band name: delta dB}); a positive delta means the render has too much of that band.
    """
    ref, ref_sr = _load_mono(reference_wav)
    ren, ren_sr = _load_mono(rendered_wav)
    for label, y in (("reference", ref), ("rendered", ren)):
        rms_db = 20 * np.log10(max(float(np.sqrt(np.mean(y**2))) if y.size else 0.0, 1e-12))
        if rms_db < SILENCE_DBFS:
            raise ClosedLoopError(f"{label} audio is silent ({rms_db:.0f} dBFS); cannot compare tone")
    ref_bands, ren_bands = _band_ratios(ref, ref_sr), _band_ratios(ren, ren_sr)
    deltas = {name: float(ren_bands[name] - ref_bands[name]) for name in BAND_NAMES}
    mae = float(np.mean(np.abs(list(deltas.values()))))
    return mae, deltas
