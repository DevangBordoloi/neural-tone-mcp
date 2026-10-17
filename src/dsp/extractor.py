"""Acoustic / tonal feature extraction for an isolated guitar stem."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.ingest.separator import extract_part_focus
from src.models import BAND_EDGES_HZ, BAND_NAMES, Focus, TonalFeatures

log = logging.getLogger(__name__)

_EPS = 1e-12


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, _EPS)))


def _band_ratios(y: np.ndarray, sr: int) -> dict[str, float]:
    import librosa

    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    power = S.sum(axis=1)
    total = power[(freqs >= 60) & (freqs < 10000)].sum() + _EPS
    return {
        name: float(10 * np.log10(power[(freqs >= lo) & (freqs < hi)].sum() / total + _EPS))
        for name, (lo, hi) in zip(BAND_NAMES, BAND_EDGES_HZ)
    }


def _attack(y: np.ndarray, sr: int) -> tuple[float, float]:
    """(attack_steepness, mean rise ms). Rise = time for the envelope to go 10% -> 90% of its local peak."""
    import librosa

    hop = 128
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False)
    if len(onsets) == 0:
        return 0.0, 0.0
    env = librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0]
    ms_per_frame = 1000.0 * hop / sr
    rises, strengths = [], []
    for o in onsets:
        seg = env[max(0, o - 4): o + int(80 / ms_per_frame)]  # look ~80 ms ahead
        if len(seg) < 3 or seg.max() < 1e-5:
            continue
        pk = int(np.argmax(seg))
        lo = np.where(seg[: pk + 1] >= 0.1 * seg[pk])[0]
        hi = np.where(seg[: pk + 1] >= 0.9 * seg[pk])[0]
        if len(lo) and len(hi):
            rises.append(max(hi[0] - lo[0], 1) * ms_per_frame)
            strengths.append(float(onset_env[o]))
    if not rises:
        return 0.0, 0.0
    rise = float(np.mean(rises))
    return float(np.mean(strengths) / rise), rise


def stereo_width_ratio(path: str | Path) -> float:
    import soundfile as sf

    d, _ = sf.read(str(path), dtype="float32", always_2d=True)
    if d.shape[1] < 2:
        return 0.0
    mid, side = (d[:, 0] + d[:, 1]) / 2, (d[:, 0] - d[:, 1]) / 2
    return float(np.sqrt(np.mean(side**2)) / (np.sqrt(np.mean(mid**2)) + _EPS))


def features_from_signal(y: np.ndarray, sr: int, focus: Focus = "full", width: float = 0.0) -> TonalFeatures:
    import librosa

    if len(y) < sr * 0.25:
        raise ValueError("audio is too short/silent to analyse (need at least 0.25 s of signal)")
    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    steep, rise = _attack(y, sr)
    # frame-level features ignore gaps between notes (digital silence reads as "flat" and "dark")
    frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    active = frame_rms > frame_rms.max() * 10 ** (-40 / 20)

    def mean_active(frames: np.ndarray) -> np.ndarray:
        frames = np.atleast_2d(frames)
        n = min(frames.shape[1], len(active))
        keep = active[:n]
        return frames[:, :n][:, keep].mean(axis=1) if keep.any() else frames.mean(axis=1)

    mfcc = mean_active(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    return TonalFeatures(
        focus=focus,
        duration_sec=float(len(y) / sr),
        rms_db=_db(rms),
        crest_factor_db=_db(peak) - _db(rms),
        attack_steepness=steep,
        attack_rise_ms=rise,
        spectral_centroid_hz=float(mean_active(librosa.feature.spectral_centroid(y=y, sr=sr))[0]),
        spectral_rolloff_hz=float(mean_active(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))[0]),
        spectral_flatness=float(mean_active(librosa.feature.spectral_flatness(y=y))[0]),
        eq_band_energy_ratios=_band_ratios(y, sr),
        mfcc_mean=[float(m) for m in mfcc],
        stereo_width_ratio=width,
    )


def extract_features(guitar_wav: str | Path, focus: Focus = "full") -> TonalFeatures:
    """Analyse a guitar stem. The focus signal (Side for rhythm, Mid for lead) drives the tonal
    measurements; stereo width is always measured on the full stereo stem."""
    y, sr = extract_part_focus(guitar_wav, focus)
    return features_from_signal(y, sr, focus, width=stereo_width_ratio(guitar_wav))
