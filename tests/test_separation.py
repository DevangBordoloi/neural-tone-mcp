"""Offline tests for the separation strategies: Demucs is replaced by fake models, so no weights are needed."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from src.ingest import separator
from src.ingest.separator import SeparationError, assess_stem, resolve_strategy, separate_guitar

SR = 48_000
MODEL_SR = 44_100


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("NEURAL_TONE_SEPARATION", "NEURAL_TONE_SHIFTS", "NEURAL_TONE_OVERLAP"):
        monkeypatch.delenv(var, raising=False)


def _tone(seconds=1.0, freq=220.0, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write(path, mono, sr=SR):
    sf.write(str(path), np.stack([mono, mono], axis=1), sr)
    return path


# ---------------------------------------------------------------------------
# strategy selection
# ---------------------------------------------------------------------------
def test_strategy_comes_from_the_argument_then_the_environment_then_the_default(monkeypatch):
    assert resolve_strategy() == separator.DEFAULT_STRATEGY
    monkeypatch.setenv("NEURAL_TONE_SEPARATION", "single")
    assert resolve_strategy() == "single"
    assert resolve_strategy("CASCADE") == "cascade"  # an explicit argument wins, case-insensitively


def test_unknown_strategy_is_rejected(monkeypatch):
    with pytest.raises(SeparationError, match="strategy"):
        resolve_strategy("bs-roformer")
    monkeypatch.setenv("NEURAL_TONE_SEPARATION", "nonsense")
    with pytest.raises(SeparationError):
        resolve_strategy()


# ---------------------------------------------------------------------------
# separate_guitar with fake models
# ---------------------------------------------------------------------------
class _Fakes:
    """Stands in for _load_model/_run_model. Each model returns a distinct constant per stem so the routing is visible."""

    def __init__(self, monkeypatch):
        self.calls = []
        self.models = {
            "htdemucs": SimpleNamespace(name="htdemucs", sources=["drums", "bass", "other", "vocals"], samplerate=MODEL_SR),
            "htdemucs_6s": SimpleNamespace(name="htdemucs_6s", sources=["drums", "bass", "other", "vocals", "guitar", "piano"],
                                           samplerate=MODEL_SR),
            "no_guitar": SimpleNamespace(name="no_guitar", sources=["drums", "bass", "other", "vocals"], samplerate=MODEL_SR),
        }
        monkeypatch.setattr(separator, "_load_model", lambda name, device: self.models[name])
        monkeypatch.setattr(separator, "_run_model", self._run)

    def _run(self, model, wav, sr, device, shifts, overlap):
        self.calls.append(SimpleNamespace(model=model.name, wav=wav.copy(), sr=sr, shifts=shifts, overlap=overlap))
        n = int(wav.shape[1] * model.samplerate / sr)
        out = np.zeros((len(model.sources), 2, n), dtype=np.float32)
        for i, name in enumerate(model.sources):
            out[i] = {"other": 0.2, "guitar": 0.1}.get(name, 0.0) * _tone(n / model.samplerate, 330.0, 1.0, model.samplerate)
        return out


def test_single_runs_only_the_six_stem_model_on_the_original_mix(tmp_path, monkeypatch):
    fakes = _Fakes(monkeypatch)
    mix = _write(tmp_path / "mix.wav", _tone())
    dst = separate_guitar(mix, out_dir=tmp_path / "out", strategy="single", device="cpu")
    assert [c.model for c in fakes.calls] == ["htdemucs_6s"]
    assert fakes.calls[0].sr == SR and np.allclose(fakes.calls[0].wav[0], _tone(), atol=1e-3)  # 16-bit WAV round trip
    y, sr = sf.read(str(dst))
    assert dst.name == "mix_guitar.wav" and sr == SR and y.shape == (SR, 2)  # resampled back to the input rate


def test_cascade_feeds_the_other_stem_of_stage_one_into_the_guitar_model(tmp_path, monkeypatch):
    fakes = _Fakes(monkeypatch)
    mix = _write(tmp_path / "mix.wav", _tone())
    separate_guitar(mix, out_dir=tmp_path / "out", strategy="cascade", device="cpu")
    assert [c.model for c in fakes.calls] == ["htdemucs", "htdemucs_6s"]
    stage1, stage2 = fakes.calls
    assert stage1.sr == SR and stage2.sr == MODEL_SR  # stage 2 listens at the 4-stem model's rate
    expected_other = 0.2 * _tone(1.0, 330.0, 1.0, MODEL_SR)
    assert stage2.wav.shape == (2, MODEL_SR) and np.allclose(stage2.wav[0], expected_other, atol=1e-5)


def test_cascade_uses_a_custom_stage_one_model_and_passes_quality_settings(tmp_path, monkeypatch):
    fakes = _Fakes(monkeypatch)
    fakes.models["custom4"] = SimpleNamespace(name="custom4", sources=["other"], samplerate=MODEL_SR)
    separate_guitar(_write(tmp_path / "m.wav", _tone()), out_dir=tmp_path, strategy="cascade", stage1_model="custom4",
                    shifts=2, overlap=0.5, device="cpu")
    assert [c.model for c in fakes.calls] == ["custom4", "htdemucs_6s"]
    assert all(c.shifts == 2 and c.overlap == 0.5 for c in fakes.calls)


def test_shifts_and_overlap_default_from_the_environment(tmp_path, monkeypatch):
    fakes = _Fakes(monkeypatch)
    monkeypatch.setenv("NEURAL_TONE_SHIFTS", "1")
    monkeypatch.setenv("NEURAL_TONE_OVERLAP", "0.4")
    separate_guitar(_write(tmp_path / "m.wav", _tone()), out_dir=tmp_path, strategy="single", device="cpu")
    assert (fakes.calls[0].shifts, fakes.calls[0].overlap) == (1, 0.4)


def test_a_model_without_a_guitar_stem_is_an_error(tmp_path, monkeypatch):
    _Fakes(monkeypatch)
    with pytest.raises(SeparationError, match="no guitar"):
        separate_guitar(_write(tmp_path / "m.wav", _tone()), out_dir=tmp_path, model_name="no_guitar", device="cpu")


def test_mono_input_is_widened_to_stereo_for_the_models(tmp_path, monkeypatch):
    fakes = _Fakes(monkeypatch)
    sf.write(str(tmp_path / "mono.wav"), _tone(), SR)
    separate_guitar(tmp_path / "mono.wav", out_dir=tmp_path, strategy="single", device="cpu")
    assert fakes.calls[0].wav.shape[0] == 2


def test_lossy_input_logs_a_warning_but_still_runs(tmp_path, monkeypatch, caplog):
    _Fakes(monkeypatch)
    wav = _write(tmp_path / "from_web.wav", _tone())
    lossy = wav.rename(tmp_path / "from_web.mp3")  # only the extension matters to the check; soundfile sniffs content
    with caplog.at_level(logging.WARNING, logger="src.ingest.separator"):
        separate_guitar(lossy, out_dir=tmp_path, strategy="single", device="cpu")
    assert "lossy format" in caplog.text


def test_a_nearly_silent_guitar_stem_is_flagged(tmp_path, monkeypatch, caplog):
    _Fakes(monkeypatch)
    monkeypatch.setattr(separator, "_stem", lambda model, sources, name: sources[0] * 0)  # the model returns silence
    with caplog.at_level(logging.WARNING, logger="src.ingest.separator"):
        separate_guitar(_write(tmp_path / "m.wav", _tone()), out_dir=tmp_path, strategy="single", device="cpu")
    assert "stem check" in caplog.text and "silent" in caplog.text


# ---------------------------------------------------------------------------
# assess_stem
# ---------------------------------------------------------------------------
def test_assess_stem_passes_a_healthy_stem(tmp_path):
    mix = _write(tmp_path / "mix.wav", _tone(amp=0.4))
    stem = _write(tmp_path / "stem.wav", _tone(amp=0.3))
    r = assess_stem(mix, stem)
    assert r["warnings"] == [] and r["active_fraction"] == 1.0
    assert r["stem_to_mix_db"] == pytest.approx(20 * np.log10(0.3 / 0.4), abs=0.1)


def test_assess_stem_warns_when_the_stem_is_far_below_the_mix(tmp_path):
    mix = _write(tmp_path / "mix.wav", _tone(amp=0.5))
    stem = _write(tmp_path / "stem.wav", _tone(amp=0.5 * 10 ** (-30 / 20)))
    assert any("below the mix" in w for w in assess_stem(mix, stem)["warnings"])


def test_assess_stem_warns_about_silence_and_sparse_stems(tmp_path):
    mix = _write(tmp_path / "mix.wav", _tone(amp=0.5))
    silent = _write(tmp_path / "silent.wav", np.zeros(SR, dtype=np.float32))
    assert any("almost silent" in w for w in assess_stem(mix, silent)["warnings"])

    sparse = np.zeros(SR * 2, dtype=np.float32)
    sparse[: SR // 10] = _tone(0.1, amp=0.3)  # 5% of the stem is active
    r = assess_stem(mix, _write(tmp_path / "sparse.wav", sparse))
    assert r["active_fraction"] < 0.25 and any("mostly silent" in w for w in r["warnings"])
