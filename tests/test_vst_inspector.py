"""Inspector tests against a mock pedalboard plugin (no real VST3 needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.dsp import vst_inspector as vi
from src.models import NeuralDSPPresetJSON, TrackConfig


class MockParam:
    def __init__(self, display, values=None, lo=None, hi=None, units=None, raw=0.5):
        self._display, self.valid_values = display, values
        self.min_value, self.max_value, self.units = lo, hi, units
        self.step_size = None
        self.name = None
        self.raw_value = raw

    @property
    def string_value(self):
        return self._display(self.raw_value)


def _linear_db(x):
    return f"{-24 + 48 * x:.1f} dB"


def _log_hz(x):
    hz = 80 * (8000 / 80) ** x
    return f"{hz / 1000:.2f} kHz" if hz >= 1000 else f"{hz:.0f} Hz"


class MockPlugin:
    name = "Archetype Mock"
    is_instrument = False
    path_to_plugin_file = "C:/mock/Archetype Mock.vst3"

    def __init__(self):
        opts = ["Clean", "Crunch", "Lead"]
        self.parameters = {
            "input_gain": MockParam(_linear_db, [round(-24 + i * 0.1, 1) for i in range(481)], -24.0, 24.0, "dB"),
            "gate_active": MockParam(lambda x: "On" if x >= 0.5 else "Off", [False, True], False, True, raw=1.0),
            "amp_channel": MockParam(lambda x: opts[round(x * 2)], opts, raw=0.0),
            "eq_freq": MockParam(_log_hz, [float(i) for i in range(100)], 80.0, 8000.0, "Hz"),
        }
        self.__post_init__()

    def __post_init__(self):
        for k, v in self.parameters.items():
            v.name = k.replace("_", " ").title()

    def __call__(self, audio, sr, reset=True):
        gain = 10 ** ((-24 + 48 * self.parameters["input_gain"].raw_value) / 20)
        return np.clip(audio * gain, -1, 1).astype(np.float32)


def test_kinds_and_ranges():
    s = vi.inspect_plugin(MockPlugin())
    p = s.by_name()
    assert s.parameter_count == 4 and s.plugin_name == "Archetype Mock"
    assert [p[k].kind for k in ("input_gain", "gate_active", "amp_channel", "eq_freq")] == \
        ["numeric", "boolean", "enum", "numeric"]
    assert (p["input_gain"].min_value, p["input_gain"].max_value) == (-24.0, 24.0)
    assert p["amp_channel"].valid_values == ["Clean", "Crunch", "Lead"]
    assert p["gate_active"].valid_values == [False, True]


def test_linear_detection_and_samples():
    p = vi.inspect_plugin(MockPlugin()).by_name()
    assert p["input_gain"].linear is True
    assert p["eq_freq"].linear is False
    assert [x.display for x in p["input_gain"].samples][0] == "-24.0 dB"
    assert p["eq_freq"].samples[-1].display == "8.00 kHz"


def test_host_name_curve_and_option_check():
    p = vi.inspect_plugin(MockPlugin()).by_name()
    assert p["input_gain"].host_name == "Input Gain" and p["input_gain"].curve is None
    assert len(p["eq_freq"].curve) == vi.CURVE_POINTS
    assert p["amp_channel"].options_verified is True


def test_inspection_restores_plugin_state():
    plug = MockPlugin()
    before = {k: v.raw_value for k, v in plug.parameters.items()}
    vi.inspect_plugin(plug)
    assert {k: v.raw_value for k, v in plug.parameters.items()} == before


def test_normalized_for_and_snap():
    p = vi.inspect_plugin(MockPlugin()).by_name()
    assert p["input_gain"].normalized_for(0.0) == pytest.approx(0.5)
    assert p["amp_channel"].normalized_for("Lead") == 1.0
    assert p["gate_active"].normalized_for(True) == 1.0
    assert p["eq_freq"].normalized_for(800) == pytest.approx(0.5, abs=0.02)  # log knob, via measured curve
    with pytest.raises(ValueError, match="outside"):
        p["eq_freq"].normalized_for(50)
    with pytest.raises(ValueError):
        p["amp_channel"].normalized_for("Metal")
    assert p["amp_channel"].snap(0.4) == 0.5  # 3 options -> 0, .5, 1
    assert p["gate_active"].snap(0.7) == 1.0


def test_schema_roundtrip_and_find(tmp_path, monkeypatch):
    s = vi.inspect_plugin(MockPlugin())
    out = vi.save_schema(s, tmp_path)
    assert out.name == "Archetype Mock.json"
    assert vi.load_schema(out) == s
    assert vi.load_schema("Archetype Mock", tmp_path) == s
    assert vi.load_schema("Archetype Mock X.vst3", tmp_path) == s  # partial/odd names still resolve
    assert [x.name for x in s.search("eq")] == ["eq_freq"]
    with pytest.raises(vi.InspectorError):
        vi.load_schema("nope", tmp_path)


def test_find_vst3_by_name(tmp_path, monkeypatch):
    (tmp_path / "Neural DSP").mkdir()
    for n in ("Archetype Nolly X", "Archetype Petrucci X"):
        (tmp_path / "Neural DSP" / f"{n}.vst3").mkdir()
    monkeypatch.setattr(vi, "vst3_search_dirs", lambda: [tmp_path])
    assert vi.find_vst3("nolly x").name == "Archetype Nolly X.vst3"
    with pytest.raises(vi.InspectorError, match="ambiguous"):
        vi.find_vst3("archetype")
    with pytest.raises(vi.InspectorError, match="no .vst3"):
        vi.find_vst3("gojira")


def test_preset_validation():
    s = vi.inspect_plugin(MockPlugin())
    mk = lambda params: NeuralDSPPresetJSON(  # noqa: E731
        preset_name="t", target_plugin="Archetype Mock.vst3",
        track_config=TrackConfig(track_name="t"), vst_parameters=params)
    assert mk({"input_gain": 0.5}).validate_against(s) == []
    with pytest.raises(ValueError, match="unknown parameter"):
        mk({"nonexistent": 0.1}).validate_against(s)
    assert len(mk({"input_gain": 1.4}).validate_against(s, strict=False)) == 1
    assert mk({"amp_channel": 0.4}).snapped(s) == {"amp_channel": 0.5}


def test_render_preview(tmp_path):
    from pedalboard.io import AudioFile

    di = tmp_path / "di.wav"
    sig = (0.05 * np.sin(2 * math.pi * 110 * np.arange(48000) / 48000)).astype(np.float32)[None, :]
    with AudioFile(str(di), "w", 48000, 1) as f:
        f.write(sig)
    preset = {"preset_name": "t", "target_plugin": "x", "track_config": {"track_name": "t"},
              "vst_parameters": {"input_gain": 0.75}}  # +12 dB
    out = vi.render_preview(di, preset, tmp_path / "o.wav", plugin=MockPlugin())
    with AudioFile(str(out)) as f:
        y = f.read(f.frames)
    assert y.shape[0] == 2  # mono DI was widened
    assert np.abs(y).max() == pytest.approx(0.05 * 10 ** (12 / 20), rel=0.02)
    with pytest.raises(vi.InspectorError, match="no parameters named"):
        vi.render_preview(di, {**preset, "vst_parameters": {"bogus": 1}}, tmp_path / "x.wav", plugin=MockPlugin())
