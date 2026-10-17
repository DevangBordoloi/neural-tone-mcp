"""Offline tests: synthetic audio -> features -> preset JSON (real saved schemas) -> mocked DAW calls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.daw.base import DawController
from src.daw.reaper_driver import BridgeClient, ReaperDriver, match_reaper_params
from src.dsp.extractor import extract_features
from src.dsp.vst_inspector import load_schema
from src.ingest.separator import _gate, extract_part_focus
from src.llm.tone_mapper import build_preset, derive_intent, parse_prompt, write_preset
from src.models import NeuralDSPPresetJSON, TonalFeatures

SR = 48_000
SCHEMA_DIR = Path(__file__).resolve().parents[1] / "config" / "plugin_schemas"
needs_schemas = pytest.mark.skipif(not (SCHEMA_DIR / "Archetype Nolly X.json").exists(), reason="run scripts/inspect_plugin.py --all")


# ---------------------------------------------------------------------------
# synthetic guitars
# ---------------------------------------------------------------------------
def _riff(distortion: float, seconds: float = 6.0, bright: float = 1.0) -> np.ndarray:
    """Palm-muted-ish power chords; `distortion` soft-clips them (tanh) like an amp would."""
    t = np.arange(int(SR * seconds)) / SR
    out = np.zeros_like(t)
    for start in np.arange(0, seconds - 0.4, 0.5):
        i0, n = int(start * SR), int(0.35 * SR)
        tt = np.arange(n) / SR
        env = np.minimum(tt / 0.003, 1) * np.exp(-9 * tt)
        note = sum(a * np.sin(2 * np.pi * 82.4 * h * tt) for h, a in [(1, 1), (2, .6), (3, .5 * bright), (4, .3 * bright), (6, .2 * bright)])
        out[i0:i0 + n] += env * note
    out = out / np.abs(out).max() * 0.5
    return np.tanh(out * (1 + 25 * distortion)) / np.tanh(1 + 25 * distortion) * 0.5


def _write(path: Path, left: np.ndarray, right: np.ndarray | None = None) -> Path:
    sf.write(str(path), np.stack([left, left if right is None else right], axis=1).astype(np.float32), SR)
    return path


@pytest.fixture(scope="module")
def clean_wav(tmp_path_factory):
    return _write(tmp_path_factory.mktemp("a") / "clean.wav", _riff(0.0))


@pytest.fixture(scope="module")
def heavy_wav(tmp_path_factory):
    return _write(tmp_path_factory.mktemp("a") / "heavy.wav", _riff(1.0))


# ---------------------------------------------------------------------------
# separator helpers
# ---------------------------------------------------------------------------
def test_focus_rhythm_is_side_lead_is_mid(tmp_path):
    rng = np.random.default_rng(0)
    centre, side = 0.2 * rng.standard_normal(SR * 2), 0.2 * rng.standard_normal(SR * 2)
    p = _write(tmp_path / "s.wav", centre + side, centre - side)
    r, _ = extract_part_focus(p, "rhythm", gate=False)
    m, _ = extract_part_focus(p, "lead")
    assert np.corrcoef(r, side[: len(r)])[0, 1] > 0.99
    assert np.corrcoef(m, centre[: len(m)])[0, 1] > 0.99
    with pytest.raises(Exception):
        extract_part_focus(p, "bogus")


def test_gate_removes_quiet_frames():
    loud = 0.3 * np.ones(SR)
    quiet = 1e-4 * np.ones(SR)  # -80 dBFS
    out = _gate(np.concatenate([loud, quiet, loud]), SR)
    assert len(out) == pytest.approx(2 * SR, abs=SR * 0.1)


def test_rhythm_falls_back_to_mid_when_side_is_silent(clean_wav):
    sig, _ = extract_part_focus(clean_wav, "rhythm")  # mono file -> no Side content
    assert len(sig) > SR


# ---------------------------------------------------------------------------
# extractor
# ---------------------------------------------------------------------------
def test_features_distinguish_clean_from_heavy(clean_wav, heavy_wav):
    c, h = extract_features(clean_wav, "lead"), extract_features(heavy_wav, "lead")
    assert isinstance(c, TonalFeatures) and len(c.mfcc_mean) == 13 and set(c.eq_band_energy_ratios) == {
        "sub_thump", "low_body", "low_mid_boxiness", "mid_bark", "high_mid_bite", "presence_air"}
    assert h.crest_factor_db < c.crest_factor_db          # clipped waveform is flatter
    assert h.spectral_flatness > c.spectral_flatness       # more harmonics
    assert h.spectral_centroid_hz > c.spectral_centroid_hz
    assert c.stereo_width_ratio == pytest.approx(0.0, abs=1e-3)
    assert c.attack_rise_ms > 0


def test_stereo_width_measured_on_stem(tmp_path):
    rng = np.random.default_rng(1)
    a = _riff(0.8); b = np.roll(a, int(0.25 * SR))  # decorrelated take (hard-panned double)
    p = _write(tmp_path / "wide.wav", a, b)
    assert extract_features(p, "rhythm").stereo_width_ratio > 0.3


def test_short_audio_rejected(tmp_path):
    p = _write(tmp_path / "x.wav", np.zeros(1000))
    with pytest.raises(Exception):
        extract_features(p, "lead")


# ---------------------------------------------------------------------------
# tone mapper (real saved schemas)
# ---------------------------------------------------------------------------
def test_parse_prompt():
    v, f = parse_prompt("tight modern djent rhythm, a bit bright")
    assert v["gain"] == 0.85 and v["tight"] == 0.85 and v["brightness"] == 0.75 and f == "rhythm"
    assert parse_prompt("clean ambient lead")[1] == "lead"


@needs_schemas
@pytest.mark.parametrize("plugin", ["Archetype Nolly X", "Archetype Gojira", "Archetype Petrucci X", "Archetype Plini",
                                    "Archetype Abasi", "Archetype Misha Mansoor X", "Archetype John Mayer X"])
def test_preset_valid_for_every_archetype(heavy_wav, plugin):
    feats = extract_features(heavy_wav, "rhythm")
    preset = build_preset(feats, "tight djent rhythm", plugin)
    schema = load_schema(plugin)
    assert preset.validate_against(schema) == []
    assert preset.vst_parameters, "mapper set nothing"
    assert all(0.0 <= v <= 1.0 for v in preset.vst_parameters.values())
    assert any("gain" in k for k in preset.vst_parameters)
    assert preset.signal_chain_summary["Amp"]


@needs_schemas
def test_heavy_vs_clean_picks_different_amp_and_gain(clean_wav, heavy_wav):
    schema = load_schema("Nolly X")
    heavy = build_preset(extract_features(heavy_wav, "rhythm"), "high gain metal rhythm", "Nolly X", schema)
    clean = build_preset(extract_features(clean_wav, "lead"), "clean chimey", "Nolly X", schema)
    amp = schema.by_name()["amp_type"]
    assert amp.valid_values[round(heavy.vst_parameters["amp_type"] * 3)] in ("Rhythm", "Lead")
    assert amp.valid_values[round(clean.vst_parameters["amp_type"] * 3)] == "Clean"
    assert heavy.vst_parameters["rhythm_amp_gain"] > 0.6
    assert clean.vst_parameters["clean_amp_gain"] < 0.4
    assert heavy.vst_parameters["gate_active"] == 1.0 and clean.vst_parameters["gate_active"] == 0.0


@needs_schemas
def test_gojira_prefix_and_nonlinear_filters(heavy_wav):
    p = build_preset(extract_features(heavy_wav, "rhythm"), "heavy tight rhythm", "Archetype Gojira")
    assert "hot_amp_gain" in p.vst_parameters or "rust_amp_gain" in p.vst_parameters
    assert p.vst_parameters["amp_type"] in (0.5, 1.0)
    p2 = build_preset(extract_features(heavy_wav, "rhythm"), "heavy tight rhythm", "Archetype Nolly X")
    hp = [v for k, v in p2.vst_parameters.items() if k.endswith("eq_high_pass")]
    assert hp and 0.0 < hp[0] < 1.0  # set through the measured log curve, not a linear guess


@needs_schemas
def test_double_tracking_follows_width(tmp_path):
    wide = extract_features(_write(tmp_path / "w.wav", _riff(0.8), np.roll(_riff(0.8), int(0.25 * SR))), "rhythm")
    mono = extract_features(_write(tmp_path / "m.wav", _riff(0.8)), "lead")
    assert build_preset(wide, "rhythm", "Nolly X").track_config.double_tracked is True
    assert build_preset(mono, "lead", "Nolly X").track_config.double_tracked is False


@needs_schemas
def test_ambient_prompt_enables_reverb(heavy_wav):
    feats = extract_features(heavy_wav, "lead")
    on = build_preset(feats, "ambient lead with reverb", "Nolly X")
    off = build_preset(feats, "dry lead", "Nolly X")
    assert on.vst_parameters["reverb_active"] == 1.0 and off.vst_parameters["reverb_active"] == 0.0


@needs_schemas
def test_write_preset_roundtrip(heavy_wav, tmp_path):
    p = build_preset(extract_features(heavy_wav, "rhythm"), "djent", "Nolly X", preset_name="rt")
    path = write_preset(p, tmp_path)
    assert NeuralDSPPresetJSON.model_validate_json(path.read_text()) == p


# ---------------------------------------------------------------------------
# DAW drivers with a fake bridge
# ---------------------------------------------------------------------------
class FakeBridge(BridgeClient):
    def __init__(self, schema):
        self.calls: list[tuple[str, dict]] = []
        self.schema = schema
        self.names = [p.host_name for p in schema.parameters]

    def call(self, cmd, timeout=None, **args):
        self.calls.append((cmd, args))
        if cmd == "ping":
            return {"bridge_version": "test", "reaper_version": "7.0"}
        if cmd == "create_track":
            return {"guid": f"G{sum(c == 'create_track' for c, _ in self.calls)}"}
        if cmd == "add_fx":
            return {"slot": 1, "loaded": "VST3: " + args["plugin"]}
        if cmd == "get_params":
            return {"fx": "VST3: x", "params": self.names}
        if cmd == "set_params":
            return {"results": [{"ok": True} for _ in args["params"]]}
        if cmd == "get_chain":
            return {"chain": "<FXCHAIN>"}
        return {}


@needs_schemas
def test_match_by_host_name_not_pedalboard_index():
    s = load_schema("Nolly X")
    shuffled = list(reversed([p.host_name for p in s.parameters]))  # REAPER order differs from pedalboard's
    idx, missing = match_reaper_params(s, shuffled, ["input_gain", "rhythm_amp_gain", "no_such"])
    assert shuffled[idx["input_gain"]] == "Input Gain" and shuffled[idx["rhythm_amp_gain"]] == "Rhythm Amp Gain"
    assert missing == ["no_such"]


@needs_schemas
@pytest.mark.parametrize("double", [False, True])
def test_reaper_driver_applies_preset(heavy_wav, tmp_path, double):
    feats = extract_features(heavy_wav, "rhythm")
    preset = build_preset(feats, "djent rhythm", "Nolly X", preset_name="drv")
    preset.track_config.double_tracked = double
    preset.track_config.panning = [-100.0, 100.0] if double else [0.0]
    path = write_preset(preset, tmp_path)

    drv = ReaperDriver(backend="bridge")
    drv.bridge = FakeBridge(load_schema("Nolly X"))
    drv.connect()
    rep = drv.create_tracks_and_apply_preset(path)

    calls = drv.bridge.calls
    sets = [a for c, a in calls if c == "set_params"]
    assert len(sets) == 1 and len(sets[0]["params"]) == len(preset.vst_parameters)
    assert all(i["mode"] == "normalized" and 0 <= i["value"] <= 1 for i in sets[0]["params"])
    assert rep["tracks"][0]["failed"] == []
    kinds = [c for c, _ in calls]
    if double:
        assert kinds.count("create_track") == 3 and "set_chain" in kinds and "adjust_folder_depth" in kinds
        pans = [a.get("pan") for c, a in calls if c == "create_track"]
        assert pans == [None, -100.0, 100.0]
    else:
        assert kinds.count("create_track") == 1 and "set_chain" not in kinds


def test_driver_reports_unreachable_reaper(tmp_path, monkeypatch):
    drv = ReaperDriver(backend="bridge", bridge_dir=tmp_path)
    drv.bridge.timeout = 0.2
    with pytest.raises(Exception, match="didn't answer"):
        drv.connect()


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        DawController()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MCP server surface
# ---------------------------------------------------------------------------
def test_mcp_server_exposes_required_tools():
    import asyncio

    from src import mcp_server

    tools = mcp_server.mcp._tool_manager.list_tools() if hasattr(mcp_server.mcp, "_tool_manager") else asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {"inspect_neural_dsp_plugin", "extract_tone_to_json", "apply_tone_json_to_daw", "match_url_tone_end_to_end"} <= names


@needs_schemas
def test_pipeline_from_local_stem(heavy_wav, tmp_path):
    from src.pipeline import extract_tone

    res = extract_tone(str(heavy_wav), "tight djent rhythm", "Archetype Nolly X", is_guitar_stem=True, out_dir=tmp_path)
    assert Path(res["json_path"]).exists() and res["preset"].vst_parameters
