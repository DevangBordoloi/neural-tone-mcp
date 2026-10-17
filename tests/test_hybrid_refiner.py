"""Offline tests for the dual-LLM hybrid engine: every API, plugin and render call is mocked or synthetic."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from src.dsp import closed_loop
from src.dsp.closed_loop import (
    ClosedLoopError, compute_band_spectral_deltas, ensure_default_di, generate_default_di, render_preset_preview,
)
from src.llm import claude_refiner, gemini_refiner, hybrid_arbiter
from src.llm.claude_refiner import (
    NeuralDSPRefinement, RefinerError, TOOL_NAME, compact_schema, refine_parameters_with_claude,
)
from src.llm.gemini_refiner import DEFAULT_GEMINI_MODEL, _GeminiRefinement, refine_parameters_with_gemini
from src.llm.hybrid_arbiter import orchestrate_tone_refinement, sanitize_refinement
from src.models import NeuralDSPPresetJSON, ParameterSpec, PluginSchema, TonalFeatures, TrackConfig

SR = 48_000


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """src.pipeline loads the project's real .env on import; none of these tests may see keys or backend choices."""
    for var in ("NEURAL_TONE_CLAUDE_BACKEND", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                "CLAUDE_CLI_PATH", "NEURAL_TONE_MODEL", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)


def _schema_dict() -> dict:
    params = [
        ParameterSpec(index=0, name="amp_gain", kind="numeric", default_normalized=0.5, default_display="50 %",
                      min_value=0, max_value=100, units="%", linear=True),
        ParameterSpec(index=1, name="amp_type", kind="enum", default_normalized=0.0, default_display="Clean",
                      valid_values=["Clean", "Crunch", "Lead"], num_steps=3),
        ParameterSpec(index=2, name="gate_active", kind="boolean", default_normalized=0.0, default_display="Off",
                      valid_values=[False, True], num_steps=2),
        ParameterSpec(index=3, name="amp_treble", kind="numeric", default_normalized=0.5, default_display="50 %",
                      min_value=0, max_value=100, units="%", linear=True),
    ]
    return PluginSchema(plugin_name="Fake Amp", vst3_path="nowhere.vst3", parameter_count=len(params),
                        inspected_with="test", parameters=params).model_dump()


def _draft(**params: float) -> NeuralDSPRefinement:
    return NeuralDSPRefinement(reasoning="draft", double_track_rhythm=False, vst_parameters=params or {"amp_gain": 0.4})


# ---------------------------------------------------------------------------
# unified schema
# ---------------------------------------------------------------------------
def test_refinement_clamps_to_unit_range_and_drops_junk():
    r = NeuralDSPRefinement(
        reasoning="x", double_track_rhythm=True,
        vst_parameters={"hi": 1.7, "lo": -0.2, "ok": "0.4", "nan": float("nan"), "inf": float("inf"), "bad": "abc", "none": None},
    )
    assert r.vst_parameters == {"hi": 1.0, "lo": 0.0, "ok": 0.4}


def test_refinement_requires_all_fields():
    with pytest.raises(ValidationError):
        NeuralDSPRefinement.model_validate({"double_track_rhythm": False, "vst_parameters": {}})


def test_compact_schema_keeps_names_ranges_and_options_only():
    rows = {r["name"]: r for r in compact_schema(_schema_dict())}
    assert rows["amp_gain"]["display_range"] == [0, 100, "%"]
    assert rows["amp_type"]["options"] == ["Clean", "Crunch", "Lead"]
    assert "samples" not in rows["amp_gain"] and "curve" not in rows["amp_gain"]


# ---------------------------------------------------------------------------
# Claude refiner
# ---------------------------------------------------------------------------
def _fake_anthropic(monkeypatch, content=None, exc: Exception | None = None):
    calls: list[dict] = []

    def create(**kw):
        calls.append(kw)
        if exc:
            raise exc
        return SimpleNamespace(content=content)

    monkeypatch.setattr(claude_refiner, "Anthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return calls


def test_claude_forces_the_tool_and_finds_it_after_a_text_block(monkeypatch):
    tool_input = {"reasoning": "more gain", "double_track_rhythm": True, "vst_parameters": {"amp_gain": 1.4, "amp_treble": 0.3}}
    calls = _fake_anthropic(monkeypatch, content=[
        SimpleNamespace(type="text", text="thinking..."),
        SimpleNamespace(type="tool_use", name=TOOL_NAME, input=tool_input),
    ])
    out = refine_parameters_with_claude({"rms_db": -20}, _schema_dict(), "tight rhythm", {"mid_bark": 3.0}, {"amp_gain": 0.5})
    assert out.vst_parameters == {"amp_gain": 1.0, "amp_treble": 0.3}  # 1.4 clamped
    assert out.double_track_rhythm is True
    kw = calls[0]
    assert kw["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert kw["tools"][0]["input_schema"] == NeuralDSPRefinement.model_json_schema()
    sent = json.loads(kw["messages"][0]["content"])
    assert sent["spectral_error_delta_db_rendered_minus_reference"] == {"mid_bark": 3.0}
    assert sent["baseline_parameters"] == {"amp_gain": 0.5}
    assert {r["name"] for r in sent["valid_plugin_schema"]} == {"amp_gain", "amp_type", "gate_active", "amp_treble"}


def test_claude_api_backend_without_api_key_fails_before_any_request(monkeypatch):
    monkeypatch.setenv("NEURAL_TONE_CLAUDE_BACKEND", "api")
    monkeypatch.setattr(claude_refiner, "Anthropic", lambda: pytest.fail("must not create a client"))
    with pytest.raises(RefinerError, match="ANTHROPIC_API_KEY"):
        refine_parameters_with_claude({}, _schema_dict(), "x")


def test_claude_reply_without_tool_call_is_an_error(monkeypatch):
    _fake_anthropic(monkeypatch, content=[SimpleNamespace(type="text", text="sorry")])
    with pytest.raises(RefinerError, match="no commit_neural_dsp_preset"):
        refine_parameters_with_claude({}, _schema_dict(), "x")


def test_claude_api_failure_becomes_refiner_error(monkeypatch):
    _fake_anthropic(monkeypatch, exc=ConnectionError("boom"))
    with pytest.raises(RefinerError, match="boom"):
        refine_parameters_with_claude({}, _schema_dict(), "x")


def test_claude_malformed_tool_arguments_are_an_error(monkeypatch):
    _fake_anthropic(monkeypatch, content=[SimpleNamespace(type="tool_use", name=TOOL_NAME, input={"vst_parameters": {}})])
    with pytest.raises(RefinerError, match="did not validate"):
        refine_parameters_with_claude({}, _schema_dict(), "x")


# --- claude CLI backend (no API key) ---------------------------------------------------------------------
CLI_TOOL_INPUT = {"reasoning": "via cli", "double_track_rhythm": False, "vst_parameters": {"amp_gain": 0.6, "amp_treble": 1.8}}


def _fake_cli(monkeypatch, stdout=None, returncode=0, stderr="", exc: Exception | None = None, exe="claude.exe"):
    runs: list[tuple[list[str], dict]] = []

    def run(cmd, **kw):
        runs.append((cmd, kw))
        if exc:
            raise exc
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(claude_refiner, "_find_claude_cli", lambda: exe)
    monkeypatch.setattr(claude_refiner.subprocess, "run", run)
    monkeypatch.setattr(claude_refiner, "Anthropic", lambda: pytest.fail("the CLI backend must not use the API"))
    return runs


def test_without_an_api_key_claude_goes_through_the_cli(monkeypatch):
    runs = _fake_cli(monkeypatch, stdout=json.dumps({"is_error": False, "structured_output": CLI_TOOL_INPUT}))
    out = refine_parameters_with_claude({"rms_db": -20}, _schema_dict(), "tight rhythm", {"mid_bark": 2.0}, {"amp_gain": 0.5})
    assert out.vst_parameters == {"amp_gain": 0.6, "amp_treble": 1.0}  # 1.8 clamped
    cmd, kw = runs[0]
    assert cmd[:2] == ["claude.exe", "-p"] and "--output-format" in cmd and "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--tools") + 1] == ""  # no tools: the CLI can only answer
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == NeuralDSPRefinement.model_json_schema()
    assert "--model" not in cmd
    sent = json.loads(kw["input"])  # the long prompt travels on stdin
    assert sent["baseline_parameters"] == {"amp_gain": 0.5} and sent["spectral_error_delta_db_rendered_minus_reference"] == {"mid_bark": 2.0}


def test_cli_backend_can_be_forced_even_when_an_api_key_exists_and_passes_the_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("NEURAL_TONE_CLAUDE_BACKEND", "cli")
    monkeypatch.setenv("NEURAL_TONE_MODEL", "claude-test-model")
    runs = _fake_cli(monkeypatch, stdout=json.dumps({"structured_output": CLI_TOOL_INPUT}))
    refine_parameters_with_claude({}, _schema_dict(), "x")
    cmd = runs[0][0]
    assert cmd[cmd.index("--model") + 1] == "claude-test-model"


def test_auto_backend_prefers_the_api_when_a_key_is_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert claude_refiner._backend() == "api"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert claude_refiner._backend() == "cli"


def test_cli_falls_back_to_json_text_in_result_for_older_clis(monkeypatch):
    _fake_cli(monkeypatch, stdout=json.dumps({"is_error": False, "result": json.dumps(CLI_TOOL_INPUT)}))
    assert refine_parameters_with_claude({}, _schema_dict(), "x").reasoning == "via cli"


@pytest.mark.parametrize("kwargs, match", [
    ({"stdout": json.dumps({"is_error": True, "result": "Not logged in"})}, "Not logged in"),
    ({"stdout": "", "returncode": 1, "stderr": "crashed"}, "without JSON"),
    ({"stdout": json.dumps({"is_error": False, "result": "no json here"})}, "no structured output"),
    ({"stdout": json.dumps({"structured_output": {"vst_parameters": {}}})}, "did not validate"),
    ({"exc": __import__("subprocess").TimeoutExpired("claude", 1)}, "timed out"),
    ({"exc": FileNotFoundError("gone")}, "could not run"),
])
def test_cli_failures_become_refiner_errors(monkeypatch, kwargs, match):
    _fake_cli(monkeypatch, **kwargs)
    with pytest.raises(RefinerError, match=match):
        refine_parameters_with_claude({}, _schema_dict(), "x")


def test_missing_cli_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(claude_refiner, "_find_claude_cli", lambda: None)
    with pytest.raises(RefinerError, match="CLAUDE_CLI_PATH"):
        refine_parameters_with_claude({}, _schema_dict(), "x")


def test_find_claude_cli_prefers_the_explicit_path(monkeypatch, tmp_path):
    exe = tmp_path / "claude.exe"
    exe.write_text("")
    monkeypatch.setenv("CLAUDE_CLI_PATH", str(exe))
    assert claude_refiner._find_claude_cli() == str(exe)
    monkeypatch.setenv("CLAUDE_CLI_PATH", str(tmp_path / "missing.exe"))
    monkeypatch.setattr(claude_refiner.shutil, "which", lambda name: "from-path")
    assert claude_refiner._find_claude_cli() == "from-path"


# ---------------------------------------------------------------------------
# Gemini refiner
# ---------------------------------------------------------------------------
class _FakeGemini:
    def __init__(self, reply: str | None = None, generate_exc: Exception | None = None, fail_upload_at: int | None = None):
        self.uploaded: list[str] = []
        self.deleted: list[str] = []
        self.generate_kwargs: dict = {}
        self._reply, self._exc, self._fail_at = reply, generate_exc, fail_upload_at
        self.files = SimpleNamespace(upload=self._upload, get=lambda name: None, delete=self._delete)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _upload(self, file):
        if self._fail_at is not None and len(self.uploaded) == self._fail_at:
            raise ConnectionError("upload failed")
        self.uploaded.append(file)
        return SimpleNamespace(name=f"files/{len(self.uploaded)}", state=SimpleNamespace(name="ACTIVE"))

    def _delete(self, name):
        self.deleted.append(name)

    def _generate(self, **kw):
        self.generate_kwargs = kw
        if self._exc:
            raise self._exc
        return SimpleNamespace(parsed=None, text=self._reply)


GOOD_REPLY = json.dumps({
    "reasoning": "tighter attack", "double_track_rhythm": False,
    "vst_parameters": [{"name": "amp_treble", "value": 0.7}, {"name": "amp_gain", "value": 1.3}],
})


def _call_gemini(fake: _FakeGemini, monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setattr(gemini_refiner, "_make_client", lambda: fake)
    return refine_parameters_with_gemini(
        "ref.wav", "render.wav", _draft(amp_gain=0.4), {"mid_bark": -3.0}, _schema_dict(), "tight rhythm")


def test_gemini_uploads_both_files_returns_unified_model_and_cleans_up(monkeypatch):
    fake = _FakeGemini(reply=GOOD_REPLY)
    out = _call_gemini(fake, monkeypatch)
    assert isinstance(out, NeuralDSPRefinement)
    assert out.vst_parameters == {"amp_treble": 0.7, "amp_gain": 1.0}  # list converted to dict, 1.3 clamped
    assert fake.uploaded == ["ref.wav", "render.wav"]
    assert sorted(fake.deleted) == ["files/1", "files/2"]
    kw = fake.generate_kwargs
    assert kw["model"] == DEFAULT_GEMINI_MODEL == "gemini-pro-latest"
    assert kw["config"].response_mime_type == "application/json"
    assert kw["config"].response_schema is _GeminiRefinement
    assert any("Audio 1" in str(c) for c in kw["contents"]) and any("Audio 2" in str(c) for c in kw["contents"])


def test_gemini_model_can_be_pinned_with_an_env_var(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
    fake = _FakeGemini(reply=GOOD_REPLY)
    _call_gemini(fake, monkeypatch)
    assert fake.generate_kwargs["model"] == "gemini-3.1-pro-preview"


def test_gemini_deletes_uploads_when_the_request_fails(monkeypatch):
    fake = _FakeGemini(generate_exc=RuntimeError("quota"))
    with pytest.raises(RefinerError, match="quota"):
        _call_gemini(fake, monkeypatch)
    assert sorted(fake.deleted) == ["files/1", "files/2"]


def test_gemini_deletes_the_first_upload_when_the_second_fails(monkeypatch):
    fake = _FakeGemini(reply=GOOD_REPLY, fail_upload_at=1)
    with pytest.raises(ConnectionError):
        _call_gemini(fake, monkeypatch)
    assert fake.deleted == ["files/1"]


def test_gemini_malformed_reply_is_an_error_and_still_cleans_up(monkeypatch):
    fake = _FakeGemini(reply="not json")
    with pytest.raises(RefinerError, match="did not validate"):
        _call_gemini(fake, monkeypatch)
    assert len(fake.deleted) == 2


def test_gemini_cleanup_failure_does_not_mask_the_result(monkeypatch):
    fake = _FakeGemini(reply=GOOD_REPLY)
    fake.files.delete = lambda name: (_ for _ in ()).throw(RuntimeError("delete failed"))
    assert _call_gemini(fake, monkeypatch).reasoning == "tighter attack"


def test_gemini_without_api_key_is_a_refiner_error(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RefinerError, match="GEMINI_API_KEY"):
        refine_parameters_with_gemini("a.wav", "b.wav", _draft(), {}, _schema_dict(), "x")


def test_gemini_response_schema_is_accepted_by_the_developer_api_sdk():
    """NeuralDSPRefinement itself (a dict field) is rejected by the SDK in Developer API mode; ours must not be."""
    genai = pytest.importorskip("google.genai")
    from google.genai import _transformers

    client = genai.Client(api_key="test-key")
    with pytest.raises(ValueError, match="additionalProperties"):
        _transformers.t_schema(client._api_client, NeuralDSPRefinement)
    _transformers.t_schema(client._api_client, _GeminiRefinement)


# ---------------------------------------------------------------------------
# closed loop
# ---------------------------------------------------------------------------
def test_default_di_is_a_four_second_drop_d_palm_mute(tmp_path):
    p = generate_default_di(tmp_path / "di.wav")
    y, sr = sf.read(str(p))
    assert sr == SR and len(y) == 4 * SR and y.ndim == 1
    assert 0.45 < np.abs(y).max() <= 0.5 + 1e-3
    spec = np.abs(np.fft.rfft(y[: SR // 4])) ** 2  # first chug
    freqs = np.fft.rfftfreq(SR // 4, 1 / SR)
    assert abs(freqs[spec.argmax()] - 73.4) < 6
    early, late = np.abs(y[: SR // 10]).max(), np.abs(y[int(0.4 * SR): int(0.5 * SR)]).max()
    assert late < 0.05 * early  # exponential decay between chugs


def test_ensure_default_di_generates_only_the_default_asset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(closed_loop, "PROJECT_ROOT", tmp_path)
    p = ensure_default_di()
    assert p == tmp_path / "assets" / "default_di_riff.wav" and p.exists()
    with pytest.raises(ClosedLoopError, match="not found"):
        ensure_default_di("assets/some_other.wav")


def _noise(seconds=3.0, seed=0):
    from scipy.signal import butter, sosfilt

    y = np.random.default_rng(seed).standard_normal(int(seconds * SR))
    return sosfilt(butter(2, 8000, fs=SR, output="sos"), y).astype(np.float32) * 0.1


def test_band_deltas_are_zero_for_identical_audio_and_ignore_level(tmp_path):
    y = _noise()
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), y, SR)
    sf.write(str(tmp_path / "same.wav"), y, SR)
    sf.write(str(tmp_path / "quiet.wav"), y * 0.1, SR)
    for name in ("same", "quiet"):
        mae, deltas = compute_band_spectral_deltas(str(ref), str(tmp_path / f"{name}.wav"))
        assert mae < 0.05 and set(deltas) == {"sub_thump", "low_body", "low_mid_boxiness", "mid_bark", "high_mid_bite", "presence_air"}


def test_band_deltas_report_a_bright_render_as_positive_in_the_high_bands(tmp_path):
    from scipy.signal import butter, sosfilt

    y = _noise()
    bright = y + 3 * sosfilt(butter(4, 4000, "high", fs=SR, output="sos"), y).astype(np.float32)
    sf.write(str(tmp_path / "ref.wav"), y, SR)
    sf.write(str(tmp_path / "bright.wav"), bright.astype(np.float32), SR)
    mae, d = compute_band_spectral_deltas(str(tmp_path / "ref.wav"), str(tmp_path / "bright.wav"))
    # rendered minus reference; band energy is relative to the total, so the other bands shrink
    assert d["presence_air"] > 2
    assert all(v < -5 for k, v in d.items() if k != "presence_air")
    assert mae == pytest.approx(np.mean(np.abs(list(d.values()))))


def test_band_deltas_refuse_silent_audio(tmp_path):
    sf.write(str(tmp_path / "ref.wav"), _noise(), SR)
    sf.write(str(tmp_path / "silent.wav"), np.zeros(SR, dtype=np.float32), SR)
    with pytest.raises(ClosedLoopError, match="silent"):
        compute_band_spectral_deltas(str(tmp_path / "ref.wav"), str(tmp_path / "silent.wav"))


class _FakeParam:
    raw_value = 0.0


class _FakePlugin:
    def __init__(self):
        self.parameters = {"amp_gain": _FakeParam(), "amp_treble": _FakeParam()}
        self.seen_shape = None

    def __call__(self, audio, sr, reset=True):
        self.seen_shape = audio.shape
        return audio * 0.5


def test_render_clamps_values_skips_unknown_names_and_writes_stereo(tmp_path):
    di = generate_default_di(tmp_path / "di.wav", seconds=1.0)
    plugin = _FakePlugin()
    out = tmp_path / "sub" / "render.wav"
    render_preset_preview("unused.vst3", {"amp_gain": 1.7, "amp_treble": -3.0, "nonexistent": 0.5}, str(di), str(out), plugin=plugin)
    assert plugin.parameters["amp_gain"].raw_value == 1.0 and plugin.parameters["amp_treble"].raw_value == 0.0
    assert plugin.seen_shape[0] == 2  # mono DI was widened for the amp sim
    y, sr = sf.read(str(out))
    assert sr == SR and y.shape[1] == 2 and len(y) == SR


def test_render_loads_the_plugin_when_none_is_given(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(closed_loop, "load_plugin", lambda path: loaded.append(path) or _FakePlugin())
    di = generate_default_di(tmp_path / "di.wav", seconds=0.5)
    render_preset_preview("my.vst3", {"amp_gain": 0.5}, str(di), str(tmp_path / "o.wav"))
    assert loaded == ["my.vst3"]


# ---------------------------------------------------------------------------
# arbiter
# ---------------------------------------------------------------------------
@pytest.fixture
def loop(monkeypatch, tmp_path):
    """Mocks every collaborator of the arbiter; tests set `ctl.error` / failures and read the call log."""
    ctl = SimpleNamespace(
        error=1.5, deltas={"mid_bark": 1.5}, calls=[], vst3=tmp_path / "Fake.vst3",
        claude=lambda *a, **k: _draft(amp_gain=0.4), gemini=None, render_exc=None,
    )
    ctl.vst3.write_text("x")

    def claude(*a, **k):
        ctl.calls.append("claude")
        return ctl.claude(*a, **k)

    def gemini(*a, **k):
        ctl.calls.append("gemini")
        ctl.gemini_args = a
        if ctl.gemini is None:
            return NeuralDSPRefinement(reasoning="by ear", double_track_rhythm=False, vst_parameters={"amp_treble": 0.8})
        return ctl.gemini(*a, **k)

    def render(vst3, params, di, out, plugin=None):
        ctl.calls.append("render")
        if ctl.render_exc:
            raise ctl.render_exc

    monkeypatch.setattr(hybrid_arbiter, "refine_parameters_with_claude", claude)
    monkeypatch.setattr(hybrid_arbiter, "refine_parameters_with_gemini", gemini)
    monkeypatch.setattr(hybrid_arbiter, "render_preset_preview", render)
    monkeypatch.setattr(hybrid_arbiter, "compute_band_spectral_deltas", lambda ref, ren: (ctl.error, ctl.deltas))
    monkeypatch.setattr(hybrid_arbiter, "load_plugin", lambda p: object())
    monkeypatch.setattr(hybrid_arbiter, "ensure_default_di", lambda p: tmp_path / "di.wav")
    ctl.run = lambda **kw: orchestrate_tone_refinement(
        str(tmp_path / "stem.wav"), {"rms_db": -20}, _schema_dict(), str(kw.pop("vst3", ctl.vst3)), "tight rhythm",
        baseline_parameters={"gate_active": 1.0, "amp_gain": 0.5}, work_dir=tmp_path / "work", **kw)
    return ctl


def test_acceptable_error_keeps_claude_and_never_calls_gemini(loop, caplog):
    caplog.set_level(logging.INFO, logger="src.llm.hybrid_arbiter")
    preset, source = loop.run()
    assert source == "claude" and loop.calls == ["claude", "render"]
    assert preset.vst_parameters == {"gate_active": 1.0, "amp_gain": 0.4}  # baseline switches survive the merge
    assert "Claude numeric model achieved acceptable match: 1.50dB" in caplog.text


def test_error_exactly_at_the_threshold_is_accepted(loop):
    loop.error = 2.0
    assert loop.run()[1] == "claude"


def test_error_above_threshold_escalates_to_gemini_with_the_render_and_deltas(loop, caplog):
    caplog.set_level(logging.INFO, logger="src.llm.hybrid_arbiter")
    loop.error, loop.deltas = 5.0, {"mid_bark": -5.0}
    preset, source = loop.run()
    assert source == "gemini" and loop.calls == ["claude", "render", "gemini", "render"]  # last render is the log-only post-check
    target, rendered, draft, deltas, schema, prompt = loop.gemini_args
    assert target.endswith("stem.wav") and rendered.endswith("claude_rendered.wav")
    assert draft.vst_parameters["amp_gain"] == 0.4 and deltas == {"mid_bark": -5.0} and prompt == "tight rhythm"
    assert preset.vst_parameters == {"gate_active": 1.0, "amp_gain": 0.4, "amp_treble": 0.8}  # Gemini overlays Claude's draft
    assert preset.reasoning == "by ear"
    assert "Escalating to Gemini multimodal audio listening (Error: 5.00dB)" in caplog.text


def test_force_gemini_escalates_even_when_claude_is_close(loop):
    loop.error = 0.3
    assert loop.run(force_gemini_audio=True)[1] == "gemini"


def test_missing_vst3_returns_claude_without_rendering_or_escalating(loop, tmp_path):
    for force in (False, True):
        loop.calls.clear()
        preset, source = loop.run(vst3=tmp_path / "missing.vst3", force_gemini_audio=force)
        assert source == "claude" and loop.calls == ["claude"]


def test_gemini_failure_falls_back_to_the_claude_draft(loop):
    loop.error = 9.0
    loop.gemini = lambda *a, **k: (_ for _ in ()).throw(RefinerError("quota"))
    preset, source = loop.run()
    assert source == "claude_fallback" and preset.vst_parameters["amp_gain"] == 0.4


def test_claude_failure_uses_the_heuristic_baseline(loop):
    loop.claude = lambda *a, **k: (_ for _ in ()).throw(RefinerError("no key"))
    preset, source = loop.run()
    assert source == "heuristic" and preset.vst_parameters == {"gate_active": 1.0, "amp_gain": 0.5}
    loop.error = 9.0
    loop.gemini = lambda *a, **k: (_ for _ in ()).throw(RefinerError("quota"))
    assert loop.run()[1] == "heuristic"


def test_render_failure_returns_the_draft_unverified(loop):
    loop.render_exc = RuntimeError("plugin crashed")
    preset, source = loop.run(force_gemini_audio=True)
    assert source == "claude" and "gemini" not in loop.calls


def test_sanitize_drops_unknown_names_snaps_stepped_values_and_overlays_base():
    r = NeuralDSPRefinement(reasoning="r", double_track_rhythm=False,
                            vst_parameters={"amp_type": 0.4, "bogus": 0.5, "amp_gain": 0.9})
    out = sanitize_refinement(r, _schema_dict(), base={"amp_gain": 0.1, "gate_active": 1.0})
    assert out.vst_parameters == {"amp_gain": 0.9, "gate_active": 1.0, "amp_type": 0.5}
    assert r.vst_parameters["bogus"] == 0.5  # the input is not mutated


def test_sanitize_accepts_everything_when_no_schema_is_available():
    r = NeuralDSPRefinement(reasoning="r", double_track_rhythm=False, vst_parameters={"anything": 0.2})
    assert sanitize_refinement(r, {}).vst_parameters == {"anything": 0.2}


# ---------------------------------------------------------------------------
# pipeline + MCP tool
# ---------------------------------------------------------------------------
def _features(focus="rhythm") -> TonalFeatures:
    return TonalFeatures(
        focus=focus, duration_sec=5, rms_db=-20, crest_factor_db=12, attack_steepness=1, attack_rise_ms=10,
        spectral_centroid_hz=2000, spectral_rolloff_hz=6000, spectral_flatness=0.01,
        eq_band_energy_ratios={k: -6.0 for k in ("sub_thump", "low_body", "low_mid_boxiness", "mid_bark", "high_mid_bite", "presence_air")},
        mfcc_mean=[0.0] * 13, stereo_width_ratio=0.3)


def _pipeline_run(monkeypatch, tmp_path, focus: str, double: bool):
    from src import pipeline
    from src.llm.tone_mapper import Intent

    schema = PluginSchema.model_validate(_schema_dict())
    baseline = NeuralDSPPresetJSON(
        preset_name="base", target_plugin="Fake Amp.vst3", track_config=TrackConfig(track_name="Gtr"),
        signal_chain_summary={"Amp": "Clean"}, vst_parameters={"amp_gain": 0.5})
    stem = tmp_path / "stem.wav"
    monkeypatch.setattr(pipeline, "_analyse", lambda *a, **k: (schema, tmp_path / "a.wav", stem, _features(focus), Intent(focus=focus), baseline))
    seen = {}

    def fake_orchestrate(**kw):
        seen.update(kw)
        return NeuralDSPRefinement(reasoning="why", double_track_rhythm=double, vst_parameters={"amp_gain": 0.9}), "gemini"

    monkeypatch.setattr(pipeline, "orchestrate_tone_refinement", fake_orchestrate)
    res = pipeline.extract_tone_hybrid("x.wav", "tight rhythm", "Fake Amp", force_gemini=True, out_dir=tmp_path / "out")
    return res, seen


def test_hybrid_pipeline_writes_the_refined_preset_and_records_the_refiner(monkeypatch, tmp_path):
    res, seen = _pipeline_run(monkeypatch, tmp_path, "rhythm", double=True)
    preset = NeuralDSPPresetJSON.model_validate_json(Path(res["json_path"]).read_text())
    assert preset.vst_parameters == {"amp_gain": 0.9}
    assert preset.track_config.double_tracked and preset.track_config.panning == [-100.0, 100.0]
    assert preset.signal_chain_summary["Refiner"] == "gemini" and preset.signal_chain_summary["Reasoning"] == "why"
    assert res["refiner"] == "gemini"
    assert seen["force_gemini_audio"] is True and seen["baseline_parameters"] == {"amp_gain": 0.5}


def test_hybrid_pipeline_never_double_tracks_a_lead(monkeypatch, tmp_path):
    res, _ = _pipeline_run(monkeypatch, tmp_path, "lead", double=True)
    assert res["preset"].track_config.double_tracked is False and res["preset"].track_config.panning == [0.0]


def test_mcp_exposes_match_url_tone_hybrid(monkeypatch, tmp_path):
    from src import mcp_server

    tools = {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}
    props = tools["match_url_tone_hybrid"].input_schema["properties"]
    assert {"url", "user_prompt", "target_plugin_name", "force_gemini"} <= set(props)
    assert tools["match_url_tone_hybrid"].input_schema["required"] == ["url", "user_prompt", "target_plugin_name"]

    res, _ = _pipeline_run(monkeypatch, tmp_path, "rhythm", double=False)
    captured = {}
    monkeypatch.setattr(mcp_server, "extract_tone_hybrid", lambda *a: captured.setdefault("args", a) and res)

    async def fake_apply(path, daw="reaper"):
        return {"applied": path, "daw": daw}

    monkeypatch.setattr(mcp_server, "apply_tone_json_to_daw", fake_apply)
    out = asyncio.run(mcp_server.match_url_tone_hybrid("http://x", "tight rhythm", "Fake Amp", force_gemini=True))
    assert captured["args"][-1] is True  # force_gemini reached the pipeline
    assert out["extracted"]["refiner"] == "gemini" and out["extracted"]["reasoning"] == "why"
    assert out["applied"] == {"applied": res["json_path"], "daw": "reaper"}
