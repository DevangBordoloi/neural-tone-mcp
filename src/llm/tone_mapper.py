"""Map TonalFeatures + a user prompt to a NeuralDSPPresetJSON for an inspected Neural DSP plugin.

Heuristic engine (deterministic, offline): features and prompt words become a handful of 0-1 "intent"
values (gain, brightness, bass, mid scoop, tightness, ambience, width), which a name-pattern resolver
turns into the actual parameters of whichever Archetype's schema is given. Archetype plugins name things
differently (`rhythm_amp_gain`, `rhythm_gain`, `hot_amp_gain`, `gate_threshold`, `gate_amount` ...), so
nothing here hard-codes one plugin. Optional `refine_with_llm` lets Claude adjust the result; every value
is re-validated against the schema.

Assumptions worth knowing (they are guesses, not measurements of the plugin):
  * the 9-band graphic EQ sits on octave centres 63 Hz ... 16 kHz;
  * feature -> knob curves are rules of thumb, so treat the preset as a starting point to adjust by ear.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.dsp.vst_inspector import load_schema
from src.models import (
    Focus, NeuralDSPPresetJSON, ParameterSpec, PluginSchema, TonalFeatures, TrackConfig,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
EQ_CENTRES_HZ = (63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, x)))


def _scale(x: float, lo: float, hi: float) -> float:
    """Linear 0..1 position of x between lo and hi (works with lo > hi for inverted scales)."""
    return _clip((x - lo) / (hi - lo))


# ---------------------------------------------------------------------------
# intent
# ---------------------------------------------------------------------------
@dataclass
class Intent:
    focus: Focus = "rhythm"
    gain: float = 0.6            # 0 clean ... 1 extreme high gain
    brightness: float = 0.5
    bass: float = 0.5
    mid: float = 0.5             # 0 scooped ... 1 mid-forward
    tight: float = 0.5           # gate / boost / low-cut amount
    ambience: float = 0.0        # reverb + delay
    width: float = 0.0           # doubler / L-R pair
    amp_class: str = "high_gain"  # clean | crunch | high_gain | lead
    notes: list[str] = field(default_factory=list)


# word -> (field, value). Later matches override earlier ones; "blend" fields are mixed with the audio.
_PROMPT_RULES: list[tuple[str, str, float]] = [
    (r"\bclean\b|\bchime|\bsparkl", "gain", 0.12), (r"\bcrunch|\bedge of breakup|\bblues", "gain", 0.42),
    (r"\bhigh[- ]?gain\b|\bmetal\b|\bdjent\b|\bheavy\b|\bbrutal|\bchug", "gain", 0.85),
    (r"\bsaturated\b|\bfuzz", "gain", 0.7), (r"\blow[- ]?gain\b", "gain", 0.35),
    (r"\btight\b|\bpalm[- ]?mut|\bchug|\bdjent\b|\bpercussive", "tight", 0.85), (r"\bloose\b|\bspongy", "tight", 0.2),
    (r"\bbright\b|\bbite\b|\bcutting\b|\bglassy", "brightness", 0.75), (r"\bdark\b|\bwarm\b|\bmellow|\bsmooth", "brightness", 0.28),
    (r"\bscoop|\bV[- ]shape", "mid", 0.15), (r"\bmid[- ]?(forward|heavy|range)\b|\bhonk", "mid", 0.8),
    (r"\bthick\b|\bfat\b|\bbass[- ]?heavy|\bdeep\b", "bass", 0.75), (r"\bthin\b|\bscooped bass", "bass", 0.3),
    (r"\bambient\b|\bspacey|\bspace\b|\batmospher|\blush\b|\breverb|\bdelay\b|\becho", "ambience", 0.7),
    (r"\bwide\b|\bdouble[- ]?tracked|\bdoubled|\bstereo\b", "width", 0.8),
]


def parse_prompt(prompt: str) -> tuple[dict[str, float], Focus | None]:
    p = prompt.lower()
    vals: dict[str, float] = {}
    for pat, key, v in _PROMPT_RULES:
        if re.search(pat, p, re.IGNORECASE):
            vals[key] = v
    focus: Focus | None = None
    if re.search(r"\blead\b|\bsolo\b|\blegato", p):
        focus = "lead"
    if re.search(r"\brhythm\b|\bchugs?\b|\briff", p):
        focus = "rhythm"
    return vals, focus


def derive_intent(features: TonalFeatures, prompt: str = "", focus: Focus | None = None) -> Intent:
    f, bands = features, features.eq_band_energy_ratios
    from_prompt, prompt_focus = parse_prompt(prompt)
    focus = focus or prompt_focus or (features.focus if features.focus != "full" else "rhythm")

    # saturation: low crest factor + noisy (flat) spectrum
    gain_audio = 0.6 * _scale(f.crest_factor_db, 22, 9) + 0.4 * _scale(np.log10(max(f.spectral_flatness, 1e-6)), -3.2, -1.4)
    bright_audio = _scale(np.log2(max(f.spectral_centroid_hz, 100) / 700), 0, np.log2(3500 / 700))
    low = (bands["sub_thump"] + bands["low_body"]) / 2
    bass_audio = _scale(low - bands["mid_bark"], -16, 4)
    mid_audio = _scale(bands["mid_bark"] - (bands["low_body"] + bands["high_mid_bite"]) / 2, -8, 6)
    tight_audio = _scale(f.attack_rise_ms, 40, 6) if f.attack_rise_ms > 0 else 0.5
    width_audio = _scale(f.stereo_width_ratio, 0.05, 0.6)

    def blend(key: str, audio: float, prompt_weight: float = 0.65) -> float:
        return _clip(prompt_weight * from_prompt[key] + (1 - prompt_weight) * audio) if key in from_prompt else audio

    gain = blend("gain", gain_audio)
    intent = Intent(
        focus=focus, gain=gain, brightness=blend("brightness", bright_audio), bass=blend("bass", bass_audio),
        mid=blend("mid", mid_audio), tight=blend("tight", tight_audio),
        ambience=from_prompt.get("ambience", 0.7 if focus == "lead" and "ambient" in prompt.lower() else 0.0),
        width=blend("width", width_audio) if focus == "rhythm" else from_prompt.get("width", 0.0),
    )
    if focus == "lead":
        intent.gain = _clip(gain + 0.05)
        intent.mid = _clip(intent.mid + 0.1)
        intent.tight = min(intent.tight, 0.6)
    intent.amp_class = "clean" if gain < 0.25 else "crunch" if gain < 0.55 else ("lead" if focus == "lead" else "high_gain")
    intent.notes.append(
        f"audio: crest {f.crest_factor_db:.1f} dB, centroid {f.spectral_centroid_hz:.0f} Hz, rise {f.attack_rise_ms:.1f} ms, "
        f"width {f.stereo_width_ratio:.2f}"
    )
    if from_prompt:
        intent.notes.append(f"prompt hints: {from_prompt}")
    return intent


# ---------------------------------------------------------------------------
# schema-driven parameter resolver
# ---------------------------------------------------------------------------
class Resolver:
    """Finds parameters by name pattern and records normalized settings, validated against the schema."""

    def __init__(self, schema: PluginSchema):
        self.schema = schema
        self.params = schema.by_name()
        self.values: dict[str, float] = {}
        self.skipped: list[str] = []

    def find(self, *patterns: str, kind: str | None = None) -> ParameterSpec | None:
        for pat in patterns:
            rx = re.compile(pat, re.IGNORECASE)
            for p in self.schema.parameters:
                if rx.search(p.name) and (kind is None or p.kind == kind):
                    return p
        return None

    def findall(self, pattern: str) -> list[ParameterSpec]:
        rx = re.compile(pattern, re.IGNORECASE)
        return [p for p in self.schema.parameters if rx.search(p.name)]

    def _put(self, p: ParameterSpec, x: float) -> None:
        self.values[p.name] = round(p.snap(x), 6)

    def frac(self, pat: str | ParameterSpec | None, x: float) -> ParameterSpec | None:
        """Set a knob by normalized 0-1 position (right for 0-100 % / 0-1 knobs)."""
        p = self.find(pat) if isinstance(pat, str) else pat
        if p is None:
            return None
        self._put(p, x)
        return p

    def display(self, pat: str | ParameterSpec | None, value: float) -> ParameterSpec | None:
        """Set a numeric knob by its displayed value (dB, Hz ...); clamps to the knob's range."""
        p = self.find(pat) if isinstance(pat, str) else pat
        if p is None or p.kind != "numeric":
            return None
        lo, hi = p.min_value, p.max_value
        if p.linear is False and p.curve:
            vs = [v for _, v in p.curve]
            value = min(max(value, min(vs)), max(vs))
        elif lo is not None and hi is not None:
            value = min(max(value, min(lo, hi)), max(lo, hi))
        try:
            self._put(p, p.normalized_for(value))
        except ValueError as e:
            self.skipped.append(str(e))
            return None
        return p

    def onoff(self, pat: str | ParameterSpec | None, on: bool) -> ParameterSpec | None:
        """Boolean, or an enum whose last option is the active state (Inactive/Active, Off/On)."""
        p = self.find(pat) if isinstance(pat, str) else pat
        if p is None or p.kind == "numeric":
            return None
        self._put(p, 1.0 if on else 0.0)
        return p

    def choose(self, pat: str, preferred_words: list[str], fallback_frac: float) -> tuple[ParameterSpec, str] | None:
        """Pick an enum option whose label contains one of `preferred_words` (in priority order)."""
        p = self.find(pat, kind="enum")
        if p is None or not p.valid_values:
            return None
        opts = [str(o) for o in p.valid_values]
        for w in preferred_words:
            for i, o in enumerate(opts):
                if w in o.lower():
                    self._put(p, i / (len(opts) - 1) if len(opts) > 1 else 0.0)
                    return p, o
        i = round(fallback_frac * (len(opts) - 1))
        self._put(p, i / (len(opts) - 1) if len(opts) > 1 else 0.0)
        return p, opts[i]


AMP_WORDS = {
    "clean": ["clean", "cln", "piezo", "smooth", "headroom"],
    "crunch": ["crunch", "rust", "edge", "signature", "drive"],
    "high_gain": ["rhythm", "hot", "high", "heavy"],
    "lead": ["lead", "solo", "hot"],
}


def _eq_gain_db(intent: Intent) -> list[float]:
    """Gentle 9-band tilt from the intent (octave centres 63 Hz ... 16 kHz), clamped to +/-5 dB."""
    out = []
    for f in EQ_CENTRES_HZ:
        octs = np.log2(f / 1000)
        low = _clip(-octs / 4)                       # 1 at 63 Hz ... 0 at 1 kHz
        high = _clip(octs / 3)                       # 0 at 1 kHz ... 1 at 8 kHz+
        mid = float(np.exp(-(octs - 0.3) ** 2 / 1.2))  # bump around 1.2 kHz
        db = (intent.bass - 0.5) * 8 * low + (intent.brightness - 0.5) * 8 * high + (intent.mid - 0.5) * 8 * mid
        if f >= 16000:
            db -= 3 * intent.gain  # tame fizz on high gain
        out.append(float(np.clip(db, -5, 5)))
    return out


def build_preset(
    features: TonalFeatures | None,
    user_prompt: str,
    target_plugin: str,
    schema: PluginSchema | None = None,
    preset_name: str | None = None,
    intent: Intent | None = None,
    focus: Focus | None = None,
) -> NeuralDSPPresetJSON:
    schema = schema or load_schema(target_plugin)
    if intent is None:
        if features is None:
            raise ValueError("need features or an explicit Intent")
        intent = derive_intent(features, user_prompt, focus)
    r = Resolver(schema)
    summary: dict[str, str] = {}

    # --- sections on, bypass off ------------------------------------------------------------------
    for p in r.findall(r"^active_\w+$|_section_active$|^active_amp$"):
        r.onoff(p, True)
    r.onoff(r"^bypass$", False)

    # --- input / gate -------------------------------------------------------------------------------
    r.display(r"^input_gain$", 0.0)
    r.display(r"^output_gain$", 0.0)
    gate_db = -78 + 24 * intent.tight          # -78 (loose) ... -54 (tight)
    gate_on = r.onoff(r"^gate_active$", intent.tight > 0.25 and intent.gain > 0.3)
    gate_p = r.find(r"^gate_threshold$", r"^gate_amount$", r"^gate$", kind="numeric")
    if gate_p is not None and gate_p.min_value is not None and gate_p.min_value <= -60 and gate_p.max_value <= 6:
        r.display(gate_p, gate_db)
    gate_txt = f"{gate_db:.0f} dB" if gate_p is not None else "n/a"
    summary["Gate"] = f"{'on' if intent.tight > 0.25 and intent.gain > 0.3 else 'off'} @ {gate_txt}" if (gate_on or gate_p) else "none"

    # --- boost / overdrive ---------------------------------------------------------------------------
    od = r.find(r"^(?:overdrive(?:_1)?|od|\w*precision_drive)_active$")
    if od is not None:
        use_boost = intent.gain >= 0.5 and intent.tight >= 0.5
        r.onoff(od, use_boost)
        base = od.name[: -len("_active")]
        r.frac(rf"^{re.escape(base)}_(drive|dist|gain)$", 0.08 + 0.12 * (1 - intent.tight))
        r.frac(rf"^{re.escape(base)}_(level|volume)$", 0.62)
        r.frac(rf"^{re.escape(base)}_tone$", 0.55 + 0.15 * intent.brightness)
        summary["Overdrive"] = (f"tightening boost on (low drive, level 62%)" if use_boost else "off")
    else:
        summary["Overdrive"] = "none"

    # --- amp -----------------------------------------------------------------------------------------
    amp_label, prefix = None, None
    pick = r.choose(r"^amp_type$|^amp_channel$|^channel$", AMP_WORDS[intent.amp_class],
                    {"clean": 0.0, "crunch": 0.4, "high_gain": 0.7, "lead": 1.0}[intent.amp_class])
    if pick:
        amp_label = pick[1]
        tok = re.sub(r"[^a-z0-9]+", " ", amp_label.lower()).split()[0]
        if r.findall(rf"^{tok}_"):
            prefix = tok
    if prefix:
        pre = rf"^{prefix}(?:_amp)?"
        r.frac(rf"{pre}_gain$", _clip(intent.gain * 0.85 + 0.1))
        r.frac(rf"{pre}_(bass|low)$", _clip(0.25 + 0.55 * intent.bass))
        r.frac(rf"{pre}_(mid|middle)$", _clip(0.2 + 0.6 * intent.mid))
        r.frac(rf"{pre}_(treble|high)$", _clip(0.25 + 0.55 * intent.brightness))
        r.frac(rf"{pre}_presence$", _clip(0.2 + 0.6 * intent.brightness))
        r.frac(rf"{pre}_(depth|resonance)$", _clip(0.25 + 0.5 * intent.bass))
        r.frac(rf"{pre}_master$", 0.5)
    summary["Amp"] = (
        f"{amp_label}, gain {intent.gain * 100:.0f}%, bass {intent.bass:.2f} mid {intent.mid:.2f} treble {intent.brightness:.2f}"
        if amp_label else "amp selection not available on this plugin")

    # --- graphic EQ + filters -----------------------------------------------------------------------
    eq_pre = rf"^{prefix}(?:_amp)?_eq" if prefix else r"^(?:\w+_)?eq"
    bands = sorted(r.findall(rf"{eq_pre}_band_\d+(?:_db)?$"), key=lambda p: int(re.search(r"band_(\d+)", p.name).group(1)))
    if len(bands) == len(EQ_CENTRES_HZ):
        r.onoff(rf"{eq_pre}_active$", True)
        gains = _eq_gain_db(intent)
        for p, db in zip(bands, gains):
            r.display(p, db)
        summary["Graphic EQ"] = "9-band: " + " ".join(f"{g:+.0f}" for g in gains) + " dB (octave centres 63 Hz-16 kHz, assumed)"
    else:
        summary["Graphic EQ"] = "no 9-band EQ matched; amp tone controls only"
    if features is not None:
        r.display(rf"{eq_pre}_high_pass$", 60 + 80 * intent.tight)
        r.display(rf"{eq_pre}_low_pass$", _clip(features.spectral_rolloff_hz, 3500, 12000) * 1.15)

    # --- cab -------------------------------------------------------------------------------------------
    r.onoff(r"^cab_(l|1)_active$", True)
    r.onoff(r"^cab_(r|2)_active$", intent.width > 0.5 or intent.focus == "rhythm")
    summary["Cab"] = "default mic/position (left/primary cab on, second cab for width)"

    # --- doubler, ambience -------------------------------------------------------------------------------
    dbl = r.onoff(r"^doubler_active$", intent.width > 0.4)
    if dbl is not None:
        r.display(r"^doubler_spread$", 6 + 8 * intent.width)
    amb_on = intent.ambience > 0.25
    post = []
    for label, act, mix in (
        ("reverb", r"^(?:\w*_)?(?:reverb|rev|spring_reverb)_active$", r"^(?:\w*_)?(?:reverb|rev|spring_reverb)_(mix|dry_wet)$"),
        ("delay", r"^(?:stereo_|dream_|millipede_)?(?:delay(?:_[12])?|dly)_active$", r"^(?:stereo_|dream_|millipede_)?(?:delay(?:_[12])?|dly)_(mix|dry_wet)$"),
    ):
        a = r.onoff(act, amb_on)
        if a is not None:
            if amb_on:
                r.frac(mix, 0.12 + 0.25 * intent.ambience)
            post.append(f"{label} {'on' if amb_on else 'off'}")
    summary["Post-FX"] = ", ".join(post) + (" ; doubler on" if dbl is not None and intent.width > 0.4 else "") if post or dbl else "none"

    double = intent.focus == "rhythm" and intent.width > 0.3
    name = preset_name or _make_name(target_plugin, intent, user_prompt)
    preset = NeuralDSPPresetJSON(
        preset_name=name,
        target_plugin=f"{schema.plugin_name}.vst3",
        track_config=TrackConfig(
            track_name=f"{intent.focus.title()} Gtr", double_tracked=double,
            panning=[-100.0, 100.0] if double else [0.0], input_gain_db=0.0),
        signal_chain_summary={**summary, "Intent": "; ".join(intent.notes)},
        vst_parameters=r.values,
    )
    preset.validate_against(schema)
    if r.skipped:
        log.warning("skipped: %s", r.skipped)
    return preset


def _make_name(plugin: str, intent: Intent, prompt: str) -> str:
    words = re.sub(r"[^A-Za-z0-9 ]+", "", prompt).split()[:4]
    base = "_".join(words) if words else f"{intent.focus}_{intent.amp_class}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", f"{base}_{intent.focus}")[:60]


def write_preset(preset: NeuralDSPPresetJSON, out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{preset.preset_name}.json"
    path.write_text(preset.model_dump_json(indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# optional LLM refinement (needs ANTHROPIC_API_KEY; not exercised by the offline tests)
# ---------------------------------------------------------------------------
def refine_with_llm(
    preset: NeuralDSPPresetJSON, features: TonalFeatures, user_prompt: str, schema: PluginSchema,
    model: str | None = None,
) -> NeuralDSPPresetJSON:
    """Ask Claude to adjust the heuristic preset. Only known parameter names with 0-1 values are kept."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    import anthropic

    known = schema.by_name()
    ctx = {k: {"now": v, "display": known[k].default_display, "kind": known[k].kind,
               "range": [known[k].min_value, known[k].max_value, known[k].units], "options": known[k].valid_values}
           for k, v in preset.vst_parameters.items()}
    msg = (
        f"User wants: {user_prompt!r}\nMeasured guitar features: {features.model_dump_json()}\n"
        f"Plugin: {schema.plugin_name}. Current parameter settings (normalized 0-1; options listed in order, "
        f"option i sits at i/(n-1)):\n{json.dumps(ctx)}\n\n"
        "Return ONLY a JSON object mapping parameter names (from the list above) to new normalized values 0-1, "
        "for the parameters you want to change. No prose."
    )
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model or os.environ.get("NEURAL_TONE_MODEL", "claude-sonnet-5-5"), max_tokens=2000,
        system="You are an expert guitar tone engineer tuning Neural DSP Archetype plugins.",
        messages=[{"role": "user", "content": msg}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError("LLM returned no JSON")
    changes = {k: float(v) for k, v in json.loads(m.group()).items()
               if k in preset.vst_parameters and isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0}
    merged = {**preset.vst_parameters, **{k: known[k].snap(v) for k, v in changes.items()}}
    out = preset.model_copy(update={"vst_parameters": merged})
    out.validate_against(schema)
    return out
