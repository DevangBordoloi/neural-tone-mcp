"""Pydantic schemas: the inspected plugin schema and the Neural DSP preset JSON."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ParamKind = Literal["boolean", "enum", "numeric"]


Focus = Literal["rhythm", "lead", "full"]
BAND_NAMES = ("sub_thump", "low_body", "low_mid_boxiness", "mid_bark", "high_mid_bite", "presence_air")
BAND_EDGES_HZ = ((60, 120), (120, 250), (250, 600), (600, 1600), (1600, 4000), (4000, 10000))


class TonalFeatures(BaseModel):
    """Measurements of an isolated guitar stem (see src/dsp/extractor.py for exact definitions)."""

    focus: Focus = "full"
    duration_sec: float
    rms_db: float = Field(description="RMS level, dBFS")
    crest_factor_db: float = Field(description="peak dBFS - RMS dBFS; low = compressed/saturated")
    attack_steepness: float = Field(description="mean onset strength per ms of rise time; high = snappy, gated")
    attack_rise_ms: float = Field(description="mean 10%->90% rise time of note onsets")
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    spectral_flatness: float = Field(description="0 tonal ... 1 noise-like; rises with distortion")
    eq_band_energy_ratios: dict[str, float] = Field(
        description="dB of each band's energy relative to the 60-10000 Hz total, keys = BAND_NAMES"
    )
    mfcc_mean: list[float] = Field(min_length=13, max_length=13)
    stereo_width_ratio: float = Field(description="Side/Mid energy ratio of the stereo guitar stem")


class ParamSample(BaseModel):
    """What the plugin displays at one normalized position (captures non-linear knob scaling)."""

    normalized: float
    display: str


class ParameterSpec(BaseModel):
    index: int = Field(description="pedalboard's enumeration order; NOT guaranteed to equal the host/REAPER param index")
    name: str = Field(description="pedalboard key (snake_case) in plugin.parameters; use this in vst_parameters")
    host_name: str | None = Field(None, description="the plugin's own display name, e.g. 'Input Gain' (what a DAW shows)")
    kind: ParamKind
    default_normalized: float = Field(description="0-1 value the plugin held when it was loaded")
    default_display: str
    units: str | None = None
    min_value: float | None = Field(None, description="numeric params: value at normalized 0.0")
    max_value: float | None = Field(None, description="numeric params: value at normalized 1.0")
    step_size: float | None = None
    num_steps: int | None = Field(None, description="number of distinct settings, when finite")
    valid_values: list[bool | str] | None = Field(
        None, description="boolean/enum params: options in normalized order (index / (n - 1))"
    )
    linear: bool | None = Field(
        None, description="numeric params: True if display value is linear in the normalized value"
    )
    samples: list[ParamSample] = Field(default_factory=list)
    curve: list[tuple[float, float]] | None = Field(
        None, description="non-linear numeric params: dense (normalized, value) pairs parsed from the display text"
    )
    options_verified: bool | None = Field(
        None, description="enum params: True if option i really sits at normalized i/(n-1) (checked on the plugin)"
    )

    def snap(self, normalized: float) -> float:
        """Clamp to 0-1 and snap to the nearest real setting for stepped params."""
        x = min(1.0, max(0.0, float(normalized)))
        n = self.num_steps
        if n and n > 1:
            x = round(x * (n - 1)) / (n - 1)
        return x

    def normalized_for(self, value: bool | str | float) -> float:
        """Convert a display-side value (option label, bool, or number) to normalized 0-1."""
        if self.valid_values:
            opts = self.valid_values
            if value not in opts:
                raise ValueError(f"{self.name}: {value!r} not in {opts}")
            return opts.index(value) / (len(opts) - 1) if len(opts) > 1 else 0.0
        if self.min_value is None or self.max_value is None or self.max_value == self.min_value:
            raise ValueError(f"{self.name}: has no numeric range")
        if self.linear is False:
            if not self.curve:
                raise ValueError(f"{self.name}: non-linear scale and no curve; use a normalized value (see samples)")
            xs, vs = zip(*self.curve)
            v = float(value)
            if not min(vs) <= v <= max(vs):
                raise ValueError(f"{self.name}: {v} outside {min(vs)}..{max(vs)}")
            return self.snap(float(__import__("numpy").interp(v, vs, xs)))
        return self.snap((float(value) - self.min_value) / (self.max_value - self.min_value))


class PluginSchema(BaseModel):
    plugin_name: str
    vst3_path: str
    is_instrument: bool = False
    parameter_count: int
    inspected_with: str = Field(description="pedalboard version")
    parameters: list[ParameterSpec]

    def by_name(self) -> dict[str, ParameterSpec]:
        return {p.name: p for p in self.parameters}

    def search(self, *words: str) -> list[ParameterSpec]:
        toks = [w.lower() for w in words]
        return [p for p in self.parameters if all(t in p.name.lower() for t in toks)]


class TrackConfig(BaseModel):
    track_name: str
    double_tracked: bool = False
    panning: list[float] = Field(default_factory=lambda: [0.0], description="-100 (L) to 100 (R) per track")
    input_gain_db: float = 0.0


class NeuralDSPPresetJSON(BaseModel):
    preset_name: str
    target_plugin: str = Field(description='e.g. "Archetype Gojira.vst3"')
    track_config: TrackConfig
    signal_chain_summary: dict[str, str] = Field(default_factory=dict)
    vst_parameters: dict[str, float] = Field(
        default_factory=dict, description="exact parameter name -> normalized 0-1 value"
    )

    def validate_against(self, schema: PluginSchema, strict: bool = True) -> list[str]:
        """Return problems (unknown names, out-of-range values). Raises if strict and any exist."""
        known = schema.by_name()
        problems = []
        for k, v in self.vst_parameters.items():
            if k not in known:
                problems.append(f"unknown parameter {k!r}")
            elif not 0.0 <= v <= 1.0:
                problems.append(f"{k}: {v} outside 0-1")
        if strict and problems:
            raise ValueError("; ".join(problems))
        return problems

    def snapped(self, schema: PluginSchema) -> dict[str, Any]:
        """vst_parameters with each stepped value snapped to a real setting."""
        known = schema.by_name()
        return {k: known[k].snap(v) for k, v in self.vst_parameters.items() if k in known}
