"""Introspect a VST3 (e.g. Neural DSP Archetype) with Spotify's pedalboard and dump its parameters.

The plugin's binary preset format is never touched: the plugin is loaded and its host-visible
parameters are enumerated (name, range, default, option list, display samples).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from src.models import NeuralDSPPresetJSON, ParameterSpec, ParamSample, PluginSchema

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_DIR = PROJECT_ROOT / "config" / "plugin_schemas"
SAMPLE_POINTS = (0.0, 0.25, 0.5, 0.75, 1.0)
CURVE_POINTS = 33


class InspectorError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# locating plugins / schema files
# ---------------------------------------------------------------------------
def vst3_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    if os.environ.get("NEURAL_DSP_VST3_DIR"):
        dirs.append(Path(os.environ["NEURAL_DSP_VST3_DIR"]).expanduser())
    if sys.platform == "win32":
        for var in ("COMMONPROGRAMFILES", "COMMONPROGRAMFILES(X86)"):
            if os.environ.get(var):
                dirs.append(Path(os.environ[var]) / "VST3")
        # MCP hosts may launch us with a scrubbed environment, so also try the usual install locations
        dirs += [Path(r"C:\Program Files\Common Files\VST3"), Path(r"C:\Program Files (x86)\Common Files\VST3")]
        dirs.append(Path.home() / "AppData/Local/Programs/Common/VST3")
    elif sys.platform == "darwin":
        dirs += [Path("/Library/Audio/Plug-Ins/VST3"), Path.home() / "Library/Audio/Plug-Ins/VST3"]
    else:
        dirs += [Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3"), Path.home() / ".vst3"]
    return [d for d in dict.fromkeys(dirs) if d.is_dir()]


def find_vst3(name_or_path: str | os.PathLike[str]) -> Path:
    """Resolve a path, or a (partial, case-insensitive) plugin name such as "Nolly X", to a .vst3."""
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    want = re.sub(r"[^a-z0-9]", "", str(name_or_path).lower().removesuffix(".vst3"))
    found: list[Path] = []
    for d in vst3_search_dirs():
        found += [c for c in d.rglob("*.vst3") if want and want in re.sub(r"[^a-z0-9]", "", c.stem.lower())]
    found = list({c.resolve(): c for c in found}.values())  # same bundle reached via overlapping search dirs
    exact = [c for c in found if re.sub(r"[^a-z0-9]", "", c.stem.lower()) == want]
    found = exact or found
    if len(found) == 1:
        return found[0]
    if not found:
        raise InspectorError(f"no .vst3 matching {str(name_or_path)!r} in {[str(d) for d in vst3_search_dirs()]}")
    raise InspectorError(f"{str(name_or_path)!r} is ambiguous: {[c.name for c in found]}")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def schema_dir() -> Path:
    env = os.environ.get("PLUGIN_SCHEMA_DIR")
    if not env:
        return DEFAULT_SCHEMA_DIR
    d = Path(env).expanduser()
    return d if d.is_absolute() else PROJECT_ROOT / d


def schema_path(plugin_name: str, out_dir: Path | None = None) -> Path:
    safe = re.sub(r'[<>:"/\\|?*]+', "_", plugin_name).strip()
    return (out_dir or schema_dir()) / f"{safe}.json"


def load_schema(name_or_file: str | os.PathLike[str], out_dir: Path | None = None) -> PluginSchema:
    p = Path(name_or_file)
    if not (p.suffix == ".json" and p.exists()):
        d = out_dir or schema_dir()
        want = _norm(str(name_or_file).removesuffix(".vst3"))
        have = {c: _norm(c.stem) for c in d.glob("*.json")} if d.is_dir() else {}
        hits = [c for c, n in have.items() if n == want] or [c for c, n in have.items() if want and (want in n or n in want)]
        if len(hits) != 1:
            why = "ambiguous: " + ", ".join(c.stem for c in hits) if hits else "none found"
            raise InspectorError(f"schema for {str(name_or_file)!r} in {d}: {why}; run scripts/inspect_plugin.py first")
        p = hits[0]
    return PluginSchema.model_validate_json(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# inspection
# ---------------------------------------------------------------------------
def load_plugin(vst3_path: str | os.PathLike[str]) -> Any:
    try:
        import pedalboard
    except ImportError as e:  # pragma: no cover
        raise InspectorError("pedalboard is not installed") from e
    path = find_vst3(vst3_path)
    log.info("loading %s (large plugins can take 20+ seconds)", path)
    try:
        return pedalboard.load_plugin(str(path))
    except Exception as e:
        raise InspectorError(f"pedalboard could not load {path}: {e}") from e


def _classify(valid: list[Any] | None, min_v: Any, max_v: Any) -> str:
    if valid and all(isinstance(v, bool) for v in valid):
        return "boolean"
    if valid and any(isinstance(v, str) for v in valid):
        return "enum"
    if min_v is None or max_v is None:
        return "enum" if valid else "numeric"
    return "numeric"


def _sample_display(param: Any, x: float) -> str:
    param.raw_value = x
    return str(param.string_value)


def _parse_number(display: str) -> float | None:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(k)?", display.replace(",", ""), re.IGNORECASE)
    return float(m.group(1)) * (1000 if m.group(2) else 1) if m else None


def _is_linear(spec_min: float, spec_max: float, samples: list[ParamSample]) -> bool | None:
    """Does the displayed number track the normalized value linearly? Parse numbers out of the samples."""
    pts = []
    for s in samples:
        v = _parse_number(s.display)
        if v is None:
            return None
        pts.append((s.normalized, v))
    span = spec_max - spec_min
    if span == 0:
        return None
    tol = max(abs(span) * 0.03, 1e-6)
    return all(abs((spec_min + n * span) - v) <= tol for n, v in pts)


def _measure_curve(param: Any, restore: float) -> list[tuple[float, float]] | None:
    """Dense (normalized, value) table for a non-linear knob, parsed from the plugin's display text."""
    pts: list[tuple[float, float]] = []
    try:
        for i in range(CURVE_POINTS):
            x = i / (CURVE_POINTS - 1)
            v = _parse_number(_sample_display(param, x))
            if v is None:
                return None
            pts.append((x, v))
    except Exception as e:
        log.warning("could not measure curve: %s", e)
        return None
    finally:
        try:
            param.raw_value = restore
        except Exception:
            pass
    vs = [v for _, v in pts]
    return pts if all(b >= a for a, b in zip(vs, vs[1:])) or all(b <= a for a, b in zip(vs, vs[1:])) else None


def _verify_options(param: Any, options: list[Any], restore: float) -> bool:
    """Confirm option i displays at normalized i/(n-1) (spot-checks long lists)."""
    n = len(options)
    idxs = range(n) if n <= 64 else sorted({0, n - 1, *range(0, n, max(1, n // 32))})
    try:
        return all(str(param_display) == str(options[i])
                   for i in idxs for param_display in [_sample_display(param, i / (n - 1) if n > 1 else 0.0)])
    except Exception:
        return False
    finally:
        try:
            param.raw_value = restore
        except Exception:
            pass


def inspect_parameter(index: int, name: str, param: Any, samples: Iterable[float] = SAMPLE_POINTS) -> ParameterSpec:
    valid = list(param.valid_values) if getattr(param, "valid_values", None) is not None else None
    min_v, max_v = getattr(param, "min_value", None), getattr(param, "max_value", None)
    kind = _classify(valid, min_v, max_v)
    default = float(param.raw_value)
    default_display = str(param.string_value)

    sample_list: list[ParamSample] = []
    try:
        for x in samples:
            sample_list.append(ParamSample(normalized=x, display=_sample_display(param, x)))
    except Exception as e:  # a parameter that rejects writes still gets its metadata recorded
        log.warning("could not sample %s: %s", name, e)
        sample_list = []
    finally:
        try:
            param.raw_value = default  # leave the plugin as we found it
        except Exception:
            pass

    spec = ParameterSpec(
        index=index, name=name, host_name=getattr(param, "name", None), kind=kind, default_normalized=default, default_display=default_display,
        units=(getattr(param, "units", None) or getattr(param, "label", None) or None),
        step_size=getattr(param, "step_size", None) or getattr(param, "approximate_step_size", None),
        num_steps=len(valid) if valid else None,
        samples=sample_list,
    )
    if kind == "numeric":
        spec.min_value, spec.max_value = float(min_v), float(max_v)
        spec.linear = _is_linear(spec.min_value, spec.max_value, sample_list)
        if spec.linear is not True:
            spec.curve = _measure_curve(param, default)
    else:
        spec.valid_values = valid
        if kind == "enum" and valid:
            spec.options_verified = _verify_options(param, valid, default)
        if kind == "enum" and min_v is not None and max_v is not None:
            spec.min_value, spec.max_value = float(min_v), float(max_v)
    return spec


def inspect_plugin(plugin_or_path: Any, samples: Iterable[float] = SAMPLE_POINTS) -> PluginSchema:
    """Build a PluginSchema from a .vst3 path/name, or from an already-loaded pedalboard plugin."""
    import pedalboard

    if isinstance(plugin_or_path, (str, os.PathLike)):
        path = find_vst3(plugin_or_path)
        plugin = load_plugin(path)
    else:
        plugin, path = plugin_or_path, Path(getattr(plugin_or_path, "path_to_plugin_file", "") or "")
    params = plugin.parameters
    specs = [inspect_parameter(i, k, v, samples) for i, (k, v) in enumerate(params.items())]
    name = str(getattr(plugin, "name", "") or path.stem)
    return PluginSchema(
        plugin_name=name, vst3_path=str(path), is_instrument=bool(getattr(plugin, "is_instrument", False)),
        parameter_count=len(specs), inspected_with=getattr(pedalboard, "__version__", "unknown"), parameters=specs,
    )


def save_schema(schema: PluginSchema, out_dir: Path | None = None) -> Path:
    out = schema_path(schema.plugin_name, out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(schema.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    return out


def dump_plugin_schema(vst3: str | os.PathLike[str], out_dir: Path | None = None) -> tuple[PluginSchema, Path]:
    schema = inspect_plugin(vst3)
    return schema, save_schema(schema, out_dir)


# ---------------------------------------------------------------------------
# headless preview
# ---------------------------------------------------------------------------
def render_preview(
    di_wav_path: str | os.PathLike[str],
    preset_json: NeuralDSPPresetJSON | dict[str, Any] | str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    plugin: Any | None = None,
) -> Path:
    """Run a dry guitar DI through the target plugin with the preset's parameters; write a WAV.

    `preset_json` may be a NeuralDSPPresetJSON, its dict form, or a path to the JSON file.
    `plugin` is an already-loaded plugin (skips the slow load); otherwise preset.target_plugin is loaded.
    """
    from pedalboard.io import AudioFile

    if isinstance(preset_json, (str, os.PathLike)):
        preset = NeuralDSPPresetJSON.model_validate_json(Path(preset_json).read_text(encoding="utf-8"))
    elif isinstance(preset_json, dict):
        preset = NeuralDSPPresetJSON.model_validate(preset_json)
    else:
        preset = preset_json

    plugin = plugin or load_plugin(preset.target_plugin)
    params = plugin.parameters
    missing = [k for k in preset.vst_parameters if k not in params]
    if missing:
        raise InspectorError(f"plugin has no parameters named {missing[:8]}")
    for k, v in preset.vst_parameters.items():
        params[k].raw_value = min(1.0, max(0.0, float(v)))

    with AudioFile(str(di_wav_path)) as f:
        audio, sr = f.read(f.frames), f.samplerate
    if audio.shape[0] == 1:  # amp sims are happiest with a stereo bus
        import numpy as np

        audio = np.repeat(audio, 2, axis=0)
    out = plugin(audio, sr, reset=True)

    out_path = Path(output_path) if output_path else Path("output") / f"{preset.preset_name}_preview.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with AudioFile(str(out_path), "w", sr, out.shape[0]) as f:
        f.write(out)
    return out_path
