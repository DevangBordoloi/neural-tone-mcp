"""FastMCP server: the neural-tone pipeline as tools Claude can call.

Run:  python -m src.mcp_server
Register:  claude mcp add neural-tone -- <venv>\\Scripts\\python.exe -m src.mcp_server   (from the project dir)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]

try:  # mcp >= 2: only ToolError messages reach the model; anything else becomes "Error executing tool"
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    try:
        from mcp.server.fastmcp.exceptions import ToolError  # type: ignore[no-redef]
    except ImportError:
        ToolError = Exception  # type: ignore[misc,assignment]

from src.daw.ableton_driver import AbletonDriver
from src.daw.base import DawController, DawError
from src.daw.reaper_driver import ReaperDriver
from src.dsp.vst_inspector import InspectorError, dump_plugin_schema, load_schema
from src.pipeline import extract_tone, extract_tone_hybrid

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")

INSTRUCTIONS = """\
Builds Neural DSP tones from reference audio.
1. inspect_neural_dsp_plugin(vst3_path) once per plugin (slow, ~20 s) - caches its exact parameter schema.
2. extract_tone_to_json(url, user_prompt, target_plugin) - downloads the clip, isolates the guitar, measures the
   tone and writes a preset JSON (normalized 0-1 values keyed by exact parameter name).
3. apply_tone_json_to_daw(json_path, daw) - creates the track(s) in REAPER (or Ableton) and applies every parameter.
match_url_tone_end_to_end does 2+3 in one call.
match_url_tone_hybrid does the same with the dual-LLM engine: Claude drafts the parameters, a headless render of the
plugin is compared with the reference, and Gemini listens to both recordings only when the band error is above
2 dB (or force_gemini). It needs ANTHROPIC_API_KEY and, for the escalation, GEMINI_API_KEY; without them it falls
back to the offline heuristic preset, and the result's `refiner` field says which stage produced the tone. The result is a rule-of-thumb starting point, not a clone of the
record: say so to the user, and tell them what was assumed (see signal_chain_summary in the result).
The REAPER bridge is shared with the separate `reaper` MCP server: do not call `reaper` tools at the same time as
apply_tone_json_to_daw / match_url_tone_end_to_end. Extraction takes minutes on CPU (Demucs).
Use `rhythm` or `lead` in user_prompt to pick the Side (double-tracked rhythm) or Mid (lead) channel.
"""

mcp = _Server("neural-tone", instructions=INSTRUCTIONS)


def _driver(daw: str) -> DawController:
    d = daw.lower()
    if d == "reaper":
        return ReaperDriver()
    if d == "ableton":
        return AbletonDriver()
    raise DawError(f"unknown daw {daw!r}; use 'reaper' or 'ableton'")


def _brief(res: dict[str, Any]) -> dict[str, Any]:
    p, f = res["preset"], res["features"]
    return {
        "json_path": res["json_path"],
        "preset_name": p.preset_name,
        "target_plugin": p.target_plugin,
        "track_config": p.track_config.model_dump(),
        "signal_chain_summary": p.signal_chain_summary,
        "parameters_set": len(p.vst_parameters),
        "features": {
            "focus": f.focus, "rms_db": round(f.rms_db, 1), "crest_factor_db": round(f.crest_factor_db, 1),
            "attack_rise_ms": round(f.attack_rise_ms, 1), "spectral_centroid_hz": round(f.spectral_centroid_hz),
            "stereo_width_ratio": round(f.stereo_width_ratio, 2),
            "eq_band_energy_db": {k: round(v, 1) for k, v in f.eq_band_energy_ratios.items()},
        },
        "guitar_stem": res["guitar_stem"],
    }


@mcp.tool()
async def inspect_neural_dsp_plugin(vst3_path: str) -> dict:
    """Load a Neural DSP .vst3 (path or name like 'Nolly X') and save its parameter schema. Slow (~20 s)."""
    try:
        schema, out = await asyncio.to_thread(dump_plugin_schema, vst3_path)
    except InspectorError as e:
        raise ToolError(str(e)) from e
    kinds: dict[str, int] = {}
    for p in schema.parameters:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    return {"plugin": schema.plugin_name, "parameters": schema.parameter_count, "kinds": kinds, "schema_file": str(out)}


@mcp.tool()
async def extract_tone_to_json(
    url: str, user_prompt: str, target_plugin: str, start_sec: int = 30, duration_sec: int = 25,
    use_llm: bool = False,
) -> dict:
    """Download audio from `url` (or a local WAV path), isolate the guitar, measure it and write the preset JSON.

    target_plugin: name of an inspected plugin, e.g. "Archetype Nolly X". user_prompt: the tone wanted, e.g.
    "tight modern djent rhythm". use_llm=True lets Claude refine the heuristic preset (needs ANTHROPIC_API_KEY).
    """
    try:
        res = await asyncio.to_thread(
            extract_tone, url, user_prompt, target_plugin, start_sec, duration_sec, None, False, use_llm)
    except (InspectorError, DawError, RuntimeError, ValueError) as e:
        raise ToolError(str(e)) from e
    return _brief(res)


@mcp.tool()
async def apply_tone_json_to_daw(json_path: str, daw: Literal["reaper", "ableton"] = "reaper") -> dict:
    """Create the track(s), insert the Neural DSP plugin and apply every parameter from a preset JSON."""
    def run() -> dict[str, Any]:
        d = _driver(daw)
        d.connect()
        return d.create_tracks_and_apply_preset(json_path)

    try:
        return await asyncio.to_thread(run)
    except (DawError, InspectorError, FileNotFoundError, ValueError, RuntimeError) as e:
        raise ToolError(str(e)) from e


@mcp.tool()
async def match_url_tone_end_to_end(
    url: str, user_prompt: str, target_plugin: str, daw: Literal["reaper", "ableton"] = "reaper",
    start_sec: int = 30, duration_sec: int = 25,
) -> dict:
    """URL -> guitar stem -> tone features -> preset JSON -> tracks and plugin settings in the DAW."""
    extracted = await extract_tone_to_json(url, user_prompt, target_plugin, start_sec, duration_sec)
    applied = await apply_tone_json_to_daw(extracted["json_path"], daw)
    return {"extracted": extracted, "applied": applied}


@mcp.tool()
async def match_url_tone_hybrid(
    url: str, user_prompt: str, target_plugin_name: str, force_gemini: bool = False,
    daw: Literal["reaper", "ableton"] = "reaper", start_sec: int = 30, duration_sec: int = 25,
) -> dict:
    """URL -> stem -> features -> Claude draft -> headless render check -> (Gemini listens if off) -> DAW.

    force_gemini=True always sends the reference and the render to Gemini. `refiner` in the result is "claude",
    "gemini", "claude_fallback" (Gemini was needed but failed) or "heuristic" (Claude unavailable).
    """
    try:
        res = await asyncio.to_thread(
            extract_tone_hybrid, url, user_prompt, target_plugin_name, start_sec, duration_sec, None, False, force_gemini)
    except (InspectorError, DawError, RuntimeError, ValueError) as e:
        raise ToolError(str(e)) from e
    applied = await apply_tone_json_to_daw(res["json_path"], daw)
    return {"extracted": {**_brief(res), "refiner": res["refiner"], "reasoning": res["reasoning"]}, "applied": applied}


@mcp.tool()
async def get_plugin_schema_summary(plugin: str, name_filter: str | None = None, limit: int = 80) -> dict:
    """Read a saved plugin schema (no plugin load): parameter names, kinds, ranges. Filter by words in the name."""
    try:
        s = load_schema(plugin)
    except InspectorError as e:
        raise ToolError(str(e)) from e
    ps = s.search(*(name_filter or "").split())
    return {"plugin": s.plugin_name, "total": s.parameter_count, "matched": len(ps), "parameters": [
        {"name": p.name, "host_name": p.host_name, "kind": p.kind, "default": p.default_display,
         "range": [p.min_value, p.max_value, p.units] if p.kind == "numeric" else p.valid_values}
        for p in ps[:limit]]}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
