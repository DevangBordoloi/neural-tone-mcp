"""End-to-end orchestration shared by the MCP server and the CLI: source -> stem -> features -> preset JSON."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.dsp.extractor import extract_features
from src.dsp.vst_inspector import InspectorError, find_vst3, load_schema
from src.llm.hybrid_arbiter import orchestrate_tone_refinement
from src.llm.tone_mapper import build_preset, derive_intent, parse_prompt, refine_with_llm, write_preset
from src.models import Focus, PluginSchema

log = logging.getLogger(__name__)

# API keys and other settings: <project>/.env, without overriding variables already set in the environment
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def resolve_source(
    source: str, start_sec: float | None, duration_sec: float | None, work_dir: Path | None = None,
) -> Path:
    """A local audio file is used as-is; anything else is treated as a URL and downloaded."""
    p = Path(source).expanduser()
    if p.exists():
        return p
    from src.ingest.downloader import download_audio

    end = None if start_sec is None or duration_sec is None else start_sec + duration_sec
    return download_audio(source, start_sec=start_sec, end_sec=end, out_dir=work_dir)


def _analyse(
    source: str, user_prompt: str, target_plugin: str, start_sec: float | None, duration_sec: float | None,
    focus: Focus | None, is_guitar_stem: bool, work_dir: Path | None, separation: str | None = None,
):
    """Shared first half: source -> guitar stem -> TonalFeatures -> intent -> heuristic baseline preset."""
    schema = load_schema(target_plugin)
    _, prompt_focus = parse_prompt(user_prompt)
    focus = focus or prompt_focus or "rhythm"

    audio = resolve_source(source, start_sec, duration_sec, work_dir)
    if is_guitar_stem:
        stem = audio
    else:
        from src.ingest.separator import separate_guitar

        stem = separate_guitar(audio, out_dir=(work_dir or audio.parent) / "stems", strategy=separation)
    features = extract_features(stem, focus)

    intent = derive_intent(features, user_prompt, focus)
    preset = build_preset(features, user_prompt, target_plugin, schema=schema, intent=intent)
    return schema, audio, stem, features, intent, preset


def extract_tone(
    source: str,
    user_prompt: str,
    target_plugin: str,
    start_sec: float | None = 30,
    duration_sec: float | None = 25,
    focus: Focus | None = None,
    is_guitar_stem: bool = False,
    use_llm: bool = False,
    out_dir: Path | None = None,
    work_dir: Path | None = None,
    separation: str | None = None,
) -> dict[str, Any]:
    """URL/WAV -> guitar stem -> TonalFeatures -> NeuralDSPPresetJSON file. Returns paths and the objects.

    separation: "cascade" or "single" (default: NEURAL_TONE_SEPARATION, else cascade); see src/ingest/separator.py.
    """
    schema, audio, stem, features, _, preset = _analyse(
        source, user_prompt, target_plugin, start_sec, duration_sec, focus, is_guitar_stem, work_dir, separation)
    if use_llm:
        try:
            preset = refine_with_llm(preset, features, user_prompt, schema)
        except Exception as e:  # keep the heuristic result if the LLM is unavailable
            log.warning("LLM refinement skipped: %s", e)
    path = write_preset(preset, out_dir)
    return {"json_path": str(path), "preset": preset, "features": features, "guitar_stem": str(stem), "audio": str(audio)}


def _locate_vst3(schema: PluginSchema) -> str:
    """Where the plugin binary is now: the path recorded at inspection time, else a name search."""
    if Path(schema.vst3_path).exists():
        return schema.vst3_path
    try:
        return str(find_vst3(schema.plugin_name))
    except InspectorError:
        return schema.vst3_path  # the arbiter skips the render check when this does not exist


def extract_tone_hybrid(
    source: str,
    user_prompt: str,
    target_plugin: str,
    start_sec: float | None = 30,
    duration_sec: float | None = 25,
    focus: Focus | None = None,
    is_guitar_stem: bool = False,
    force_gemini: bool = False,
    out_dir: Path | None = None,
    work_dir: Path | None = None,
    separation: str | None = None,
) -> dict[str, Any]:
    """Like extract_tone, but Claude refines the heuristic preset, a headless render audits it, and Gemini listens
    when the band error is too large (or force_gemini). `refiner` in the result says which stage produced the preset."""
    schema, audio, stem, features, intent, baseline = _analyse(
        source, user_prompt, target_plugin, start_sec, duration_sec, focus, is_guitar_stem, work_dir, separation)

    refinement, refiner = orchestrate_tone_refinement(
        target_stem_wav=str(stem),
        dsp_features=features.model_dump(),
        plugin_schema=schema.model_dump(),
        vst3_path=_locate_vst3(schema),
        user_prompt=user_prompt,
        force_gemini_audio=force_gemini,
        baseline_parameters=baseline.vst_parameters,
        baseline_double_track=baseline.track_config.double_tracked,
        work_dir=(work_dir or Path(stem).parent) / "closed_loop",
    )

    double = refinement.double_track_rhythm and intent.focus == "rhythm"  # the lead is the mono Mid channel
    preset = baseline.model_copy(update={
        "vst_parameters": refinement.vst_parameters,
        "track_config": baseline.track_config.model_copy(
            update={"double_tracked": double, "panning": [-100.0, 100.0] if double else [0.0]}),
        "signal_chain_summary": {**baseline.signal_chain_summary, "Refiner": refiner, "Reasoning": refinement.reasoning},
    })
    preset.validate_against(schema)
    path = write_preset(preset, out_dir)
    return {
        "json_path": str(path), "preset": preset, "features": features, "guitar_stem": str(stem), "audio": str(audio),
        "refiner": refiner, "reasoning": refinement.reasoning,
    }
