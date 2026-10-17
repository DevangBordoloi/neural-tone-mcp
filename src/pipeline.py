"""End-to-end orchestration shared by the MCP server and the CLI: source -> stem -> features -> preset JSON."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.dsp.extractor import extract_features
from src.dsp.vst_inspector import load_schema
from src.llm.tone_mapper import build_preset, derive_intent, parse_prompt, refine_with_llm, write_preset
from src.models import Focus

log = logging.getLogger(__name__)


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
) -> dict[str, Any]:
    """URL/WAV -> guitar stem -> TonalFeatures -> NeuralDSPPresetJSON file. Returns paths and the objects."""
    schema = load_schema(target_plugin)
    _, prompt_focus = parse_prompt(user_prompt)
    focus = focus or prompt_focus or "rhythm"

    audio = resolve_source(source, start_sec, duration_sec, work_dir)
    if is_guitar_stem:
        stem = audio
    else:
        from src.ingest.separator import separate_guitar

        stem = separate_guitar(audio, out_dir=(work_dir or audio.parent) / "stems")
    features = extract_features(stem, focus)

    intent = derive_intent(features, user_prompt, focus)
    preset = build_preset(features, user_prompt, target_plugin, schema=schema, intent=intent)
    if use_llm:
        try:
            preset = refine_with_llm(preset, features, user_prompt, schema)
        except Exception as e:  # keep the heuristic result if the LLM is unavailable
            log.warning("LLM refinement skipped: %s", e)
    path = write_preset(preset, out_dir)
    return {"json_path": str(path), "preset": preset, "features": features, "guitar_stem": str(stem), "audio": str(audio)}
