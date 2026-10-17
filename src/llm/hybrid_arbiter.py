"""Hybrid coordinator: Claude drafts numerically, a headless render checks it, Gemini listens only when it must.

    Claude draft -> pedalboard render -> band error vs reference
        error <= threshold  -> keep Claude's draft                       (source "claude")
        error >  threshold  -> Gemini listens to reference + render      (source "gemini")

Sources returned alongside the preset:
    "claude"           accepted Claude's draft after (or without being able to run) the render check
    "gemini"           Gemini's corrected preset
    "claude_fallback"  escalation was needed but Gemini failed (no key, API error); Claude's draft is returned
    "heuristic"        Claude was unavailable, so the offline heuristic baseline was used instead
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.dsp.closed_loop import (
    DEFAULT_DI_PATH, compute_band_spectral_deltas, ensure_default_di, render_preset_preview,
)
from src.dsp.vst_inspector import load_plugin
from src.llm.claude_refiner import NeuralDSPRefinement, refine_parameters_with_claude
from src.llm.gemini_refiner import refine_parameters_with_gemini
from src.models import ParameterSpec, PluginSchema

log = logging.getLogger(__name__)


def _param_specs(plugin_schema: dict) -> dict[str, ParameterSpec | None]:
    """name -> ParameterSpec (for snapping stepped values); names only if the dict is not a full PluginSchema."""
    try:
        return dict(PluginSchema.model_validate(plugin_schema).by_name())
    except ValueError:
        return {p["name"]: None for p in plugin_schema.get("parameters", []) if "name" in p}


def sanitize_refinement(
    refinement: NeuralDSPRefinement, plugin_schema: dict, base: dict[str, float] | None = None,
) -> NeuralDSPRefinement:
    """Overlay a refiner's parameters onto `base`, dropping names the plugin does not have and snapping stepped values.

    Values are already clamped to 0-1 by NeuralDSPRefinement. Parameters the refiner left out keep their `base` value,
    which is how switches the refiner did not mention (amp section on, cab on, ...) survive.
    """
    known = _param_specs(plugin_schema)
    merged = dict(base or {})
    dropped = []
    for name, value in refinement.vst_parameters.items():
        if known and name not in known:
            dropped.append(name)
            continue
        spec = known.get(name)
        merged[name] = round(spec.snap(value), 6) if spec is not None else value
    if dropped:
        log.warning("dropped %d parameter(s) not in the plugin schema: %s", len(dropped), dropped[:8])
    return refinement.model_copy(update={"vst_parameters": merged})


def orchestrate_tone_refinement(
    target_stem_wav: str,
    dsp_features: dict,
    plugin_schema: dict,
    vst3_path: str,
    user_prompt: str,
    di_wav_path: str = DEFAULT_DI_PATH,
    error_threshold_db: float = 2.0,
    force_gemini_audio: bool = False,
    baseline_parameters: dict[str, float] | None = None,
    baseline_double_track: bool = False,
    work_dir: str | Path | None = None,
) -> tuple[NeuralDSPRefinement, str]:
    """Run the Claude -> render -> (Gemini) flow. `baseline_*` is the offline heuristic preset: it is shown to
    Claude, used as the starting point for every merge, and used as the draft if Claude is unavailable."""
    # 1. Stage 1 draft
    source = "claude"
    try:
        draft = refine_parameters_with_claude(dsp_features, plugin_schema, user_prompt, None, baseline_parameters)
        draft = sanitize_refinement(draft, plugin_schema, baseline_parameters)
    except Exception as e:  # missing key, API error, malformed reply: keep going with the offline baseline
        log.warning("Claude refinement unavailable (%s); using the heuristic baseline as the draft", e)
        source = "heuristic"
        draft = NeuralDSPRefinement(
            reasoning="Claude refinement unavailable; offline heuristic baseline.",
            double_track_rhythm=baseline_double_track,
            vst_parameters=dict(baseline_parameters or {}),
        )

    # 2. Headless render of the draft
    if not Path(vst3_path).expanduser().exists():
        log.warning("VST3 not found at %s; skipping the render check%s", vst3_path,
                    " (cannot escalate to Gemini without a render)" if force_gemini_audio else "")
        return draft, source
    out_dir = Path(work_dir) if work_dir else Path(target_stem_wav).parent / "closed_loop"
    rendered = out_dir / "claude_rendered.wav"
    try:
        di = ensure_default_di(di_wav_path)
        plugin = load_plugin(vst3_path)  # loaded once: a second render for the post-check reuses it
        render_preset_preview(vst3_path, draft.vst_parameters, str(di), str(rendered), plugin=plugin)

        # 3. Band error against the reference stem
        mean_error_db, spectral_deltas = compute_band_spectral_deltas(target_stem_wav, str(rendered))
    except Exception as e:
        log.error("render check failed (%s); returning the draft unverified", e)
        return draft, source
    log.info("band deltas (dB, rendered - reference): %s", {k: round(v, 1) for k, v in spectral_deltas.items()})

    # 4. Decide
    if mean_error_db <= error_threshold_db and not force_gemini_audio:
        log.info("Claude numeric model achieved acceptable match: %.2fdB", mean_error_db)
        return draft, source

    log.info("Escalating to Gemini multimodal audio listening (Error: %.2fdB)", mean_error_db)
    try:
        refined = refine_parameters_with_gemini(
            target_stem_wav, str(rendered), draft, spectral_deltas, plugin_schema, user_prompt)
        refined = sanitize_refinement(refined, plugin_schema, draft.vst_parameters)
    except Exception as e:
        log.warning("Gemini escalation failed (%s); returning the Claude draft", e)
        return draft, "claude_fallback" if source == "claude" else source

    _log_post_check(plugin, vst3_path, refined, di, out_dir, target_stem_wav, mean_error_db)
    return refined, "gemini"


def _log_post_check(
    plugin: Any, vst3_path: str, refined: NeuralDSPRefinement, di: Path, out_dir: Path,
    target_stem_wav: str, before_db: float,
) -> None:
    """Informational only: re-render Gemini's preset and log the band error. Gemini's ear wins over the numbers."""
    try:
        out = out_dir / "gemini_rendered.wav"
        render_preset_preview(vst3_path, refined.vst_parameters, str(di), str(out), plugin=plugin)
        after_db, _ = compute_band_spectral_deltas(target_stem_wav, str(out))
        log.info("band error after Gemini: %.2fdB (Claude draft: %.2fdB)", after_db, before_db)
    except Exception as e:
        log.debug("post-escalation check skipped: %s", e)
