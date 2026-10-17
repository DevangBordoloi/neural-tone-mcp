"""Stage 2 of the hybrid engine: Gemini listens to the reference and to Claude's render, then corrects the draft.

Gemini sees the actual audio, so it can judge what static band-energy numbers miss (attack tightness, gain
structure, mic placement, modulation and delay).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from pydantic import BaseModel, Field

from src.llm.claude_refiner import NeuralDSPRefinement, RefinerError, compact_schema

log = logging.getLogger(__name__)

# The spec names gemini-2.5-pro, but Google has retired it for new API keys (404 "no longer available to new users").
# "-latest" follows Google's current Pro model; set GEMINI_MODEL to pin one.
DEFAULT_GEMINI_MODEL = "gemini-pro-latest"
UPLOAD_TIMEOUT_SEC = 120


class _ParamValue(BaseModel):
    name: str = Field(description="Exact VST3 parameter name from the plugin schema.")
    value: float = Field(description="Normalized value between 0.0 and 1.0.")


class _GeminiRefinement(BaseModel):
    """NeuralDSPRefinement as Gemini can accept it.

    The Gemini Developer API rejects `dict[str, float]` in a response schema ("additionalProperties is only
    supported in ... Enterprise Agent Platform mode"), so parameters travel as a list of name/value pairs and
    are converted back to the unified NeuralDSPRefinement below.
    """

    reasoning: str
    double_track_rhythm: bool
    vst_parameters: list[_ParamValue]

    def to_refinement(self) -> NeuralDSPRefinement:
        return NeuralDSPRefinement(
            reasoning=self.reasoning,
            double_track_rhythm=self.double_track_rhythm,
            vst_parameters={p.name: p.value for p in self.vst_parameters},
        )


SYSTEM_INSTRUCTION = (
    "You are an expert guitar tone engineer matching a Neural DSP plugin to a reference recording by ear. "
    "Only use parameter names present in the provided plugin schema, with normalized values from 0.0 to 1.0. "
    "For enum and boolean parameters, option i of n sits at i/(n-1)."
)


def _make_client() -> Any:
    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RefinerError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
    return genai.Client(api_key=key)


def _wait_until_ready(client: Any, uploaded: Any) -> Any:
    """Uploaded files can sit in PROCESSING for a moment; poll until Gemini can read them."""
    deadline = time.monotonic() + UPLOAD_TIMEOUT_SEC
    while getattr(getattr(uploaded, "state", None), "name", getattr(uploaded, "state", None)) == "PROCESSING":
        if time.monotonic() > deadline:
            raise RefinerError(f"Gemini file {uploaded.name} was still processing after {UPLOAD_TIMEOUT_SEC}s")
        time.sleep(1.0)
        uploaded = client.files.get(name=uploaded.name)
    return uploaded


def refine_parameters_with_gemini(
    target_stem_wav: str,
    claude_rendered_wav: str,
    claude_draft: NeuralDSPRefinement,
    spectral_delta: dict[str, float],
    plugin_schema: dict,
    user_prompt: str,
) -> NeuralDSPRefinement:
    """Upload both WAVs, ask Gemini to reconcile them, and return a validated NeuralDSPRefinement.

    Uploaded files are always deleted from Google's storage afterwards.
    """
    from google.genai import types

    client = _make_client()
    uploaded_names: list[str] = []
    try:
        reference = client.files.upload(file=target_stem_wav)
        uploaded_names.append(reference.name)
        render = client.files.upload(file=claude_rendered_wav)
        uploaded_names.append(render.name)
        reference, render = _wait_until_ready(client, reference), _wait_until_ready(client, render)

        prompt = (
            f"The user wants: {user_prompt}\n\n"
            "Audio 1 is the REFERENCE guitar tone to match. Audio 2 is Claude's attempt, rendered through the "
            "plugin with the draft settings below.\n"
            "Listen to both. Static DSP numbers already tell us the band-energy error (dB, rendered minus "
            "reference, negative = Claude's render is too quiet in that band):\n"
            f"{json.dumps(spectral_delta)}\n\n"
            "Judge what those numbers miss: pick attack tightness, gain structure (pedal boost vs. tube sag), "
            "cabinet mic placement (cone vs. edge fizz), and modulation/delay presence. Then adjust the "
            "vst_parameters so Audio 2 moves toward Audio 1. Return every parameter you change; unchanged "
            "parameters keep Claude's value.\n\n"
            f"Claude's draft reasoning: {claude_draft.reasoning}\n"
            f"Claude's draft parameters: {json.dumps(claude_draft.vst_parameters, separators=(',', ':'))}\n"
            f"Double-tracked rhythm (Claude's call): {claude_draft.double_track_rhythm}\n\n"
            f"Valid plugin schema: {json.dumps(compact_schema(plugin_schema), default=str, separators=(',', ':'))}"
        )
        try:
            response = client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
                contents=["Audio 1 (Reference):", reference, "Audio 2 (Claude's render):", render, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=_GeminiRefinement,
                ),
            )
        except Exception as e:
            raise RefinerError(f"Gemini request failed: {e}") from e

        parsed = getattr(response, "parsed", None)
        try:
            if isinstance(parsed, _GeminiRefinement):
                return parsed.to_refinement()
            return _GeminiRefinement.model_validate_json(response.text).to_refinement()
        except (ValueError, TypeError) as e:
            raise RefinerError(f"Gemini's reply did not validate: {e}") from e
    finally:
        for name in uploaded_names:
            try:
                client.files.delete(name=name)
            except Exception as e:  # cleanup must never mask the real result or error
                log.warning("could not delete Gemini file %s: %s", name, e)
