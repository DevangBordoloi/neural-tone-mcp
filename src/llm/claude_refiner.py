"""Stage 1 of the hybrid engine: Claude maps measured DSP numbers + the user's intent to VST3 parameters.

Also holds the unified `NeuralDSPRefinement` model that both refiners (Claude, Gemini) return, and the helpers
they share: a compact, token-cheap view of the plugin schema and the forced-tool-call plumbing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from typing import Any

from anthropic import Anthropic
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

# Same default as tone_mapper.refine_with_llm; override with NEURAL_TONE_MODEL.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
TOOL_NAME = "commit_neural_dsp_preset"
CLI_TIMEOUT_SEC = 240


class RefinerError(RuntimeError):
    """An LLM refiner could not produce a usable preset (missing key, API failure, malformed reply)."""


class NeuralDSPRefinement(BaseModel):
    reasoning: str = Field(description="Brief DSP reasoning linking spectral/crest metrics to amp, overdrive, and EQ settings.")
    double_track_rhythm: bool = Field(description="True if stereo side-channel energy indicates hard-panned double tracking.")
    vst_parameters: dict[str, float] = Field(description="Exact VST3 parameter names mapped to normalized floats between 0.0 and 1.0.")

    @field_validator("vst_parameters", mode="before")
    @classmethod
    def _clamp_to_unit_range(cls, v: Any) -> Any:
        """Clamp to 0-1 and drop values that are not finite numbers, instead of rejecting the whole reply."""
        if not isinstance(v, dict):
            return v
        out: dict[str, float] = {}
        for key, raw in v.items():
            try:
                x = float(raw)
            except (TypeError, ValueError):
                log.warning("dropping non-numeric value for %r: %r", key, raw)
                continue
            if not math.isfinite(x):
                log.warning("dropping non-finite value for %r", key)
                continue
            out[str(key)] = min(1.0, max(0.0, x))
        return out


def compact_schema(plugin_schema: dict) -> list[dict[str, Any]]:
    """Token-cheap view of a PluginSchema dict: name, kind, default, and range or option list per parameter."""
    out: list[dict[str, Any]] = []
    for p in plugin_schema.get("parameters", []):
        entry: dict[str, Any] = {"name": p["name"], "kind": p.get("kind"), "default": p.get("default_normalized")}
        if p.get("kind") == "numeric":
            entry["display_range"] = [p.get("min_value"), p.get("max_value"), p.get("units")]
            if p.get("linear") is False:
                entry["nonlinear"] = True  # normalized position does not map linearly to the displayed value
        elif p.get("valid_values"):
            entry["options"] = p["valid_values"]
        out.append(entry)
    return out


SYSTEM_PROMPT = (
    "You are a Neural DSP tone-matching engine. Map the extracted librosa DSP metrics "
    "and user intent into exact normalized (0.0 to 1.0) VST3 parameters. "
    "Only use parameter keys present in the provided plugin_schema. "
    "For enum and boolean parameters the options are listed in order and option i of n sits at i/(n-1); "
    "numeric parameters list their displayed range. A baseline preset may be supplied: keep its amp, cab and "
    "section on/off settings unless the measurements justify changing them, and return every parameter you "
    "want changed (omitted parameters keep their baseline value). "
    "eq_band_energy_ratios are dB relative to the 60-10000 Hz total, so only their shape matters, not their level."
)


def _find_claude_cli() -> str | None:
    """The `claude` executable: CLAUDE_CLI_PATH, else PATH (MCP hosts sometimes launch us with a trimmed PATH)."""
    explicit = os.environ.get("CLAUDE_CLI_PATH")
    if explicit and os.path.exists(explicit):
        return explicit
    return shutil.which("claude")


def _backend() -> str:
    """"api" (Anthropic SDK + ANTHROPIC_API_KEY) or "cli" (the logged-in `claude` CLI).

    NEURAL_TONE_CLAUDE_BACKEND=api|cli forces one; by default the API is used when a key is set, else the CLI.
    """
    choice = (os.environ.get("NEURAL_TONE_CLAUDE_BACKEND") or "auto").lower()
    if choice in ("api", "cli"):
        return choice
    return "api" if os.environ.get("ANTHROPIC_API_KEY") else "cli"


def _claude_via_cli(user_message: str) -> dict:
    """One headless `claude -p` call with the refinement JSON schema; returns the structured output dict.

    Tools and other MCP servers are disabled (so this cannot recurse into the neural-tone server) and the prompt
    goes in on stdin because the schema listing is too long for a command line.
    """
    exe = _find_claude_cli()
    if exe is None:
        raise RefinerError("the `claude` CLI was not found: put it on PATH or set CLAUDE_CLI_PATH "
                           "(or set ANTHROPIC_API_KEY and NEURAL_TONE_CLAUDE_BACKEND=api)")
    cmd = [exe, "-p", "--output-format", "json", "--json-schema", json.dumps(NeuralDSPRefinement.model_json_schema()),
           "--tools", "", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
           "--system-prompt", SYSTEM_PROMPT]
    if os.environ.get("NEURAL_TONE_MODEL"):
        cmd += ["--model", os.environ["NEURAL_TONE_MODEL"]]
    try:
        proc = subprocess.run(
            cmd, input=user_message, capture_output=True, text=True, encoding="utf-8",
            timeout=CLI_TIMEOUT_SEC, cwd=tempfile.gettempdir())
    except subprocess.TimeoutExpired as e:
        raise RefinerError(f"claude CLI timed out after {CLI_TIMEOUT_SEC}s") from e
    except OSError as e:
        raise RefinerError(f"could not run the claude CLI: {e}") from e
    try:
        reply = json.loads(proc.stdout)
    except ValueError as e:
        raise RefinerError(f"claude CLI exited {proc.returncode} without JSON: {(proc.stderr or proc.stdout)[:300]}") from e
    if proc.returncode != 0 or reply.get("is_error"):
        raise RefinerError(f"claude CLI failed: {str(reply.get('result') or proc.stderr)[:300]}")
    structured = reply.get("structured_output")
    if structured is None:  # older CLIs put the JSON text in `result`
        try:
            structured = json.loads(reply.get("result") or "")
        except ValueError as e:
            raise RefinerError("claude CLI returned no structured output") from e
    return structured


def _claude_via_api(user_message: str) -> dict:
    client = Anthropic()
    try:
        response = client.messages.create(
            model=os.environ.get("NEURAL_TONE_MODEL") or DEFAULT_CLAUDE_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            tools=[{
                "name": TOOL_NAME,
                "description": "Output the refined Neural DSP parameter dictionary.",
                "input_schema": NeuralDSPRefinement.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )
    except Exception as e:
        raise RefinerError(f"Claude request failed: {e}") from e
    # content[0] is not guaranteed to be the tool call, so look for it by type and name
    block = next((b for b in response.content if getattr(b, "type", None) == "tool_use" and b.name == TOOL_NAME), None)
    if block is None:
        raise RefinerError("Claude returned no commit_neural_dsp_preset tool call")
    return block.input


def refine_parameters_with_claude(
    dsp_features: dict,
    plugin_schema: dict,
    user_prompt: str,
    spectral_delta: dict | None = None,
    baseline_parameters: dict[str, float] | None = None,
) -> NeuralDSPRefinement:
    """Ask Claude for a parameter dictionary, via the Anthropic API or the `claude` CLI (see `_backend`).

    `spectral_delta` is the dB error per band when re-refining.
    """
    backend = _backend()
    if backend == "api" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RefinerError("ANTHROPIC_API_KEY is not set")
    payload = {
        "user_prompt": user_prompt,
        "dsp_features": dsp_features,
        "spectral_error_delta_db_rendered_minus_reference": spectral_delta,
        "baseline_parameters": baseline_parameters,
        "plugin": plugin_schema.get("plugin_name"),
        "valid_plugin_schema": compact_schema(plugin_schema),
    }
    message = json.dumps(payload, default=str, separators=(",", ":"))
    tool_args = _claude_via_cli(message) if backend == "cli" else _claude_via_api(message)
    try:
        return NeuralDSPRefinement.model_validate(tool_args)
    except ValueError as e:
        raise RefinerError(f"Claude's reply did not validate: {e}") from e
