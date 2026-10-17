# neural-tone-mcp

URL (YouTube/SoundCloud) or WAV -> guitar stem -> tone measurements -> Neural DSP preset JSON -> tracks and
plugin settings in REAPER, exposed as MCP tools for Claude.

```
downloader (yt-dlp+ffmpeg, 48k WAV)  ->  separator (Demucs htdemucs_6s, Mid/Side)  ->  extractor (librosa/scipy)
      -> tone_mapper (features + prompt + plugin schema -> NeuralDSPPresetJSON)  ->  daw driver (REAPER / Ableton)
vst_inspector (pedalboard) builds the per-plugin schema the mapper and drivers rely on.
```

## Use

```bash
# once per plugin (≈20 s each): writes config/plugin_schemas/<plugin>.json  (all 8 are already generated)
python scripts/inspect_plugin.py --all

# URL -> preset JSON (-> REAPER with --daw reaper)
python scripts/run_pipeline.py "https://youtu.be/..." "tight djent rhythm" "Archetype Nolly X" --start 60 --duration 25 --daw reaper
python scripts/run_pipeline.py riff.wav "clean ambient lead" "Archetype Plini" --stem   # already-isolated guitar
```

MCP server (registered with Claude Code as `neural-tone`): `inspect_neural_dsp_plugin`, `extract_tone_to_json`,
`apply_tone_json_to_daw`, `match_url_tone_end_to_end`, `match_url_tone_hybrid`, `get_plugin_schema_summary`.
Put `rhythm` or `lead` in the prompt to choose the Side (double-tracked rhythm) or Mid (lead) channel.

## Guitar stem separation

`src/ingest/separator.py` isolates the guitar with Demucs. Two strategies, chosen by `NEURAL_TONE_SEPARATION` (or the
`strategy=` / `separation=` argument):

| Strategy | What it does | Cost |
|---|---|---|
| `cascade` (default) | `htdemucs` (4 stems) removes vocals, drums and bass; `htdemucs_6s` then separates the guitar from the remaining "other" stem | two model passes |
| `single` | one `htdemucs_6s` pass on the full mix | one pass |

The cascade follows the two-stage workflow in the stem-separation survey this project is based on: once the drums are
gone, the guitar model no longer has to decide whether a sharp transient is a snare or a pick attack. Measured on one
25 s slice of a live recording (CPU): the cascade took 63 s against 31 s, and the drum pattern's correlation with the
guitar stem's onsets fell from 0.17 to 0.09. The features the tone mapper uses were essentially unchanged (attack rise
21.1 ms in both, crest factor 14.7 vs 14.3 dB, centroid 2312 vs 2243 Hz), so the cleaner stem did not cost pick attack.
That is one clip and a proxy, not an SDR score; there is no ground truth for real recordings. Use `single` if speed matters.

Full write-up (design, settings, measurements, limitations): [docs/STEM_SEPARATION.md](docs/STEM_SEPARATION.md).

Other knobs: `NEURAL_TONE_SHIFTS` (test-time augmentation, each shift is another full pass) and `NEURAL_TONE_OVERLAP`
(chunk overlap, default 0.25). Input quality matters more than any of them: separation is clearly better from lossless
audio than from MP3/Opus, and YouTube audio is lossy, so a studio WAV/FLAC of the same section will separate better.
After each separation a sanity check warns if the guitar stem is nearly silent, far below the mix, or mostly gaps,
which usually means there is no guitar in that section.

Not implemented: a BS-RoFormer first stage (the survey's top-SDR model) and the community `htdemucs-6s-guitar-ft`
checkpoint. The common BS-RoFormer checkpoints split vocals from "instrumental" only, so drums would stay in the stem
that goes to the guitar model, and both need extra dependencies and downloads that were not tested here.

## Hybrid refinement (Claude + Gemini)

`match_url_tone_hybrid(url, user_prompt, target_plugin_name, force_gemini=False, daw="reaper")` adds a closed loop
on top of the heuristic mapper:

```
stem -> features -> heuristic baseline -> Claude draft (numeric) -> headless render (pedalboard) -> band error
        error <= 2 dB  -> keep Claude's draft
        error  > 2 dB  -> Gemini (Pro) listens to the reference and the render, corrects the draft
```

| File | Role |
|---|---|
| `src/llm/claude_refiner.py` | `NeuralDSPRefinement` model and the forced-tool-call Claude refiner |
| `src/llm/gemini_refiner.py` | uploads both WAVs, asks Gemini for a structured correction, always deletes the uploads |
| `src/dsp/closed_loop.py` | `render_preset_preview`, `compute_band_spectral_deltas`, and the default DI generator |
| `src/llm/hybrid_arbiter.py` | `orchestrate_tone_refinement`: runs the flow and decides when to escalate |

**Claude via the CLI, no API key:** with `NEURAL_TONE_CLAUDE_BACKEND=cli` (also the default when no `ANTHROPIC_API_KEY` is set) the
Claude stage runs `claude -p` with a JSON schema, tools and other MCP servers disabled, using your logged-in Claude Code. Each call
takes about 15-20 s (mostly CLI start-up). Set `CLAUDE_CLI_PATH` if `claude` is not on the server's PATH.

**Gemini model:** the spec's `gemini-2.5-pro` is closed to new API keys, so the default is `gemini-pro-latest`. Pro models need a
billing-enabled key (free-tier keys get `429 ... limit: 0`); on a free key set `GEMINI_MODEL=gemini-flash-latest`, which worked in
testing. If the escalation fails for any reason the Claude draft is returned with `refiner: claude_fallback`.

Setup: copy `.env.example` to `.env` in the project root and fill in `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` (or
`GOOGLE_API_KEY`) for the escalation. `.env` is git-ignored and is read when the server or CLI starts; variables already
set in the environment win over it. Restart the MCP server after changing keys. `NEURAL_TONE_MODEL` overrides the Claude model (default `claude-sonnet-5-5`).

The result's `refiner` field says which stage produced the tone:

| `refiner` | Meaning |
|---|---|
| `claude` | Claude's draft was accepted (or the render check could not run, e.g. the `.vst3` was not found) |
| `gemini` | Gemini's corrected preset |
| `claude_fallback` | the error was too high but Gemini failed (no key, quota, bad reply); Claude's draft is returned |
| `heuristic` | Claude was unavailable, so the offline baseline was used |

Every parameter a model returns is clamped to 0-1, checked against the plugin schema (unknown names are dropped) and
snapped to real steps. Parameters a model leaves out keep their baseline value, so amp-section switches survive.

**The band error is only as good as the DI.** The render plays `assets/default_di_riff.wav` (generated on first use):
a synthetic low-passed sawtooth palm mute on Drop D. That is not guitar-like across the spectrum, so against a real
recording it alone sits about 10 dB away from the reference, and the amp render lands around 16 dB. With that
default the 2 dB gate will nearly always escalate to Gemini. For the gate to mean something, pass a clean,
full-range guitar DI (`di_wav_path` of `orchestrate_tone_refinement`) that resembles the part being matched.
The error compares band balance, not notes.

Gemini's structured-output schema sends parameters as a list of `{name, value}` pairs: the Gemini Developer API
rejects `dict[str, float]` (`additionalProperties`). The reply is converted back to `NeuralDSPRefinement`.

## REAPER

The driver talks to the Lua bridge from the `reaper-mcp` project (`reaper_mcp_bridge.lua` running in REAPER;
Actions > Show action list > Load ReaScript). `reapy` is the fallback and is untested. Parameters are matched to
REAPER by the plugin's display name (`host_name` in the schema), never by pedalboard's index. Tested live:
40/40 parameters applied and read back identically.

## What is and isn't solid

* Tested: schema inspection on all 8 plugins, `render_preview` through the real Nolly X, feature extraction on
  synthetic audio, mapper output validated against every Archetype schema, REAPER apply + read-back, and the real
  URL -> Demucs -> features -> JSON chain on a short clip. `python -m pytest` runs 70 offline tests.
* Heuristic: feature -> knob rules are rules of thumb. The 9-band EQ centres (octaves from 63 Hz) are assumed.
  Treat a result as a starting point and adjust by ear.
* Untested: Ableton driver (AbletonMCP wire format assumed), `reapy` fallback, and the live Claude/Gemini calls
  (`--llm` and the hybrid engine are exercised with mocked APIs only; the render and band measurement are tested
  against the real plugin).
* Demucs runs on CPU (torch CPU build): about 30 s for a 25 s clip with `single`, about 1 minute with the default `cascade`.


## Examples

`examples/` holds presets the pipeline produced (normalized 0-1 values keyed by exact parameter name):
`lead_guitar_solo_singing_lead.json` (Archetype John Mayer X, a lead solo, extracted from a live-concert clip) and two
Nolly X rhythm presets. Apply one with `python scripts/run_pipeline.py` or the `apply_tone_json_to_daw` MCP tool.
These are rule-of-thumb starting points, not clones of the recordings.
