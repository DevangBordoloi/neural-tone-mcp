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
`apply_tone_json_to_daw`, `match_url_tone_end_to_end`, `get_plugin_schema_summary`.
Put `rhythm` or `lead` in the prompt to choose the Side (double-tracked rhythm) or Mid (lead) channel.

## REAPER

The driver talks to the Lua bridge from the `reaper-mcp` project (`reaper_mcp_bridge.lua` running in REAPER;
Actions > Show action list > Load ReaScript). `reapy` is the fallback and is untested. Parameters are matched to
REAPER by the plugin's display name (`host_name` in the schema), never by pedalboard's index. Tested live:
40/40 parameters applied and read back identically.

## What is and isn't solid

* Tested: schema inspection on all 8 plugins, `render_preview` through the real Nolly X, feature extraction on
  synthetic audio, mapper output validated against every Archetype schema, REAPER apply + read-back, and the real
  URL -> Demucs -> features -> JSON chain on a short clip. `python -m pytest` runs 35 offline tests.
* Heuristic: feature -> knob rules are rules of thumb. The 9-band EQ centres (octaves from 63 Hz) are assumed.
  Treat a result as a starting point and adjust by ear.
* Untested: Ableton driver (AbletonMCP wire format assumed), `reapy` fallback, and LLM refinement (`--llm`).
* Demucs runs on CPU (torch CPU build): expect about 1-2 minutes for a 25 s clip.

## Windows + Norton note

Norton's Web Shield re-signs HTTPS and mangles some PyPI replies. Downloads use the Windows certificate store
(`src/net.py`, verification stays on). If `pip install` fails with `InvalidChunkLength`, use
`python tools/pip_identity.py --python <venv python> install ...`, or install that one package from its wheel.

## YouTube downloads: 403 and certificate errors

* **`HTTP Error 403: Forbidden` on the media download.** YouTube now needs a JavaScript runtime for yt-dlp to solve
  its stream challenge. Install Node.js and add `--js-runtimes node` to your command, or put that single line in
  `%APPDATA%\yt-dlp\config` (Linux/macOS: `~/.config/yt-dlp/config`). The config file only affects the `yt-dlp`
  command line, not yt-dlp used as a Python library, so for the MCP tools the most reliable route is to pass a local
  WAV instead of a URL: `yt-dlp --js-runtimes node -f bestaudio <url>`, convert with the bundled ffmpeg, then give
  the WAV path to `extract_tone_to_json`.
* **`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`.** Python is not using the Windows
  certificate store (typical behind antivirus HTTPS scanning). Install `pip-system-certs`, using
  `pip install --use-feature=truststore -U pip-system-certs certifi yt-dlp`, which keeps verification on. Do not
  use `--no-check-certificate`.

## Examples

`examples/` holds presets the pipeline produced (normalized 0-1 values keyed by exact parameter name):
`lead_guitar_solo_singing_lead.json` (Archetype John Mayer X, a lead solo, extracted from a live-concert clip) and two
Nolly X rhythm presets. Apply one with `python scripts/run_pipeline.py` or the `apply_tone_json_to_daw` MCP tool.
These are rule-of-thumb starting points, not clones of the recordings.
