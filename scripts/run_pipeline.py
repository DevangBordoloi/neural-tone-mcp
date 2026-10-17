"""Standalone runner: URL (or WAV) -> preset JSON -> optional DAW.

  python scripts/run_pipeline.py "https://youtu.be/..." "tight djent rhythm" "Archetype Nolly X" --start 60 --duration 25
  python scripts/run_pipeline.py riff_guitar.wav "clean ambient lead" "Archetype Plini" --stem --daw reaper
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.daw.base import DawError  # noqa: E402
from src.pipeline import extract_tone  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="URL or local audio file")
    ap.add_argument("prompt", help='the tone you want, e.g. "tight djent rhythm"')
    ap.add_argument("plugin", help='inspected plugin, e.g. "Archetype Nolly X"')
    ap.add_argument("--start", type=float, default=30, help="start second (default 30)")
    ap.add_argument("--duration", type=float, default=25, help="seconds to analyse (default 25)")
    ap.add_argument("--focus", choices=["rhythm", "lead", "full"])
    ap.add_argument("--stem", action="store_true", help="the input is already an isolated guitar (skip Demucs)")
    ap.add_argument("--llm", action="store_true", help="refine with Claude (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--daw", choices=["reaper", "ableton"], help="also build the track(s) in this DAW")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")

    res = extract_tone(a.source, a.prompt, a.plugin, a.start, a.duration, a.focus, a.stem, a.llm)
    p = res["preset"]
    print(f"preset: {res['json_path']}")
    for k, v in p.signal_chain_summary.items():
        print(f"  {k}: {v}")
    if a.daw:
        from src.mcp_server import _driver

        d = _driver(a.daw)
        try:
            print("connected:", d.connect())
            print(json.dumps(d.create_tracks_and_apply_preset(res["json_path"]), indent=2))
        except DawError as e:
            print(f"DAW step failed: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
