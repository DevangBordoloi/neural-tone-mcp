"""CLI: scan a Neural DSP .vst3 and save its parameter schema.

  python scripts/inspect_plugin.py "Archetype Nolly X"
  python scripts/inspect_plugin.py --all
  python scripts/inspect_plugin.py "Nolly X" --find gain      # also print matching parameters
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dsp.vst_inspector import (  # noqa: E402
    InspectorError, dump_plugin_schema, find_vst3, vst3_search_dirs,
)


def _print_params(schema, words: list[str]) -> None:
    for p in schema.search(*words):
        rng = f"{p.valid_values}" if p.valid_values else f"{p.min_value}..{p.max_value} {p.units or ''}".strip()
        print(f"  #{p.index:<3} {p.name:<40} {p.kind:<8} default={p.default_display!r:<12} {rng[:70]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plugin", nargs="*", help="path to a .vst3, or part of its name")
    ap.add_argument("--all", action="store_true", help="inspect every Neural DSP plugin found")
    ap.add_argument("--out-dir", type=Path, help="schema folder (default config/plugin_schemas)")
    ap.add_argument("--find", nargs="+", metavar="WORD", help="print parameters whose name contains all WORDs")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")

    targets = list(a.plugin)
    if a.all:
        for d in vst3_search_dirs():
            targets += [str(p) for p in d.rglob("*.vst3") if "neural dsp" in str(p).lower() or "archetype" in p.stem.lower()]
    if not targets:
        ap.error("give a plugin or --all")

    failed = 0
    for t in dict.fromkeys(targets):
        try:
            print(f"Inspecting {find_vst3(t).name} ...", flush=True)
            schema, out = dump_plugin_schema(t, a.out_dir)
        except InspectorError as e:
            failed += 1
            print(f"  FAILED: {e}", file=sys.stderr)
            continue
        kinds = {k: sum(p.kind == k for p in schema.parameters) for k in ("numeric", "enum", "boolean")}
        print(f"  {schema.parameter_count} parameters {kinds} -> {out}")
        if a.find:
            _print_params(schema, a.find)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
