"""REAPER driver.

Primary path: the file-based Lua bridge from the reaper-mcp project (reaper_mcp_bridge.lua running in
REAPER, polling ~/.reaper-mcp). It needs no REAPER Python setup. Fallback: `reapy` (python-reapy), which
needs REAPER's Python bridge enabled (`python -c "import reapy; reapy.configure_reaper()"`) and is
UNTESTED here.

Parameters are matched to REAPER's parameter list by the plugin's host display name (schema `host_name`),
never by pedalboard's enumeration index.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from src.daw.base import DawController, DawError
from src.models import PluginSchema

log = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_reaper_params(schema: PluginSchema, reaper_names: list[str], wanted: list[str]) -> tuple[dict[str, int], list[str]]:
    """Map schema parameter names -> DAW param indices by host display name. Returns (found, missing)."""
    by_norm: dict[str, list[int]] = {}
    for i, n in enumerate(reaper_names):
        by_norm.setdefault(_norm(n), []).append(i)
    spec = schema.by_name()
    found: dict[str, int] = {}
    missing: list[str] = []
    for name in wanted:
        p = spec.get(name)
        cands = by_norm.get(_norm(p.host_name or name), []) if p else []
        if not cands:
            cands = by_norm.get(_norm(name), [])
        if not cands:
            missing.append(name)
        else:  # duplicate labels: prefer the one nearest pedalboard's order
            found[name] = min(cands, key=lambda i: abs(i - (p.index if p else i)))
    return found, missing


class BridgeClient:
    """Synchronous client for the reaper-mcp request.json / response.json protocol."""

    def __init__(self, bridge_dir: str | Path | None = None, timeout: float = 30.0):
        env = os.environ.get("REAPER_MCP_BRIDGE_DIR")
        self.dir = Path(bridge_dir or env or Path.home() / ".reaper-mcp").expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(os.environ.get("REAPER_MCP_TIMEOUT", timeout))

    def call(self, cmd: str, timeout: float | None = None, **args: Any) -> Any:
        args = {k: v for k, v in args.items() if v is not None}
        req, resp = self.dir / "request.json", self.dir / "response.json"
        rid = uuid.uuid4().hex
        resp.unlink(missing_ok=True)
        req.unlink(missing_ok=True)
        tmp = self.dir / f"request.{rid}.tmp"
        tmp.write_text(json.dumps({"id": rid, "cmd": cmd, "args": args}), encoding="utf-8")
        os.replace(tmp, req)
        wait = timeout or self.timeout
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                data = json.loads(resp.read_text(encoding="utf-8"))
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                data = None
            if data is not None:
                resp.unlink(missing_ok=True)
                if data.get("id") == rid:
                    if data.get("ok"):
                        return data.get("result")
                    raise DawError(str(data.get("error") or "unknown REAPER error"))
            time.sleep(0.02)
        req.unlink(missing_ok=True)
        raise DawError(
            f"REAPER didn't answer within {wait:.0f}s. Open REAPER and run reaper_mcp_bridge.lua "
            f"(Actions > Show action list > Load ReaScript). It watches {self.dir}."
        )


class ReaperDriver(DawController):
    name = "reaper"

    def __init__(self, backend: str = "auto", bridge_dir: str | Path | None = None):
        self.backend_pref = backend  # auto | bridge | reapy
        self.bridge = BridgeClient(bridge_dir)
        self.backend: str | None = None
        self._reapy: Any = None

    # -- connection --------------------------------------------------------------------------------
    def connect(self) -> dict[str, Any]:
        errors = []
        if self.backend_pref in ("auto", "bridge"):
            try:
                info = self.bridge.call("ping", timeout=4)
                self.backend = "bridge"
                return {"backend": "bridge", **info}
            except DawError as e:
                errors.append(f"bridge: {e}")
        if self.backend_pref in ("auto", "reapy"):
            try:
                import reapy

                self._reapy = reapy
                reapy.Project()  # raises if REAPER's reapy bridge is not running
                self.backend = "reapy"
                return {"backend": "reapy"}
            except Exception as e:
                errors.append(f"reapy: {e}")
        raise DawError("could not reach REAPER. " + " | ".join(errors))

    def _ensure(self) -> str:
        if not self.backend:
            self.connect()
        return self.backend or ""

    # -- operations ----------------------------------------------------------------------------------
    def create_track(self, name: str, pan: float = 0.0, folder_start: bool = False) -> Any:
        if self._ensure() == "bridge":
            res = self.bridge.call("create_track", name=name, pan=pan if pan else None, folder_depth=1 if folder_start else None)
            return res["guid"]
        project = self._reapy.Project()
        track = project.add_track(index=len(project.tracks), name=name)
        track.pan = pan / 100.0
        return track

    def add_plugin(self, track: Any, plugin_name: str) -> Any:
        if self._ensure() == "bridge":
            return self.bridge.call("add_fx", track=track, plugin=plugin_name, timeout=120)["slot"]
        return track.add_fx(plugin_name)

    def set_parameters(self, track: Any, plugin: Any, values: dict[str, float], schema: PluginSchema) -> dict[str, Any]:
        if self._ensure() == "bridge":
            info = self.bridge.call("get_params", track=track, fx=plugin, names_only=True, limit=5000)
            idx, missing = match_reaper_params(schema, info["params"], list(values))
            items = [{"index": i, "value": values[n], "mode": "normalized"} for n, i in idx.items()]
            failed: list[dict[str, Any]] = [{"param": n, "error": "no matching parameter in REAPER"} for n in missing]
            applied = 0
            if items:
                res = self.bridge.call("set_params", track=track, fx=plugin, params=items, timeout=120)
                for n, r in zip(idx, res["results"]):
                    if r.get("ok"):
                        applied += 1
                    else:
                        failed.append({"param": n, "error": r.get("error")})
            return {"plugin": info["fx"], "applied": applied, "failed": failed}
        names = [p.name for p in plugin.params]
        idx, missing = match_reaper_params(schema, names, list(values))
        for n, i in idx.items():
            plugin.params[i] = values[n]
        return {"plugin": plugin.name, "applied": len(idx), "failed": [{"param": n, "error": "not found"} for n in missing]}

    def copy_plugin_state(self, src_track: Any, dst_track: Any) -> None:
        if self._ensure() != "bridge":
            raise DawError("exact chain copy needs the bridge backend")
        chain = self.bridge.call("get_chain", track=src_track, snapshot=False)
        self.bridge.call("set_chain", track=dst_track, chain=chain["chain"], mode="replace", timeout=120)

    def end_folder(self, last_track: Any) -> None:
        if self._ensure() == "bridge":
            self.bridge.call("adjust_folder_depth", track=last_track, delta=-1)
