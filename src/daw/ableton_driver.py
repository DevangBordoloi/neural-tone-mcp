"""Ableton Live driver over AbletonMCP's TCP socket (localhost:9877).

UNTESTED against a live Ableton. The wire format follows AbletonMCP's convention: one JSON object
`{"type": <command>, "params": {...}}` per request, answered by `{"status": "success"|"error",
"result"|"message": ...}`. Command names are overridable via `commands` in case your AbletonMCP version
differs. Values are matched to device parameters by host display name (schema `host_name`).
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Any

from src.daw.base import DawController, DawError
from src.daw.reaper_driver import match_reaper_params
from src.models import PluginSchema

log = logging.getLogger(__name__)

DEFAULT_COMMANDS = {
    "session": "get_session_info",
    "create_track": "create_audio_track",
    "rename_track": "set_track_name",
    "load_device": "load_instrument_or_effect",
    "device_params": "get_device_parameters",
    "set_param": "set_device_parameter",
}


class AbletonDriver(DawController):
    name = "ableton"

    def __init__(self, host: str = "localhost", port: int = 9877, timeout: float = 20.0, commands: dict[str, str] | None = None):
        self.host, self.port, self.timeout = host, port, timeout
        self.cmd = {**DEFAULT_COMMANDS, **(commands or {})}

    def _send(self, kind: str, **params: Any) -> Any:
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
                s.sendall(json.dumps({"type": self.cmd.get(kind, kind), "params": params}).encode())
                buf = b""
                while True:
                    chunk = s.recv(65536)
                    if not chunk:
                        raise DawError("AbletonMCP closed the connection before a full reply")
                    buf += chunk
                    try:
                        msg = json.loads(buf)
                        break
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            raise DawError(f"cannot reach AbletonMCP at {self.host}:{self.port}: {e}") from e
        if msg.get("status") == "error":
            raise DawError(str(msg.get("message") or msg))
        return msg.get("result", msg)

    def connect(self) -> dict[str, Any]:
        res = self._send("session")
        return {"backend": "ableton", **(res if isinstance(res, dict) else {})}

    def create_track(self, name: str, pan: float = 0.0, folder_start: bool = False) -> int:
        res = self._send("create_track", index=-1)
        idx = int(res["index"]) if isinstance(res, dict) and "index" in res else int(res)
        self._send("rename_track", track_index=idx, name=name)
        return idx

    def add_plugin(self, track: int, plugin_name: str) -> int:
        res = self._send("load_device", track_index=track, uri=plugin_name)
        return int(res.get("device_index", 0)) if isinstance(res, dict) else 0

    def set_parameters(self, track: int, plugin: int, values: dict[str, float], schema: PluginSchema) -> dict[str, Any]:
        info = self._send("device_params", track_index=track, device_index=plugin)
        params = info["parameters"] if isinstance(info, dict) else info
        idx, missing = match_reaper_params(schema, [p["name"] for p in params], list(values))
        failed: list[dict[str, Any]] = [{"param": n, "error": "no matching device parameter"} for n in missing]
        applied = 0
        for n, i in idx.items():
            lo, hi = float(params[i].get("min", 0.0)), float(params[i].get("max", 1.0))
            try:
                self._send("set_param", track_index=track, device_index=plugin, parameter_index=i, value=lo + values[n] * (hi - lo))
                applied += 1
            except DawError as e:
                failed.append({"param": n, "error": str(e)})
        return {"plugin": schema.plugin_name, "applied": applied, "failed": failed}
