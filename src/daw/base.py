"""Abstract DAW controller: the pipeline only talks to this interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.dsp.vst_inspector import load_schema
from src.models import NeuralDSPPresetJSON, PluginSchema

log = logging.getLogger(__name__)


class DawError(RuntimeError):
    pass


class DawController(ABC):
    name = "daw"

    @abstractmethod
    def connect(self) -> dict[str, Any]:
        """Verify the DAW is reachable. Returns version info; raises DawError if not."""

    @abstractmethod
    def create_track(self, name: str, pan: float = 0.0, folder_start: bool = False) -> Any:
        """Create an audio track (pan -100..100). Returns a track handle."""

    @abstractmethod
    def add_plugin(self, track: Any, plugin_name: str) -> Any:
        """Insert the plugin on the track. Returns a plugin handle."""

    @abstractmethod
    def set_parameters(self, track: Any, plugin: Any, values: dict[str, float], schema: PluginSchema) -> dict[str, Any]:
        """Apply {pedalboard parameter name: normalized 0-1}. Returns {"applied": n, "failed": [...]}."""

    def copy_plugin_state(self, src_track: Any, dst_track: Any) -> None:
        """Duplicate the plugin chain (used for the R side of a double-tracked pair)."""
        raise NotImplementedError

    def end_folder(self, last_track: Any) -> None:
        """Close a folder opened with create_track(folder_start=True)."""

    # ------------------------------------------------------------------
    def create_tracks_and_apply_preset(self, preset_json_path: str | Path) -> dict[str, Any]:
        preset = NeuralDSPPresetJSON.model_validate_json(Path(preset_json_path).read_text(encoding="utf-8"))
        schema = load_schema(preset.target_plugin)
        preset.validate_against(schema)
        cfg = preset.track_config
        report: dict[str, Any] = {"preset": preset.preset_name, "daw": self.name, "tracks": []}

        if cfg.double_tracked:
            pans = cfg.panning[:2] if len(cfg.panning) >= 2 else [-100.0, 100.0]
            self.create_track(cfg.track_name, folder_start=True)
            left = self.create_track(f"{cfg.track_name} L", pan=pans[0])
            plug = self.add_plugin(left, schema.plugin_name)
            res = self.set_parameters(left, plug, preset.vst_parameters, schema)
            right = self.create_track(f"{cfg.track_name} R", pan=pans[1])
            self.copy_plugin_state(left, right)
            self.end_folder(right)
            report["tracks"] = [{"track": f"{cfg.track_name} L", **res}, {"track": f"{cfg.track_name} R", "copied_from": "L"}]
        else:
            track = self.create_track(cfg.track_name, pan=cfg.panning[0] if cfg.panning else 0.0)
            plug = self.add_plugin(track, schema.plugin_name)
            report["tracks"] = [{"track": cfg.track_name, **self.set_parameters(track, plug, preset.vst_parameters, schema)}]
        return report
