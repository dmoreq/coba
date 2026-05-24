"""Preset management for simulation configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Preset:
    """Named run preset."""

    preset_id: str
    title: str
    payload: dict[str, Any]


class PresetManager:
    """File-backed preset collection."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (Path.home() / ".coba_flet_presets.json")

    def save_presets(self, presets: list[Preset]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"preset_id": preset.preset_id, "title": preset.title, "payload": preset.payload}
            for preset in presets
        ]
        self.path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def load_presets(self) -> list[Preset]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [
            Preset(
                preset_id=str(item["preset_id"]),
                title=str(item["title"]),
                payload=dict(item["payload"]),
            )
            for item in payload
        ]
