"""Local preference persistence for redesign UI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UserPreferences:
    """Persisted local UI settings."""

    world_id: str = "rural_clinic"
    policy_id: str = "random"
    speed: str = "1x"


class PreferencesStore:
    """File-backed preference storage."""

    def __init__(self, file_path: Path | None = None) -> None:
        self._path = file_path or (Path.home() / ".coba_flet_preferences.json")

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> UserPreferences:
        if not self._path.exists():
            return UserPreferences()
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return UserPreferences(
            world_id=str(payload.get("world_id", "rural_clinic")),
            policy_id=str(payload.get("policy_id", "random")),
            speed=str(payload.get("speed", "1x")),
        )

    def save(self, prefs: UserPreferences) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = asdict(prefs)
        self._path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
