"""Scene panel view-model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenePanelModel:
    """View-model for scenario context rendering."""

    world_title: str
    world_description: str
    context_items: dict[str, Any] = field(default_factory=dict)
