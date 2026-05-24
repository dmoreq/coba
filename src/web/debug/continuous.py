"""Continuous-action debugger pane helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContinuousDebugPane:
    """Renderable debug summary for continuous action policies."""

    title: str
    rows: tuple[tuple[str, str], ...]


def build_continuous_debug_pane(snapshot: dict[str, Any]) -> ContinuousDebugPane:
    return ContinuousDebugPane(
        title="Continuous Debug",
        rows=(
            ("best_action", str(snapshot.get("best_action", "n/a"))),
            ("best_reward", str(snapshot.get("best_reward", "n/a"))),
            ("history_size", str(snapshot.get("history_size", 0))),
        ),
    )
