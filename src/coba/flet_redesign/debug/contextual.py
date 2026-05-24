"""Contextual policy debugger pane builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextualDebugPane:
    """Renderable debugger summary for contextual policies."""

    title: str
    rows: tuple[tuple[str, str], ...]


def build_linucb_debug_pane(snapshot: dict[str, Any]) -> ContextualDebugPane:
    scores = snapshot.get("scores", {})
    arm_count = len(snapshot.get("arms", {}))
    best = max(scores.items(), key=lambda item: item[1])[0] if scores else "n/a"
    rows = (
        ("arms", str(arm_count)),
        ("best_arm", str(best)),
        ("feature_count", str(len(snapshot.get("feature_order", ())))),
    )
    return ContextualDebugPane(title="LinUCB Debug", rows=rows)


def build_logistic_debug_pane(snapshot: dict[str, Any]) -> ContextualDebugPane:
    scores = snapshot.get("scores", {})
    arm_count = len(snapshot.get("arms", {}))
    best = max(scores.items(), key=lambda item: item[1])[0] if scores else "n/a"
    rows = (
        ("arms", str(arm_count)),
        ("best_arm", str(best)),
        ("feature_count", str(len(snapshot.get("feature_order", ())))),
    )
    return ContextualDebugPane(title="Logistic Debug", rows=rows)
