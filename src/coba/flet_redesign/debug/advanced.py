"""Advanced policy debugger pane builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdvancedDebugPane:
    """Renderable summary for advanced policies."""

    title: str
    details: tuple[tuple[str, str], ...]


def build_gp_debug_pane(snapshot: dict[str, Any]) -> AdvancedDebugPane:
    arms = snapshot.get("arms", {})
    return AdvancedDebugPane(
        title="GP-UCB Debug",
        details=(
            ("beta", str(snapshot.get("beta", "n/a"))),
            ("arm_count", str(len(arms))),
        ),
    )


def build_ensemble_debug_pane(snapshot: dict[str, Any]) -> AdvancedDebugPane:
    return AdvancedDebugPane(
        title="Ensemble Debug",
        details=(
            ("n_heads", str(snapshot.get("n_heads", "n/a"))),
            ("arm_count", str(len(snapshot.get("arms", {})))),
        ),
    )


def build_tree_debug_pane(snapshot: dict[str, Any]) -> AdvancedDebugPane:
    return AdvancedDebugPane(
        title="Tree Debug",
        details=(
            ("context_key", str(snapshot.get("context_key", "n/a"))),
            ("arm_count", str(len(snapshot.get("arms", {})))),
        ),
    )


def build_hybrid_debug_pane(snapshot: dict[str, Any]) -> AdvancedDebugPane:
    return AdvancedDebugPane(
        title="Hybrid Debug",
        details=(
            ("n_shared", str(snapshot.get("n_shared", "n/a"))),
            ("arm_count", str(len(snapshot.get("arms", {})))),
        ),
    )
