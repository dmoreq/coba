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
    feature_order = snapshot.get("feature_order", ())
    best = max(scores.items(), key=lambda item: item[1])[0] if scores else "n/a"
    rows: list[tuple[str, str]] = [
        ("arms", str(arm_count)),
        ("best_arm", str(best)),
        ("feature_count", str(len(feature_order))),
    ]

    arm_data = snapshot.get("arms", {})
    for arm_name, arm_state in arm_data.items():
        if not isinstance(arm_state, dict):
            continue
        a_matrix = arm_state.get("a")
        b_vector = arm_state.get("b")
        if a_matrix and b_vector:
            rows.append((f"  {arm_name} A-trace", f"{_trace(a_matrix):.3f}"))
            rows.append((f"  {arm_name} b-norm", f"{_norm(b_vector):.3f}"))
            rows.append((f"  {arm_name} score", f"{scores.get(arm_name, 0.0):.4f}"))

    return ContextualDebugPane(title="LinUCB Debug", rows=tuple(rows))


def build_logistic_debug_pane(snapshot: dict[str, Any]) -> ContextualDebugPane:
    scores = snapshot.get("scores", {})
    arm_count = len(snapshot.get("arms", {}))
    feature_order = snapshot.get("feature_order", ())
    best = max(scores.items(), key=lambda item: item[1])[0] if scores else "n/a"
    rows: list[tuple[str, str]] = [
        ("arms", str(arm_count)),
        ("best_arm", str(best)),
        ("feature_count", str(len(feature_order))),
        ("learning_rate", str(snapshot.get("learning_rate", "n/a"))),
    ]

    arm_data = snapshot.get("arms", {})
    for arm_name, arm_state in arm_data.items():
        if not isinstance(arm_state, dict):
            continue
        theta = arm_state.get("theta")
        if theta:
            rows.append((f"  {arm_name} theta-norm", f"{_norm(theta):.4f}"))
            rows.append((f"  {arm_name} score", f"{scores.get(arm_name, 0.0):.4f}"))

    return ContextualDebugPane(title="Logistic Debug", rows=tuple(rows))


def _trace(matrix: list[list[float]]) -> float:
    d = min(len(matrix), len(matrix[0]) if matrix else 0)
    return sum(matrix[i][i] for i in range(d))


def _norm(vector: list[float]) -> float:
    return sum(v * v for v in vector) ** 0.5
