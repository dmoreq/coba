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
    rows: list[tuple[str, str]] = [
        ("beta", str(snapshot.get("beta", "n/a"))),
        ("arm_count", str(len(arms))),
    ]
    for arm_name, arm_data in arms.items():
        if isinstance(arm_data, dict):
            rows.append(
                (
                    f"  {arm_name}",
                    f"n={arm_data.get('count', 0)} "
                    f"μ={arm_data.get('mean', 0.0):.4f} "
                    f"σ²={arm_data.get('variance', 1.0):.4f}",
                )
            )
    return AdvancedDebugPane(title="GP-UCB Debug", details=tuple(rows))


def build_ensemble_debug_pane(snapshot: dict[str, Any]) -> AdvancedDebugPane:
    rows: list[tuple[str, str]] = [
        ("n_heads", str(snapshot.get("n_heads", "n/a"))),
        ("arm_count", str(len(snapshot.get("arms", {})))),
    ]
    arms = snapshot.get("arms", {})
    for arm_name, arm_data in arms.items():
        if isinstance(arm_data, dict) and "predictions" in arm_data:
            preds = arm_data["predictions"]
            if len(preds) > 0:
                rows.append(
                    (
                        f"  {arm_name}",
                        f"μ={sum(preds) / len(preds):.4f} "
                        f"σ={_std(preds):.4f} "
                        f"agree={arm_data.get('agreement_ratio', 0.0):.2f}",
                    )
                )
    return AdvancedDebugPane(title="Ensemble Debug", details=tuple(rows))


def build_tree_debug_pane(snapshot: dict[str, Any]) -> AdvancedDebugPane:
    rows: list[tuple[str, str]] = [
        ("context_key", str(snapshot.get("context_key", "n/a"))),
        ("arm_count", str(len(snapshot.get("arms", {})))),
    ]
    arms = snapshot.get("arms", {})
    for arm_name, arm_data in arms.items():
        if isinstance(arm_data, dict):
            bucket_count = arm_data.get("bucket_count", 0)
            rows.append((f"  {arm_name} buckets", str(bucket_count)))
    return AdvancedDebugPane(title="Tree Debug", details=tuple(rows))


def build_hybrid_debug_pane(snapshot: dict[str, Any]) -> AdvancedDebugPane:
    rows: list[tuple[str, str]] = [
        ("n_shared", str(snapshot.get("n_shared", "n/a"))),
        ("arm_count", str(len(snapshot.get("arms", {})))),
    ]
    arms = snapshot.get("arms", {})
    for arm_name, arm_data in arms.items():
        if isinstance(arm_data, dict):
            shared_theta = arm_data.get("shared_theta")
            arm_theta = arm_data.get("arm_theta")
            if shared_theta:
                rows.append((f"  {arm_name} shared-θ norm", f"{_norm_list(shared_theta):.4f}"))
            if arm_theta:
                rows.append((f"  {arm_name} arm-θ norm", f"{_norm_list(arm_theta):.4f}"))
    return AdvancedDebugPane(title="Hybrid Debug", details=tuple(rows))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _norm_list(values: list[float]) -> float:
    return sum(v * v for v in values) ** 0.5
