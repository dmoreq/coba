"""Context-free policy debugger pane builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextFreeDebugPane:
    """Renderable debugger summary for context-free policies."""

    title: str
    details: tuple[tuple[str, str], ...]


def build_cf_debug_pane(snapshot: dict[str, Any]) -> ContextFreeDebugPane:
    policy_name = snapshot.get("policy", "unknown")
    arms = snapshot.get("arms", {})
    rows: list[tuple[str, str]] = [
        ("policy", policy_name),
        ("total_pulls", str(snapshot.get("total_pulls", 0))),
        ("arm_count", str(len(arms))),
    ]
    if policy_name == "epsilon_greedy":
        rows.append(("epsilon", str(snapshot.get("epsilon", 0.0))))
    elif policy_name == "ucb1":
        rows.append(("alpha", str(snapshot.get("alpha", 0.0))))
    elif policy_name == "thompson":
        rows.append(("prior_alpha", str(snapshot.get("prior_alpha", 0.0))))
        rows.append(("prior_beta", str(snapshot.get("prior_beta", 0.0))))
    elif policy_name == "softmax":
        rows.append(("tau", str(snapshot.get("tau", 0.0))))

    for arm_name, arm_data in arms.items():
        if isinstance(arm_data, dict):
            for k, v in arm_data.items():
                if isinstance(v, float):
                    rows.append((f"  {arm_name}/{k}", f"{v:.4f}"))
                else:
                    rows.append((f"  {arm_name}/{k}", str(v)))

    return ContextFreeDebugPane(title=f"{policy_name} Debug", details=tuple(rows))
