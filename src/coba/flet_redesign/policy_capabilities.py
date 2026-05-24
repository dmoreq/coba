"""Policy capability registry for UI/debugger composition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyCapability:
    """Static capability declaration for one policy."""

    policy_id: str
    family: str
    needs_context: bool
    debug_views: tuple[str, ...]


POLICY_CAPABILITIES: dict[str, PolicyCapability] = {
    "random": PolicyCapability("random", "context_free", False, ("summary",)),
    "epsilon_greedy": PolicyCapability("epsilon_greedy", "context_free", False, ("summary",)),
    "ucb1": PolicyCapability("ucb1", "context_free", False, ("summary",)),
    "thompson": PolicyCapability("thompson", "context_free", False, ("summary",)),
    "softmax": PolicyCapability("softmax", "context_free", False, ("summary",)),
    "linucb": PolicyCapability("linucb", "linear_contextual", True, ("linucb_debug",)),
    "linucb_sw": PolicyCapability("linucb_sw", "linear_contextual", True, ("linucb_debug",)),
    "logistic_ucb": PolicyCapability("logistic_ucb", "logistic", True, ("logistic_debug",)),
    "gp_ucb": PolicyCapability("gp_ucb", "bayesian", True, ("gp_debug",)),
    "bootstrapped_ensemble": PolicyCapability(
        "bootstrapped_ensemble",
        "ensemble",
        True,
        ("ensemble_debug",),
    ),
    "linucb_hybrid": PolicyCapability("linucb_hybrid", "hybrid", True, ("hybrid_debug",)),
    "tree_ucb": PolicyCapability("tree_ucb", "tree_ensemble", True, ("tree_debug",)),
    "tree_ts": PolicyCapability("tree_ts", "tree_ensemble", True, ("tree_debug",)),
}


def get_policy_capability(policy_id: str) -> PolicyCapability:
    try:
        return POLICY_CAPABILITIES[policy_id]
    except KeyError as exc:
        raise KeyError(f"Unknown policy_id '{policy_id}'") from exc
