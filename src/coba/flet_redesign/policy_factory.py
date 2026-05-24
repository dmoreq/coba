"""Policy factory for redesign lessons and arena runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from coba.flet_redesign.contracts import BanditPolicy
from coba.flet_redesign.policies import (
    BootstrappedEnsemblePolicy,
    EpsilonGreedyPolicy,
    GPUCBPolicy,
    LinUCBHybridPolicy,
    LinUCBSWPolicy,
    LinUCBPolicy,
    LogisticUCBPolicy,
    RandomPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy,
    TreeTSPolicy,
    TreeUCBPolicy,
    UCB1Policy,
)


def build_policy(
    policy_id: str,
    *,
    feature_order: Sequence[str] = (),
    seed: int = 0,
    params: dict[str, Any] | None = None,
) -> BanditPolicy[Any, Any]:
    """Instantiate one policy by id."""
    params = params or {}

    if policy_id == "random":
        return RandomPolicy(seed=seed)
    if policy_id == "epsilon_greedy":
        return EpsilonGreedyPolicy(epsilon=float(params.get("epsilon", 0.1)), seed=seed)
    if policy_id == "ucb1":
        return UCB1Policy(alpha=float(params.get("alpha", 1.0)), seed=seed)
    if policy_id == "thompson":
        return ThompsonSamplingPolicy(
            prior_alpha=float(params.get("prior_alpha", 1.0)),
            prior_beta=float(params.get("prior_beta", 1.0)),
            seed=seed,
        )
    if policy_id == "softmax":
        return SoftmaxPolicy(tau=float(params.get("tau", 0.2)), seed=seed)
    if policy_id == "linucb":
        return LinUCBPolicy(
            feature_order=feature_order,
            alpha=float(params.get("alpha", 1.0)),
            l2_lambda=float(params.get("l2_lambda", 1.0)),
        )
    if policy_id == "linucb_sw":
        return LinUCBSWPolicy(
            feature_order=feature_order,
            window_size=int(params.get("window_size", 200)),
            alpha=float(params.get("alpha", 1.0)),
            l2_lambda=float(params.get("l2_lambda", 1.0)),
        )
    if policy_id == "logistic_ucb":
        return LogisticUCBPolicy(
            feature_order=feature_order,
            alpha=float(params.get("alpha", 0.5)),
            learning_rate=float(params.get("learning_rate", 0.1)),
        )
    if policy_id == "gp_ucb":
        return GPUCBPolicy(beta=float(params.get("beta", 1.5)))
    if policy_id == "bootstrapped_ensemble":
        return BootstrappedEnsemblePolicy(
            n_heads=int(params.get("n_heads", 8)),
            seed=seed,
        )
    if policy_id == "linucb_hybrid":
        return LinUCBHybridPolicy(
            feature_order=feature_order,
            n_shared=int(params.get("n_shared", 1)),
            alpha=float(params.get("alpha", 1.0)),
        )
    if policy_id == "tree_ucb":
        return TreeUCBPolicy(
            context_key=str(
                params.get("context_key", feature_order[0] if feature_order else "step")
            ),
            alpha=float(params.get("alpha", 0.8)),
        )
    if policy_id == "tree_ts":
        return TreeTSPolicy(
            context_key=str(
                params.get("context_key", feature_order[0] if feature_order else "step")
            ),
            seed=seed,
        )

    raise ValueError(f"Unsupported policy_id '{policy_id}'")
