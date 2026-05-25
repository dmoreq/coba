"""Tests for Phase-7 advanced policies and diagnostics."""

from __future__ import annotations

from web.analysis import compute_comparison_diagnostics
from web.debug import (
    build_ensemble_debug_pane,
    build_gp_debug_pane,
    build_hybrid_debug_pane,
    build_tree_debug_pane,
)
from web.policy_capabilities import get_policy_capability
from web.policy_factory import build_policy
from web.policies import (
    BootstrappedEnsemblePolicy,
    GPUCBPolicy,
    LinUCBHybridPolicy,
    TreeTSPolicy,
    TreeUCBPolicy,
)


def _context(step: int) -> dict[str, float]:
    return {"x1": float(step), "x2": float(step % 3), "x3": 1.0}


def test_advanced_policy_factory_builds_expected_types() -> None:
    features = ("x1", "x2", "x3")
    assert isinstance(build_policy("gp_ucb", feature_order=features), GPUCBPolicy)
    assert isinstance(
        build_policy("bootstrapped_ensemble", feature_order=features, seed=1),
        BootstrappedEnsemblePolicy,
    )
    assert isinstance(
        build_policy("linucb_hybrid", feature_order=features, params={"n_shared": 1}),
        LinUCBHybridPolicy,
    )
    assert isinstance(
        build_policy("tree_ucb", feature_order=features, params={"context_key": "x1"}),
        TreeUCBPolicy,
    )
    assert isinstance(
        build_policy("tree_ts", feature_order=features, params={"context_key": "x1"}, seed=1),
        TreeTSPolicy,
    )


def test_advanced_policies_select_and_update() -> None:
    arms = ["a", "b", "c"]
    policies = [
        GPUCBPolicy(beta=1.5),
        BootstrappedEnsemblePolicy(n_heads=4, seed=2),
        LinUCBHybridPolicy(feature_order=("x1", "x2", "x3"), n_shared=1, alpha=1.0),
        TreeUCBPolicy(context_key="x1", alpha=0.8),
        TreeTSPolicy(context_key="x1", seed=2),
    ]
    for policy in policies:
        arm = policy.select_arm(_context(1), arms)
        assert arm in arms
        policy.update(_context(1), arm, reward=1.0)
        snapshot = policy.get_debug_snapshot()
        assert "scores" in snapshot


def test_policy_capability_registry_and_debug_panes() -> None:
    cap = get_policy_capability("gp_ucb")
    assert cap.family == "bayesian"
    assert cap.needs_context
    assert "gp_debug" in cap.debug_views

    gp_pane = build_gp_debug_pane({"beta": 1.5, "arms": {"a": {}}})
    ensemble_pane = build_ensemble_debug_pane({"n_heads": 8, "arms": {"a": {}, "b": {}}})
    tree_pane = build_tree_debug_pane({"context_key": "x1", "arms": {"a": {}}})
    hybrid_pane = build_hybrid_debug_pane({"n_shared": 1, "arms": {"a": {}, "b": {}}})

    assert gp_pane.title == "GP-UCB Debug"
    assert ensemble_pane.title == "Ensemble Debug"
    assert tree_pane.title == "Tree Debug"
    assert hybrid_pane.title == "Hybrid Debug"


def test_comparison_diagnostics_computes_expected_metrics() -> None:
    baseline = [
        {"step_index": 1, "cumulative_reward": 1.0, "metadata": {"uncertainty": 0.5}},
        {"step_index": 2, "cumulative_reward": 2.0, "metadata": {"uncertainty": 0.4}},
        {"step_index": 3, "cumulative_reward": 3.0, "metadata": {"uncertainty": 0.3}},
    ]
    candidate = [
        {"step_index": 1, "cumulative_reward": 1.2, "metadata": {"uncertainty": 0.3}},
        {"step_index": 2, "cumulative_reward": 2.6, "metadata": {"uncertainty": 0.2}},
        {"step_index": 3, "cumulative_reward": 3.6, "metadata": {"uncertainty": 0.1}},
    ]
    diag = compute_comparison_diagnostics(
        baseline_records=baseline,
        candidate_records=candidate,
    )
    assert diag.final_reward_delta > 0.0
    assert diag.mean_uncertainty > 0.0
