"""Tests for debug pane builders across all policy families."""

from __future__ import annotations

from web.debug import (
    AdvancedDebugPane,
    ContinuousDebugPane,
    ContextFreeDebugPane,
    ContextualDebugPane,
    build_cf_debug_pane,
    build_continuous_debug_pane,
    build_ensemble_debug_pane,
    build_gp_debug_pane,
    build_hybrid_debug_pane,
    build_linucb_debug_pane,
    build_logistic_debug_pane,
    build_tree_debug_pane,
)


def test_cf_debug_pane_random() -> None:
    snap = {"policy": "random", "total_pulls": 100, "arms": {}}
    pane = build_cf_debug_pane(snap)
    assert isinstance(pane, ContextFreeDebugPane)
    assert pane.title == "random Debug"
    assert any("total_pulls" in k for k, _ in pane.details)


def test_cf_debug_pane_epsilon_greedy() -> None:
    snap = {
        "policy": "epsilon_greedy",
        "epsilon": 0.1,
        "total_pulls": 50,
        "arms": {"a": {"pulls": 25, "mean_reward": 0.6}},
    }
    pane = build_cf_debug_pane(snap)
    assert isinstance(pane, ContextFreeDebugPane)
    assert pane.title == "epsilon_greedy Debug"


def test_cf_debug_pane_ucb1() -> None:
    snap = {
        "policy": "ucb1",
        "alpha": 1.0,
        "total_pulls": 50,
        "arms": {"x": {"pulls": 10, "mean_reward": 0.4, "ucb_bonus": 0.5}},
    }
    pane = build_cf_debug_pane(snap)
    assert isinstance(pane, ContextFreeDebugPane)
    assert pane.title == "ucb1 Debug"


def test_cf_debug_pane_thompson() -> None:
    snap = {
        "policy": "thompson",
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "arms": {"a": {"successes": 5, "failures": 3}},
    }
    pane = build_cf_debug_pane(snap)
    assert isinstance(pane, ContextFreeDebugPane)


def test_cf_debug_pane_softmax() -> None:
    snap = {
        "policy": "softmax",
        "tau": 0.2,
        "arms": {"a": {"pulls": 10, "mean_reward": 0.5, "probability": 0.7}},
    }
    pane = build_cf_debug_pane(snap)
    assert isinstance(pane, ContextFreeDebugPane)


def test_linucb_debug_pane_with_matrix_data() -> None:
    snap = {
        "feature_order": ("f1", "f2"),
        "scores": {"arm1": 1.5, "arm2": 0.8},
        "arms": {
            "arm1": {"a": [[2.0, 0.0], [0.0, 2.0]], "b": [0.5, 0.3]},
            "arm2": {"a": [[3.0, 0.0], [0.0, 3.0]], "b": [0.1, 0.2]},
        },
    }
    pane = build_linucb_debug_pane(snap)
    assert isinstance(pane, ContextualDebugPane)
    assert len(pane.rows) >= 5


def test_linucb_debug_pane_empty_arms() -> None:
    pane = build_linucb_debug_pane({"scores": {}, "arms": {}, "feature_order": ()})
    assert isinstance(pane, ContextualDebugPane)


def test_logistic_debug_pane() -> None:
    snap = {
        "feature_order": ("f1",),
        "scores": {"arm1": 0.6},
        "learning_rate": 0.1,
        "arms": {"arm1": {"theta": [0.1, -0.2]}},
    }
    pane = build_logistic_debug_pane(snap)
    assert isinstance(pane, ContextualDebugPane)


def test_gp_debug_pane() -> None:
    snap = {"beta": 1.5, "arms": {"arm1": {"count": 20, "mean": 0.55, "variance": 0.04}}}
    pane = build_gp_debug_pane(snap)
    assert isinstance(pane, AdvancedDebugPane)
    assert len(pane.details) >= 2


def test_ensemble_debug_pane() -> None:
    snap = {
        "n_heads": 8,
        "arms": {"arm1": {"predictions": [0.5, 0.6, 0.4], "agreement_ratio": 0.75}},
    }
    pane = build_ensemble_debug_pane(snap)
    assert isinstance(pane, AdvancedDebugPane)


def test_tree_debug_pane() -> None:
    snap = {"context_key": "age", "arms": {"arm1": {"bucket_count": 5}}}
    pane = build_tree_debug_pane(snap)
    assert isinstance(pane, AdvancedDebugPane)


def test_hybrid_debug_pane() -> None:
    snap = {
        "n_shared": 2,
        "arms": {"arm1": {"shared_theta": [0.1, 0.2], "arm_theta": [0.05]}},
    }
    pane = build_hybrid_debug_pane(snap)
    assert isinstance(pane, AdvancedDebugPane)


def test_continuous_debug_pane() -> None:
    snap = {"best_action": 0.5, "best_reward": 0.7, "history_size": 30}
    pane = build_continuous_debug_pane(snap)
    assert isinstance(pane, ContinuousDebugPane)
    assert len(pane.rows) == 3


def test_cf_debug_pane_empty_snapshot() -> None:
    pane = build_cf_debug_pane({})
    assert isinstance(pane, ContextFreeDebugPane)
    assert pane.title != ""
