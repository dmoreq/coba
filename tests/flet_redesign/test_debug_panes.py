"""Tests for debug pane builders across all policy families (context-free and continuous panes removed as dead code)."""

from __future__ import annotations

from web.debug import (
    AdvancedDebugPane,
    ContextualDebugPane,
    build_ensemble_debug_pane,
    build_gp_debug_pane,
    build_hybrid_debug_pane,
    build_linucb_debug_pane,
    build_logistic_debug_pane,
    build_tree_debug_pane,
)


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
