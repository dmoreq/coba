"""Tests for Phase-6 contextual policies."""

from __future__ import annotations

from coba.flet_redesign.policy_factory import build_policy
from coba.flet_redesign.policies import LinUCBSWPolicy, LinUCBPolicy, LogisticUCBPolicy

FEATURES = ("x1", "x2", "x3")


def _context(step: int) -> dict[str, float]:
    return {"x1": float(step), "x2": float(step % 2), "x3": 1.0}


def test_linucb_select_update_cycle() -> None:
    policy = LinUCBPolicy(feature_order=FEATURES, alpha=1.0, l2_lambda=1.0)
    arms = ["a", "b"]
    arm = policy.select_arm(_context(1), arms)
    assert arm in arms
    policy.update(_context(1), arm, reward=1.0)
    snapshot = policy.get_debug_snapshot()
    assert set(snapshot["arms"].keys()).issuperset({arm})


def test_linucb_sw_respects_window_size() -> None:
    policy = LinUCBSWPolicy(feature_order=FEATURES, window_size=2, alpha=1.0, l2_lambda=1.0)
    for step in range(1, 6):
        arm = policy.select_arm(_context(step), ["a", "b"])
        policy.update(_context(step), arm, reward=float(step % 2))
    snapshot = policy.get_debug_snapshot()
    counts = [arm_data["count"] for arm_data in snapshot["arms"].values()]
    assert all(count <= 2 for count in counts)


def test_logistic_ucb_updates_theta() -> None:
    policy = LogisticUCBPolicy(feature_order=FEATURES, alpha=0.5, learning_rate=0.1)
    arm = policy.select_arm(_context(1), ["a", "b"])
    before = policy.get_debug_snapshot()["arms"][arm]["theta"]
    policy.update(_context(1), arm, reward=1.0)
    after = policy.get_debug_snapshot()["arms"][arm]["theta"]
    assert before != after


def test_policy_factory_builds_contextual_variants() -> None:
    linucb = build_policy("linucb", feature_order=FEATURES)
    linucb_sw = build_policy("linucb_sw", feature_order=FEATURES)
    logistic = build_policy("logistic_ucb", feature_order=FEATURES)
    assert isinstance(linucb, LinUCBPolicy)
    assert isinstance(linucb_sw, LinUCBSWPolicy)
    assert isinstance(logistic, LogisticUCBPolicy)
