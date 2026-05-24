"""Low-level math regression tests for contextual policy updates."""

from __future__ import annotations

import numpy as np

from web.policies import LinUCBPolicy


def test_linucb_matrix_update_matches_outer_product_formula() -> None:
    policy = LinUCBPolicy(feature_order=("x1", "x2"), alpha=1.0, l2_lambda=1.0)
    context = {"x1": 2.0, "x2": 1.0}
    arm = policy.select_arm(context, ["a"])
    policy.update(context, arm, reward=1.0)

    snapshot = policy.get_debug_snapshot()
    a = np.array(snapshot["arms"][arm]["a"], dtype=float)
    b = np.array(snapshot["arms"][arm]["b"], dtype=float)

    x = np.array([2.0, 1.0], dtype=float)
    expected_a = np.eye(2, dtype=float) + np.outer(x, x)
    expected_b = x
    assert np.allclose(a, expected_a)
    assert np.allclose(b, expected_b)
