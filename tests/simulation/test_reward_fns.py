"""Tests for coba.simulation.reward_fns."""

import numpy as np
import pytest

from coba.simulation.reward_fns import (
    RewardFn,
    categorical_reward,
    context_free_reward,
    linear_reward,
)

# ── categorical_reward ────────────────────────────────────────────────────────


class TestCategoricalReward:
    """Tests for categorical_reward factory."""

    def test_returns_callable(self) -> None:
        fn = categorical_reward({"A": 0.5})
        assert callable(fn)

    def test_output_is_binary(self) -> None:
        fn = categorical_reward({"A": 0.8, "B": 0.2})
        ctx = np.zeros(3)
        for _ in range(30):
            assert categorical_reward({"A": 0.8})(  "A", ctx) in (0.0, 1.0)

    def test_known_arms_receive_correct_rates(self) -> None:
        """With 100 trials, a p=1.0 arm always returns 1."""
        fn = categorical_reward({"always": 1.0, "never": 0.0})
        ctx = np.zeros(1)
        assert all(fn("always", ctx) == 1.0 for _ in range(50))
        assert all(fn("never", ctx) == 0.0 for _ in range(50))

    def test_unknown_arm_uses_fallback(self) -> None:
        fn = categorical_reward({}, fallback=0.0)
        ctx = np.zeros(1)
        assert all(fn("unknown", ctx) == 0.0 for _ in range(50))

    def test_context_is_ignored(self) -> None:
        """Reward should not depend on context for categorical reward."""
        fn = categorical_reward({"A": 1.0})
        assert fn("A", np.zeros(5)) == fn("A", np.ones(5))


# ── linear_reward ─────────────────────────────────────────────────────────────


class TestLinearReward:
    """Tests for linear_reward factory."""

    def test_returns_callable(self) -> None:
        fn = linear_reward({"A": [1.0, 0.0]})
        assert callable(fn)

    def test_output_is_binary(self) -> None:
        fn = linear_reward({"A": [0.5, 0.5]})
        ctx = np.array([1.0, 1.0])
        for _ in range(30):
            assert fn("A", ctx) in (0.0, 1.0)

    def test_positive_weights_increase_reward(self) -> None:
        """Large positive weights push prob → 1."""
        fn = linear_reward({"A": [10.0, 10.0]})
        ctx = np.array([1.0, 1.0])
        rewards = [fn("A", ctx) for _ in range(100)]
        assert np.mean(rewards) > 0.8

    def test_negative_weights_decrease_reward(self) -> None:
        """Large negative weights push prob → 0."""
        fn = linear_reward({"A": [-10.0, -10.0]})
        ctx = np.array([1.0, 1.0])
        rewards = [fn("A", ctx) for _ in range(100)]
        assert np.mean(rewards) < 0.2

    def test_fallback_for_unknown_arm(self) -> None:
        fn = linear_reward({}, fallback=0.0)
        ctx = np.zeros(2)
        assert all(fn("unknown", ctx) == 0.0 for _ in range(30))

    def test_fallback_for_mismatched_dim(self) -> None:
        fn = linear_reward({"A": [1.0, 2.0, 3.0]}, fallback=0.0)
        ctx = np.zeros(5)  # wrong dimension
        assert all(fn("A", ctx) == 0.0 for _ in range(30))


# ── context_free_reward ───────────────────────────────────────────────────────


class TestContextFreeReward:
    """Tests for context_free_reward (alias of categorical_reward)."""

    def test_is_reward_fn_type(self) -> None:
        fn: RewardFn = context_free_reward({"X": 0.5})
        assert callable(fn)

    def test_same_behaviour_as_categorical(self) -> None:
        rates = {"A": 1.0, "B": 0.0}
        fn = context_free_reward(rates)
        ctx = np.zeros(1)
        assert fn("A", ctx) == 1.0
        assert fn("B", ctx) == 0.0
