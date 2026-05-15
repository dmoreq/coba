"""Tests for ContinuousBandit public façade."""

import numpy as np
import pytest

from coba.continuous.bandit import ContinuousBandit
from coba.continuous.schemas import ContinuousDecision
from coba.config import BanditConfig


class TestContinuousBanditInitialization:
    """Test ContinuousBandit creation and configuration."""

    def test_bandit_creation_basic(self) -> None:
        """ContinuousBandit initializes with minimal parameters."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=8)
        assert bandit.a_min == 0.0
        assert bandit.a_max == 1.0
        assert bandit.n_features == 8
        assert not bandit.is_fitted
        assert bandit.n_leaves == 64  # depth=6 by default

    def test_bandit_creation_with_config(self) -> None:
        """ContinuousBandit accepts a BanditConfig."""
        config = BanditConfig(cats_depth=4, alpha=2.0)
        bandit = ContinuousBandit(
            a_min=0.50,
            a_max=5.00,
            n_features=5,
            config=config,
        )
        assert bandit.n_leaves == 16  # 2^4
        assert bandit._policy._alpha == 2.0

    def test_bandit_creation_with_kwargs_override(self) -> None:
        """ContinuousBandit kwargs override config values."""
        config = BanditConfig(alpha=1.0, cats_depth=5)
        bandit = ContinuousBandit(
            a_min=0.0,
            a_max=1.0,
            n_features=4,
            config=config,
            alpha=3.0,  # Override
            depth=3,  # Override
        )
        assert bandit._policy._alpha == 3.0
        assert bandit.n_leaves == 8  # 2^3


class TestContinuousBanditDecision:
    """Test decision making."""

    def test_decide_returns_continuous_decision(self) -> None:
        """decide() returns a valid ContinuousDecision."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=5)
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        decision = bandit.decide(x)

        assert isinstance(decision, ContinuousDecision)
        assert bandit.a_min <= decision.chosen_action <= bandit.a_max

    def test_decide_batch(self) -> None:
        """decide_batch() returns list of decisions."""
        bandit = ContinuousBandit(a_min=0.5, a_max=5.0, n_features=4)
        contexts = np.random.standard_normal((10, 4))
        decisions = bandit.decide_batch(contexts)

        assert len(decisions) == 10
        assert all(isinstance(d, ContinuousDecision) for d in decisions)
        assert all(bandit.a_min <= d.chosen_action <= bandit.a_max for d in decisions)

    def test_decide_context_validation(self) -> None:
        """decide() validates context shape."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=5)

        with pytest.raises(ValueError, match="context shape mismatch"):
            bandit.decide(np.array([0.1, 0.2, 0.3]))


class TestContinuousBanditUpdate:
    """Test learning from observations."""

    def test_update_single(self) -> None:
        """update() processes one observation."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=3)
        x = np.array([0.1, 0.2, 0.3])

        assert not bandit.is_fitted
        bandit.update(x, action=0.5, reward=0.8)
        assert bandit.is_fitted

    def test_update_batch(self) -> None:
        """update_batch() processes multiple observations."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=4)

        rng = np.random.default_rng(42)
        contexts = rng.standard_normal((20, 4))
        actions = rng.uniform(0.0, 1.0, 20)
        rewards = rng.uniform(0.0, 1.0, 20)

        bandit.update_batch(contexts, actions, rewards)
        assert bandit.is_fitted

    def test_update_batch_empty_noop(self) -> None:
        """update_batch() with empty arrays is a no-op."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=3)

        bandit.update_batch(
            np.empty((0, 3)),
            np.array([]),
            np.array([]),
        )
        assert not bandit.is_fitted


class TestContinuousBanditOfflineBootstrap:
    """Test offline training."""

    def test_fit_offline_basic(self) -> None:
        """fit_offline() trains from historical data."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=5)

        rng = np.random.default_rng(42)
        contexts = rng.standard_normal((50, 5))
        actions = rng.uniform(0.0, 1.0, 50)
        rewards = rng.uniform(0.0, 1.0, 50)

        bandit.fit_offline(contexts, actions, rewards)
        assert bandit.is_fitted

    def test_fit_offline_returns_self(self) -> None:
        """fit_offline() returns self for method chaining."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=3)

        contexts = np.random.standard_normal((10, 3))
        actions = np.random.uniform(0.0, 1.0, 10)
        rewards = np.random.uniform(0.0, 1.0, 10)

        result = bandit.fit_offline(contexts, actions, rewards)
        assert result is bandit

    def test_fit_offline_with_propensities(self) -> None:
        """fit_offline() respects historical propensities."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=4)

        rng = np.random.default_rng(42)
        contexts = rng.standard_normal((20, 4))
        actions = rng.uniform(0.0, 1.0, 20)
        rewards = rng.uniform(0.0, 1.0, 20)
        propensities = rng.uniform(0.1, 1.0, 20)

        bandit.fit_offline(contexts, actions, rewards, propensities)
        assert bandit.is_fitted


class TestContinuousBanditMonitoring:
    """Test statistics and monitoring."""

    def test_get_stats(self) -> None:
        """get_stats() returns per-leaf statistics."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=3, depth=2)

        # Train
        rng = np.random.default_rng(42)
        contexts = rng.standard_normal((20, 3))
        actions = rng.uniform(0.0, 1.0, 20)
        rewards = rng.uniform(0.0, 1.0, 20)
        bandit.fit_offline(contexts, actions, rewards)

        stats = bandit.get_stats()
        assert len(stats) == 4  # 2^2 leaves
        assert all(isinstance(s, dict) for s in stats.values())


class TestContinuousBanditProperties:
    """Test properties and utilities."""

    def test_n_leaves_property(self) -> None:
        """n_leaves reflects tree depth."""
        for depth in [1, 2, 4, 6, 8]:
            bandit = ContinuousBandit(
                a_min=0.0,
                a_max=1.0,
                n_features=3,
                depth=depth,
            )
            assert bandit.n_leaves == 2**depth

    def test_repr(self) -> None:
        """__repr__ produces informative string."""
        bandit = ContinuousBandit(
            a_min=0.5,
            a_max=5.0,
            n_features=8,
            depth=4,
        )
        repr_str = repr(bandit)
        assert "ContinuousBandit" in repr_str
        assert "0.5" in repr_str
        assert "5.0" in repr_str
        assert "8" in repr_str
        assert "16" in repr_str  # n_leaves


class TestContinuousBanditIntegration:
    """End-to-end integration tests."""

    def test_online_learning_workflow(self) -> None:
        """Full online learning workflow: decide → update → decide."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=4, seed=42)

        rng = np.random.default_rng(42)
        for step in range(30):
            x = rng.standard_normal(4)
            decision = bandit.decide(x)

            # Simulate reward based on action proximity to 0.5
            true_reward = 1.0 - abs(decision.chosen_action - 0.5) * 2
            reward = float(np.clip(true_reward + rng.normal(0, 0.05), 0, 1))

            bandit.update(x, decision.chosen_action, reward, decision.propensity)

        assert bandit.is_fitted

    def test_offline_then_online(self) -> None:
        """Train offline, then update online."""
        bandit = ContinuousBandit(a_min=0.0, a_max=1.0, n_features=5, depth=3)

        # Offline phase
        rng = np.random.default_rng(42)
        contexts_hist = rng.standard_normal((50, 5))
        actions_hist = rng.uniform(0.0, 1.0, 50)
        rewards_hist = rng.uniform(0.0, 1.0, 50)
        bandit.fit_offline(contexts_hist, actions_hist, rewards_hist)

        # Online phase
        for _ in range(10):
            x = rng.standard_normal(5)
            decision = bandit.decide(x)
            reward = rng.uniform(0.0, 1.0)
            bandit.update(x, decision.chosen_action, reward, decision.propensity)

        assert bandit.is_fitted
