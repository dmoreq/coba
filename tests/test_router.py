"""Tests for ClusterRouter."""

import numpy as np
import pytest

from coba.router import ClusterRouter
from coba.types import PolicyType


def make_synthetic_data(n: int = 200, n_features: int = 7, seed: int = 0):
    rng = np.random.default_rng(seed)
    contexts = rng.standard_normal((n, n_features))
    arms = [1.0, 1.1, 1.2, 1.5]
    decisions = rng.choice(arms, size=n)
    rewards = rng.uniform(0, 1, n)
    return contexts, decisions, rewards, arms


class TestClusterRouter:
    """Tests for ClusterRouter."""

    def setup_method(self):
        contexts, decisions, rewards, arms = make_synthetic_data()
        self.contexts = contexts
        self.decisions = decisions
        self.rewards = rewards
        self.arms = arms

    def test_init(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        assert not router.is_fitted
        assert len(router.arms) == 4

    def test_fit_marks_as_fitted(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        router.fit(self.contexts, self.decisions, self.rewards)
        assert router.is_fitted

    def test_predict_returns_valid_arm(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        router.fit(self.contexts, self.decisions, self.rewards)
        ctx = np.random.default_rng(1).standard_normal(7)
        arm = router.predict(ctx)
        assert arm in self.arms

    def test_score_all_returns_all_arms(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        router.fit(self.contexts, self.decisions, self.rewards)
        ctx = np.random.default_rng(1).standard_normal(7)
        scores = router.score_all(ctx)
        assert set(scores.keys()) == set(self.arms)

    def test_partial_fit(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        router.fit(self.contexts[:100], self.decisions[:100], self.rewards[:100])
        total_before = router._total_pulls
        router.partial_fit(self.contexts[100:], self.decisions[100:], self.rewards[100:])
        assert router._total_pulls > total_before

    def test_update_single(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        router.fit(self.contexts, self.decisions, self.rewards)
        ctx = self.contexts[0]
        arm = self.arms[0]
        before_pulls = router._total_pulls
        router.update(ctx, arm, 0.5)
        assert router._total_pulls == before_pulls + 1

    def test_add_arm(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        router.fit(self.contexts, self.decisions, self.rewards)
        router.add_arm(2.0)
        assert 2.0 in router.arms
        assert len(router.arms) == 5

    def test_add_arm_warm_start(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        router.fit(self.contexts, self.decisions, self.rewards)
        router.add_arm(2.0, warm_start_from=1.5)
        ctx = self.contexts[0]
        # Should be able to predict with new arm
        arm = router.predict(ctx)
        assert arm in router.arms

    def test_remove_arm(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        router.fit(self.contexts, self.decisions, self.rewards)
        router.remove_arm(1.0)
        assert 1.0 not in router.arms

    def test_remove_last_arm_raises(self):
        router = ClusterRouter(arms=[1.0], n_clusters=2, n_features=7)
        with pytest.raises(ValueError, match="Cannot remove the last arm"):
            router.remove_arm(1.0)

    def test_add_duplicate_arm_raises(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        with pytest.raises(ValueError, match="already exists"):
            router.add_arm(1.0)

    def test_update_unknown_arm_raises(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        router.fit(self.contexts, self.decisions, self.rewards)
        with pytest.raises(ValueError, match="not found"):
            router.update(self.contexts[0], arm=99.9, reward=0.5)

    def test_predict_before_fit_does_not_crash(self):
        """Before fit, predict should fall back gracefully (returns first arm)."""
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        ctx = np.zeros(7)
        # Should not raise — returns the arm with highest score (all zero → first arm)
        arm = router.predict(ctx)
        assert arm in self.arms

    def test_partial_fit_before_fit_triggers_fit(self):
        """partial_fit before fit should delegate to fit()."""
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        router.partial_fit(self.contexts, self.decisions, self.rewards)
        assert router.is_fitted

    @pytest.mark.parametrize(
        "policy",
        [
            PolicyType.LIN_UCB,
            PolicyType.LIN_TS,
            PolicyType.THOMPSON,
            PolicyType.UCB1,
            PolicyType.LOGISTIC_UCB,
            PolicyType.LOGISTIC_TS,
        ],
    )
    def test_all_policies_work(self, policy):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7, policy=policy, seed=42)
        router.fit(self.contexts, self.decisions, self.rewards)
        ctx = self.contexts[0]
        arm = router.predict(ctx)
        assert arm in self.arms

    def test_n_clusters_minimum_of_1(self) -> None:
        """n_clusters=1 is valid (used for context-free UCB1/Thompson policies)."""
        router = ClusterRouter(arms=self.arms, n_clusters=1, n_features=7)
        assert router.n_clusters == 1

    def test_n_clusters_zero_raises(self) -> None:
        """n_clusters=0 is invalid and must raise ValueError."""
        with pytest.raises(ValueError):
            ClusterRouter(arms=self.arms, n_clusters=0, n_features=7)

    def test_empty_arms_raises(self):
        with pytest.raises(ValueError):
            ClusterRouter(arms=[], n_clusters=2, n_features=7)

    def test_fit_with_weights(self):
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        weights = np.ones(len(self.rewards)) * 2.0
        router.fit(self.contexts, self.decisions, self.rewards, weights=weights)
        assert router.is_fitted

    def test_add_arm_warm_start_from_missing_falls_back_to_cold(self):
        """When warm_start_from arm does not exist, should create a fresh cold model."""
        router = ClusterRouter(arms=self.arms, n_clusters=3, n_features=7)
        router.fit(self.contexts, self.decisions, self.rewards)
        # 9.9 is not in arms — should fall back to cold start without raising
        router.add_arm(3.0, warm_start_from=9.9)
        assert 3.0 in router.arms
        # Should still be able to predict
        arm = router.predict(self.contexts[0])
        assert arm in router.arms

    def test_n_clusters_must_be_at_least_1(self):
        """n_clusters=0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_clusters must be at least 1"):
            ClusterRouter(arms=self.arms, n_clusters=0, n_features=7)
