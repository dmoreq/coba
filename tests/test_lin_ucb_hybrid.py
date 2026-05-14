"""Tests for LinUCBHybridArmModel and ClusterBandit hybrid policy integration."""

import numpy as np
import pytest

from coba import ClusterBandit
from coba.policies.lin_ucb_hybrid import LinUCBHybridArmModel
from coba.policies.ridge import RidgeRegression
from coba.types import PolicyType

N_SHARED = 3
N_ARM = 4
N_FEATURES = N_SHARED + N_ARM
ARMS = ["A", "B", "C"]


def make_shared_ridge() -> RidgeRegression:
    return RidgeRegression(n_features=N_SHARED, l2_lambda=1.0)


def make_arm_model(arm: str = "A") -> LinUCBHybridArmModel:
    return LinUCBHybridArmModel(
        arm=arm,
        n_shared=N_SHARED,
        n_arm=N_ARM,
        shared_ridge=make_shared_ridge(),
        alpha=1.0,
        l2_lambda=1.0,
    )


class TestLinUCBHybridArmModel:
    def test_init_basic(self):
        m = make_arm_model()
        assert m.n_shared == N_SHARED
        assert m.n_arm == N_ARM
        assert not m.is_fitted

    def test_invalid_n_shared_negative(self):
        with pytest.raises(ValueError, match="n_shared"):
            LinUCBHybridArmModel("A", n_shared=-1, n_arm=3, shared_ridge=make_shared_ridge())

    def test_invalid_n_arm_negative(self):
        with pytest.raises(ValueError, match="n_arm"):
            LinUCBHybridArmModel("A", n_shared=3, n_arm=-1, shared_ridge=make_shared_ridge())

    def test_zero_total_features_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            LinUCBHybridArmModel("A", n_shared=0, n_arm=0, shared_ridge=make_shared_ridge())

    def test_score_before_update_returns_finite(self):
        m = make_arm_model()
        x = np.ones(N_FEATURES)
        score = m.score(x)
        assert np.isfinite(score)

    def test_update_marks_fitted(self):
        m = make_arm_model()
        x = np.random.default_rng(0).standard_normal(N_FEATURES)
        m.update(x, reward=0.7)
        assert m.is_fitted

    def test_update_changes_score(self):
        m = make_arm_model()
        rng = np.random.default_rng(1)
        x = rng.standard_normal(N_FEATURES)
        score_before = m.score(x)
        for _ in range(10):
            m.update(rng.standard_normal(N_FEATURES), reward=0.8)
        score_after = m.score(x)
        assert score_before != score_after

    def test_shared_ridge_is_same_object_across_arms(self):
        shared = make_shared_ridge()
        m1 = LinUCBHybridArmModel("A", N_SHARED, N_ARM, shared_ridge=shared)
        m2 = LinUCBHybridArmModel("B", N_SHARED, N_ARM, shared_ridge=shared)
        assert m1._shared_ridge is m2._shared_ridge

    def test_shared_ridge_updated_when_any_arm_pulls(self):
        shared = make_shared_ridge()
        m1 = LinUCBHybridArmModel("A", N_SHARED, N_ARM, shared_ridge=shared)
        m2 = LinUCBHybridArmModel("B", N_SHARED, N_ARM, shared_ridge=shared)
        rng = np.random.default_rng(2)
        m1.update(rng.standard_normal(N_FEATURES), reward=0.6)
        # m2's score should also reflect the shared ridge update from m1's pull
        x = rng.standard_normal(N_FEATURES)
        score_after = m2.score(x)
        assert np.isfinite(score_after)
        # Shared beta is nonzero after m1's update
        assert not np.allclose(m2.beta_shared, np.zeros(N_SHARED))

    def test_reset_clears_per_arm_ridge_only(self):
        shared = make_shared_ridge()
        m = LinUCBHybridArmModel("A", N_SHARED, N_ARM, shared_ridge=shared)
        rng = np.random.default_rng(3)
        m.update(rng.standard_normal(N_FEATURES), reward=0.9)
        beta_shared_before = m.beta_shared.copy()
        m.reset()
        # Shared beta preserved
        np.testing.assert_array_equal(m.beta_shared, beta_shared_before)
        # Per-arm beta reset to zeros
        assert np.allclose(m.beta_arm, np.zeros(N_ARM))
        assert not m.is_fitted

    def test_batch_update(self):
        m = make_arm_model()
        rng = np.random.default_rng(4)
        x_batch = rng.standard_normal((20, N_FEATURES))
        y = rng.uniform(0, 1, 20)
        m.update_batch(x_batch, y)
        assert m.is_fitted

    def test_shared_only_mode(self):
        shared = RidgeRegression(n_features=N_SHARED, l2_lambda=1.0)
        m = LinUCBHybridArmModel("A", n_shared=N_SHARED, n_arm=0, shared_ridge=shared)
        x = np.ones(N_SHARED)
        m.update(x, reward=0.5)
        assert m.is_fitted
        assert np.isfinite(m.score(x))

    def test_arm_only_mode(self):
        shared = RidgeRegression(n_features=1, l2_lambda=1.0)
        m = LinUCBHybridArmModel("A", n_shared=0, n_arm=N_ARM, shared_ridge=shared)
        x = np.ones(N_ARM)
        m.update(x, reward=0.5)
        assert m.is_fitted
        assert np.isfinite(m.score(x))


class TestClusterBanditHybridPolicy:
    def _make_bandit(self) -> ClusterBandit:
        return ClusterBandit(
            arms=ARMS,
            n_features=N_FEATURES,
            policy=PolicyType.LIN_UCB_HYBRID,
            n_shared_features=N_SHARED,
            n_clusters=2,
            seed=0,
        )

    def test_hybrid_bandit_creates(self):
        bandit = self._make_bandit()
        assert bandit._router._cfg.n_shared_features == N_SHARED

    def test_invalid_n_shared_raises(self):
        with pytest.raises(ValueError):
            ClusterBandit(
                arms=ARMS,
                n_features=N_FEATURES,
                policy=PolicyType.LIN_UCB_HYBRID,
                n_shared_features=0,  # must be > 0 for hybrid
                n_clusters=2,
            )

    def test_fit_and_decide(self):
        bandit = self._make_bandit()
        rng = np.random.default_rng(5)
        n = 200
        bandit.fit_offline(
            contexts=rng.standard_normal((n, N_FEATURES)),
            decisions=rng.choice(ARMS, size=n),
            rewards=rng.uniform(0, 1, n),
        )
        decision = bandit.decide(rng.standard_normal(N_FEATURES))
        assert decision.chosen_arm in ARMS

    def test_update_online(self):
        bandit = self._make_bandit()
        rng = np.random.default_rng(6)
        bandit.fit_offline(
            contexts=rng.standard_normal((100, N_FEATURES)),
            decisions=rng.choice(ARMS, size=100),
            rewards=rng.uniform(0, 1, 100),
        )
        ctx = rng.standard_normal(N_FEATURES)
        decision = bandit.decide(ctx)
        bandit.update(context=ctx, arm=decision.chosen_arm, reward=0.75)

    def test_shared_ridge_benefits_data_poor_arm(self):
        """Arm B gets zero direct pulls; shared features from A/C still inform its score."""
        bandit = self._make_bandit()
        rng = np.random.default_rng(7)
        n = 100
        # Only pull A and C
        contexts = rng.standard_normal((n, N_FEATURES))
        decisions = rng.choice(["A", "C"], size=n)
        rewards = rng.uniform(0, 1, n)
        bandit.fit_offline(contexts, decisions, rewards)
        # B should still produce a finite score (via shared features)
        scores = bandit.score_all(rng.standard_normal(N_FEATURES))
        assert np.isfinite(scores["B"])
