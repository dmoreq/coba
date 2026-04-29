"""Tests for core arm models: RidgeRegression, LinUCB, LinTS, Thompson, UCB1."""

import math

import numpy as np
import pytest

from coba.policies.linucb import LinUCBArmModel
from coba.policies.lin_ts import LinTSArmModel
from coba.policies.ridge import RidgeRegression
from coba.policies.thompson import ThompsonArmModel
from coba.policies.ucb1 import UCB1ArmModel


class TestRidgeRegression:
    """Tests for the Sherman-Morrison online ridge regression."""

    def test_init_state(self):
        ridge = RidgeRegression(n_features=3, l2_lambda=1.0)
        assert ridge.n_obs == 0
        np.testing.assert_array_almost_equal(ridge.beta, np.zeros(3))
        np.testing.assert_array_almost_equal(ridge.A, np.eye(3))

    def test_single_update_changes_beta(self):
        ridge = RidgeRegression(n_features=2, l2_lambda=1.0)
        x = np.array([1.0, 0.5])
        ridge.update(x, y=0.8)
        assert ridge.n_obs == 1
        # Beta should be non-zero after one observation
        assert not np.allclose(ridge.beta, np.zeros(2))

    def test_predict_before_any_update(self):
        """Before any data, prediction should be 0 (zero-initialized beta)."""
        ridge = RidgeRegression(n_features=3)
        x = np.array([1.0, 2.0, 3.0])
        assert ridge.predict(x) == pytest.approx(0.0)

    def test_sherman_morrison_consistent_with_direct_inverse(self):
        """SM update should give same beta as directly computing A^-1 Xty."""
        rng = np.random.default_rng(0)
        n_features = 4
        l2_lambda = 2.0
        n_obs = 20

        x_mat = rng.standard_normal((n_obs, n_features))
        y_vec = rng.uniform(0, 1, n_obs)

        # SM incremental update
        ridge_sm = RidgeRegression(n_features, l2_lambda)
        for xi, yi in zip(x_mat, y_vec):
            ridge_sm.update(xi, float(yi))

        # Direct computation
        a_direct = l2_lambda * np.eye(n_features) + x_mat.T @ x_mat
        xty_direct = x_mat.T @ y_vec
        beta_direct = np.linalg.inv(a_direct) @ xty_direct

        np.testing.assert_array_almost_equal(ridge_sm.beta, beta_direct, decimal=6)

    def test_batch_update(self):
        ridge = RidgeRegression(n_features=2)
        x_mat = np.array([[1.0, 0.5], [0.2, 0.8], [0.7, 0.3]])
        y_vec = np.array([0.5, 0.7, 0.3])
        ridge.update_batch(x_mat, y_vec)
        assert ridge.n_obs == 3

    def test_reset(self):
        ridge = RidgeRegression(n_features=2)
        ridge.update(np.array([1.0, 0.5]), 0.8)
        ridge.reset()
        assert ridge.n_obs == 0
        np.testing.assert_array_almost_equal(ridge.beta, np.zeros(2))

    def test_weighted_update(self):
        """Weighted updates should differ from unweighted."""
        ridge_unweighted = RidgeRegression(n_features=2)
        ridge_weighted = RidgeRegression(n_features=2)
        x = np.array([1.0, 0.5])
        ridge_unweighted.update(x, 0.5, weight=1.0)
        ridge_weighted.update(x, 0.5, weight=2.0)
        # With weight=2, the observation has double influence
        assert not np.allclose(ridge_unweighted.beta, ridge_weighted.beta)

    def test_gamma_decay(self):
        """Gamma < 1.0 should decay the history (A_inv scales up)."""
        ridge1 = RidgeRegression(n_features=2, gamma=1.0)
        ridge2 = RidgeRegression(n_features=2, gamma=0.5)

        x = np.array([1.0, 0.0])
        ridge1.update(x, y=1.0)
        ridge2.update(x, y=1.0)

        # ridge2's A_inv should be larger because it was decayed
        # (multiplied by 1/0.5 = 2.0)
        assert np.isclose(ridge1.A_inv[1, 1], 1.0)
        assert np.isclose(ridge2.A_inv[1, 1], 2.0)


class TestLinUCBArmModel:
    """Tests for LinUCB arm model."""

    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.model = LinUCBArmModel(arm=1.2, n_features=3, alpha=1.0, l2_lambda=1.0, rng=self.rng)

    def test_initial_state(self):
        assert not self.model.is_fitted
        assert self.model.n_obs == 0
        assert self.model.arm == 1.2

    def test_score_before_update_is_nonnegative(self):
        """UCB score for unexplored arms should be positive."""
        x = np.array([1.0, 0.5, 0.3])
        score = self.model.score(x)
        # Exploitation term = 0, exploration term = alpha * sqrt(x A_inv x^T) > 0
        assert score >= 0.0

    def test_update_marks_as_fitted(self):
        x = np.array([1.0, 0.5, 0.3])
        self.model.update(x, reward=0.7)
        assert self.model.is_fitted
        assert self.model.n_obs == 1

    def test_score_increases_with_alpha(self):
        """Higher alpha should give higher exploration bonus."""
        x = np.array([1.0, 0.5, 0.3])
        model_low = LinUCBArmModel(arm=1.0, n_features=3, alpha=0.1, rng=self.rng)
        model_high = LinUCBArmModel(arm=1.0, n_features=3, alpha=5.0, rng=self.rng)
        assert model_high.score(x) > model_low.score(x)

    def test_reset(self):
        x = np.array([1.0, 0.5, 0.3])
        self.model.update(x, 0.7)
        self.model.reset()
        assert not self.model.is_fitted
        assert self.model.n_obs == 0

    def test_clone(self):
        x = np.array([1.0, 0.5, 0.3])
        self.model.update(x, 0.7)
        clone = self.model.clone()
        # Beta should be identical in clone
        np.testing.assert_array_equal(clone.beta, self.model.beta)
        # Modifying original should not affect clone
        self.model.update(x, 0.9)
        assert not np.allclose(clone.beta, self.model.beta)


class TestLinTSArmModel:
    """Tests for LinTS arm model."""

    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.model = LinTSArmModel(arm=1.0, n_features=3, v_sq=1.0, rng=self.rng)

    def test_score_is_stochastic(self):
        """LinTS scores should differ between calls (sampling-based)."""
        x = np.array([1.0, 0.5, 0.3])
        scores = [self.model.score(x) for _ in range(10)]
        # At least some scores should differ (unless extremely unlikely)
        assert len(set(round(s, 10) for s in scores)) > 1

    def test_update_and_fit(self):
        x = np.array([1.0, 0.5, 0.3])
        self.model.update(x, 0.8)
        assert self.model.is_fitted

    def test_higher_v_sq_more_variable(self):
        """Higher v_sq should produce higher variance in scores."""
        x = np.array([1.0, 0.5, 0.3])
        data = [
            (np.array([1.0, 0.5, 0.3]), 0.5),
            (np.array([0.3, 0.2, 0.8]), 0.7),
        ]

        model_low = LinTSArmModel(arm=1.0, n_features=3, v_sq=0.01, rng=np.random.default_rng(2))
        model_high = LinTSArmModel(arm=1.0, n_features=3, v_sq=10.0, rng=np.random.default_rng(3))

        for xi, yi in data:
            model_low.update(xi, yi)
            model_high.update(xi, yi)

        scores_low = [model_low.score(x) for _ in range(100)]
        scores_high = [model_high.score(x) for _ in range(100)]

        assert np.std(scores_high) > np.std(scores_low)


class TestThompsonArmModel:
    """Tests for context-free Thompson Sampling arm model."""

    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.model = ThompsonArmModel(arm=1.0, alpha_prior=1.0, beta_prior=1.0, rng=self.rng)

    def test_initial_mean_reward(self):
        """With equal priors, initial mean should be 0.5."""
        assert self.model.mean_reward == pytest.approx(0.5)

    def test_score_in_unit_interval(self):
        """Beta samples should be in [0, 1]."""
        for _ in range(100):
            s = self.model.score(None)
            assert 0.0 <= s <= 1.0

    def test_mean_shifts_with_rewards(self):
        """High rewards should push alpha up, increasing mean_reward."""
        for _ in range(50):
            self.model.update(None, reward=0.9)
        assert self.model.mean_reward > 0.7

    def test_reset_restores_prior(self):
        for _ in range(20):
            self.model.update(None, reward=0.9)
        self.model.reset()
        assert self.model.mean_reward == pytest.approx(0.5)
        assert self.model.n_obs == 0

    def test_context_is_ignored(self):
        """Score should not depend on the context vector."""
        rng = np.random.default_rng(99)
        model = ThompsonArmModel(arm=1.0, rng=rng)
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([9.9, 8.8, 7.7])
        # With same RNG state, scores should be drawn from the same distribution
        # (context doesn't matter). We just verify no exception is raised.
        model.score(x1)
        model.score(x2)

    def test_ips_weight_scaling(self):
        """IPS weight should scale the effective reward contribution."""
        model_normal = ThompsonArmModel(arm=1.0, rng=np.random.default_rng(1))
        model_weighted = ThompsonArmModel(arm=1.0, rng=np.random.default_rng(2))
        model_normal.update(None, 0.5, weight=1.0)
        model_weighted.update(None, 0.5, weight=2.0)
        # Weighted model should have a different posterior
        assert model_weighted._alpha != model_normal._alpha


class TestUCB1ArmModel:
    """Tests for UCB1 arm model."""

    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.model = UCB1ArmModel(arm=1.0, alpha=1.0, rng=self.rng)

    def test_unexplored_score_is_infinite(self):
        assert self.model.score(None, total_pulls=10) == float("inf")

    def test_score_after_one_pull(self):
        self.model.update(None, reward=0.5)
        score = self.model.score(None, total_pulls=5)
        assert math.isfinite(score)
        assert score > 0.5  # Mean + UCB bonus

    def test_higher_total_pulls_increases_score(self):
        """More total pulls increases the log(N) term → higher UCB bonus."""
        self.model.update(None, 0.5)
        score_low = self.model.score(None, total_pulls=10)
        score_high = self.model.score(None, total_pulls=1000)
        assert score_high > score_low

    def test_mean_reward(self):
        self.model.update(None, 0.6)
        self.model.update(None, 0.4)
        assert self.model.mean_reward == pytest.approx(0.5)

    def test_reset(self):
        self.model.update(None, 0.7)
        self.model.reset()
        assert self.model.n_obs == 0
        assert self.model.score(None, total_pulls=1) == float("inf")

    def test_weighted_updates_use_effective_pulls_for_mean(self):
        model = UCB1ArmModel(arm=1.0, alpha=1.0, rng=np.random.default_rng(1))
        model.update(None, reward=1.0, weight=10.0)
        model.update(None, reward=0.0, weight=1.0)
        assert model.mean_reward == pytest.approx(10.0 / 11.0, rel=1e-6)
