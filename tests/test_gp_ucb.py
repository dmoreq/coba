"""Tests for GP-UCB arm model and ClusterBandit integration."""

import numpy as np
import pytest

from coba import ClusterBandit
from coba.policies.gp_ucb import GPUCBArmModel
from coba.types import PolicyType

N_FEATURES = 4
ARMS = ["A", "B", "C"]


def make_model(arm: str = "A", beta: float = 2.0, max_obs: int = 500) -> GPUCBArmModel:
    return GPUCBArmModel(arm=arm, beta=beta, length_scale=1.0, noise_var=0.1, max_obs=max_obs)


class TestGPUCBArmModel:
    def test_cold_start_returns_inf(self):
        m = make_model()
        assert m.score(np.zeros(N_FEATURES)) == float("inf")

    def test_not_fitted_before_update(self):
        m = make_model()
        assert not m.is_fitted

    def test_fitted_after_update(self):
        m = make_model()
        m.update(np.ones(N_FEATURES), reward=1.0)
        assert m.is_fitted

    def test_score_finite_after_update(self):
        m = make_model()
        x = np.array([1.0, 0.5, -0.3, 0.8])
        m.update(x, reward=0.9)
        score = m.score(np.zeros(N_FEATURES))
        assert np.isfinite(score)

    def test_score_format_mu_plus_beta_sigma(self):
        """Higher beta → higher score (wider UCB interval)."""
        m_low = make_model(beta=0.1)
        m_high = make_model(beta=5.0)
        x_obs = np.array([1.0, 0.0, 0.0, 0.0])
        x_query = np.array([0.0, 0.0, 0.0, 0.0])
        m_low.update(x_obs, reward=0.5)
        m_high.update(x_obs, reward=0.5)
        assert m_high.score(x_query) >= m_low.score(x_query)

    def test_reset_clears_observations(self):
        m = make_model()
        m.update(np.ones(N_FEATURES), reward=1.0)
        m.reset()
        assert not m.is_fitted
        assert m.score(np.zeros(N_FEATURES)) == float("inf")

    def test_max_obs_trims_fifo(self):
        m = make_model(max_obs=5)
        for i in range(10):
            m.update(np.array([float(i)] * N_FEATURES), reward=float(i) / 10)
        assert len(m._X) == 5
        assert len(m._y) == 5

    def test_update_batch(self):
        """update_batch (from BaseArmModel) works correctly."""
        m = make_model()
        xs = np.random.default_rng(0).standard_normal((3, N_FEATURES))
        rewards = np.array([0.3, 0.7, 0.5])
        weights = np.ones(3)
        m.update_batch(xs, rewards, weights)
        assert m.is_fitted
        assert len(m._X) == 3

    def test_weighted_update(self):
        """IPS weight is applied to the stored reward."""
        m = make_model()
        x = np.ones(N_FEATURES)
        m.update(x, reward=1.0, weight=0.5)
        # stored y = reward * weight = 0.5
        assert m._y[0] == pytest.approx(0.5)

    def test_cholesky_cache_rebuilt_on_dirty(self):
        m = make_model()
        x = np.ones(N_FEATURES)
        m.update(x, reward=1.0)
        assert m._dirty
        _ = m.score(np.zeros(N_FEATURES))
        assert not m._dirty
        m.update(x * 2, reward=0.5)
        assert m._dirty

    def test_multiple_observations_decrease_uncertainty(self):
        """More observations → lower posterior variance → tighter UCB interval."""
        rng = np.random.default_rng(42)
        x_query = rng.standard_normal(N_FEATURES)

        m1 = make_model()
        m1.update(rng.standard_normal(N_FEATURES), reward=0.5)
        score_1 = m1.score(x_query)

        m5 = make_model()
        for _ in range(20):
            m5.update(rng.standard_normal(N_FEATURES), reward=0.5)
        score_5 = m5.score(x_query)

        # More data → tighter bounds → lower score for the same query point
        # (assuming rewards are similar; UCB shrinks as σ shrinks)
        assert score_5 < score_1

    def test_clone_preserves_observations(self):
        m = make_model()
        x = np.ones(N_FEATURES)
        m.update(x, reward=0.8)
        c = m.clone()
        assert c.arm == m.arm
        assert len(c._X) == len(m._X)
        assert c.is_fitted


class TestGPUCBClusterBandit:
    def test_end_to_end_decide_update(self):
        bandit = ClusterBandit(
            arms=ARMS,
            n_features=N_FEATURES,
            policy=PolicyType.GP_UCB,
            n_clusters=2,
            gp_beta=2.0,
            gp_length_scale=1.0,
            gp_noise_var=0.1,
            gp_max_obs=100,
            seed=0,
        )
        rng = np.random.default_rng(0)
        ctx = rng.standard_normal(N_FEATURES)
        decision = bandit.decide(ctx)
        assert decision.chosen_arm in ARMS
        bandit.update(ctx, decision.chosen_arm, reward=0.7)

    def test_fit_then_decide(self):
        bandit = ClusterBandit(
            arms=ARMS,
            n_features=N_FEATURES,
            policy=PolicyType.GP_UCB,
            n_clusters=2,
            seed=0,
        )
        rng = np.random.default_rng(1)
        n = 30
        contexts = rng.standard_normal((n, N_FEATURES))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(size=n)
        bandit.fit_offline(contexts, decisions, rewards)
        ctx = rng.standard_normal(N_FEATURES)
        decision = bandit.decide(ctx)
        assert decision.chosen_arm in ARMS

    def test_gp_params_propagated(self):
        bandit = ClusterBandit(
            arms=ARMS,
            n_features=N_FEATURES,
            policy=PolicyType.GP_UCB,
            n_clusters=1,
            gp_beta=5.0,
            gp_max_obs=50,
            seed=0,
        )
        model = bandit._router._cluster_bandits[0][ARMS[0]]
        assert isinstance(model, GPUCBArmModel)
        assert model.beta == 5.0
        assert model.max_obs == 50
