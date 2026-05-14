"""Tests for off-policy IPS/DR estimators and evaluation metrics."""

import numpy as np
import pytest

from coba.config import BanditConfig
from coba.evaluation import (
    EvalResult,
    doubly_robust_eval,
    ncis_eval,
    rejection_sampling_eval,
)
from coba.offpolicy import DoublyRobustUpdater, IPSConfig, IPSEstimator
from coba.router import ClusterRouter
from coba.types import PolicyType

ARMS = [1.0, 1.1, 1.2, 1.5]


def make_fitted_router(
    n: int = 300, n_features: int = 7, seed: int = 0, n_clusters: int = 3
) -> ClusterRouter:
    rng = np.random.default_rng(seed)
    contexts = rng.standard_normal((n, n_features))
    decisions = rng.choice(ARMS, size=n)
    rewards = rng.uniform(0, 1, n)
    cfg = BanditConfig(n_clusters=n_clusters, policy=PolicyType.LIN_UCB, seed=seed)
    router = ClusterRouter(arms=ARMS, n_features=n_features, config=cfg)
    router.fit(contexts, decisions, rewards)
    return router


class TestIPSEstimator:
    def test_unit_propensity_gives_unit_weights(self):
        propensities = np.ones(3)
        weights = IPSEstimator.compute_weights(propensities)
        np.testing.assert_array_almost_equal(weights, np.ones(3))

    def test_half_propensity_gives_double_weights(self):
        propensities = np.array([0.5, 0.5])
        weights = IPSEstimator.compute_weights(propensities)
        np.testing.assert_array_almost_equal(weights, np.array([2.0, 2.0]))

    def test_weights_clipped_at_min(self):
        """Very small propensities should be clipped."""
        propensities = np.array([1e-10])  # very small
        config = IPSConfig(clip_min=1e-4, clip_max=None)
        weights = IPSEstimator.compute_weights(propensities, config)
        assert weights[0] == pytest.approx(1.0 / 1e-4)

    def test_weights_clipped_at_max(self):
        propensities = np.array([0.001])
        config = IPSConfig(clip_min=1e-4, clip_max=5.0)
        weights = IPSEstimator.compute_weights(propensities, config)
        assert weights[0] == pytest.approx(5.0)

    def test_dr_rewards_formula(self):
        """DR = rhat + (r - rhat) / p when rhat and p are exact."""
        rewards = np.array([0.8])
        propensities = np.array([0.5])
        rhat = np.array([0.6])
        config = IPSConfig(clip_min=1e-4, clip_max=None, use_dr=True)
        dr = IPSEstimator.compute_dr_rewards(rewards, propensities, rhat, config)
        # DR = 0.6 + (0.8 - 0.6) / 0.5 = 0.6 + 0.4 = 1.0
        assert dr[0] == pytest.approx(1.0)

    def test_dr_rewards_clip_max(self):
        """DR values should be clipped when clip_max is set."""
        rewards = np.array([1.0])
        propensities = np.array([0.001])  # very small → large IPS correction
        rhat = np.array([0.0])
        config = IPSConfig(clip_min=1e-4, clip_max=5.0, use_dr=True)
        dr = IPSEstimator.compute_dr_rewards(rewards, propensities, rhat, config)
        assert abs(dr[0]) <= 5.0


class TestDoublyRobustUpdater:
    def test_fit_offline(self):
        router = make_fitted_router(n=200, n_features=7, seed=0, n_clusters=3)
        # Clear the fitted state to test fit_offline
        router.is_fitted = False
        rng = np.random.default_rng(0)
        contexts = rng.standard_normal((200, 7))
        decisions = rng.choice(ARMS, size=200)
        rewards = rng.uniform(0, 1, 200)
        propensities = np.full(200, 0.25)

        updater = DoublyRobustUpdater(router)
        updater.fit_offline(contexts, decisions, rewards, propensities)
        assert router.is_fitted

    def test_update_from_logs(self):
        rng = np.random.default_rng(1)
        n = 200
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)

        router = make_fitted_router(n_clusters=3, n_features=7, seed=1)
        updater = DoublyRobustUpdater(router)
        updater.fit_offline(contexts, decisions, rewards, propensities)
        before = router._total_pulls

        # incremental update
        updater.update_from_logs(contexts[:50], decisions[:50], rewards[:50], propensities[:50])
        assert router._total_pulls > before

    def test_dr_requires_reward_estimates(self):
        rng = np.random.default_rng(2)
        n = 100
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)

        router = make_fitted_router(n_clusters=3, n_features=7, seed=2)
        config = IPSConfig(use_dr=True)
        updater = DoublyRobustUpdater(router, config=config)

        with pytest.raises(ValueError, match="reward_estimates"):
            updater.fit_offline(contexts, decisions, rewards, propensities)


class TestRejectionSamplingEval:
    def test_result_type(self):
        router = make_fitted_router()
        rng = np.random.default_rng(5)
        n = 100
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        result = rejection_sampling_eval(router, contexts, decisions, rewards)
        assert isinstance(result, EvalResult)

    def test_utilization_rate_in_range(self):
        router = make_fitted_router()
        rng = np.random.default_rng(6)
        n = 200
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        result = rejection_sampling_eval(router, contexts, decisions, rewards)
        assert 0.0 <= result.utilization_rate <= 1.0

    def test_unfitted_router_raises(self):
        cfg = BanditConfig(n_clusters=3, policy=PolicyType.LIN_UCB)
        router = ClusterRouter(arms=ARMS, n_features=7, config=cfg)
        with pytest.raises(ValueError, match="fitted"):
            rejection_sampling_eval(
                router,
                np.zeros((10, 7)),
                np.array([1.0] * 10),
                np.ones(10),
            )


class TestDoublyRobustEval:
    def test_result_type(self):
        router = make_fitted_router()
        rng = np.random.default_rng(7)
        n = 100
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)
        reward_estimates = rng.uniform(0, 1, n)
        result = doubly_robust_eval(
            router, contexts, decisions, rewards, propensities, reward_estimates
        )
        assert isinstance(result, EvalResult)
        assert result.method == "doubly_robust"
        assert isinstance(result.estimated_reward, float)

    def test_accepts_target_reward_estimates(self):
        router = make_fitted_router()
        rng = np.random.default_rng(17)
        n = 100
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)
        reward_estimates = rng.uniform(0, 1, n)
        target_reward_estimates = rng.uniform(0, 1, n)
        result = doubly_robust_eval(
            router,
            contexts,
            decisions,
            rewards,
            propensities,
            reward_estimates,
            target_reward_estimates=target_reward_estimates,
        )
        assert isinstance(result, EvalResult)

    def test_length_mismatch_raises(self):
        router = make_fitted_router()
        rng = np.random.default_rng(18)
        n = 20
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)
        reward_estimates = rng.uniform(0, 1, n - 1)
        with pytest.raises(ValueError, match="same length"):
            doubly_robust_eval(
                router,
                contexts,
                decisions,
                rewards,
                propensities,
                reward_estimates,
            )


class TestNCISEval:
    def test_result_type(self):
        rng = np.random.default_rng(8)
        n = 100
        policy_scores = rng.uniform(0.1, 1.0, n)
        logging_scores = rng.uniform(0.1, 1.0, n)
        rewards = rng.uniform(0, 1, n)
        result = ncis_eval(policy_scores, logging_scores, rewards)
        assert isinstance(result, EvalResult)
        assert result.method == "ncis"

    def test_estimated_reward_in_range(self):
        rng = np.random.default_rng(9)
        n = 500
        policy_scores = rng.uniform(0.1, 1.0, n)
        logging_scores = rng.uniform(0.1, 1.0, n)
        rewards = rng.uniform(0, 1, n)
        result = ncis_eval(policy_scores, logging_scores, rewards)
        # Estimated reward should be in a reasonable range
        assert -10 <= result.estimated_reward <= 10

    def test_all_above_clip_max_raises(self):
        """If all weights exceed clip_max, should raise ValueError."""
        n = 10
        policy_scores = np.full(n, 1000.0)
        logging_scores = np.full(n, 0.001)
        rewards = np.ones(n)
        with pytest.raises(ValueError, match="NCIS: no samples below clip_max"):
            ncis_eval(policy_scores, logging_scores, rewards, clip_max=1.0)


class TestEvalResult:
    def test_utilization_rate_zero_division(self):
        result = EvalResult("test", 0.0, 0, 0)
        assert result.utilization_rate == 0.0

    def test_repr(self):
        result = EvalResult("rejection_sampling", 0.75, 50, 100)
        r = repr(result)
        assert "rejection_sampling" in r
        assert "50/100" in r


class TestDoublyRobustUpdaterDRMode:
    """Tests for DoublyRobustUpdater with use_dr=True (previously uncovered path)."""

    def test_update_from_logs_dr_mode(self) -> None:
        """update_from_logs with DR mode should apply DR-corrected rewards."""
        rng = np.random.default_rng(10)
        n = 100
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)
        reward_estimates = rng.uniform(0, 1, n)

        router = make_fitted_router(n_clusters=3, n_features=7, seed=10)
        config = IPSConfig(use_dr=True)
        updater = DoublyRobustUpdater(router, config=config)

        # Initial fit (non-DR) to warm up the router
        updater_plain = DoublyRobustUpdater(router)
        updater_plain.fit_offline(contexts, decisions, rewards, propensities)
        before_pulls = router._total_pulls

        # Now incremental update with DR correction
        updater.update_from_logs(
            contexts[:30],
            decisions[:30],
            rewards[:30],
            propensities[:30],
            reward_estimates=reward_estimates[:30],
        )
        assert router._total_pulls > before_pulls

    def test_update_from_logs_dr_none_reward_estimates_raises(self) -> None:
        """DR mode without reward_estimates should raise unless fallback is enabled."""
        rng = np.random.default_rng(11)
        n = 100
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)

        router = make_fitted_router(n_clusters=3, n_features=7, seed=11)
        # First fit to enable partial_fit
        router.fit(contexts, decisions, rewards)
        before_pulls = router._total_pulls

        config = IPSConfig(use_dr=True)
        updater = DoublyRobustUpdater(router, config=config)
        with pytest.raises(ValueError, match="reward_estimates"):
            updater.update_from_logs(
                contexts[:20],
                decisions[:20],
                rewards[:20],
                propensities[:20],
                reward_estimates=None,
            )
        assert router._total_pulls == before_pulls

    def test_update_from_logs_dr_none_reward_estimates_allows_fallback(self) -> None:
        """If explicitly enabled, DR mode may fallback to IPS."""
        rng = np.random.default_rng(12)
        n = 100
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)

        router = make_fitted_router(n_clusters=3, n_features=7, seed=12)
        router.fit(contexts, decisions, rewards)
        before_pulls = router._total_pulls

        config = IPSConfig(use_dr=True, allow_ips_fallback_when_dr_missing=True)
        updater = DoublyRobustUpdater(router, config=config)
        updater.update_from_logs(
            contexts[:20],
            decisions[:20],
            rewards[:20],
            propensities[:20],
            reward_estimates=None,
        )
        assert router._total_pulls > before_pulls
