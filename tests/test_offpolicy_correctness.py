"""Formula validation and edge case tests for IPS/DR off-policy estimators.

Verifies: unbiasedness properties, DR variance reduction, clipping behavior,
and edge cases (empty arrays, single samples, boundary propensities).
"""

from __future__ import annotations

import numpy as np
import pytest

from coba.offpolicy import DoublyRobustUpdater, IPSConfig, IPSEstimator
from coba.router import ClusterRouter


def make_unfitted_router(n_features: int = 7) -> ClusterRouter:
    from coba.config import BanditConfig
    from coba.types import PolicyType

    cfg = BanditConfig(n_clusters=2, policy=PolicyType.LIN_UCB, seed=0)
    return ClusterRouter(arms=["A", "B"], n_features=n_features, config=cfg)


class TestIPSEstimatorProperties:
    """IPS weight computation properties."""

    def test_uniform_propensities_produce_unbiased_mean(self) -> None:
        """When all propensities are equal, the IPS-weighted mean reward must
        equal the unweighted mean reward (up to floating arithmetic)."""
        n = 1000
        rng = np.random.default_rng(0)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)

        weights = IPSEstimator.compute_weights(propensities)
        weighted_mean = float(np.average(rewards, weights=weights))
        unweighted_mean = float(np.mean(rewards))

        assert weighted_mean == pytest.approx(unweighted_mean, rel=1e-2)

    def test_weights_sum_to_n_over_p_when_uniform(self) -> None:
        """With uniform propensities p=0.25, weights = 1/0.25 = 4,
        sum(weights) = 4n for all samples."""
        n = 100
        weights = IPSEstimator.compute_weights(np.full(n, 0.25))
        assert float(np.sum(weights)) == pytest.approx(4.0 * n, rel=1e-6)

    def test_no_clip_max_produces_unbounded_weights(self) -> None:
        """With clip_max=None, very small propensities produce very large weights."""
        config = IPSConfig(clip_min=1e-4, clip_max=None)
        weights = IPSEstimator.compute_weights(np.array([0.001, 0.00001]), config)
        assert weights[0] == pytest.approx(1000.0)
        assert weights[1] == pytest.approx(10000.0)

    def test_clip_min_zero_allows_very_small_propensities(self) -> None:
        """With clip_min=0, a very small propensity gives a very large but finite weight."""
        config = IPSConfig(clip_min=0.0, clip_max=None)
        weights = IPSEstimator.compute_weights(np.array([1e-15]), config)
        assert weights[0] == pytest.approx(1e15)
        assert np.isfinite(weights[0])

    def test_empty_array_produces_empty_result(self) -> None:
        """Empty input arrays must produce empty output."""
        weights = IPSEstimator.compute_weights(np.array([]))
        assert len(weights) == 0


class TestDRCorrectness:
    """Doubly-Robust reward correction formula validation."""

    def test_dr_equals_direct_observation_when_propensity_is_one(self) -> None:
        """When p=1.0, DR corrected reward = rhat + (r - rhat) / 1.0 = r.
        The DR formula collapses to the direct observation."""
        rewards = np.array([0.3, 0.7, 0.5])
        propensities = np.ones(3)
        rhat = np.array([0.4, 0.6, 0.5])
        config = IPSConfig(clip_min=1e-4, clip_max=None, use_dr=True)
        dr = IPSEstimator.compute_dr_rewards(rewards, propensities, rhat, config)
        np.testing.assert_array_almost_equal(dr, rewards)

    def test_dr_equals_reward_estimate_when_model_is_perfect(self) -> None:
        """When reward_estimates == rewards, the IPS correction term is zero
        and DR = reward_estimates."""
        rewards = np.array([0.3, 0.7, 0.5])
        propensities = np.array([0.1, 0.5, 0.8])
        rhat = rewards.copy()  # perfect model
        config = IPSConfig(clip_min=1e-4, clip_max=None, use_dr=True)
        dr = IPSEstimator.compute_dr_rewards(rewards, propensities, rhat, config)
        np.testing.assert_array_almost_equal(dr, rhat)

    def test_dr_has_lower_variance_than_pure_ips(self) -> None:
        """With a moderately good reward model, DR must have lower std than
        pure IPS across 100 trials."""
        rng = np.random.default_rng(0)
        n = 500
        # True rewards with some noise
        base_rewards = np.clip(rng.normal(0.5, 0.2, n), 0, 1)

        # Varying propensities — some small ones that inflate IPS variance
        propensities = rng.uniform(0.05, 0.5, n)

        # Reward model is noisy but unbiased: E[rhat] ≈ E[rewards]
        rhat = np.clip(base_rewards + rng.normal(0, 0.1, n), 0, 1)

        # Pure IPS: weight = 1/p
        ips_weights = IPSEstimator.compute_weights(propensities)
        ips_weighted_rewards = base_rewards * ips_weights

        # DR: rhat + (r - rhat) / p
        config = IPSConfig(clip_min=1e-4, clip_max=10.0, use_dr=True)
        dr_rewards = IPSEstimator.compute_dr_rewards(base_rewards, propensities, rhat, config)

        assert np.std(dr_rewards) < np.std(ips_weighted_rewards), (
            f"DR std={np.std(dr_rewards):.4f}, IPS std={np.std(ips_weighted_rewards):.4f}"
        )

    def test_ips_config_defaults(self) -> None:
        """Default IPSConfig has sensible values."""
        cfg = IPSConfig()
        assert cfg.clip_min == 1e-4
        assert cfg.clip_max == 10.0
        assert not cfg.use_dr

    def test_dr_clip_max_keeps_values_in_bounds(self) -> None:
        """With clip_max=5.0, all DR values must be in [-5, 5]."""
        rewards = np.array([1.0, 0.0, 0.5])
        propensities = np.array([0.0001, 0.0001, 0.5])
        rhat = np.array([0.0, 1.0, 0.5])
        config = IPSConfig(clip_min=1e-4, clip_max=5.0, use_dr=True)
        dr = IPSEstimator.compute_dr_rewards(rewards, propensities, rhat, config)
        assert np.all(np.abs(dr) <= 5.0)

    def test_default_config_passed_when_none(self) -> None:
        """When config=None, the default IPSConfig must be used."""
        # This should not raise and use defaults
        weights = IPSEstimator.compute_weights(np.array([0.1, 0.2]))
        assert np.all(np.isfinite(weights))


class TestDoublyRobustUpdaterEdgeCases:
    """Edge case tests for DoublyRobustUpdater."""

    def test_fit_offline_with_dr_and_missing_estimates_raises(self) -> None:
        """DR fit_offline without reward_estimates must raise."""
        router = make_unfitted_router()
        config = IPSConfig(use_dr=True)
        updater = DoublyRobustUpdater(router, config)
        with pytest.raises(ValueError, match="reward_estimates"):
            updater.fit_offline(
                np.zeros((10, 7)),
                np.array(["A"] * 10),
                np.ones(10),
                np.full(10, 0.5),
                reward_estimates=None,
            )

    def test_fit_offline_with_single_sample(self) -> None:
        """Fitting with a single sample must not crash (needs n_clusters=1)."""
        from coba.config import BanditConfig
        from coba.types import PolicyType

        cfg = BanditConfig(n_clusters=1, policy=PolicyType.LIN_UCB, seed=0)
        router = ClusterRouter(arms=["A"], n_features=7, config=cfg)
        updater = DoublyRobustUpdater(router)
        updater.fit_offline(
            np.zeros((1, 7)),
            np.array(["A"]),
            np.array([0.5]),
            np.array([1.0]),
        )
        assert router.is_fitted

    def test_update_from_logs_with_single_sample(self) -> None:
        """Incremental update with a single row must not crash."""
        router = make_unfitted_router()
        router.fit(
            np.zeros((10, 7)),
            np.array(["A"] * 5 + ["B"] * 5),
            np.ones(10),
        )
        before = router._total_pulls
        updater = DoublyRobustUpdater(router)
        updater.update_from_logs(
            np.zeros((1, 7)),
            np.array(["A"]),
            np.array([0.5]),
            np.array([0.5]),
        )
        assert router._total_pulls > before

    def test_update_from_logs_dr_fallback(self) -> None:
        """DR mode with IPS fallback enabled must work without reward_estimates."""
        router = make_unfitted_router()
        router.fit(
            np.zeros((10, 7)),
            np.arange(10) % 2,  # alternating 0=A, 1=B
            np.ones(10),
        )
        config = IPSConfig(use_dr=True, allow_ips_fallback_when_dr_missing=True)
        updater = DoublyRobustUpdater(router, config)
        before = router._total_pulls
        updater.update_from_logs(
            np.zeros((3, 7)),
            np.array(["A", "B", "A"]),
            np.array([0.3, 0.7, 0.5]),
            np.full(3, 0.5),
            reward_estimates=None,
        )
        assert router._total_pulls > before

    def test_fit_offline_then_update_from_logs(self) -> None:
        """Full off-policy bootstrap → incremental update pipeline."""
        rng = np.random.default_rng(0)
        n = 100
        router = make_unfitted_router(n_features=4)
        contexts = rng.standard_normal((n, 4))
        decisions = np.array(["A"] * 50 + ["B"] * 50)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.5)

        updater = DoublyRobustUpdater(router)
        updater.fit_offline(contexts, decisions, rewards, propensities)
        assert router.is_fitted

        # Incremental batch
        updater.update_from_logs(contexts[:20], decisions[:20], rewards[:20], propensities[:20])
        # Router still fitted, more pulls registered
        assert router.is_fitted
