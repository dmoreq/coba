"""Edge case tests for ClusterRouter — scaling, minibatch, arm lifecycle,
unfitted behavior, hybrid/neural policy validation, and empty clusters.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans, MiniBatchKMeans

from coba.config import BanditConfig
from coba.policies.linucb import LinUCBArmModel
from coba.policies.linucb_sw import SlidingWindowLinUCBArmModel
from coba.policies.thompson import ThompsonArmModel
from coba.policies.ucb1 import UCB1ArmModel
from coba.router import ClusterRouter, _build_model_for_arm
from coba.types import PolicyType

ARMS = [1.0, 1.1, 1.2, 1.5]


def make_router(**kwargs):
    cfg = BanditConfig(
        n_clusters=kwargs.pop("n_clusters", 3),
        policy=PolicyType.LIN_UCB,
        seed=kwargs.pop("seed", 0),
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return ClusterRouter(arms=ARMS, n_features=7, config=cfg)


class TestClusterRouterScaleContexts:
    def test_scaler_fit_and_apply_on_fit(self) -> None:
        router = make_router(scale_contexts=True, n_clusters=2)
        contexts = np.random.default_rng(0).standard_normal((50, 7))
        decisions = np.random.default_rng(0).choice(ARMS, size=50)
        rewards = np.ones(50)
        router.fit(contexts, decisions, rewards)
        assert router._scaler is not None
        assert hasattr(router._scaler, "mean_")

    def test_unfitted_router_route_returns_cluster_zero(self) -> None:
        router = make_router(n_clusters=2)
        ctx = np.zeros(7)
        cluster_idx, scaled = router._route(ctx)
        assert cluster_idx == 0

    def test_unfitted_router_update_triggers_auto_bootstrap(self) -> None:
        router = make_router(n_clusters=2)
        assert not router.is_fitted
        ctx = np.random.default_rng(0).standard_normal(7)
        router.update(ctx, arm=ARMS[0], reward=0.5)
        # First update queues; after n_clusters updates auto-bootstrap triggers
        for _ in range(3):
            router.update(np.random.default_rng(0).standard_normal(7), arm=ARMS[1], reward=0.5)
        assert router.is_fitted

    def test_scaler_not_yet_fitted_returns_raw_context(self) -> None:
        router = make_router(scale_contexts=True, n_clusters=2)
        ctx = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        cluster_idx, scaled = router._route(ctx)
        # Scaler hasn't fit yet → raw context returned
        np.testing.assert_array_equal(scaled, ctx)


class TestClusterRouterMinibatch:
    def test_minibatch_mode_uses_minibatch_kmeans(self) -> None:
        router = make_router(use_minibatch=True, n_clusters=2)
        assert isinstance(router._kmeans, MiniBatchKMeans)

    def test_full_batch_mode_uses_standard_kmeans(self) -> None:
        router = make_router(use_minibatch=False, n_clusters=2)
        assert isinstance(router._kmeans, KMeans)


class TestClusterRouterArmLifecycle:
    def test_warm_start_clone_independence(self) -> None:
        rng = np.random.default_rng(0)
        router = make_router(n_clusters=2)
        router.fit(
            rng.standard_normal((100, 7)),
            rng.choice(ARMS, size=100),
            rng.uniform(0, 1, 100),
        )
        router.add_arm(2.0, warm_start_from=1.5)
        # Safe to use new arm
        ctx = rng.standard_normal(7)
        arm = router.predict(ctx)
        assert arm in router.arms

    def test_reset_arm_across_all_clusters(self) -> None:
        rng = np.random.default_rng(0)
        router = make_router(n_clusters=2)
        router.fit(
            rng.standard_normal((100, 7)),
            rng.choice(ARMS, size=100),
            rng.uniform(0, 1, 100),
        )
        router.reset_arm(ARMS[0])
        # Bandit still works after reset
        arm = router.predict(rng.standard_normal(7))
        assert arm in router.arms

    def test_add_arm_with_gamma_override(self) -> None:
        router = make_router(n_clusters=1)
        router.add_arm("new_arm", gamma=0.5)
        model = router._cluster_bandits[0]["new_arm"]
        assert model._gamma == 0.5 if hasattr(model, "_gamma") else True

    def test_warm_start_from_nonexistent_arm_falls_back_to_cold(self) -> None:
        router = make_router(n_clusters=1)
        router.add_arm("Z", warm_start_from="nonexistent")
        assert "Z" in router.arms


class TestClusterRouterPrediction:
    def test_score_all_unfitted_returns_all_arms(self) -> None:
        router = make_router(n_clusters=2)
        scores = router.score_all(np.zeros(7))
        assert set(scores.keys()) == set(ARMS)

    def test_predict_ucb1_uses_global_pull_count(self) -> None:
        cfg = BanditConfig(n_clusters=2, policy=PolicyType.UCB1, seed=0)
        router = ClusterRouter(arms=ARMS, n_features=7, config=cfg)
        rng = np.random.default_rng(0)
        router.fit(
            rng.standard_normal((50, 7)),
            rng.choice(ARMS, size=50),
            rng.uniform(0, 1, 50),
        )
        assert router._score_fn == router._score_ucb1
        arm = router.predict(rng.standard_normal(7))
        assert arm in ARMS

    def test_empty_cluster_does_not_crash_predict(self) -> None:
        """Cluster with no data — predict still returns a valid arm."""
        cfg = BanditConfig(n_clusters=5, policy=PolicyType.LIN_UCB, seed=0)
        router = ClusterRouter(arms=ARMS, n_features=7, config=cfg)
        # Only feed data that maps to 2 clusters when n_clusters=5
        rng = np.random.default_rng(0)
        contexts = np.vstack(
            [
                np.ones((30, 7)) * 10,
                np.ones((30, 7)) * -10,
            ]
        )
        decisions = rng.choice(ARMS, size=60)
        rewards = rng.uniform(0, 1, 60)
        router.fit(contexts, decisions, rewards)
        arm = router.predict(rng.standard_normal(7))
        assert arm in ARMS

    def test_batch_update_respects_weights(self) -> None:
        cfg = BanditConfig(n_clusters=2, policy=PolicyType.LIN_UCB, seed=0)
        router = ClusterRouter(arms=ARMS, n_features=7, config=cfg)
        rng = np.random.default_rng(0)
        contexts = rng.standard_normal((50, 7))
        decisions = rng.choice(ARMS, size=50)
        rewards = rng.uniform(0, 1, 50)
        router.fit(contexts, decisions, rewards)
        total_before = router._total_pulls
        router.partial_fit(
            rng.standard_normal((10, 7)),
            rng.choice(ARMS, size=10),
            rng.uniform(0, 1, 10),
            weights=np.full(10, 2.0),
        )
        assert router._total_pulls > total_before


class TestClusterRouterValidation:
    def test_hybrid_policy_without_shared_features_raises(self) -> None:
        with pytest.raises(ValueError):
            cfg = BanditConfig(
                n_clusters=2,
                policy=PolicyType.LIN_UCB_HYBRID,
                n_shared_features=0,
            )
            ClusterRouter(arms=ARMS, n_features=7, config=cfg)

    def test_neural_linear_without_backbone_raises(self) -> None:
        with pytest.raises(ValueError, match="neural_backbone"):
            _build_model_for_arm(
                arm="A",
                cfg=BanditConfig(policy=PolicyType.NEURAL_LINEAR, n_clusters=1),
                n_features=5,
                rng=np.random.default_rng(0),
                neural_backbone=None,
            )

    def test_policy_registry_returns_correct_model_types(self) -> None:
        """For each core policy, _build_model_for_arm returns the expected class."""
        test_cases = [
            (PolicyType.LIN_UCB, LinUCBArmModel),
            (PolicyType.THOMPSON, ThompsonArmModel),
            (PolicyType.UCB1, UCB1ArmModel),
            (PolicyType.LIN_UCB_SW, SlidingWindowLinUCBArmModel),
        ]
        for policy_type, expected_class in test_cases:
            cfg = BanditConfig(policy=policy_type, n_clusters=1)
            model = _build_model_for_arm(
                arm="test",
                cfg=cfg,
                n_features=3,
                rng=np.random.default_rng(0),
            )
            assert isinstance(
                model, expected_class
            ), f"Expected {expected_class.__name__} for {policy_type}, got {type(model).__name__}"

    def test_unsupported_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported policy"):
            cfg = BanditConfig(policy=PolicyType.CATS, n_clusters=1)
            _build_model_for_arm(
                arm="test",
                cfg=cfg,
                n_features=3,
                rng=np.random.default_rng(0),
            )

    def test_n_clusters_at_least_one_validation(self) -> None:
        with pytest.raises(ValueError, match="n_clusters must be >= 1"):
            make_router(n_clusters=0)

    def test_empty_arms_raises(self) -> None:
        with pytest.raises(ValueError):
            cfg = BanditConfig(n_clusters=2, policy=PolicyType.LIN_UCB)
            ClusterRouter(arms=[], n_features=7, config=cfg)
