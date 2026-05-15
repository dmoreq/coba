"""Integration tests for tree-ensemble policies in ClusterBandit."""

import numpy as np

from coba import ClusterBandit
from coba.config import BanditConfig
from coba.router import _build_model_for_arm
from coba.types import PolicyType
from coba.policies.tree_ensemble import RandomForestTSArmModel, RandomForestUCBArmModel


def _dataset(n: int = 60, d: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    contexts = rng.standard_normal((n, d))
    arms = np.where(contexts[:, 0] > 0, "a", "b")
    rewards = np.where(arms == "a", 0.8, 0.3) + rng.normal(0, 0.02, n)
    return contexts, arms, np.clip(rewards, 0, 1)


class TestTreeEnsemblePolicyTypes:
    def test_policy_types_exist(self) -> None:
        assert PolicyType.RANDOM_FOREST_UCB.value == "random_forest_ucb"
        assert PolicyType.RANDOM_FOREST_TS.value == "random_forest_ts"

    def test_config_has_random_forest_defaults(self) -> None:
        cfg = BanditConfig()
        assert cfg.rf_n_estimators == 50
        assert cfg.rf_max_depth == 6
        assert cfg.rf_min_samples_leaf == 1
        assert cfg.rf_max_obs == 1000
        assert cfg.rf_min_uncertainty > 0


class TestRouterFactoryTreeEnsembles:
    def test_factory_builds_random_forest_ucb_model(self) -> None:
        cfg = BanditConfig(policy=PolicyType.RANDOM_FOREST_UCB, rf_n_estimators=7)
        model = _build_model_for_arm("a", cfg, n_features=4, rng=np.random.default_rng(0))
        assert isinstance(model, RandomForestUCBArmModel)
        assert model.n_estimators == 7

    def test_factory_builds_random_forest_ts_model(self) -> None:
        cfg = BanditConfig(policy=PolicyType.RANDOM_FOREST_TS, rf_n_estimators=7)
        model = _build_model_for_arm("a", cfg, n_features=4, rng=np.random.default_rng(0))
        assert isinstance(model, RandomForestTSArmModel)
        assert model.n_estimators == 7


class TestClusterBanditTreeEnsembles:
    def test_random_forest_ucb_offline_then_decide(self) -> None:
        contexts, decisions, rewards = _dataset()
        cfg = BanditConfig(policy=PolicyType.RANDOM_FOREST_UCB, n_clusters=2, rf_n_estimators=7)
        bandit = ClusterBandit(arms=["a", "b"], n_features=4, config=cfg)
        bandit.fit_offline(contexts, decisions, rewards)
        decision = bandit.decide(contexts[0])
        assert decision.chosen_arm in {"a", "b"}
        assert all(np.isfinite(v) or np.isinf(v) for v in decision.all_scores.values())

    def test_random_forest_ts_offline_then_decide(self) -> None:
        contexts, decisions, rewards = _dataset()
        cfg = BanditConfig(policy=PolicyType.RANDOM_FOREST_TS, n_clusters=2, rf_n_estimators=7)
        bandit = ClusterBandit(arms=["a", "b"], n_features=4, config=cfg)
        bandit.fit_offline(contexts, decisions, rewards)
        decision = bandit.decide(contexts[0])
        assert decision.chosen_arm in {"a", "b"}

    def test_random_forest_ucb_online_update(self) -> None:
        contexts, decisions, rewards = _dataset(n=20)
        cfg = BanditConfig(policy=PolicyType.RANDOM_FOREST_UCB, n_clusters=1, rf_n_estimators=5)
        bandit = ClusterBandit(arms=["a", "b"], n_features=4, config=cfg)
        bandit.fit_offline(contexts[:10], decisions[:10], rewards[:10])
        decision = bandit.decide(contexts[11])
        bandit.update(contexts[11], decision.chosen_arm, reward=0.7)
        assert bandit.is_fitted

    def test_config_copy_preserves_random_forest_fields(self) -> None:
        cfg = BanditConfig(policy=PolicyType.RANDOM_FOREST_UCB, rf_n_estimators=9, rf_max_obs=13)
        bandit = ClusterBandit(arms=["a", "b"], n_features=4, config=cfg)
        assert bandit._config.rf_n_estimators == 9
        assert bandit._config.rf_max_obs == 13
