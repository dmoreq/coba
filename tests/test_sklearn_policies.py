import numpy as np
import pytest
from sklearn.linear_model import SGDRegressor

from coba.bandit import ClusterBandit
from coba.policies.sklearn_models import (
    BootstrappedTSArmModel,
    BootstrappedUCBArmModel,
    EpsilonGreedyArmModel,
)
from coba.types import PolicyType


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def base_estimator():
    return SGDRegressor(random_state=42)


def test_epsilon_greedy_arm_model(rng, base_estimator):
    model = EpsilonGreedyArmModel(
        arm="test_arm", rng=rng, base_estimator=base_estimator, epsilon=0.0
    )

    assert not model.is_fitted

    # Should return exploration value if not fitted
    score = model.score(np.array([1.0, 2.0]))
    assert score > 100.0  # Exploration returns a large value (1e3 to 1e4)

    # Train
    model.update(np.array([1.0, 2.0]), 0.5)
    assert model.is_fitted

    # Now should predict a real value (not randomly large) since epsilon=0.0
    score2 = model.score(np.array([1.0, 2.0]))
    assert score2 < 100.0

    # Batch update
    model.update_batch(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 0.8]))

    model.reset()
    assert not model.is_fitted


def test_bootstrapped_ts_arm_model(rng, base_estimator):
    model = BootstrappedTSArmModel(
        arm="test_arm", rng=rng, base_estimator=base_estimator, n_bootstraps=3
    )

    assert not model.is_fitted
    assert len(model.models) == 3

    model.update(np.array([1.0, 2.0]), 0.5)
    assert model.is_fitted

    score = model.score(np.array([1.0, 2.0]))
    assert isinstance(score, float)

    # Batch update
    model.update_batch(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 0.8]))

    model.reset()
    assert not model.is_fitted


def test_bootstrapped_ucb_arm_model(rng, base_estimator):
    model = BootstrappedUCBArmModel(
        arm="test_arm",
        rng=rng,
        base_estimator=base_estimator,
        n_bootstraps=5,
        percentile=90.0,
    )

    assert not model.is_fitted

    model.update(np.array([1.0, 2.0]), 0.5)
    assert model.is_fitted

    score = model.score(np.array([1.0, 2.0]))
    assert isinstance(score, float)

    model.reset()
    assert not model.is_fitted


def test_cluster_bandit_with_epsilon_greedy(base_estimator):
    bandit = ClusterBandit(
        arms=[1.0, 1.1],
        n_features=2,
        policy=PolicyType.EPSILON_GREEDY,
        n_clusters=1,
        base_estimator=base_estimator,
        epsilon=0.1,
    )

    # Fit offline
    bandit.fit_offline(
        contexts=np.array([[1.0, 2.0], [3.0, 4.0]]),
        decisions=np.array([1.0, 1.1]),
        rewards=np.array([0.5, 0.8]),
    )

    assert bandit.is_fitted
    decision = bandit.decide(np.array([1.0, 2.0]))
    assert decision.chosen_arm in [1.0, 1.1]

    # Update online
    bandit.update(np.array([1.0, 2.0]), 1.0, 0.7)


def test_cluster_bandit_with_bootstrapped_ts(base_estimator):
    bandit = ClusterBandit(
        arms=[1.0, 1.1],
        n_features=2,
        policy=PolicyType.BOOTSTRAPPED_TS,
        n_clusters=2,
        base_estimator=base_estimator,
        n_bootstraps=5,
    )

    bandit.fit_offline(
        contexts=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        decisions=np.array([1.0, 1.1, 1.0, 1.1]),
        rewards=np.array([0.5, 0.8, 0.1, 0.9]),
    )

    assert bandit.is_fitted
    decision = bandit.decide(np.array([1.0, 2.0]))
    assert decision.chosen_arm in [1.0, 1.1]


def test_cluster_bandit_with_bootstrapped_ucb(base_estimator):
    bandit = ClusterBandit(
        arms=[1.0, 1.1],
        n_features=2,
        policy=PolicyType.BOOTSTRAPPED_UCB,
        n_clusters=2,
        base_estimator=base_estimator,
        n_bootstraps=5,
    )

    bandit.fit_offline(
        contexts=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        decisions=np.array([1.0, 1.1, 1.0, 1.1]),
        rewards=np.array([0.5, 0.8, 0.1, 0.9]),
    )

    assert bandit.is_fitted
    decision = bandit.decide(np.array([1.0, 2.0]))
    assert decision.chosen_arm in [1.0, 1.1]


def test_bootstrapped_ts_gamma_method(rng: np.random.Generator, base_estimator: object) -> None:
    """Gamma bootstrap method should produce non-negative weights and still update."""
    model = BootstrappedTSArmModel(
        arm="gamma_arm",
        rng=rng,
        base_estimator=base_estimator,
        n_bootstraps=3,
        bootstrap_method="gamma",
    )
    # Single update should not raise
    model.update(np.array([1.0, 2.0]), 0.5)
    assert model.is_fitted

    # Batch update with gamma weights
    model.update_batch(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([0.5, 0.8]),
    )
    assert model.is_fitted


def test_bootstrapped_ts_optimistic_before_fit(
    rng: np.random.Generator, base_estimator: object
) -> None:
    """Before any update, _predict_all should return high optimistic values."""
    model = BootstrappedTSArmModel(
        arm="opt_arm",
        rng=rng,
        base_estimator=base_estimator,
        n_bootstraps=5,
    )
    # _predict_all is used internally by score(); call it directly for clarity
    preds = model._predict_all(np.array([1.0, 2.0]))

    assert len(preds) == 5
    # Optimistic scores should be in the large range [1e3, 1e4]
    assert (preds >= 1e3).all()
    assert (preds <= 1e4).all()


def test_epsilon_greedy_explores_when_forced(
    rng: np.random.Generator, base_estimator: object
) -> None:
    """With epsilon=1.0, EpsilonGreedy should always return an exploration score."""
    model = EpsilonGreedyArmModel(
        arm="explore_arm",
        rng=rng,
        base_estimator=base_estimator,
        epsilon=1.0,
    )
    # Train so is_fitted=True
    model.update(np.array([1.0, 2.0]), 0.5)
    assert model.is_fitted

    # With epsilon=1.0, should still return exploration score
    score = model.score(np.array([1.0, 2.0]))
    assert score >= 1e3
