"""Shared fixtures and utilities for the coba core test suite.

This conftest replaces the duplicated make_bandit(), make_router(), and
make_context() helpers that were scattered across test files, reducing
~80 lines of duplication.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from coba.bandit import ClusterBandit
from coba.config import BanditConfig
from coba.router import ClusterRouter
from coba.types import PolicyType


# ── Deterministic RNG ───────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def seed_rng() -> np.random.Generator:
    """Session-scoped deterministic random number generator."""
    return np.random.default_rng(42)


# ── Shared data fixtures ────────────────────────────────────────────────────


@pytest.fixture
def arms() -> list[str | int | float]:
    """Standard four-arm setup used by most tests."""
    return [1.0, 1.1, 1.2, 1.5]


@pytest.fixture(scope="session")
def n_features() -> int:
    return 7


@pytest.fixture
def context(n_features: int) -> np.ndarray:
    """A single 7-dimensional context vector used across tests."""
    return np.array([50.0, 10.0, 100.0, 5.0, 8.0, 1.0, 5.0], dtype=np.float64)


@pytest.fixture
def synthetic_logs(arms: list, n_features: int) -> dict[str, np.ndarray]:
    """200-sample synthetic logged data with contexts, decisions, rewards, propensities."""
    rng = np.random.default_rng(0)
    n = 200
    keyed = arms
    return {
        "contexts": rng.standard_normal((n, n_features)),
        "decisions": np.array(rng.choice(keyed, size=n)),
        "rewards": rng.uniform(0, 1, n),
        "propensities": np.full(n, 1.0 / len(arms), dtype=np.float64),
    }


# ── Shared helper factories ─────────────────────────────────────────────────


def make_bandit(
    arms: list | None = None,
    fitted: bool = False,
    policy: PolicyType = PolicyType.LIN_UCB,
    n_features: int = 7,
    n_clusters: int = 3,
    seed: int = 0,
    **kwargs: Any,
) -> ClusterBandit:
    """Create a ClusterBandit, optionally pre-fitted with synthetic data."""
    if arms is None:
        arms = [1.0, 1.1, 1.2, 1.5]
    bandit = ClusterBandit(
        arms=arms,
        n_features=n_features,
        policy=policy,
        n_clusters=n_clusters,
        seed=seed,
        **kwargs,
    )
    if fitted:
        rng = np.random.default_rng(seed)
        n = 200
        contexts = rng.standard_normal((n, n_features))
        decisions = rng.choice(arms, size=n)
        rewards = rng.uniform(0, 1, n)
        bandit.fit_offline(contexts, decisions, rewards)
    return bandit


def make_router(
    arms: list | None = None,
    n_clusters: int = 3,
    n_features: int = 7,
    policy: PolicyType = PolicyType.LIN_UCB,
    seed: int = 0,
) -> ClusterRouter:
    """Create a ClusterRouter (unfitted by default)."""
    if arms is None:
        arms = [1.0, 1.1, 1.2, 1.5]
    cfg = BanditConfig(n_clusters=n_clusters, policy=policy, seed=seed)
    return ClusterRouter(arms=arms, n_features=n_features, config=cfg)


def make_fitted_router(
    arms: list | None = None,
    n: int = 200,
    n_clusters: int = 3,
    n_features: int = 7,
    policy: PolicyType = PolicyType.LIN_UCB,
    seed: int = 0,
) -> ClusterRouter:
    """Create and fit a ClusterRouter with synthetic data."""
    if arms is None:
        arms = [1.0, 1.1, 1.2, 1.5]
    rng = np.random.default_rng(seed)
    contexts = rng.standard_normal((n, n_features))
    decisions = rng.choice(arms, size=n)
    rewards = rng.uniform(0, 1, n)
    cfg = BanditConfig(n_clusters=n_clusters, policy=policy, seed=seed)
    router = ClusterRouter(arms=arms, n_features=n_features, config=cfg)
    router.fit(contexts, decisions, rewards)
    return router


# ── Assertion utilities ─────────────────────────────────────────────────────


def assert_array_almost_equal(actual: np.ndarray, desired: np.ndarray, decimal: int = 6) -> None:
    """Wrapper around np.testing.assert_array_almost_equal with a readable name."""
    np.testing.assert_array_almost_equal(actual, desired, decimal=decimal)
