"""Shared pytest fixtures for the web module test suite."""

from __future__ import annotations

import random

import pytest

from web.worlds import create_world

ALL_POLICY_IDS = [
    "random",
    "epsilon_greedy",
    "ucb1",
    "thompson",
    "softmax",
    "linucb",
    "linucb_sw",
    "lints",
    "logistic_ucb",
    "gp_ucb",
    "bootstrapped_ensemble",
    "linucb_hybrid",
    "tree_ucb",
    "tree_ts",
    "cats",
]

ALL_WORLD_IDS = [
    "rural_clinic",
    "moviematch",
    "newsfeed",
    "shopsmart",
    "ridepilot",
    "gamebot",
    "labtrial",
]

CONTEXTUAL_POLICY_IDS = [
    "linucb",
    "linucb_sw",
    "lints",
    "logistic_ucb",
    "linucb_hybrid",
    "tree_ucb",
    "tree_ts",
]

CONTEXT_FREE_POLICY_IDS = [
    "random",
    "epsilon_greedy",
    "ucb1",
    "thompson",
    "softmax",
    "gp_ucb",
    "bootstrapped_ensemble",
]


@pytest.fixture(scope="session")
def seed_rng() -> random.Random:
    """Seeded RNG for deterministic test value generation."""
    return random.Random(0)


@pytest.fixture
def clinic_world():
    """Fresh rural_clinic world with seed=0."""
    w = create_world("rural_clinic")
    w.reset(seed=0)
    return w


@pytest.fixture(params=ALL_WORLD_IDS)
def any_world(request):
    """Parametrized fixture for all 7 worlds."""
    w = create_world(request.param)
    w.reset(seed=0)
    return w


@pytest.fixture(params=ALL_POLICY_IDS)
def any_policy_id(request):
    """Parametrized fixture for all 15 policy IDs."""
    return request.param


@pytest.fixture
def clinic_feature_order():
    """Feature order tuple for rural_clinic."""
    return ("symptom_severity", "comorbidity", "age_bucket")


@pytest.fixture
def contextual_arms():
    """Standard three-arm list for contextual tests."""
    return ["arm_a", "arm_b", "arm_c"]


@pytest.fixture
def known_context():
    """Simple known context for deterministic testing."""
    return {"step": 1, "f1": 0.5, "f2": 1.0, "f3": 0.0}


@pytest.fixture
def three_arms():
    """Three string arms."""
    return ["arm_1", "arm_2", "arm_3"]
