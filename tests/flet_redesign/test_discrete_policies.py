"""Tests for phase-1 discrete policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from web.policies import (
    EpsilonGreedyPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy,
    UCB1Policy,
)


@pytest.mark.parametrize(
    ("policy_cls", "kwargs"),
    [
        (EpsilonGreedyPolicy, {"epsilon": 0.2}),
        (UCB1Policy, {"alpha": 1.0}),
        (ThompsonSamplingPolicy, {"prior_alpha": 1.0, "prior_beta": 1.0}),
        (SoftmaxPolicy, {"tau": 0.2}),
    ],
)
def test_policy_seed_replay_matches_after_reset(policy_cls: Any, kwargs: dict[str, Any]) -> None:
    policy = policy_cls(seed=7, **kwargs)
    arms = ["a", "b", "c"]

    seq_first = [policy.select_arm(context={}, arms=arms) for _ in range(6)]
    for idx, arm in enumerate(seq_first):
        policy.update(context={"step": idx}, arm=arm, reward=1.0 if arm == "a" else 0.0)

    policy.reset()
    seq_second = [policy.select_arm(context={}, arms=arms) for _ in range(6)]
    assert seq_first == seq_second


def test_epsilon_greedy_rejects_invalid_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedyPolicy(epsilon=1.5)


def test_ucb1_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        UCB1Policy(alpha=0.0)


def test_thompson_rejects_invalid_priors() -> None:
    with pytest.raises(ValueError, match="prior_alpha"):
        ThompsonSamplingPolicy(prior_alpha=0.0, prior_beta=1.0)


def test_softmax_rejects_invalid_tau() -> None:
    with pytest.raises(ValueError, match="tau"):
        SoftmaxPolicy(tau=0.0)


@pytest.mark.parametrize(
    "policy",
    [
        EpsilonGreedyPolicy(epsilon=0.1, seed=1),
        UCB1Policy(alpha=1.0, seed=1),
        ThompsonSamplingPolicy(seed=1),
        SoftmaxPolicy(tau=0.2, seed=1),
    ],
)
def test_all_discrete_policies_require_non_empty_arms(policy: Any) -> None:
    with pytest.raises(ValueError, match="at least one arm"):
        policy.select_arm(context=None, arms=[])


def test_ucb1_covers_unseen_arms_first() -> None:
    policy = UCB1Policy(alpha=1.0, seed=0)
    arms: Sequence[str] = ["a", "b", "c"]
    chosen: list[str] = []
    for _ in range(3):
        arm = policy.select_arm(context=None, arms=arms)
        chosen.append(arm)
        policy.update(context=None, arm=arm, reward=0.0)
    # First three pulls should cover all arms exactly once (order can vary).
    assert set(chosen) == set(arms)
