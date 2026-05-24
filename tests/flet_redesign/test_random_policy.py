"""Tests for the random baseline policy."""

from __future__ import annotations

import pytest

from web.policies import RandomPolicy


def test_random_policy_same_seed_same_sequence() -> None:
    arms = ["a", "b", "c"]
    p1 = RandomPolicy(seed=42)
    p2 = RandomPolicy(seed=42)
    seq1 = [p1.select_arm(context={}, arms=arms) for _ in range(8)]
    seq2 = [p2.select_arm(context={}, arms=arms) for _ in range(8)]
    assert seq1 == seq2


def test_random_policy_reset_restores_sequence() -> None:
    arms = ["a", "b", "c"]
    policy = RandomPolicy(seed=7)
    first = [policy.select_arm(context=None, arms=arms) for _ in range(5)]
    _ = [policy.select_arm(context=None, arms=arms) for _ in range(5)]
    policy.reset()
    replay = [policy.select_arm(context=None, arms=arms) for _ in range(5)]
    assert replay == first


def test_random_policy_requires_non_empty_arms() -> None:
    policy = RandomPolicy(seed=1)
    with pytest.raises(ValueError, match="at least one arm"):
        policy.select_arm(context={}, arms=[])
