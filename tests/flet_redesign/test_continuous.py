"""Tests for continuous action policies and simulator."""

from __future__ import annotations

import random

from web.continuous import (
    CATSLikePolicy,
    ContinuousActionSpace,
    ContinuousSimulator,
    ContinuousWorld,
)
from web.policy_factory import build_policy


class SimpleContinuousWorld(ContinuousWorld):
    def __init__(self) -> None:
        self._rng = random.Random(0)

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(0 if seed is None else seed)

    def sample_context(self, step_index: int) -> dict[str, float]:
        return {"step": float(step_index)}

    def sample_reward(self, context: dict[str, float], action: float) -> float:
        _ = context
        # Peak reward near action=0.7
        noise = (self._rng.random() - 0.5) * 0.02
        return max(0.0, 1.0 - abs(action - 0.7) + noise)


def test_cats_like_policy_and_continuous_simulator() -> None:
    policy = CATSLikePolicy(action_space=ContinuousActionSpace(0.0, 1.0), exploration=0.3, seed=5)
    simulator = ContinuousSimulator(policy=policy, world=SimpleContinuousWorld(), seed=5)
    simulator.reset()
    steps = simulator.run_steps(20)
    assert len(steps) == 20
    assert simulator.cumulative_reward > 0.0
    assert all(0.0 <= step.action <= 1.0 for step in steps)


def test_continuous_policy_factory() -> None:
    policy = build_policy("cats", seed=1, params={"action_min": 0.0, "action_max": 2.0})
    assert isinstance(policy, CATSLikePolicy)
    action = policy.select_action()
    policy.update(action, reward=0.5)
    snapshot = policy.get_debug_snapshot()
    assert "best_action" in snapshot
