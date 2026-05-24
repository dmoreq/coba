"""Epsilon-greedy policy for discrete arms."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from web.contracts import BanditPolicy, DebugSnapshotProvider


class EpsilonGreedyPolicy(BanditPolicy[Any, Any], DebugSnapshotProvider):
    """Classic epsilon-greedy with deterministic RNG seeding."""

    def __init__(self, epsilon: float = 0.1, seed: int = 0) -> None:
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon
        self._seed = seed
        self._rng = random.Random(seed)
        self._reward_sum: dict[Any, float] = {}
        self._pulls: dict[Any, int] = {}

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._reward_sum.clear()
        self._pulls.clear()

    def select_arm(self, context: Any, arms: Sequence[Any]) -> Any:
        _ = context
        if not arms:
            raise ValueError("EpsilonGreedyPolicy requires at least one arm")
        arm_list = list(arms)
        self._ensure_arms(arm_list)

        if self._rng.random() < self.epsilon:
            return self._rng.choice(arm_list)

        return max(arm_list, key=lambda arm: self._mean_reward(arm))

    def update(self, context: Any, arm: Any, reward: float) -> None:
        _ = context
        self._ensure_arms([arm])
        self._pulls[arm] += 1
        self._reward_sum[arm] += reward

    def _ensure_arms(self, arms: Sequence[Any]) -> None:
        for arm in arms:
            if arm not in self._pulls:
                self._pulls[arm] = 0
                self._reward_sum[arm] = 0.0

    def _mean_reward(self, arm: Any) -> float:
        pulls = self._pulls[arm]
        if pulls == 0:
            return 0.0
        return self._reward_sum[arm] / float(pulls)

    def get_debug_snapshot(self) -> dict[str, Any]:
        arms = {}
        for arm in self._pulls:
            arms[str(arm)] = {
                "pulls": self._pulls[arm],
                "mean_reward": self._mean_reward(arm),
            }
        return {
            "policy": "epsilon_greedy",
            "epsilon": self.epsilon,
            "total_pulls": sum(self._pulls.values()),
            "arms": arms,
        }
