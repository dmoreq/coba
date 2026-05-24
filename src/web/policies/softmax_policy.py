"""Softmax exploration policy."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from web.contracts import BanditPolicy, DebugSnapshotProvider


class SoftmaxPolicy(BanditPolicy[Any, Any], DebugSnapshotProvider):
    """Softmax policy using running per-arm means as logits."""

    def __init__(self, tau: float = 0.2, seed: int = 0) -> None:
        if tau <= 0.0:
            raise ValueError("tau must be > 0")
        self.tau = tau
        self._seed = seed
        self._rng = random.Random(seed)
        self._reward_sum: dict[Any, float] = {}
        self._pulls: dict[Any, int] = {}
        self._last_probs: dict[str, float] = {}

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._reward_sum.clear()
        self._pulls.clear()
        self._last_probs.clear()

    def select_arm(self, context: Any, arms: Sequence[Any]) -> Any:
        _ = context
        if not arms:
            raise ValueError("SoftmaxPolicy requires at least one arm")
        arm_list = list(arms)
        self._ensure_arms(arm_list)

        values = [self._mean_reward(arm) / self.tau for arm in arm_list]
        max_val = max(values)
        exp_vals = [math.exp(val - max_val) for val in values]
        total = sum(exp_vals)
        probs = [val / total for val in exp_vals]
        probs_dict = {str(arm): prob for arm, prob in zip(arm_list, probs)}
        self._last_probs = probs_dict

        draw = self._rng.random()
        cum_prob = 0.0
        for arm, prob in zip(arm_list, probs):
            cum_prob += prob
            if draw <= cum_prob:
                return arm
        return arm_list[-1]

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
                "probability": self._last_probs.get(str(arm), 0.0),
            }
        return {
            "policy": "softmax",
            "tau": self.tau,
            "arms": arms,
        }
